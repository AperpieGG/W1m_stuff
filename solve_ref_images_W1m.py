#!/usr/bin/env python3
"""
Script to take a per field master catalog fits file and a
field ID and then find/solve all reference images for that
field and filter the catalog for photometry stars

Based on code from Paul Chote's rasa-reduce
"""
import os
import sys
import traceback
import warnings
import tempfile
import subprocess
import argparse as ap
import sep
import numpy as np
import pymysql
import matplotlib
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from matplotlib import pyplot as plt

matplotlib.use('Agg')


# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=line-too-long
# pylint: disable=missing-docstring
# pylint: disable=too-many-locals

def arg_parse():
    """
    Parse the command line arguments
    """
    p = ap.ArgumentParser("Solve AG references images for CASUTools")
    p.add_argument('cat_file',
                   help='Master fits catalog',
                   type=str)
    p.add_argument('ref_images',
                   help='Reference images to solve with given catalog',
                   type=str,
                   nargs='+',
                   default=[])
    p.add_argument('--indir',
                   help='location of input files',
                   default='.',
                   type=str)
    p.add_argument('--outdir',
                   help='location of output files',
                   default='.',
                   type=str)
    p.add_argument('--defocus',
                   help='manual override for defocus (mm)',
                   type=float,
                   default=None)
    p.add_argument('--force3rd',
                   help='force a 3rd order distortion polyfit',
                   action='store_true',
                   default=False)
    p.add_argument('--save_matched_cat',
                   help='output the matched catalog with basic photometry',
                   action='store_true',
                   default=False)
    p.add_argument('--scale_min',
                   help='Minimum plate scale in arcsec/px',
                   type=float,
                   default=3.5)
    p.add_argument('--scale_max',
                   help='Maximum plate scale in arcsec/px',
                   type=float,
                   default=4.5)
    return p.parse_args()


def _detect_objects_sep(data, background_rms, area_min, area_max,
                        detection_sigma, defocus_mm, trim_border=50):
    """
    Find objects in an image array using SEP

    Parameters
    ----------
    data : array
        Image array to source detect on
    background_rms
        Std of the sky background
    area_min : int
        Minimum number of pixels for an object to be valid
    area_max : int
        Maximum number of pixels for an object to be valid
    detection_sigma : float
        Number of sigma above the background for source detecting
    defocus_mm : float
        Level of defocus. Used to select kernel for source detect
    trim_border : int
        Number of pixels to exclude from the edge of the image array

    Returns
    -------
    objects : astropy Table
        A list of detected objects in astropy Table format

    Raises
    ------
    None
    """
    # set up some defocused kernels for sep
    kernel1 = np.array([[1, 1, 1, 1, 1],
                        [1, 2, 3, 2, 1],
                        [1, 3, 1, 3, 1],
                        [1, 2, 3, 2, 1],
                        [1, 1, 1, 1, 1]])
    kernel2 = np.array([[1, 1, 1, 1, 1, 1],
                        [1, 2, 3, 3, 2, 1],
                        [1, 3, 1, 1, 3, 1],
                        [1, 3, 1, 1, 3, 1],
                        [1, 2, 3, 3, 2, 1],
                        [1, 1, 1, 1, 1, 1]])
    kernel3 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 3, 3, 3, 3, 3, 3, 1],
                        [1, 3, 2, 2, 2, 2, 3, 1],
                        [1, 3, 2, 1, 1, 2, 3, 1],
                        [1, 3, 2, 1, 1, 2, 3, 1],
                        [1, 3, 2, 2, 2, 2, 3, 1],
                        [1, 3, 3, 3, 3, 3, 3, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1]])

    # check for defocus
    if defocus_mm >= 0.15 and defocus_mm < 0.3:
        print("Source detect using defocused kernel 1")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel1)
    elif defocus_mm >= 0.3 and defocus_mm < 0.5:
        print("Source detect using defocused kernel 2")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel2)
    elif defocus_mm >= 0.5:
        print("Source detect using defocused kernel 3")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel3)
    else:
        print("Source detect using default kernel")
        raw_objects = sep.extract(data, detection_sigma * background_rms, minarea=area_min)

    initial_objects = len(raw_objects)

    raw_objects = Table(raw_objects[np.logical_and.reduce([
        raw_objects['npix'] < area_max,
        # Filter targets near the edge of the frame
        raw_objects['xmin'] > trim_border,
        raw_objects['xmax'] < data.shape[1] - trim_border,
        raw_objects['ymin'] > trim_border,
        raw_objects['ymax'] < data.shape[0] - trim_border
    ])])

    print(detection_sigma * background_rms, initial_objects, len(raw_objects))

    # Astrometry.net expects 1-index pixel positions
    objects = Table()
    objects['X'] = raw_objects['x'] + 1
    objects['Y'] = raw_objects['y'] + 1
    objects['FLUX'] = raw_objects['cflux']
    objects.sort('FLUX')
    objects.reverse()
    return objects


def _wcs_from_table(objects, frame_shape, scale_low, scale_high, estimate_coord=None,
                    estimate_coord_radius=None, timeout=120):  # timeout=25
    """
    Attempt to calculate a WCS solution for a given table of object detections.

    Parameters
    ----------
    table : Astropy Table
        Contains columns X, Y, FLUX, sorted by FLUX descending
    frame_shape : array
        array of frame [height, width]
    scale_low : float
        Minimum plate scale in arcsecond/px
    scale_high : float
        Maximum plate scale in arcsecond/px
    estimate_coord : SkyCoord
        Estimated position of the field
    estimate_coord_radius : float
        Radius to search around estimated coords
    timeout : int
        Abort if processing takes this long
        Default 25

    Parameters
    ----------
    solution : dict
        Dictionary of WCS header keywords

    Raises
    ------
    None
    """
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            xyls_path = os.path.join(tempdir, 'scratch.xyls')
            wcs_path = os.path.join(tempdir, 'scratch.wcs')
            objects.write(xyls_path, format='fits')

            solve_field_path = None
            possible_paths = [
                '/opt/homebrew/bin/solve-field',
                '/usr/local/astrometry.net/bin/solve-field',
                '/usr/bin/solve-field'
            ]

            for path in possible_paths:
                if os.path.isfile(path):
                    solve_field_path = path
                    break

            if solve_field_path is None:
                raise FileNotFoundError("solve-field not found in known locations.")

            astrometry_args = [
                solve_field_path,
                '--no-plots',
                '--scale-units', 'arcsecperpix',
                '--no-tweak', '--no-remove-lines',  # '--no-fits2fits',
                '--scale-high', str(scale_high), '--scale-low', str(scale_low),
                '--width', str(frame_shape[1]), '--height', str(frame_shape[0]),
                xyls_path]

            if estimate_coord is not None and estimate_coord_radius is not None:
                astrometry_args += [
                    '--ra', str(estimate_coord.ra.to_value(u.deg)),
                    '--dec', str(estimate_coord.dec.to_value(u.deg)),
                    '--radius', str(estimate_coord_radius.to_value(u.deg)),
                ]
            else:
                print("No RA/Dec constraints applied.")

            subprocess.check_call(astrometry_args, cwd=tempdir,
                                  timeout=timeout)
            # stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL)

            wcs_ignore_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'IMAGEW', 'IMAGEH']
            solution = {}
            with open(wcs_path) as wcs_file:
                header = wcs_file.read()
                # ds9 will only accept newline-delimited keys
                # so we need to reformat the 80-character cards
                for line in [header[i:i + 80] for i in range(0, len(header), 80)]:
                    key = line[0:8].strip()
                    if '=' in line and key not in wcs_ignore_cards:
                        card = fits.Card.fromstring(line)
                        solution[card.keyword] = card.value
            return solution

    except Exception:
        print('Failed to update wcs with error:')
        traceback.print_exc(file=sys.stdout)
        return {}


def check_wcs_corners(wcs_header, objects, catalog, frame_shape, check_tile_size=512):
    """
    Sanity check the WCS solution in the corners of the detector

    Parameters
    ----------
    wcs_header : dict
        Dictionary containing WCS solution
    objects : Sep Objects
        Sep output for source detection
    catalog : Astropy Table
        Catalog od objects for comparing
    frame_shape : tuple
        Shape of the detector

    Returns
    -------
    header : dict
        Updated fits header with corners + centre performance

    Raises
    ------
    None
    """
    tile_check_regions = {
        'MATCH-TL': [0, check_tile_size, frame_shape[0] - check_tile_size, frame_shape[0]],
        'MATCH-TR': [frame_shape[1] - check_tile_size, frame_shape[1], frame_shape[0] - check_tile_size,
                     frame_shape[0]],
        'MATCH-BL': [0, check_tile_size, 0, check_tile_size],
        'MATCH-BR': [frame_shape[1] - check_tile_size, frame_shape[1], 0, check_tile_size],
        'MATCH-C': [(frame_shape[1] - check_tile_size) // 2, (frame_shape[1] + check_tile_size) // 2,
                    (frame_shape[0] - check_tile_size) // 2, (frame_shape[0] + check_tile_size) // 2]
    }

    wcs_x, wcs_y = WCS(wcs_header).all_world2pix(catalog['RA_CORR'], catalog['DEC_CORR'], 1)
    delta_x = np.abs(wcs_x - objects['X'])
    delta_y = np.abs(wcs_y - objects['Y'])
    delta_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)

    header = {}
    for k, v in tile_check_regions.items():
        check_objects = np.logical_and.reduce([
            objects['X'] > v[0],
            objects['X'] < v[1],
            objects['Y'] > v[2],
            objects['Y'] < v[3],
        ])

        median = np.median(delta_xy[check_objects])
        header[k] = -1 if np.isnan(median) else median

    return header, delta_xy


def fit_hdu_distortion(wcs_header, objects, catalog, force3rd):
    """
    Fit the image distortion parameters

    Parameters
    ----------
    wcs_header : dict
        WCS solution
    objects : Sep Objects
        Catalog of source detections
    catalog : Astropy Table
        Astrometric catalog
    force3rd : boolean
        Force a new 3rd order polynomial

    Returns
    -------
    fitted_header : dict
        Updated fits header with WCS solution

    Raises
    ------
    None
    """
    # The SIP paper's U and V coordinates are found by applying the core (i.e. CD matrix)
    # transformation to the RA and Dec, ignoring distortion; relative to CRPIX
    wcs = WCS(wcs_header)
    U, V = wcs.wcs_world2pix(catalog['RA_CORR'], catalog['DEC_CORR'], 1)
    U -= wcs_header['CRPIX1']
    V -= wcs_header['CRPIX2']

    # The SIP paper's u and v coordinates are simply the image coordinates relative to CRPIX
    u = objects['X'] - wcs_header['CRPIX1']
    v = objects['Y'] - wcs_header['CRPIX2']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Solve for f(u, v) = U - u
        f_init = polynomial_from_header(wcs_header, 'A', force3rd)
        f_fit = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        f_poly, _ = f_fit(f_init, u, v, U - u)

        # Solve for g(u, v) = V - v
        g_init = polynomial_from_header(wcs_header, 'B', force3rd)
        g_fit = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        g_poly, _ = g_fit(g_init, u, v, V - v)

    # Return a copy of the header with updated distortion coefficients
    fitted_header = wcs_header.copy()

    for c, a, b in zip(f_poly.param_names, f_poly.parameters, g_poly.parameters):
        fitted_header['A_' + c[1:]] = a
        fitted_header['B_' + c[1:]] = b

    fitted_header['A_ORDER'] = f_init.degree
    fitted_header['B_ORDER'] = g_init.degree
    fitted_header['CTYPE1'] = 'RA---TAN-SIP'
    fitted_header['CTYPE2'] = 'DEC--TAN-SIP'

    return fitted_header


def polynomial_from_header(wcs_header, axis, force3rd):
    """
    Get polynomial from the header

    Parameters
    ----------
    wcs_header : dict
        WCS solution
    axis : int
        Order of axis
    force3rd : boolean
        Force a new 3rd order polynomial

    Returns
    -------
    2D polynomial model

    Raises
    ------
    None
    """
    # Astrometry.net sometimes doesn't fit distortion terms!
    if axis + '_ORDER' not in wcs_header or force3rd:
        return models.Polynomial2D(degree=3)

    coeffs = {}
    for key in wcs_header:
        if key.startswith(axis + '_') and key != axis + '_ORDER':
            coeffs['c' + key[2:]] = wcs_header[key]

    return models.Polynomial2D(degree=wcs_header[axis + '_ORDER'], **coeffs)


def fit_zeropoint_polynomial(catalog, objects, exptime, degree=2, sigma=3.0):
    """
    Fit a model of the zero point

    Parameters
    ----------
    catalog : Astropy Table
        Astrometric catalog
    objects : Sep Objects
        Source detection catalog
    exptime : float
        Exposure time
    degree : int
        Order of fit
        Default 2
    sigma : float
        Sigma for clipping outliers
        Default 3.0

    Returns
    -------
    fit : model

    Raises
    ------
    None
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        model = models.Polynomial2D(degree=degree)
        fit = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=sigma)
        zp_delta_mag = catalog['Tmag'] + 2.5 * np.log10(objects['FLUX'] / exptime)
        return fit(model, objects['X'], objects['Y'], zp_delta_mag)


def calculate_defocus_level(image):
    """
    Take an image and determine the level of defocus

    Parameters
    ----------
    image : string
        Name of input file

    Returns
    -------
    defocus : float
        Defocus in mm

    Raises
    ------
    None
    """
    # get the time and camera_id from the image header
    with fits.open(image) as ff:
        obsstart = ff[0].header['OBSSTART'].replace('T', ' ')
        camera_id = ff[0].header['CAMERAID']
        focus = float(ff[0].header['FCSR_PHY'])

    # query the current focus position first
    qry = """
        SELECT value
        FROM config
        WHERE attribute='focuser.position'
        AND scope={}
        AND valid_from < '{}'
        """.format(camera_id, obsstart)

    conn = pymysql.connect(host='10.2.4.244', db='ngts_ops', user='pipe')
    try:
        with conn.cursor() as cur:
            cur.execute(qry)
            res = cur.fetchone()

        # this is a recent image if we get a hit here
        if res:
            configured_focus = float(res[0])
            print("Image focus: {:.2f}".format(focus))
            print("Configured focus: {:.2f}".format(configured_focus))
            defocus = abs(configured_focus - focus)
            print("Defocus: {:.2f}".format(defocus))
        # otherwise we're pulling up a previously configured focus value
        else:
            qry = """
                SELECT value
                FROM config_history
                WHERE attribute='focuser.position'
                AND scope={}
                AND valid_from < '{}'
                AND valid_until > '{}'
                """.format(camera_id, obsstart, obsstart)

            with conn.cursor() as cur:
                cur.execute(qry)
                res = cur.fetchone()

            if res:
                configured_focus = float(res[0])
                print("Image focus: {:.2f}".format(focus))
                print("Configured focus: {:.2f}".format(configured_focus))
                defocus = abs(configured_focus - focus)
                print("Defocus: {:.2f}".format(defocus))
            else:
                print("Image focus: {:.2f}".format(focus))
                print("No configured focus found, defaulting defocus to 0.0mm")
                defocus = 0.0
    finally:
        conn.close()

    return defocus


def prepare_frame(input_path, output_path, catalog, defocus, force3rd, save_matched_cat, scale_min, scale_max):
    """
    Prepare the frame for WCS solution. The output is the solved image

    Parameters
    ----------
    input_path : string
        Name of input file
    output_path : string
        Name of output file
    catalog : astropy Table
        Reference catalog
    defocus : float
        Amount of defocus in mm
    force3rd : boolean
        Force a new 3rd order poly for distortion fitting
    save_matched_cat : boolean
        Output the matched catalog for analysis, includes basic photometry
    scale_min : float
        Minimum plate scale in arcsec/px
    scale_max : float
        Maximum plate scale in arcsec/px

    Returns
    -------
    None

    Raises
    ------
    None
    """
    frame = fits.open(input_path)[0]
    try:
        frame_exptime = float(frame.header['EXPOSURE'])
    except KeyError:
        frame_exptime = float(frame.header['EXPTIME'])

    # cut the catalog to those with gaia_ids and Tmag <= 15
    gids = catalog['GAIA']
    tmag = catalog['Tmag']
    # make a cross-match mask
    cm_mask = np.where(((~np.isnan(gids)) & (tmag <= 16) & (tmag >= 10)))[0]
    # make a trimmed catalog for cross-matching
    catalog_cm = catalog[cm_mask]
    catalog_cm = catalog_cm[~np.isnan(catalog_cm['RA_CORR']) & ~np.isnan(catalog_cm['DEC_CORR'])]

    # Prepare image
    # check if we have an overscan to remove
    if frame.data.shape == (2048, 2088):
        frame_data = frame.data[:, 20:2068].astype(float)
    else:
        frame_data = frame.data.astype(float)
    frame_bg = sep.Background(frame_data)
    frame_data_corr = frame_data - frame_bg

    # save the image with the same format as the input
    output = fits.PrimaryHDU(frame_data.astype(np.uint16), header=frame.header)
    output.header['BACK-LVL'] = frame_bg.globalback
    output.header['BACK-RMS'] = frame_bg.globalrms

    area_min = 10
    area_max = 5000  # subject to be changed
    detection_sigma = 3
    zp_clip_sigma = 3

    try:
        estimate_coord = SkyCoord(ra=frame.header['TELRAD'],
                                  dec=frame.header['TELDECD'],
                                  unit=(u.deg, u.deg))
        estimate_coord_radius = 4 * u.deg

    except KeyError:
        try:
            estimate_coord = SkyCoord(ra=frame.header['CMD_RA'],
                                      dec=frame.header['CMD_DEC'],
                                      unit=(u.deg, u.deg))
            estimate_coord_radius = 4 * u.deg

        except KeyError:
            try:
                estimate_coord = SkyCoord(ra=frame.header['MNTRAD'],
                                          dec=frame.header['MNTDECD'],
                                          unit=(u.deg, u.deg))
                estimate_coord_radius = 4 * u.deg

            except KeyError:
                print('No RA/DEC found in header (TELRAD, CMD_RA, or MNTRAD), skipping!')
                return None, None, None, None

    # determine if an image is defocused
    # if defocused we need a new kernel for sep
    # we query the config tables for the best focus at that time
    # if the difference is > some fraction of a mm we use a new kernel
    if defocus is None:
        defocus = calculate_defocus_level(input_path)
    # else, take the value supplied and use that manually overriden defocus

    # store the defocus value in the header
    output.header['DEFOCUS'] = defocus

    # Detect all objects in the image and attempt a full-frame solution
    objects = _detect_objects_sep(frame_data_corr, frame_bg.globalrms, area_min,
                                  area_max, detection_sigma, defocus)

    wcs_header = _wcs_from_table(objects, frame_data.shape, scale_min,
                                 scale_max, estimate_coord, estimate_coord_radius)

    if not wcs_header:
        print('Failed to find initial WCS solution - aborting')
        return None, None, None, None

    # if it fails, skip the image and stick it in a bad folder
    try:
        object_ra, object_dec = WCS(wcs_header).all_pix2world(objects['X'],
                                                              objects['Y'], 1,
                                                              ra_dec_order=True)
        objects['RA'] = object_ra
        objects['DEC'] = object_dec

        cat_coords = SkyCoord(ra=catalog_cm['RA_CORR'] * u.degree,
                              dec=catalog_cm['DEC_CORR'] * u.degree)

        # Iteratively improve the cross-match, WCS fit, and ZP estimation
        i = 0
        MIN_MATCHES_FOR_FIT = 10   # tuneable: require at least this many good matches for fitting
        while True:
            # Cross-match vs the catalog so we can exclude false detections and improve our distortion fit
            object_coordinates = SkyCoord(ra=objects['RA'] * u.degree, dec=objects['DEC'] * u.degree)
            match_idx, sep2d, _ = object_coordinates.match_to_catalog_sky(cat_coords)
            matched_cat = catalog_cm[match_idx]

            # add check here for matches, try to catch bad catalog
            print("Number of matches (raw): {}".format(len(match_idx)))

            # compute world->pixel for catalog positions
            wcs_x, wcs_y = WCS(wcs_header).all_world2pix(matched_cat['RA_CORR'],
                                                         matched_cat['DEC_CORR'], 1)
            delta_x = np.abs(wcs_x - objects['X'])
            delta_y = np.abs(wcs_y - objects['Y'])
            delta_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)

            # create mask of objects we consider bad for ZP (blends or huge positional offset)
            zp_mask = np.logical_or(matched_cat.get('BLENDED', np.zeros(len(matched_cat), dtype=bool)),
                                    delta_xy > 10)

            # compute zp_delta_mag safely
            with np.errstate(invalid='ignore', divide='ignore'):
                zp_delta_mag = matched_cat['Tmag'] + 2.5 * np.log10(objects['FLUX'] / frame_exptime)

            # Safely compute sigma clipped stats: ensure there are unmasked values
            unmasked_idx = np.where(~zp_mask)[0]
            if unmasked_idx.size >= MIN_MATCHES_FOR_FIT:
                try:
                    zp_mean, _, zp_stddev = sigma_clipped_stats(zp_delta_mag, mask=zp_mask, sigma=zp_clip_sigma)
                except Exception as e:
                    print("sigma_clipped_stats failed:", e)
                    zp_mean, zp_stddev = np.nan, np.nan
            else:
                # Not enough unmasked matches to compute a robust zp
                print(f"Not enough unmasked matches for ZP stats: {unmasked_idx.size} (<{MIN_MATCHES_FOR_FIT})")
                zp_mean, zp_stddev = np.nan, np.nan

            # Now create the zp_filter only if zp_mean and zp_stddev are finite
            if np.isfinite(zp_mean) and np.isfinite(zp_stddev) and zp_stddev > 0:
                zp_filter = np.logical_and.reduce([
                    np.logical_not(zp_mask),
                    zp_delta_mag > zp_mean - zp_clip_sigma * zp_stddev,
                    zp_delta_mag < zp_mean + zp_clip_sigma * zp_stddev])
            else:
                # fallback: define zp_filter simply as unmasked points (or stricter) for the distortion fit
                zp_filter = ~zp_mask

            n_good = np.sum(zp_filter)
            print("Number of matches used for fit (zp_filter):", n_good)

            # If too few matched points remain, either relax criteria or break and accept solution without refinement
            if n_good < MIN_MATCHES_FOR_FIT:
                print(f"Too few matches ({n_good}) for refinement. Skipping distortion fit and ZP refinement for this frame.")
                # We still keep the current wcs_header (the astrometry.net solution)
                # Set zp_mean/zp_stddev to NaN if they weren't computed
                zp_mean = zp_mean if np.isfinite(zp_mean) else np.nan
                zp_stddev = zp_stddev if np.isfinite(zp_stddev) else np.nan
                break

            # how far away is the median cross match? Guard the median call
            if np.any(zp_filter):
                median_delta_xy = np.median(delta_xy[zp_filter])
            else:
                median_delta_xy = np.inf
            print("Median delta_xy: {}".format(median_delta_xy))

            before_match, _ = check_wcs_corners(wcs_header, objects[zp_filter],
                                                matched_cat[zp_filter], frame_data.shape)

            if i == 0:
                # nfilter, before stats
                line = '{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(np.sum(zp_filter),
                                                                      *before_match.values())
                print("Pre tweak stats: {}".format(line))

            # Only attempt distortion fit if we have enough good matches
            try:
                wcs_header = fit_hdu_distortion(wcs_header, objects[zp_filter], matched_cat[zp_filter],
                                                force3rd)
            except Exception as e:
                print("fit_hdu_distortion failed:", e)
                # Stop iterative refinement — accept current WCS
                break

            i += 1

            # Update object RA/DEC after fit
            objects['RA'], objects['DEC'] = WCS(wcs_header).all_pix2world(objects['X'],
                                                                          objects['Y'], 1,
                                                                          ra_dec_order=True)

            after_match, _ = check_wcs_corners(wcs_header, objects[zp_filter],
                                               matched_cat[zp_filter], frame_data.shape)
            match_improvement = np.max([before_match[x] - after_match[x] for x in after_match])
            if (i > 100) or ((match_improvement < 0.001) and (median_delta_xy < 0.5)):
                break

        if np.any(zp_filter):
            med_val = np.median(delta_xy[zp_filter])
        else:
            med_val = np.inf

        if med_val > 0.5:
            print(f"Skipping frame {input_path}: median delta_xy={med_val:.3f}")
            return None, None, None, None

        line2 = '{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(np.sum(zp_filter),
                                                               *after_match.values())
        print("Post tweak stats: {}".format(line2))

        # Assume the last cycle of the fit did not change the cross-match, so the zero point stats remain the same
        wcs_header['MAGZP_T'] = round(float(zp_mean) if np.isfinite(zp_mean) else np.nan, 3)
        wcs_header['MAGZP_Ts'] = round(float(zp_stddev) if np.isfinite(zp_stddev) else np.nan, 3)
        wcs_header['MAGZP_Tc'] = int(np.sum(zp_filter))

        updated_wcs_header, xy_residuals = check_wcs_corners(wcs_header, objects[zp_filter],
                                                             matched_cat[zp_filter], frame_data.shape)

        # update the header from the final fit
        wcs_header.update(updated_wcs_header)

        for k, v in wcs_header.items():
            output.header[k] = v
        hdu_list = [output]

        # Refine ZP using a 2d polynomial fit
        fit_filter = np.logical_not(np.logical_or(matched_cat['BLENDED'], delta_xy > 0.5))
        zp_poly, _ = fit_zeropoint_polynomial(matched_cat[fit_filter], objects[fit_filter], frame_exptime)

        # save the matched_cat for comparing fluxes to magnitudes
        if save_matched_cat:
            matched_cat_file = "{}_matched_cat.fits".format(output_path.split('.fits')[0])
            output_matched_cat(frame_data_corr, objects[zp_filter],
                               matched_cat[zp_filter], matched_cat_file)

    except Exception:
        print("{} failed WCS, skipping...\n".format(input_path))
        traceback.print_exc(file=sys.stdout)
        return None, None, None, None

    output.header['ZP_ORDER'] = zp_poly.degree
    for c, p in zip(zp_poly.param_names, zp_poly.parameters):
        output.header['ZP_' + c[1:]] = p

    output.header['ZPCALCN2'] = np.sum(fit_filter)

    # Version 4:
    output.header['REDVER'] = 7

    # output the updated solved fits image
    fits.HDUList(hdu_list).writeto(output_path, overwrite=True)

    return wcs_header, objects[zp_filter], matched_cat[zp_filter], xy_residuals


def output_matched_cat(data, sources, catalog, outfile):
    """
    Take the sources extracted and the matched catalog
    Do photometry and save the output for analysis

    Parameters
    ----------
    sources : sep.Extract object
        list of detected sources
    catalog : astropy.Table
        table of matched catalog objects
    outfile : string
        name of output matched catalog

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # do photometry in r=3 pixel radius aperture on detections
    flux3, fluxerr3, _ = sep.sum_circle(data, sources['X'], sources['Y'], 30, subpix=0, gain=1.0)

    # finally update the catalog header with whether a star has been on chip or not
    col_f3 = Column(name='flux3', data=flux3, dtype=np.float32)
    col_fe3 = Column(name='fluxerr3', data=fluxerr3, dtype=np.float32)

    # store the X and Y positions
    col_x = Column(name='source_x', data=sources['X'], dtype=np.float32)
    col_y = Column(name='source_y', data=sources['Y'], dtype=np.float32)

    # Try inserting the new column. If it's there already, replace it
    if 'flux3' in catalog.keys():
        catalog['flux3'] = flux3
    else:
        catalog.add_column(col_f3)

    if 'fluxerr3' in catalog.keys():
        catalog['fluxerr3'] = fluxerr3
    else:
        catalog.add_column(col_fe3)

    # put out X and Y so we can do better checks
    if 'source_x' in catalog.keys():
        catalog['source_x'] = sources['X']
    else:
        catalog.add_column(col_x)

    if 'source_y' in catalog.keys():
        catalog['source_y'] = sources['Y']
    else:
        catalog.add_column(col_y)

    # write out the matched catalog and photometry
    catalog.write(outfile, format='fits', overwrite=True)


def update_master_catalog(catalog, wcs_list, cat_file, ref_image):
    """
    Take the master catalog and determine which objects are
    on a list of images (given their wcs headers)

    Parameters
    ----------
    catalog : astropy Table
        The input master catalog
    wcs_list :
        List of WCS header info
    cat_file : string
        Name of the master catalog file to update
    ref_image : string
        Name of the reference image file
    Returns
    -------
    catalog : astropy Table
        Updated master catalog

    Raises
    ------
    None
    """

    # keep a mask of images that have been on silicon
    on_chip = np.zeros(len(catalog['RA_CORR']))

    # Default full unbinned image size
    FULL_WIDTH = 6252
    FULL_HEIGHT = 4176
    EDGE_BUFFER = 80

    # Open the reference image to read CAM-BIN
    with fits.open(ref_image) as hdul:
        header = hdul[0].header

        if 'CAM-BIN' not in header:
            raise ValueError("CAM-BIN keyword is missing in the header of the reference image.")

        try:
            cam_bin = int(header['CAM-BIN'])
            bin_x = bin_y = cam_bin
            print(f"Binning detected: {bin_x}x{bin_y}")
        except ValueError:
            raise ValueError(f"CAM-BIN value '{header['CAM-BIN']}' could not be parsed as an integer.")

    # Adjust image dimensions based on binning
    IMAGE_WIDTH = FULL_WIDTH // bin_x
    IMAGE_HEIGHT = FULL_HEIGHT // bin_y

    for w in wcs_list:
        if w is not None:
            # Convert RA/DEC to pixel positions using WCS
            x, y = WCS(w).all_world2pix(catalog['RA_CORR'], catalog['DEC_CORR'], 1)

            # Mask stars that lie safely within the chip boundaries
            mask = np.where(
                (x > EDGE_BUFFER) & (x < IMAGE_WIDTH - EDGE_BUFFER) &
                (y > EDGE_BUFFER) & (y < IMAGE_HEIGHT - EDGE_BUFFER)
            )[0]

            on_chip[mask] += 1

    # finally update the catalog header with whether a star has been on chip or not
    col = Column(name='on_chip', data=on_chip, dtype=np.int16)

    # Try inserting the new column. If it's there already, replace it
    if 'on_chip' in catalog.keys():
        catalog['on_chip'] = on_chip
    else:
        catalog.add_column(col)

    # write out the updated catalog
    catalog.write(cat_file, format='fits', overwrite=True)

    # return the catalog for use in producing imcore output
    return catalog, IMAGE_WIDTH, IMAGE_HEIGHT


def write_input_catalog(catalog, wcs_list, input_cat_file, IMAGE_WIDTH, IMAGE_HEIGHT):
    """
    Here we apply the cuts to the master catalog and
    then output the final input catalog for photometry

    imcore wants:
        Sequence_number int (source id, row order)
        RA              radians
        DEC             radians

    Parameters
    ----------
    catalog : astropy Table
        Master catalog
    wcs_list : list
    input_cat_file : string
        name of the output input catalog
    IMAGE_WIDTH : int
    IMAGE_HEIGHT : int

    Returns
    -------
    None

    Raises
    ------
    None
    """
    # apply the final cuts on input catalog currently, the following:
    # The target is brighter than Gmag=16 OR it's in the guest catalogs
    # AND it's on chip in any reference image

    on_chip = np.zeros(len(catalog['RA_CORR']))

    EDGE_BUFFER = 50  # padding from the edge

    for w in wcs_list:
        if w is not None:
            # Convert RA/DEC to pixel positions using WCS
            x, y = WCS(w).all_world2pix(catalog['RA_CORR'], catalog['DEC_CORR'], 1)

            # Mask stars that lie safely within the chip boundaries
            mask = np.where(
                (x > EDGE_BUFFER) & (x < IMAGE_WIDTH - EDGE_BUFFER) &
                (y > EDGE_BUFFER) & (y < IMAGE_HEIGHT - EDGE_BUFFER)
            )[0]

            on_chip[mask] += 1

    col = Column(name='on_chip', data=on_chip, dtype=np.int16)

    # Try inserting the new column. If it's there already, replace it
    if 'on_chip' in catalog.keys():
        catalog['on_chip'] = on_chip
    else:
        catalog.add_column(col)

    # mask = np.where((((catalog['Tmag'] < 16) & (catalog['on_chip'] == 1.0)) | (
    #         (catalog['on_chip'] == 1.0) & (catalog['guest'] == True))))[0]

    # Exclude stars with 'True' in the 'blended' column
    # blended_mask = np.array([not any(blend) for blend in catalog['blended']])

    mask = np.where((catalog['Tmag'] < 16) & (catalog['on_chip'] == 1.0) & (~catalog['BLENDED']))[0]

    # mask the catalog and the source indexes
    catalog_clipped = catalog[mask]
    print(f'Initial catalog length: {len(catalog)}')
    print(f'Blended, variable and faint stars removed: {len(catalog) - len(catalog_clipped)}')

    # create the columns required for photometry, in the correct format
    input_catalog = Table([catalog_clipped['GAIA'], catalog_clipped['Gmag'], catalog_clipped['TIC'],
                           catalog_clipped['Tmag'], catalog_clipped['RA_CORR'], catalog_clipped['DEC_CORR'],
                           catalog_clipped['on_chip'], catalog_clipped['BPmag'],
                           catalog_clipped['RPmag']])

    # finally save the input catalog
    try:
        input_catalog.write(input_cat_file, format='fits', overwrite=True)
        return input_catalog
    except Exception as e:
        print(f"Error writing input catalog: {e}")
        return None


def generate_region_file(catalog, cat_file):
    """
    Take the master catalog and output some DS9 region files

    Parameters
    ----------
    catalog : astropy Table
        Table of master catalog
    cat_file : name of the master cat file
        Used to generate region file names

    Returns
    -------
    None

    Raises
    ------
    None
    """
    region_header = ["# Region file format: DS9 version 4.1\n",
                     "global color=green dashlist=8 3 width=1 ",
                     "font=\"helvetica 10  normal roman\" ",
                     "select=1 highlite=1 dash=0 fixed=0 ",
                     "edit=1 move=1 delete=1 include=1 source=1\n",
                     "icrs\n"]

    master_region = "{}_master.reg".format(cat_file.split('.fits')[0])
    master = []

    for row in catalog:
        ra = row['RA_CORR']
        dec = row['DEC_CORR']
        on_chip = row['on_chip']
        tmag = row['Tmag']

        if on_chip and (tmag <= 16):
            colour = 'green'
            master.append("circle({},{},1\") # color={}\n".format(ra, dec, colour))
        else:
            colour = 'blue'
            master.append("point({},{}) # color={} point=x\n".format(ra, dec, colour))

    # write out the master region
    with open(master_region, 'w') as of:
        for row in region_header:
            of.write(row)
        for row in master:
            of.write(row)


def generate_input_region_file(input_catalog, inp):
    """
    Take the input catalog and output some DS9 region files

    Parameters
    ----------
    input_catalog : astropy Table
        Table of input catalog
    inp : name of the input cat file
        Used to generate region file names

    Returns
    -------
    None

    Raises
    ------
    None
    """
    region_header = ["# Region file format: DS9 version 4.1\n",
                     "global color=green dashlist=8 3 width=1 ",
                     "font=\"helvetica 10  normal roman\" ",
                     "select=1 highlite=1 dash=0 fixed=0 ",
                     "edit=1 move=1 delete=1 include=1 source=1\n",
                     "icrs\n"]

    input_region = "{}.reg".format(inp.split('.fits')[0])
    input = []

    for row in input_catalog:
        ra = row['RA_CORR']
        dec = row['DEC_CORR']
        on_chip = row['on_chip']
        tmag = row['Tmag']
        gaiamag = row['Gmag']

        if on_chip and (tmag <= 16):
            colour = 'green'
            input.append("circle({},{},1\") # color={}\n".format(ra, dec, colour))
        else:
            pass  # don't plot the non-on-chip stars

    # write out the input region
    with open(input_region, 'w') as of:
        for row in region_header:
            of.write(row)
        for row in input:
            of.write(row)


if __name__ == "__main__":
    args = arg_parse()

    # check for a catalog
    if not os.path.exists(args.cat_file):
        print("{} is missing, skipping...".format(args.cat_file))
        sys.exit(1)

    # we want to make a catalog that matches the source detect closely
    master_catalog = Table.read(args.cat_file)

    # store the WCS headers for the final check if objects appear on chip
    # at least once during the list of given reference images
    wcs_store = []
    for ref_image in args.ref_images:
        input_file = "{}/{}".format(args.indir, ref_image)
        output_file = "{}/{}".format(args.outdir, ref_image)
        base_name = output_file.split(".fits")[0]
        if os.path.exists(input_file) and os.path.exists(args.outdir):
            # store the final fitted WCS header
            final_wcs, objects_matched, catalog_matched, residuals = prepare_frame(input_file,
                                                                                   output_file,
                                                                                   master_catalog,
                                                                                   args.defocus,
                                                                                   args.force3rd,
                                                                                   args.save_matched_cat,
                                                                                   args.scale_min,
                                                                                   args.scale_max)
            wcs_store.append(final_wcs)

            if final_wcs is None:
                continue

            # plot diagnostics on the fitting here using the matched catalog/objects
            # from the source detection
            ref_x = final_wcs['CRPIX1']
            ref_y = final_wcs['CRPIX2']
            radial_distance = np.sqrt((np.array(objects_matched['X']) - ref_x) ** 2 +
                                      (np.array(objects_matched['Y']) - ref_y) ** 2)

            # generate vecotrs for quiver plot
            vector_scale = 100
            wcs_pos_x, wcs_pos_y = WCS(final_wcs).all_world2pix(catalog_matched['RA_CORR'],
                                                                catalog_matched['DEC_CORR'], 1)
            x_comp = (wcs_pos_x - objects_matched['X']) * vector_scale
            y_comp = (wcs_pos_y - objects_matched['Y']) * vector_scale

            with fits.open(input_file) as hdul:
                image_data = hdul[0].data
                ny, nx = image_data.shape

            fig_q, ax_q = plt.subplots(1, figsize=(10, 10))
            ax_q.set_title('WCS - Source Detect Positions (x{})'.format(vector_scale))
            ax_q.quiver(objects_matched['X'], objects_matched['Y'], x_comp, y_comp, units='xy')
            ax_q.set_xlim(0, nx)
            ax_q.set_ylim(0, ny)
            fig_q.tight_layout()
            fig_q.savefig('{}_quiver_plot.png'.format(base_name))

            fig, ax = plt.subplots(5, figsize=(10, 20))
            # plot a zoom into the residuals to 2 pixels
            ax[0].loglog(radial_distance, residuals, 'k.')
            ax[0].set_xlabel('Radial position from CRPIX (pix)')
            ax[0].set_ylabel('Delta XY (pix)')

            # plot a histogram of delta_xy
            ax[1].hist(residuals, bins=100, log=True, color='black')
            ax[1].set_ylabel('Frequency')
            ax[1].set_xlabel('Delta XY (pix)')

            # plot the residuals versus pmRA
            ax[2].semilogy(catalog_matched['pmRA'], residuals, 'k.')
            ax[2].set_xlabel('pmRA (mas)')
            ax[2].set_ylabel('Delta XY (pix)')

            # plot the residuals versus pmDEC
            ax[3].semilogy(catalog_matched['pmDE'], residuals, 'k.')
            ax[3].set_xlabel('pmDE (mas)')
            ax[3].set_ylabel('Delta XY (pix)')

            # plot residuals versus brightness
            ax[4].semilogy(catalog_matched['Tmag'], residuals, 'k.')
            ax[4].set_xlabel('T mag')
            ax[4].set_ylabel('Delta XY (pix)')

            fig.tight_layout()
            fig.savefig('{}_wcs_residuals.png'.format(base_name))

    # check if at least one reference image passed
    if len(args.ref_images) > wcs_store.count(None):
        # Check if the input catalog file already exists
        imcore_cat_name = "{}_input.fits".format(args.cat_file.split('.')[0])

        if os.path.exists(imcore_cat_name):
            print(f"{imcore_cat_name} already exists. Skipping generation and writing.")
        else:
            # update the master catalog with stars that are on chip
            master_catalog, IMAGE_WIDTH, IMAGE_HEIGHT = update_master_catalog(master_catalog, wcs_store,
                                                                              args.cat_file, args.ref_images[0])

            # Write out some region files from the catalog
            # 1. The entire master catalog
            # 2. The stars going into the photometry, highlight guests
            generate_region_file(master_catalog, args.cat_file)

            # Apply cuts and output the file for photometry
            input_catalog = write_input_catalog(master_catalog, wcs_store, imcore_cat_name, IMAGE_WIDTH, IMAGE_HEIGHT)

            # Generate the input region file
            generate_input_region_file(input_catalog, imcore_cat_name)

        sys.exit(0)
    else:
        print('No solved reference images')
        sys.exit(1)
