"""
Functions for handling on-sky or on chip coordinates
"""
import fnmatch
import glob
import json
import os
from datetime import datetime, timedelta
from astropy.io import fits
import sep
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
import astropy.units as u
from astropy.table import Table, hstack
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from astropy.time import Time

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=c-extension-no-member


def plot_images():
    # Set plot parameters
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    # Font and fontsize
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 14


def get_location():
    """
    Get the location of the observatory

    Parameters
    ----------
    None

    Returns
    -------
    loc : EarthLocation
        Location of the observatory

    Raises
    ------
    None
    """
    site_location = EarthLocation(
        lat=-24.615662 * u.deg,
        lon=-70.391809 * u.deg,
        height=2433 * u.m
    )

    return site_location


def get_light_travel_times(ra, dec, time_to_correct):
    """
    Get the light travel times to the helio- and
    barycentric

    Parameters
    ----------
    ra : str
        The Right Ascension of the target in hour-angle
        e.g. 16:00:00
    dec : str
        The Declination of the target in degrees
        e.g. +20:00:00
    time_to_correct : astropy.Time object
        The time of observation to correct. The astropy.Time
        object must have been initialised with an EarthLocation

    Returns
    -------
    ltt_bary : float
        The light travel time to the barycentre
    ltt_helio : float
        The light travel time to the heliocentric

    Raises
    ------
    None
    """
    target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ltt_bary = time_to_correct.light_travel_time(target)
    ltt_helio = time_to_correct.light_travel_time(target, 'heliocentric')
    return ltt_bary, ltt_helio


def get_catalog(filename, ext=0):
    """
    Read a fits image and header with fitsio

    Parameters
    ----------
    filename : str
        filename to load
    ext : int
        extension to load

    Returns
    -------
    data : array
        data from the corresponding extension
    header : fitsio.header.FITSHDR
        list from file header

    Raises
    ------
    None
    """
    data, header = fits.getdata(filename, header=True, ext=ext)
    return data, header


def convert_coords_to_pixels(cat, filenames):
    """
    Convert RA and DEC coordinates to pixel coordinates using the first image for each prefix.

    Parameters
    ----------
    cat : astropy.table.table.Table
        Catalog containing the RA and DEC coordinates
    filenames : list of str
        List of filenames corresponding to each prefix

    Returns
    -------
    x, y : numpy arrays
        Pixel coordinates

    Raises
    ------
    None
    """
    # Initialize lists to store pixel coordinates
    all_x = []
    all_y = []

    # Iterate over each set of filenames corresponding to each prefix
    for prefix_filenames in filenames:
        # Take the first filename for the current prefix
        filename = prefix_filenames[0]

        # Open the FITS file and extract the WCS information
        with fits.open(filename) as hdul:
            wcs_header = hdul[0].header

            # Convert the RA and DEC to pixel coordinates
            wcs = WCS(wcs_header)
            x, y = wcs.all_world2pix(cat['ra_deg_corr'], cat['dec_deg_corr'], 1)

            # Append the pixel coordinates to the lists
            all_x.append(x)
            all_y.append(y)

    # Convert lists to numpy arrays
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    return all_x, all_y


def _detect_objects_sep(data, background_rms, area_min, area_max,
                        detection_sigma, defocus_mm, trim_border=10):
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
    if 0.15 <= defocus_mm < 0.3:
        print("Source detect using defocused kernel 1")
        raw_objects = sep.extract(data, detection_sigma * background_rms,
                                  minarea=area_min, filter_kernel=kernel1)
    elif 0.3 <= defocus_mm < 0.5:
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


def find_max_pixel_value(data, x, y, radius):
    """
    Find the maximum pixel value in the image
    in a square around the aperture centre

    Parameters
    ----------
    data : array-like
        The image to search
    x : int
        X coordinate of the search box
    y : int
        Y coordinate of the search box
    radius : int
        The half width of the search box

    Returns
    -------
    max_pixel_value : int
        The maximum pixel value in the area provided

    Raises
    ------
    None
    """
    # Check if the region is within the image bounds
    if (y - radius < 0 or y + radius > data.shape[0] or
            x - radius < 0 or x + radius > data.shape[1]):
        print(f"Skipping star at ({x}, {y}) with radius {radius}: outside image bounds.")
        return 0  # or np.nan or any appropriate default value

    return round(data[int(y - radius):int(y + radius),
                 int(x - radius):int(x + radius)].ravel().max(), 2)


def wcs_phot(data, x, y, rsi, rso, aperture_radii, gain):
    """
    Take a corrected image array and extract photometry for a set of WCS driven
    X and Y pixel positions. Do this for a series of aperture radii and apply
    a gain correction to the photometry
    """
    col_labels = ["flux", "fluxerr", "flux_w_sky", "fluxerr_w_sky", "max_pixel_value"]
    Tout = None
    for r in aperture_radii:
        flux, fluxerr, _ = sep.sum_circle(data, x, y, r,
                                          subpix=0,
                                          bkgann=(rsi, rso),
                                          gain=gain)
        flux_w_sky, fluxerr_w_sky, _ = sep.sum_circle(data, x, y, r,
                                                      subpix=0,
                                                      gain=gain)
        # calculate the max pixel value in each aperture
        max_pixel_value = np.array([find_max_pixel_value(data, int(i), int(j), int(r + 1)) for i, j in zip(x, y)])
        # build this photometry into a table
        if Tout is None:
            Tout = Table([flux, fluxerr, flux_w_sky, fluxerr_w_sky, max_pixel_value],
                         names=tuple([f"{c}_{r}" for c in col_labels]))
        else:
            T = Table([flux, fluxerr, flux_w_sky, fluxerr_w_sky, max_pixel_value],
                      names=tuple([f"{c}_{r}" for c in col_labels]))
            # stack the new columns onto the RHS of the table
            Tout = hstack([Tout, T])
    return Tout


def find_current_night_directory(directory):
    """
    Find the directory for the current night based on the current date.
    If not found, use the current working directory.

    Parameters
    ----------
    directory : str
        Base path for the directory.

    Returns
    -------
    str
        Path to the current night directory.
    """
    previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    current_date_directory = os.path.join(directory, previous_date)
    return current_date_directory if os.path.isdir(current_date_directory) else os.getcwd()


def get_phot_files(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'phot_*.fits'):
            phot_files.append(filename)
    return phot_files


def get_rel_phot_files(directory):
    """
    Get photometry files with the pattern 'phot_*.fits' from the directory.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        List of photometry files matching the pattern.
    """
    phot_files = []
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, 'rel_phot_*_1.fits'):
            phot_files.append(filename)
    return phot_files


def read_phot_file(filename):
    """
    Read the photometry file.

    Parameters
    ----------
    filename : str
        Photometry file to read.

    Returns
    -------
    astropy.table.table.Table
        Table containing the photometry data.
    """
    # Read the photometry file here using fits or any other appropriate method
    try:
        with fits.open(filename) as ff:
            # Access the data in the photometry file as needed
            tab = ff[1].data
            return tab
    except Exception as e:
        print(f"Error reading photometry file {filename}: {e}")
        return None


def bin_time_flux_error(time, flux, error, bin_fact):
    """
    Use reshape to bin light curve data, clip under filled bins
    Works with 2D arrays of flux and errors

    Note: under filled bins are clipped off the end of the series

    Parameters
    ----------
    time : array         of times to bin
    flux : array         of flux values to bin
    error : array         of error values to bin
    bin_fact : int
        Number of measurements to combine

    Returns
    -------
    times_b : array
        Binned times
    flux_b : array
        Binned fluxes
    error_b : array
        Binned errors

    Raises
    ------
    None
    """
    n_binned = int(len(time) / bin_fact)
    clip = n_binned * bin_fact
    time_b = np.average(time[:clip].reshape(n_binned, bin_fact), axis=1)
    # determine if 1 or 2d flux/err inputs
    if len(flux.shape) == 1:
        flux_b = np.average(flux[:clip].reshape(n_binned, bin_fact), axis=1)
        error_b = np.sqrt(np.sum(error[:clip].reshape(n_binned, bin_fact) ** 2, axis=1)) / bin_fact
    else:
        # assumed 2d with 1 row per star
        n_stars = len(flux)
        flux_b = np.average(flux[:clip].reshape((n_stars, n_binned, bin_fact)), axis=2)
        error_b = np.sqrt(np.sum(error[:clip].reshape((n_stars, n_binned, bin_fact)) ** 2, axis=2)) / bin_fact
    return time_b, flux_b, error_b


def remove_outliers(time, flux, flux_err, air_mass=None, zero_point=None):
    """
    Remove massive outliers in 3 rounds of clipping.

    Parameters
    ----------
    time : array
        Time values.
    flux : array
        Flux values.
    flux_err : array
        Flux error values.
    air_mass : array, optional
        Air mass values. Default is None.
    zero_point : array, optional
        Zero point values. Default is None.

    Returns
    -------
    n_time : array
        Non-outlier time values.
    n_flux : array
        Non-outlier flux values.
    n_flux_err : array
        Non-outlier flux error values.
    n_air_mass : array or None
        Non-outlier air mass values, or None if air_mass was not provided.
    n_zero_point : array or None
        Non-outlier zero point values, or None if zero_point was not provided.
    """
    n_time = np.copy(time)
    n_flux = np.copy(flux)
    n_flux_err = np.copy(flux_err)
    n_air_mass = np.copy(air_mass) if air_mass is not None else None
    n_zero_point = np.copy(zero_point) if zero_point is not None else None

    for _ in range(3):
        mad = median_abs_deviation(n_flux)
        loc = np.where((n_flux < np.median(n_flux) + 10 * mad) & (n_flux > np.median(n_flux) - 10 * mad))[0]

        # Update time, flux, and flux_err with non-outlier values
        n_time = n_time[loc]
        n_flux = n_flux[loc]
        n_flux_err = n_flux_err[loc]
        if n_air_mass is not None:
            n_air_mass = n_air_mass[loc]
        if n_zero_point is not None:
            n_zero_point = n_zero_point[loc]

        # If no outliers were removed in this round, return the filtered arrays
        if len(n_time) == len(time):
            return n_time, n_flux, n_flux_err, n_air_mass, n_zero_point

    return n_time, n_flux, n_flux_err, n_air_mass, n_zero_point


def utc_to_jd(utc_time_str):
    """
    Convert UTC time string to Julian Date

    Parameters
    ----------
    utc_time_str : str
        UTC time string in the format 'YYYY-MM-DDTHH:MM:SS'

    Returns
    -------
    jd : float
        Julian Date
    """
    t = Time(utc_time_str, format='isot', scale='utc')

    # Convert to Julian Date
    jd = t.jd

    return jd


def extract_phot_file(table, tic_id_to_plot, aper):
    """
    Extract data for a specific TIC ID from the table.

    Parameters:
    table : astropy.table.Table
        Table containing the photometry data
    tic_id_to_plot : int
        TIC ID of the star
    aper : int
        Aperture size for flux extraction

    Returns:
    jd_mid : array
        Values of BJD midpoints
    tmag : float
        Tmag value for the specified TIC ID
    fluxes : array
        Values of flux
    fluxerrs : array
        Values of flux error
    sky : array
        Values of sky flux
    """
    # Select rows with the specified TIC ID
    tic_id_data = table[table['tic_id'] == tic_id_to_plot]
    # Get jd_mid, flux_2, and fluxerr_2 for the selected rows
    jd_mid = tic_id_data['jd_bary']
    tmag = tic_id_data['Tmag'][0]
    fluxes = tic_id_data[f'flux_{aper}']
    fluxerrs = tic_id_data[f'fluxerr_{aper}']
    sky = tic_id_data[f'flux_w_sky_{aper}'] - tic_id_data[f'flux_{aper}']

    return jd_mid, tmag, fluxes, fluxerrs, sky


def calculate_trend_and_flux(time, flux, fluxerr, degree=2):
    """
    Calculate the trend of the flux values over time and adjust flux and fluxerr accordingly.

    Parameters:
    time : array-like
        Array containing the time values.
    flux : array-like
        Array containing the flux values.
    fluxerr : array-like
        Array containing the flux error values.
    degree : int, optional
        Degree of the polynomial to fit (default is 2).

    Returns:
    trend : array-like
        Array containing the trend values.
    dt_flux : array-like
        Array containing the adjusted flux values.
    dt_fluxerr : array-like
        Array containing the adjusted flux error values.
    """
    # Calculate the trend of the flux values over time
    trend = np.polyval(np.polyfit(time - int(time[0]), flux, degree), time - int(time[0]))
    # Adjust flux and fluxerr accordingly
    dt_flux = flux / trend
    dt_fluxerr = fluxerr / trend
    return trend, dt_flux, dt_fluxerr


def scintilation_noise(airmass_list, exposure):
    # Following Osborne et al. 2015 for Paranal
    D = 0.508  # telescope diameter
    h = 2349  # height of Paranal
    H = 8000  # height of atmospheric scale
    airmass = np.mean(airmass_list**3)  # airmass
    C_y = 1.3  # constant
    # C_y = 1.8
    N = np.sqrt(10e-6 * (C_y ** 2) * (D ** (-4 / 3)) * (1 / exposure) * (airmass) * np.exp((-2. * h) / H))
    # print('Scintilation noise: ', N)
    # print('Airmass: ', airmass)
    return N


def noise_sources(sky_list, bin_size, airmass_list, zp, aper, rn, dc, exposure, gain):
    """
    Returns the noise sources for a given flux

    returns arrays of noise and signal for a given flux

    Parameters
    ----------
    sky_list : list
        values of sky fluxes
    bin_size : int
        number of images to bin
    airmass_list : list
        values of airmass
    zp : list
        values of zero points
    aper : int
        aperture size
    rn : float
        value of read noise
    dc : float
        value of dark current
    exposure : float
        value of exposure time
    gain : float
        value of gain

    Returns
    -------
    synthetic_mag : array
        values of synthetic magnitudes
    photon_shot_noise : array
        values of photon shot noise
    sky_noise : array
        values of sky noise
    read_noise : array
        values of read noise
    dc_noise : array
        values of dark current noise
    N : array
        values of scintilation noise
    RNS : array
        values of read noise squared
    """

    # set aperture radius
    aperture_radius = aper
    npix = np.pi * aperture_radius ** 2

    # set exposure time and random flux
    exposure_time = exposure
    synthetic_flux = np.arange(100, 1e7, 1000)
    synthetic_mag = np.mean(zp) + 2.5*np.log10(gain) - 2.5 * np.log10(synthetic_flux/exposure_time)
    # set dark current rate from cmos characterisation
    dark_current = dc * exposure_time * npix
    dc_noise = np.sqrt(dark_current) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm

    # set read noise from cmos characterisation
    read_noise_pix = rn
    read_noise = (read_noise_pix * np.sqrt(npix)) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm
    read_signal = npix * (read_noise_pix ** 2)

    # sky_flux = np.median(sky_list)
    sky_flux = np.mean(sky_list)
    sky_noise = np.sqrt(sky_flux) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm
    print('Average sky flux: ', sky_flux)

    # set random photon shot noise from the flux
    photon_shot_noise = np.sqrt(synthetic_flux) / synthetic_flux / np.sqrt(bin_size) * 1000000  # Convert to ppm

    N = scintilation_noise(airmass_list, exposure)

    N_sc = (N * synthetic_flux) ** 2
    N = N / np.sqrt(bin_size) * 1000000  # Convert to ppm

    total_noise = np.sqrt(synthetic_flux + sky_flux + dark_current + read_signal + N_sc)
    RNS = total_noise / synthetic_flux / np.sqrt(bin_size)
    RNS = RNS * 1000000  # Convert to ppm

    return synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS


def extract_airmass_zp(table, image_directory):
    unique_frame_ids = np.unique(table['frame_id'])

    airmass_list = []
    zp_list = []

    for frame_id in unique_frame_ids:
        # Get the path to the FITS file
        fits_file_path = os.path.join(image_directory, frame_id)

        # Read FITS file header to extract airmass
        with fits.open(fits_file_path) as hdul:
            image_header = hdul[0].header
            airmass = round(image_header['AIRMASS'], 3)
            zp = image_header['MAGZP_T']

        # Append airmass value and frame ID to the list
        airmass_list.append(airmass)
        zp_list.append(zp)

    print(f"Average airmass: {np.mean(airmass_list)}")
    print(f"Average ZP: {np.mean(zp_list)}")

    return airmass_list, zp_list


def extract_airmass_and_zp(header):
    """Extract airmass and zero point from the FITS header."""
    airmass = header.get('AIRMASS', None)
    zp = header.get('MAGZP_T', None)
    return airmass, zp


def expand_and_rename_table(phot_table):
    expanded_rows = []

    for row in phot_table:
        jd_mid_values = row['Time_BJD']
        relative_flux_values = row['Relative_Flux']
        relative_flux_err_values = row['Relative_Flux_err']
        airmass = row['Airmass']
        zp = row['ZP']

        # Expand jd_mid, relative_flux, and relative_flux_err columns into individual columns
        for i in range(len(jd_mid_values)):
            expanded_row = list(row)
            expanded_row[row.colnames.index('Time_BJD')] = jd_mid_values[i]
            expanded_row[row.colnames.index('Relative_Flux')] = relative_flux_values[i]
            expanded_row[row.colnames.index('Relative_Flux_err')] = relative_flux_err_values[i]
            expanded_row[row.colnames.index('Airmass')] = airmass[i]
            expanded_row[row.colnames.index('ZP')] = zp[i]

            expanded_rows.append(expanded_row)

    # Create a new table with the expanded rows and dynamically set column names
    new_table = Table(rows=expanded_rows, names=phot_table.colnames)

    return new_table


def open_json_file():
    # Use glob to find JSON files starting with 'rel' in the current directory
    json_files = glob.glob('rms*.json')

    if not json_files:
        raise FileNotFoundError("No JSON file starting with 'rel' was found in the current directory.")

    # Open the first matching file (you can modify this if you want to handle multiple files)
    filename = json_files[0]
    with open(filename, 'r') as file:
        data = json.load(file)

    print(f"Opened JSON file: {filename}")
    return data
