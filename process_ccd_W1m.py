#! /usr/bin/env python
import logging
import os
import numpy as np
from utils_W1m import (get_location, wcs_phot, _detect_objects_sep, get_catalog,
                       extract_airmass_and_zp, get_light_travel_times)
import json
import warnings
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.wcs import WCS
import sep
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
import fitsio


# Set up logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.INFO)  # Set the overall logging level

# Create file handler
file_handler = logging.FileHandler('process.log')
file_handler.setLevel(logging.INFO)  # Set the level for the file handler

# Create stream handler (for terminal output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Set the level for the stream handler

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Ignore some annoying warnings
warnings.simplefilter('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)


GAIN = 2
N_OBJECTS_LIMIT = 200
APERTURE_RADII = [4, 5]
RSI = 15
RSO = 20
DEFOCUS = 0.0
AREA_MIN = 10
AREA_MAX = 200
DETECTION_SIGMA = 3

OK, TOO_FEW_OBJECTS, UNKNOWN = range(3)


def filter_filenames(directory):
    """
    Filter filenames based on specific criteria.

    Parameters
    ----------
    directory : str
        Directory containing the files.

    Returns
    -------
    list of str
        Filtered list of filenames.
    """
    filtered_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.fits') and not filename.endswith('.fits.bz2'):
            fits_path = os.path.join(directory, filename)
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                if 'IMGCLASS' in header and header['IMGCLASS'] == 'SCIENCE':
                    filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def get_prefix(filenames, directory):
    """
    Extract unique prefixes from a list of filenames based on the OBJECT keyword in FITS headers.

    Parameters
    ----------
    filenames : list of str
        List of filenames.
    directory : str
        Directory containing the FITS files.

    Returns
    -------
    set of str
        Set of unique prefixes extracted from the OBJECT keyword in the FITS headers.
    """
    fits_path = os.path.join(directory, filenames[0])
    with fits.open(fits_path) as hdul:
        prefix = hdul[0].header.get('OBJECT', '')
    return prefix


def load_fits(filename, ext=0):
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

    # Ensure data is converted to float if it's not already
    if data.dtype.kind != 'f':
        try:
            data = data.astype(float)
        except ValueError:
            raise ValueError("Unable to convert data to float. Data may contain non-numeric values.")

    return data, header


def main():
    # Get the current working directory and parent directory
    directory = os.getcwd()
    parent_directory = os.path.dirname(directory)

    logging.info(f"Directory: {directory}")

    # Filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    logging.info(f"Number of files: {len(filenames)}")

    # Get prefix for the current set of images
    prefix = get_prefix(filenames, directory)
    phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")

    if os.path.exists(phot_output_filename):
        logging.info(f"Photometry file for prefix {prefix} already exists, skipping.\n")
        return

    logging.info(f"Creating new photometry file for prefix {prefix}.\n")
    phot_table = None

    # Process each file in the current directory
    for filename in filenames:
        logging.info(f"Processing filename {filename}......")
        # Calibrate image and get FITS file
        ref_frame_data, ref_header = load_fits(os.path.join(directory, filename))
        logging.info(f"The average pixel value for {filename} is {fits.getdata(os.path.join(directory, filename)).mean()}")

        # Load master bias from the parent directory
        master_bias_path = os.path.join(parent_directory, 'master_bias.fits')
        if os.path.exists(master_bias_path):
            master_bias = fits.getdata(master_bias_path)
        else:
            master_bias = None

        if master_bias is not None:
            # Check if master_bias has the same dimensions as ref_frame_data
            if ref_frame_data.shape == master_bias.shape:
                ref_frame_data_corr = ref_frame_data - master_bias
                logging.info(f"After bias subtraction, mean pixel value for {filename}: {np.mean(ref_frame_data_corr)}")
            else:
                logging.info(f"Master bias shape {master_bias.shape} does not match frame shape {ref_frame_data.shape}!")
                # Trim ref_frame_data to match the dimensions of master_bias if necessary
                if master_bias.shape == (2048, 2048):
                    ref_frame_data = ref_frame_data[:, 20:2068]
                    logging.info(f"Trimmed ref_frame_data shape to match master_bias shape: {ref_frame_data.shape}")
                    ref_frame_data_corr = ref_frame_data - master_bias
                    logging.info(f"After bias subtraction, mean pixel value for {filename}: {np.mean(ref_frame_data_corr)}")
                else:
                    logging.info("Unable to trim ref_frame_data: master_bias shape is not (2048, 2088)")
                    continue
        else:
            logging.info("Master bias not found, skipping bias subtraction.")
            ref_frame_data_corr = ref_frame_data  # No bias subtraction

        # Photometry processing
        reduced_data_dict = {filename: (ref_frame_data_corr, ref_header)}
        frame_data, frame_hdr = reduced_data_dict[filename]
        logging.info(f"Extracting photometry for {filename}\n")

        airmass, zp = extract_airmass_and_zp(frame_hdr)

        # WCS handling and background subtraction
        frame_bg = sep.Background(frame_data)
        frame_data_corr_no_bg = frame_data - frame_bg
        estimate_coord = SkyCoord(ra=frame_hdr['CMD_RA'], dec=frame_hdr['CMD_DEC'], unit=(u.deg, u.deg))

        frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms, AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
        if len(frame_objects) < N_OBJECTS_LIMIT:
            logging.info(f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!\n")
            continue

            # Load the photometry catalog
        phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
        logging.info(f"Found catalog with name {prefix}_catalog_input.fits")
        # Convert RA and DEC to pixel coordinates using the WCS information from the header
        phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_cat['RA_CORR'], phot_cat['DEC_CORR'], 1)

        # Do time conversions - one time value per format per target
        half_exptime = frame_hdr['EXPTIME'] / 2.
        time_isot = Time([frame_hdr['DATE-OBS'] for i in range(len(phot_x))],
                         format='isot', scale='utc', location=get_location())
        time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
        # Correct to mid-exposure time
        time_jd = time_jd + half_exptime * u.second
        ra = phot_cat['RA_CORR']
        dec = phot_cat['DEC_CORR']
        ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
        time_bary = time_jd.tdb + ltt_bary
        time_helio = time_jd.utc + ltt_helio

        frame_ids = [filename for i in range(len(phot_x))]
        logging.info(f"Found {len(frame_ids)} sources")

        # Ensure zp is a valid float
        try:
            zp = float(zp)
        except (TypeError, ValueError):
            logging.warning(f"Invalid zp value: {zp} for {filename}, replacing with NaN.")
            zp = np.nan

        # Create photometry table
        frame_preamble = Table([frame_ids, phot_cat['GAIA'], phot_cat['Tmag'], phot_cat['TIC'],
                                phot_cat['BPmag'], phot_cat['RPmag'], time_jd.value, time_bary.value,
                                time_helio.value, phot_x, phot_y,
                                np.array([airmass] * len(phot_x), dtype='float64'),
                                np.array([zp] * len(phot_x), dtype='float64')],
                               names=("frame_id", "gaia_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                                      "jd_bary", "jd_helio", "x", "y", "airmass", "zp"))

        # Validate column type before stacking
        frame_preamble['zp'] = frame_preamble['zp'].astype('float64', copy=False)
        logging.info(f"zp column dtype in frame_preamble: {frame_preamble['zp'].dtype}")

        frame_phot = wcs_phot(frame_data, phot_x, phot_y, RSI, RSO, APERTURE_RADII, gain=GAIN)

        frame_output = hstack([frame_preamble, frame_phot])

        # Stack the photometry results
        if phot_table is None:
            phot_table = frame_output
        else:
            phot_table = vstack([phot_table, frame_output])

        logging.info(f"Finished photometry for {filename}\n")

    # Save the accumulated photometry
    if phot_table is not None:
        phot_table.write(phot_output_filename, overwrite=True)
        logging.info(f"Saved photometry for prefix {prefix} to {phot_output_filename}\n")
    else:
        logging.info(f"No photometry data for prefix {prefix}.\n")

    logging.info("Done!\n")


if __name__ == "__main__":
    main()