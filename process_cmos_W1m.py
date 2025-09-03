#! /usr/bin/env python
import argparse
import os
from datetime import datetime, timedelta
import numpy as np
from calibration_images_W1m import reduce_images
from utils_W1m import (get_location, wcs_phot, _detect_objects_sep, get_catalog,
                       extract_airmass_and_zp, get_light_travel_times)
import json
import warnings
import logging
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.wcs import WCS
import sep
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning


# -------------------------------------------------
# Parse command-line arguments
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Process FITS files for photometry.")
    parser.add_argument("--rsi", type=float, default=60,
                        help="Inner sky annulus radius in pixels (default: 60)")
    parser.add_argument("--rso", type=float, default=65,
                        help="Outer sky annulus radius in pixels (default: 65)")
    parser.add_argument("--apertures", type=float, nargs="+", default=[10, 20, 30, 40, 50],
                        help="List of aperture radii in pixels (default: 20 20 30 40 50)")
    return parser.parse_args()


# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('process.log')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

warnings.simplefilter('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

N_OBJECTS_LIMIT = 200
DEFOCUS = 0.0
AREA_MIN = 10
AREA_MAX = 2000
DETECTION_SIGMA = 3
OK, TOO_FEW_OBJECTS, UNKNOWN = range(3)


# -------------------------------------------------
# Utility functions (unchanged from your script)
# -------------------------------------------------
def load_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config


def filter_filenames(directory):
    filtered_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot", "catalog_input"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)
    return sorted(filtered_filenames)


def get_prefix(filenames):
    prefixes = set()
    for filename in filenames:
        with fits.open(filename) as hdulist:
            object_keyword = hdulist[0].header.get('OBJECT', '')
            if object_keyword:
                prefixes.add(object_keyword)
    return prefixes


# -------------------------------------------------
# Main processing function
# -------------------------------------------------
def main():
    args = parse_args()

    # Use the arguments instead of hardcoded values
    RSI = args.rsi
    RSO = args.rso
    APERTURE_RADII = args.apertures

    directory = os.getcwd()
    logging.info(f"Directory: {directory}")

    filenames = filter_filenames(directory)
    logging.info(f"Number of files: {len(filenames)}")

    prefixes = get_prefix(filenames)
    logging.info(f"The prefixes are: {prefixes}")

    if not filenames:
        logging.error("No valid FITS files found. Exiting.")
        return

        # -------------------------------------------------
        # Determine gain from first file header
        # -------------------------------------------------
    first_file = filenames[0]
    with fits.open(first_file) as hdul:
        readmode = hdul[0].header.get("READMODE", "").strip().upper()

    if readmode == "HDR":
        GAIN = 0.75
    elif readmode == "LN16":
        GAIN = 0.25
    else:
        logging.warning(f"READMODE='{readmode}' not recognized, defaulting gain=1.0")
        GAIN = 1.0

    logging.info(f"READMODE detected: {readmode}, using GAIN={GAIN}")
    
    for prefix in prefixes:
        phot_output_filename = os.path.join(directory, f"phot_{prefix}.fits")
        if os.path.exists(phot_output_filename):
            logging.info(f"Photometry file for prefix {prefix} already exists, skipping.")
            continue

        logging.info(f"Creating new photometry file for prefix {prefix}.")
        phot_table = None

        prefix_filenames = [filename for filename in filenames if filename.startswith(prefix)]
        for filename in prefix_filenames:
            logging.info(f"Processing filename {filename}......")
            # Calibrate image and get FITS file
            logging.info(
                f"The average pixel value for {filename} is {fits.getdata(os.path.join(directory, filename)).mean()}")
            reduced_data, reduced_header, _ = reduce_images([filename])
            logging.info(f"The average pixel value for {filename} is {reduced_data[0].mean()}")
            reduced_data_dict = {filename: (data, header) for data, header in zip(reduced_data, reduced_header)}
            frame_data, frame_hdr = reduced_data_dict[filename]
            logging.info(f"Extracting photometry for {filename}")

            airmass, zp = extract_airmass_and_zp(frame_hdr)

            wcs_ignore_cards = ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'DATE', 'IMAGEW', 'IMAGEH']
            wcs_header = {}
            for line in [frame_hdr[i:i + 80] for i in range(0, len(frame_hdr), 80)]:
                key = line[0:8].strip()
                if '=' in line and key not in wcs_ignore_cards:
                    card = fits.Card.fromstring(line)
                    wcs_header[card.keyword] = card.value

            frame_bg = sep.Background(frame_data)
            frame_data_corr_no_bg = frame_data - frame_bg

            try:
                ra = frame_hdr['TELRA']
                dec = frame_hdr['TELDEC']
            except KeyError:
                ra = frame_hdr['MNTRAD']
                dec = frame_hdr['MNTDECD']

            estimate_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            estimate_coord_radius = 3 * u.deg

            frame_objects = _detect_objects_sep(frame_data_corr_no_bg, frame_bg.globalrms,
                                                AREA_MIN, AREA_MAX, DETECTION_SIGMA, DEFOCUS)
            if len(frame_objects) < N_OBJECTS_LIMIT:
                logging.info(f"Fewer than {N_OBJECTS_LIMIT} objects found in {filename}, skipping photometry!")
                continue

            phot_cat, _ = get_catalog(f"{directory}/{prefix}_catalog_input.fits", ext=1)
            logging.info(f"Found catalog with name {prefix}_catalog_input.fits")
            phot_x, phot_y = WCS(frame_hdr).all_world2pix(phot_cat['RA_CORR'], phot_cat['DEC_CORR'], 1)

            half_exptime = frame_hdr['EXPTIME'] / 2.
            time_isot = Time([frame_hdr['DATE-OBS'] for _ in range(len(phot_x))],
                             format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd = time_jd + half_exptime * u.second
            ra = phot_cat['RA_CORR']
            dec = phot_cat['DEC_CORR']
            ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
            time_bary = time_jd.tdb + ltt_bary
            time_helio = time_jd.utc + ltt_helio

            frame_ids = [filename for _ in range(len(phot_x))]
            logging.info(f"Found {len(frame_ids)} sources")

            frame_preamble = Table([frame_ids, phot_cat['Tmag'], phot_cat['TIC'],
                                    phot_cat['BPmag'], phot_cat['RPmag'], time_jd.value, time_bary.value,
                                    phot_x, phot_y,
                                    np.array([airmass] * len(phot_x), dtype='float64'),
                                    np.array([zp] * len(phot_x), dtype='float64')],
                                   names=("frame_id", "Tmag", "tic_id", "gaiabp", "gaiarp", "jd_mid",
                                          "jd_bary", "x", "y", "airmass", "zp"))
            frame_preamble['zp'] = frame_preamble['zp'].astype('float64', copy=False)

            frame_phot = wcs_phot(frame_data, phot_x, phot_y, RSI, RSO, APERTURE_RADII, gain=GAIN)
            frame_output = hstack([frame_preamble, frame_phot])

            if not isinstance(frame_output, Table):
                frame_output = Table(frame_output)

            if phot_table is None:
                phot_table = frame_output
            else:
                phot_table = vstack([phot_table, frame_output])
        logging.info(f"Finished photometry for {filename}\n")
        if phot_table is not None:
            phot_table.write(phot_output_filename, overwrite=True)
            logging.info(f"Saved photometry for prefix {prefix} to {phot_output_filename}\n")
        else:
            logging.info(f"No photometry data for prefix {prefix}.\n")

    logging.info("All processing completed.")


if __name__ == "__main__":
    main()
