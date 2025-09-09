#!/usr/bin/env python

"""
This script checks the headers of the FITS files in the specified directory
and moves the files without CTYPE1 and/or CTYPE2 to a separate directory.

Usage:
python check_headers.py
"""
import sys

from donuts import Donuts
from astropy.io import fits
import numpy as np
import os
import logging
import warnings

warnings.simplefilter('ignore', category=UserWarning)

# Set up logging
logging.basicConfig(
    filename='check_cmos.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        if filename.endswith('.fits'):
            exclude_words = ["evening", "morning", "flat", "bias", "dark", "catalog", "phot"]
            if any(word in filename.lower() for word in exclude_words):
                continue
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def get_prefix(filenames):
    """
    Extract unique OBJECT header values (prefixes) from a list of FITS filenames.

    Parameters
    ----------
    filenames : list of str
        List of FITS filenames.
    Returns
    -------
    set of str
        Set of unique OBJECT-derived prefixes.
    """
    prefixes = set()
    for filename in filenames:
        try:
            with fits.open(filename) as hdul:
                object_keyword = hdul[0].header.get('OBJECT', '').strip()
                prefix = object_keyword
                prefixes.add(prefix)
        except Exception as e:
            print(f"Warning: Could not read OBJECT from {filename}: {e}")
            continue
    return prefixes


def check_headers(directory, filenames):
    """
    Check headers of all files for CTYPE1 and CTYPE2.

    Parameters
    ----------
    directory : str
        Path to the directory.
    filenames : list of str
    """
    no_wcs = os.path.join(directory, 'no_wcs')
    if not os.path.exists(no_wcs):
        os.makedirs(no_wcs)

    for file in filenames:
        try:
            with fits.open(os.path.join(directory, file)) as hdulist:
                header = hdulist[0].header
                ctype1 = header.get('CTYPE1')
                ctype2 = header.get('CTYPE2')

                if ctype1 is None or ctype2 is None:
                    logger.warning(f"{file} does not have CTYPE1 and/or CTYPE2 in the header. "
                                   f"Moving to 'no_wcs' directory.")
                    new_path = os.path.join(no_wcs, file)
                    os.rename(os.path.join(directory, file), new_path)

        except Exception as e:
            logger.error(f"Error checking header for {file}: {e}")

    logger.info(f"Done checking headers, number of files without CTYPE1 and/or CTYPE2: {len(os.listdir(no_wcs))}")


def check_donuts(directory, file_groups):
    for file_group in file_groups:
        reference_image = os.path.join(directory, file_group[0])
        logger.info(f"Reference image: {reference_image}")

        d = Donuts(reference_image)

        for filename in file_group[1:]:
            filepath = os.path.join(directory, filename)
            shift = d.measure_shift(filepath)
            sx = round(shift.x.value, 2)
            sy = round(shift.y.value, 2)
            logger.info(f'{filename} shift X: {sx} Y: {sy}')

            if np.any(np.array([abs(sx), abs(sy)]) > 50):
                logger.warning(f'{filename} image shift too big X: {sx} Y: {sy}')
                failed_dir = os.path.join(directory, 'failed_donuts')
                if not os.path.exists(failed_dir):
                    os.mkdir(failed_dir)
                os.rename(filepath, os.path.join(failed_dir, filename))


def main():
    # set directory for working
    directory = os.getcwd()
    logger.info(f"Directory: {directory}")

    log_file = 'check_cmos.log'

    # Exit if log file already exists
    if os.path.exists(log_file):
        logger.info(f"Log file '{log_file}' already exists. Exiting script.")
        sys.exit(0)

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    logger.info(f"Number of files: {len(filenames)}")

    # Iterate over each filename to get the prefix
    prefixes = get_prefix(filenames)
    logger.info(f"The prefixes are: {prefixes}")

    # Get filenames corresponding to each prefix
    prefix_filenames = [[filename for filename in filenames if filename.startswith(prefix)] for prefix in prefixes]

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames)

    # Check donuts for each group
    check_donuts(directory, prefix_filenames)

    logger.info("Done.")


if __name__ == "__main__":
    main()
