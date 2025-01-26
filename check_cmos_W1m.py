#!/usr/bin/env python

"""
This script checks the headers of the FITS files in the specified directory
and moves the files without CTYPE1 and/or CTYPE2 to a separate directory.

Usage:
python check_headers.py
"""

from datetime import datetime, timedelta
from donuts import Donuts
from astropy.io import fits
import numpy as np
import os
import json
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
    Extract unique prefixes from a list of filenames.

    Parameters
    ----------
    filenames : list of str
        List of filenames.

    Returns
    -------
    set of str
        Set of unique prefixes extracted from the filenames.
    """
    prefixes = set()
    for filename in filenames:
        prefix = filename[:11]
        prefixes.add(prefix)
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


def check_donuts(file_groups, filenames):
    """
    Check donuts for each group of images with the same prefix.

    Parameters
    ----------
    file_groups : list of str
        Prefixes for the groups of images.
    filenames : list of lists of filenames for the groups of images.
    """
    for filename, file_group in zip(filenames, file_groups):
        # Using the first filename as the reference image
        reference_image = file_group[0]
        logger.info(f"Reference image: {reference_image}")

        # Assuming Donuts class and measure_shift function are defined elsewhere
        d = Donuts(reference_image)

        for filename in file_group[1:]:
            shift = d.measure_shift(filename)
            sx = round(shift.x.value, 2)
            sy = round(shift.y.value, 2)
            logger.info(f'{filename} shift X: {sx} Y: {sy}')
            shifts = np.array([abs(sx), abs(sy)])

            if np.sum(shifts > 50) > 0:
                logger.warning(f'{filename} image shift too big X: {sx} Y: {sy}')
                if not os.path.exists('failed_donuts'):
                    os.mkdir('failed_donuts')
                comm = f'mv {filename} failed_donuts/'
                logger.info(comm)
                os.system(comm)


def main():
    # set directory for working
    directory = os.getcwd()
    logger.info(f"Directory: {directory}")

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
    check_donuts(prefix_filenames, filenames)

    logger.info("Done.")


if __name__ == "__main__":
    main()
