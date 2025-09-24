#!/usr/bin/env python

"""
This script checks the headers of the FITS files in the specified directory
and moves the files without CTYPE1 and/or CTYPE2 to a separate directory.

Usage:
python check_headers.py
"""
import sys
import glob
from donuts import Donuts
from astropy.io import fits
import numpy as np
import os
import logging
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

from utils_W1m import utc_to_jd

warnings.simplefilter('ignore', category=UserWarning)

log_file = 'check_cmos.log'
# Set up logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if analysis already done
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        if any("Done." in line for line in f):
            print(f"{log_file} shows analysis already completed. Skipping script.")
            sys.exit(0)


def acquire_header_info(directory, prefix):
    path = directory + '/'
    image_names = glob.glob(path + f'{prefix}*.fits')
    image_names = sorted(image_names[1:])
    time_jd = []

    for image in image_names:
        with fits.open(image) as hdulist:
            header = hdulist[0].header
            # Extract the UTC time string from the header
            utc_time_str = header['DATE-OBS']

            # Convert the UTC time string to JD
            jd = utc_to_jd(utc_time_str)

            time_jd.append(jd)

    return time_jd


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
        if len(file_group) < 2:
            continue  # skip if only 1 file (no reference + target)

        reference_image = os.path.join(directory, file_group[0])
        prefix = file_group[0].split('_')[0]  # or adjust how you want to get prefix
        logger.info(f"Reference image: {reference_image}")

        d = Donuts(reference_image)
        x_shifts, y_shifts, times = [], [], []

        for filename in file_group[1:]:
            filepath = os.path.join(directory, filename)
            try:
                shift = d.measure_shift(filepath)
                sx = round(shift.x.value, 2)
                sy = round(shift.y.value, 2)
                logger.info(f'{filename} shift X: {sx} Y: {sy}')

                x_shifts.append(sx)
                y_shifts.append(sy)

                # Grab time from FITS header if available
                with fits.open(filepath) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS')
                    # inside check_donuts, after reading DATE-OBS:
                    if date_obs:
                        # Extract the UTC time string from the header
                        jd = utc_to_jd(date_obs)

                        times.append(jd)
                    else:
                        times.append(len(times))  # fallback numeric sequence

                # Check for big shifts
                if np.any(np.array([abs(sx), abs(sy)]) > 4):
                    logger.warning(f'{filename} image shift too big X: {sx} Y: {sy}')
                    failed_dir = os.path.join(directory, 'failed_donuts')
                    if not os.path.exists(failed_dir):
                        os.mkdir(failed_dir)
                    os.rename(filepath, os.path.join(failed_dir, filename))

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

        # After processing the group, plot shifts
        save_path = os.path.join(directory, 'shifts_plots')
        plot_shifts(x_shifts, y_shifts, save_path, prefix, times)


def plot_shifts(x_shifts, y_shifts, save_path, prefix, time):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot of shifts
    scatter = ax.scatter(x_shifts, y_shifts, c=time, cmap='viridis')
    plt.colorbar(scatter, label='Time')

    plt.xlabel('X Shift (pixels)')
    plt.ylabel('Y Shift (pixels)')
    plt.title('Shifts to reference image')

    # Draw center lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)

    # Set limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Create a grid of 1-pixel squares
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)

    # Optional: emphasize the center square
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5))

    # Legend and colorbar
    ax.legend()
    plt.colorbar(scatter, label='Time')

    # Timestamp
    timestamp_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # Save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pdf_file_path = os.path.join(save_path, f"donuts_{prefix}_{timestamp_yesterday}.pdf")

    fig.savefig(pdf_file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"PDF plot saved to: {pdf_file_path}\n")


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
    check_donuts(directory, prefix_filenames)

    logger.info("Done.")


if __name__ == "__main__":
    main()
