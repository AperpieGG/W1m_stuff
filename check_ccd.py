#! /usr/bin/env python

"""
This script checks the headers of the FITS files in the specified directory
and moves the files without CTYPE1 and/or CTYPE2 to a separate directory.

Usage:
python check_headers.py
"""

from donuts import Donuts
from astropy.io import fits
import numpy as np
import os
import json
import warnings
from utils import get_location, get_light_travel_times
import astropy.units as u
from astropy.time import Time

warnings.simplefilter('ignore', category=UserWarning)


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
    prefixes = set()
    for filename in filenames:
        fits_path = os.path.join(directory, filename)
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            if 'OBJECT' in header:
                prefix = header['OBJECT']
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
        List of filenames.
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
                    print(f"Warning: {file} does not have CTYPE1 and/or CTYPE2 in the header. Moving to "
                          f"'no_wcs' directory.")
                    new_path = os.path.join(no_wcs, file)
                    os.rename(os.path.join(directory, file), new_path)

        except Exception as e:
            print(f"Error checking header for {file}: {e}")

    print("Done checking headers, number of files without CTYPE1 and/or CTYPE2:", len(os.listdir(no_wcs)))


def update_header(directory):
    """
    Update the header of FITS files in the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing FITS files.
    """

    for filename in filter_filenames(directory):
        filename = os.path.join(directory, filename)
        with fits.open(filename, mode='update') as hdul:
            if 'BJD' not in hdul[0].header:
                # Additional calculations based on header information
                data_exp = round(float(hdul[0].header['EXPTIME']), 2)
                half_exptime = data_exp / 2.
                time_isot = Time(hdul[0].header['DATE-OBS'], format='isot', scale='utc', location=get_location())
                time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
                time_jd += half_exptime * u.second
                ra = hdul[0].header['RA']
                dec = hdul[0].header['DEC']
                ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
                time_bary = time_jd.tdb + ltt_bary
                time_helio = time_jd.utc + ltt_helio

                # Update the header with barycentric and heliocentric times
                hdul[0].header['BJD'] = (time_bary.jd, 'Barycentric Julian Date')
                hdul[0].header['HJD'] = (time_helio.jd, 'Heliocentric Julian Date')
            else:
                print(f"TIME_BARY already present for {filename}")

                hdul.flush()
    print("All headers updated")


def check_donuts(directory, filenames):
    """
    Check donuts for each image in the directory.

    Parameters
    ----------
    directory : str
        Directory containing the images.
    filenames : list of str
        List of filenames.
    """
    # Assuming Donuts class and measure_shift function are defined elsewhere
    if filenames:
        sorted_filenames = sorted(filenames)  # Sort the filenames
        reference_image = sorted_filenames[0]  # Use the first filename as the reference image
        d = Donuts(os.path.join(directory, reference_image))
        print('The reference image is:', reference_image)
        print('The prefix is :', get_prefix([reference_image], directory))

        for filename in filenames[1:]:
            shift = d.measure_shift(os.path.join(directory, filename))
            sx = round(shift.x.value, 2)
            sy = round(shift.y.value, 2)
            print(f'{filename} shift X: {sx} Y: {sy}')
            shifts = np.array([abs(sx), abs(sy)])

            if np.sum(shifts > 50) > 0:
                print(f'{filename} image shift too big X: {sx} Y: {sy}')
                failed_donuts_dir = os.path.join(directory, 'failed_donuts')
                if not os.path.exists(failed_donuts_dir):
                    os.mkdir(failed_donuts_dir)
                comm = f'mv {os.path.join(directory, filename)} {failed_donuts_dir}/'
                print(comm)
                os.system(comm)
        else:
            print("No files to process in the directory.")


def main():
    directory = os.getcwd()
    print(f"Directory: {directory}")

    # filter filenames only for .fits data files
    filenames = filter_filenames(directory)
    print(f"Number of files: {len(filenames)}")

    # Check headers for CTYPE1 and CTYPE2
    check_headers(directory, filenames)

    # Update headers with BJD and HJD
    update_header(directory)  # Uncomment this line to update headers

    # Check donuts for the current subdirectory
    check_donuts(directory, filenames)

    print("Done.")


if __name__ == "__main__":
    main()
