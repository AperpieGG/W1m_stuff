#!/usr/bin/env python

"""
This script is used to reduce the images in the specified directory.
It will create a master bias or read it if it already exists in the calibration directory.
It will create a master dark or read it if it already exists in the calibration directory.

If this script works as a cronjob and the night directory is found then it will check if the
master_flat_<night_directory>.fits already exists in the calibration path and use that.
Otherwise, it will create it and use it for the reduction of the images.

If the current night directory is not found (running it manually) then it will create
a master_flat.fits (created from the create_flats.py) from the flat files in the
current working directory and use it for the reduction of the images.

if the master_flat is not created from the create_flats then it will take the general master_flat.fits
from the calibration directory and use it for the reduction of the images.
"""

import glob
import os
from datetime import datetime, timedelta
from astropy.io import fits
import numpy as np
from astropy.time import Time
import astropy.units as u
from utils_W1m import get_location, get_light_travel_times


def bias():
    """
    Create the master bias from the bias files.

    Parameters
    ----------
    Returns
    -------
    numpy.ndarray
        Master bias.
    """
    master_bias_path = os.path.join('.', 'master_bias.fits')

    if os.path.exists(master_bias_path):
        print('Found master bias in the current directory')
        return fits.getdata(master_bias_path)
    else:
        print('Creating master bias')

        # Find and read the bias for hdr mode
        files = [f for f in glob.glob(os.path.join('.', 'bias*.fits'))]

        # Limit the number of files to the first 21
        files = files[:21]

        first_image_shape = fits.getdata(files[0]).shape
        cube = np.zeros((*first_image_shape, len(files)))
        for i, f in enumerate(files):
            cube[:, :, i] = fits.getdata(f)
        master_bias = np.median(cube, axis=2)

        # Copy header from one of the input files
        header = fits.getheader(files[0])

        fits.PrimaryHDU(master_bias, header=header).writeto(master_bias_path, overwrite=True)
        print(f'Master bias saved to: {os.path.join(os.getcwd(), "master_bias.fits")}')
        return master_bias


def dark(master_bias):
    """
    Create the master dark from the dark files.

    Parameters
    ----------
    master_bias : numpy.ndarray
        Master bias.
    Returns
    -------
    numpy.ndarray
        Master dark.
    """
    master_dark_path = os.path.join('.', 'master_dark.fits')

    if os.path.exists(master_dark_path):
        print('Found master dark in the current directory')
        return fits.getdata(master_dark_path)
    else:
        print('Creating master dark')

        # Find and read the dark for hdr mode
        files = [f for f in glob.glob(os.path.join('.', 'dark*.fits'))]

        # Limit the number of files to the first 21
        files = files[:21]
        first_image_shape = fits.getdata(files[0]).shape
        cube = np.zeros((*first_image_shape, len(files)))

        for i, f in enumerate(files):
            cube[:, :, i] = fits.getdata(f) - master_bias
        master_dark = np.median(cube, axis=2)

        # Copy header from one of the input files
        header = fits.getheader(files[0])

        fits.PrimaryHDU(master_dark, header=header).writeto(master_dark_path, overwrite=True)
        print(f'Master dark saved to: {os.path.join(os.getcwd(), "master_dark.fits")}')
        return master_dark


def flat(master_bias, master_dark, dark_exposure=10):
    """
    Create the master flat from the flat files.

    Parameters
    ----------
    master_bias : numpy.ndarray
        Master bias.
    master_dark : numpy.ndarray
        Master dark.
    dark_exposure : int
        Dark exposure time.

    Returns
    -------
    numpy.ndarray
        Master flat.
    """
    # Find and read the flat files
    # Find appropriate files for creating the master flat
    evening_files = [f for f in glob.glob(os.path.join('.', 'evening*.fits'))]

    if not evening_files:
        # If evening files don't exist, use morning files
        evening_files = [f for f in glob.glob(os.path.join('.', 'morning*.fits'))]

    if not evening_files:
        print('No suitable flat field files found.')
        return None  # or handle the case where no files are found

    else:
        print(f'Found {len(evening_files)} evening files')

        print('Creating master flat')
        # take only the first 21
        files = evening_files[:21]

        cube = np.zeros((*master_dark.shape, len(files)))
        for i, f in enumerate(files):
            data, header = fits.getdata(f, header=True)
            # for IMX571 we will not use master-bias, only dark
            cube[:, :, i] = data - master_bias - master_dark * header['EXPTIME'] / dark_exposure
            cube[:, :, i] = cube[:, :, i] / np.average(cube[:, :, i])

        master_flat = np.median(cube, axis=2)

        # Copy header from one of the input files
        header = fits.getheader(files[0])

        # Write the master flat with the copied header
        hdu = fits.PrimaryHDU(master_flat, header=header)
        hdu.writeto(os.path.join('.', 'master_flat.fits'), overwrite=True)

        hdul = fits.open(os.path.join('.', 'master_flat.fits'), mode='update')
        hdul[0].header['FILTER'] = 'NONE'
        hdul.close()

        print(f'Master flat saved to: {os.path.join(os.getcwd(), "master_flat.fits")}')
        return master_flat


def reduce_images(prefix_filenames):
    """
    Reduce the images in the specified directory.

    Parameters
    ----------
    prefix_filenames : list of str
        List of filenames for the prefix.

    Returns
    -------
    list of numpy.ndarray
        Reduced data.
    """
    master_bias = bias()
    master_dark = dark(master_bias)
    master_flat = flat(master_bias, master_dark)

    reduced_data = []
    reduced_header_info = []
    filenames = []

    for filename in prefix_filenames:
        try:
            fd, hdr = fits.getdata(filename, header=True)

            # Additional calculations based on header information
            data_exp = round(float(hdr['EXPTIME']), 2)
            half_exptime = data_exp / 2.
            time_isot = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += half_exptime * u.second
            try:
                ra = hdr['TELRAD']
                dec = hdr['TELDECD']
            except KeyError:
                ra = hdr['MNTRAD']
                dec = hdr['MNTDECD']
            ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
            time_bary = time_jd.tdb + ltt_bary
            time_helio = time_jd.utc + ltt_helio

            # Reduce image
            # exclude master bias
            fd = (fd - master_bias - master_dark * hdr['EXPTIME'] / 10) / master_flat
            # fd = (fd - master_dark * hdr['EXPTIME'] / 10) / master_flat
            reduced_data.append(fd)  # Append the reduced image to the list
            reduced_header_info.append(hdr)

            # Append the filename to the filenames list
            filenames.append(os.path.basename(filename))

        except Exception as e:
            print(f'Failed to process {filename}. Exception: {str(e)}')
            continue

        # print(f'Reduced {filename}')

    return reduced_data, reduced_header_info, filenames



