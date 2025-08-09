#!/usr/bin/env python
import glob
import os
from astropy.io import fits
import numpy as np
from utils_W1m import find_current_night_directory


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
        print('Found master bias')
        return fits.getdata(master_bias_path)
    else:
        print('Could not find Master bias, exiting!')
        return None


def dark():
    """
    Create the master dark from the dark files.

    Parameters
    ----------
    Returns
    -------
    numpy.ndarray
        Master dark.
    """
    master_dark_path = os.path.join('.', 'master_dark.fits')

    if os.path.exists(master_dark_path):
        print('Found master dark')
        return fits.getdata(master_dark_path)
    else:
        print('Could not find Master dark, exiting!')
        return None


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
    evening_files = [f for f in glob.glob(os.path.join('.', 'evening*.fits')) if
                     'HDR' in fits.getheader(f)['READMODE']]

    if not evening_files:
        # If evening files don't exist, use morning files
        evening_files = [f for f in glob.glob(os.path.join('.', 'morning*.fits')) if
                         'HDR' in fits.getheader(f)['READMODE']]

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


def main():
    """
    Main function to create the master bias, dark, and flat.
    """
    print("Creating calibration images: master dark, and flat... NO BIAS !")
    master_dark = dark()
    flat(master_dark)


if __name__ == '__main__':
    main()
