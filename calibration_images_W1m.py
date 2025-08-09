import glob
import os
from datetime import datetime, timedelta
from astropy.io import fits
import numpy as np
from astropy.time import Time
import astropy.units as u
from utils_W1m import get_location, get_light_travel_times


def bias():
    master_bias_path = os.path.join('.', 'master_bias.fits')
    if os.path.exists(master_bias_path):
        print('Found master bias in the current directory')
        return fits.getdata(master_bias_path)
    else:
        files = [f for f in glob.glob(os.path.join('.', 'bias*.fits'))][:21]
        if not files:
            print('No bias files found. Skipping bias correction.')
            return None
        print('Creating master bias')
        first_image_shape = fits.getdata(files[0]).shape
        cube = np.zeros((*first_image_shape, len(files)))
        for i, f in enumerate(files):
            cube[:, :, i] = fits.getdata(f)
        master_bias = np.median(cube, axis=2)
        header = fits.getheader(files[0])
        fits.PrimaryHDU(master_bias, header=header).writeto(master_bias_path, overwrite=True)
        print(f'Master bias saved to: {os.path.join(os.getcwd(), "master_bias.fits")}')
        return master_bias


def dark(master_bias):
    master_dark_path = os.path.join('.', 'master_dark.fits')
    if os.path.exists(master_dark_path):
        print('Found master dark in the current directory')
        return fits.getdata(master_dark_path)
    else:
        files = [f for f in glob.glob(os.path.join('.', 'dark*.fits'))][:21]
        if not files:
            print('No dark files found. Skipping dark correction.')
            return None
        print('Creating master dark')
        first_image_shape = fits.getdata(files[0]).shape
        cube = np.zeros((*first_image_shape, len(files)))
        for i, f in enumerate(files):
            dark_data = fits.getdata(f)
            if master_bias is not None:
                dark_data -= master_bias
            cube[:, :, i] = dark_data
        master_dark = np.median(cube, axis=2)
        header = fits.getheader(files[0])
        fits.PrimaryHDU(master_dark, header=header).writeto(master_dark_path, overwrite=True)
        print(f'Master dark saved to: {os.path.join(os.getcwd(), "master_dark.fits")}')
        return master_dark


def flat(master_bias, master_dark, dark_exposure=10):
    master_flat_path = os.path.join('.', 'master_flat.fits')
    if os.path.exists(master_flat_path):
        print('Found master dark in the current directory')
        return fits.getdata(master_flat_path)
    else:
        files = glob.glob(os.path.join('.', 'evening*.fits')) or glob.glob(os.path.join('.', 'morning*.fits'))
        if not files:
            print('No flat field files found. Skipping flat correction.')
            return None
        print(f'Found {len(files)} flat files. Creating master flat.')
        files = files[:21]
        cube = np.zeros((*fits.getdata(files[0]).shape, len(files)))
        for i, f in enumerate(files):
            data, header = fits.getdata(f, header=True)
            data = data.astype(np.float64)  # or np.float32 to save memory
            if master_bias is not None:
                data -= master_bias
            if master_dark is not None:
                data -= master_dark * header['EXPTIME'] / dark_exposure
            cube[:, :, i] = data / np.average(data)
        master_flat = np.median(cube, axis=2)
        header = fits.getheader(files[0])
        hdu = fits.PrimaryHDU(master_flat, header=header)
        hdu.writeto(os.path.join('.', 'master_flat.fits'), overwrite=True)
        with fits.open(os.path.join('.', 'master_flat.fits'), mode='update') as hdul:
            hdul[0].header['FILTER'] = 'NONE'
        print(f'Master flat saved to: {os.path.join(os.getcwd(), "master_flat.fits")}')
        return master_flat


def reduce_images(prefix_filenames):
    master_bias = bias()
    master_dark = dark(master_bias)
    master_flat = flat(master_bias, master_dark)

    reduced_data = []
    reduced_header_info = []
    filenames = []

    for filename in prefix_filenames:
        try:
            fd, hdr = fits.getdata(filename, header=True)
            fd = fd.astype(np.float64)  # <-- Add this line
            data_exp = round(float(hdr['EXPTIME']), 2)
            half_exptime = data_exp / 2.
            time_isot = Time(hdr['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location()) + half_exptime * u.second
            try:
                ra = hdr['TELRAD']
                dec = hdr['TELDECD']
            except KeyError:
                ra = hdr.get('MNTRAD', 0)
                dec = hdr.get('MNTDECD', 0)
            ltt_bary, ltt_helio = get_light_travel_times(ra, dec, time_jd)
            time_bary = time_jd.tdb + ltt_bary
            time_helio = time_jd.utc + ltt_helio

            if master_bias is not None:
                fd -= master_bias
            if master_dark is not None:
                fd -= master_dark * hdr['EXPTIME'] / 10
            if master_flat is not None:
                fd /= master_flat

            reduced_data.append(fd)
            reduced_header_info.append(hdr)
            filenames.append(os.path.basename(filename))
        except Exception as e:
            print(f'Failed to process {filename}. Exception: {str(e)}')
            continue

    return reduced_data, reduced_header_info, filenames
