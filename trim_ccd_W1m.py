#! /usr/bin/env python
import os
from astropy.io import fits
import numpy as np


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
        if filename.startswith('IMAGE') and filename.endswith('.fits') and not filename.endswith('.fits.bz2'):
            filtered_filenames.append(filename)  # Append only the filename without the directory path
    return sorted(filtered_filenames)


def trim_images(directory):

    # Find and read the bias for hdr mode
    files = filter_filenames(directory)

    # join the directory path to the filenames
    files_filtered = [os.path.join(directory, f) for f in files]
    print(f'Found {len(files_filtered)} bias files in {directory}')

    if fits.open(files_filtered[0])[0].data.shape == (2048, 2048):
        print('The files are already trimmed')
    else:
        # check if we have an overscan to remove
        for filename in files_filtered:
            frame = fits.open(filename)
            frame = frame[0]
            print('Initial frame shape:', frame.data.shape)
            if frame.data.shape == (2048, 2088):  # overscan present: x,y
                frame_data = frame.data[:, 20:2068].astype(float)
                print('Final frame shape:', frame_data.shape)
                frame = fits.PrimaryHDU(frame_data.astype(np.uint16), header=frame.header)
                frame.writeto(filename, overwrite=True)

        print('Trimmed bias files to shape:', frame_data.shape)


def main():
    trim_images(os.getcwd())


if __name__ == '__main__':
    main()
