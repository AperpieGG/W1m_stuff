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
    """Check if FITS files are trimmed and trim them if necessary."""
    files = filter_filenames(directory)

    if not files:
        print(f"No valid FITS files found in {directory}")
        return

    files_filtered = [os.path.join(directory, f) for f in files]
    print(f"Found {len(files_filtered)} files in {directory}")

    already_trimmed = []
    trimmed_files = []

    for filename in files_filtered:
        try:
            with fits.open(filename) as frame:
                frame_data = frame[0].data
                print(f"Processing {filename} with shape {frame_data.shape}")

                if frame_data.shape == (2048, 2048):
                    # File is already trimmed
                    already_trimmed.append(filename)
                    print(f"{filename} is already trimmed.")
                elif frame_data.shape == (2048, 2088):
                    # Trim the overscan region
                    trimmed_data = frame_data[:, 20:2068]
                    fits.writeto(filename, trimmed_data.astype(np.uint16), header=frame[0].header, overwrite=True)
                    trimmed_files.append(filename)
                    print(f"{filename} has been trimmed to shape {trimmed_data.shape}.")
                else:
                    # Unexpected shape
                    print(f"Unexpected shape for {filename}: {frame_data.shape}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Summary of results
    print(f"Summary:")
    print(f"  Already trimmed files: {len(already_trimmed)}")
    print(f"  Trimmed files: {len(trimmed_files)}")
    print(f"  Total processed: {len(files_filtered)}")


def main():
    trim_images(os.getcwd())


if __name__ == '__main__':
    main()
