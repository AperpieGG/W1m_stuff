#! /usr/bin/env python
import os
import bz2
from astropy.io import fits

# Words to exclude from science data filenames
EXCLUDE_WORDS = ["bias", "flat", "evening", "morning", "phot", "rel"]


def is_science_image(filename):
    """Check if filename corresponds to a science image."""
    lower_name = filename.lower()
    return not any(word in lower_name for word in EXCLUDE_WORDS)


def compress_fits_files(directory):
    """Compress only science FITS files into .bz2 format."""
    fits_files = [
        f for f in os.listdir(directory)
        if f.endswith('.fits') and is_science_image(f)
    ]

    for fits_file in fits_files:
        fits_path = os.path.join(directory, fits_file)
        fits_bz2_path = fits_path + '.bz2'

        with open(fits_path, 'rb') as source:
            with bz2.BZ2File(fits_bz2_path, 'wb') as target:
                target.write(source.read())

        print(f"Compressed: {fits_file} → {fits_file}.bz2")


def delete_png_files(directory):
    """Delete all PNG files in the directory."""
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    for file in png_files:
        os.remove(os.path.join(directory, file))
        print(f"Deleted: {file}")


def main():
    directory = os.getcwd()
    print(f"The directory is: {directory}")

    compress_fits_files(directory)
    delete_png_files(directory)


if __name__ == "__main__":
    main()
