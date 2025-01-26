#! /usr/bin/env python
import os
import bz2


def compress_fits_files(directory):
    # Get a list of all .fits files in the directory
    fits_files = [f for f in os.listdir(directory) if f.startswith('IMAGE') and f.endswith('.fits')]

    # Compress each .fits file into .fits.bz2 format
    for fits_file in fits_files:
        fits_path = os.path.join(directory, fits_file)
        fits_bz2_path = fits_path + '.bz2'
        with open(fits_path, 'rb') as source:
            with bz2.BZ2File(fits_bz2_path, 'wb') as target:
                target.write(source.read())
        print(f"{fits_file} compressed to {fits_bz2_path}")


def delete_uncompressed_images(directory):
    # Get a list of all uncompressed .fits files in the directory
    uncompressed_files = [f for f in os.listdir(directory) if f.startswith('IMAGE') and f.endswith('.fits')]

    # Delete each uncompressed .fits file
    for file in uncompressed_files:
        os.remove(os.path.join(directory, file))
        print(f"{file} deleted.")


def delete_png_files(directory):
    # Get a list of all .png files in the directory
    png_files = [f for f in os.listdir(directory) if f.startswith('IMAGE') and f.endswith('.png')]

    # Delete each .png file
    for file in png_files:
        os.remove(os.path.join(directory, file))
        print(f"{file} deleted.")


def main():
    # Get the current working directory
    directory = os.getcwd()

    print('The directory is:', directory)

    compress_fits_files(directory)
    delete_uncompressed_images(directory)
    delete_png_files(directory)


if __name__ == "__main__":
    main()
