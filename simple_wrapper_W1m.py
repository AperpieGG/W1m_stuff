#!/usr/bin/env python3
"""
Run through many reference images, generate catalogs
and try solving them one by one
"""
import os
import glob as g
from astropy.io import fits
import argparse as ap

# pylint: disable=invalid-name
# pylint: disable=no-member


def arg_parse():
    """
    Parse the command line arguments
    """
    p = ap.ArgumentParser("Solve AG references images for CASUTools")
    p.add_argument('--defocus',
                   help='manual override for defocus (mm)',
                   type=float,
                   default=0.0)
    p.add_argument('--force3rd',
                   help='force a 3rd order distortion polyfit',
                   action='store_true',
                   default=False)
    p.add_argument('--save_matched_cat',
                   help='output the matched catalog with basic photometry',
                   action='store_true',
                   default=False)
    return p.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = arg_parse()

    # Get the current working directory
    parent_directory = os.getcwd()

    # Get a list of subdirectories inside the parent directory
    subdirectories = [name for name in os.listdir(parent_directory) if
                      os.path.isdir(os.path.join(parent_directory, name))]

    print('The subdirectories are:', subdirectories)

    # Iterate over each subdirectory
    for subdirectory in subdirectories:
        if subdirectory.startswith("action") and subdirectory.endswith("_observeField"):
            # Form the full path to the subdirectory
            subdirectory_path = os.path.join(parent_directory, subdirectory)

            # Change the working directory to the subdirectory
            os.chdir(subdirectory_path)

            print(f"Processing subdirectory: {subdirectory_path}")

            # Get a list of all FITS images
            all_fits = sorted(g.glob("*.fits"))

            # Filter the FITS images based on criteria
            ref_images = []
            for fits_file in all_fits:
                if fits_file.startswith("IMAGE") and fits_file.endswith(".fits") and not fits_file.endswith(".fits.bz2"):
                    header = fits.getheader(fits_file)
                    if header.get("IMGCLASS") == "SCIENCE":
                        ref_images.append(fits_file)

            ref_image = ref_images[0]

            if ref_image is None:
                print("No reference image found for subdirectory:", subdirectory)
                continue

            print("Reference image found:", ref_image)
            base_name = ref_image.split('.fits')[0]
            print("Base name:", base_name)
            prefix = fits.getheader(ref_image)['OBJECT']
            cat_file = f"{prefix}_catalog.fits"

            # Get the coords from the header of the first FITS file
            with fits.open(ref_image) as ff:
                ra = str(ff[0].header['CMD_RA'])
                dec = str(ff[0].header['CMD_DEC'])
                epoch = str(ff[0].header['DATE-OBS'])
                box_size = "2.8"  # You can adjust this as needed

            if not os.path.exists(cat_file):
                # call the catalog maker only for the first image of each prefix
                cmd_args = ["/home/ops/refcatpipe2/cmos/make_ref_catalog_ccd.py",
                            ra, dec, box_size, box_size, epoch, cat_file]
                cmd = " ".join(cmd_args)
                os.system(cmd)
                print("Catalog created for image:", ref_image)

            if os.path.exists(cat_file):
                print("Found cat file:", cat_file)
                print()
                print("Solving image:", ref_image)

                # Solve the reference image with this catalog file
                cmd2_args = ["/home/ops/refcatpipe2/cmos/solve_ref_images_ccd.py",
                             cat_file, ref_image]

                # Add optional arguments based on command-line arguments
                if args.save_matched_cat:
                    cmd2_args.append("--save_matched_cat")
                if args.defocus is not None:
                    cmd2_args.append("--defocus {:.2f}".format(args.defocus))
                if args.force3rd:
                    cmd2_args.append("--force3rd")

                cmd2 = " ".join(cmd2_args)
                os.system(cmd2)
                print("Solved image:", ref_image)

                # Now, iterate over all FITS files again and solve for each image
                for fits_file in ref_images:
                    with fits.open(fits_file) as hdulist:
                        object_keyword = hdulist[0].header.get('OBJECT', '')
                        if "_cat" not in fits_file and fits_file != ref_image:
                            # Check if keywords exist
                            if 'CTYPE1' in hdulist[0].header and 'CTYPE2' in hdulist[0].header and 'ZP_ORDER' in hdulist[0].header:
                                print("Image {} is already solved. Skipping..\n".format(fits_file))
                                continue

                            print("Solving image {} \n".format(fits_file))
                            # Solve the image with the same catalog file
                            cmd2_args = ["/home/ops/refcatpipe2/cmos/solve_ref_images_ccd.py",
                                         cat_file, fits_file]

                            # Add optional arguments based on command-line arguments
                            if args.save_matched_cat:
                                cmd2_args.append("--save_matched_cat")
                            if args.defocus is not None:
                                cmd2_args.append("--defocus {:.2f}".format(args.defocus))
                            if args.force3rd:
                                cmd2_args.append("--force3rd")

                            cmd2 = " ".join(cmd2_args)
                            os.system(cmd2)
                            print("Solved image {}\n".format(fits_file))

            # Move back to the parent directory for the next subdirectory iteration
            os.chdir(parent_directory)
