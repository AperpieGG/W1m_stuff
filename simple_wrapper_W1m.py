#!/usr/bin/env python3
"""
Run through many reference images, generate catalogs
and try solving them one by one
"""
import os
import glob as g
import argparse as ap
from astropy.io import fits


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
    p.add_argument('--camera',
                   help='camera type (ccd or cmos)',
                   type=str,
                   default='ccd')
    return p.parse_args()


if __name__ == "__main__":
    # Grab command line args
    args = arg_parse()

    # Set scale values based on camera type
    if args.camera.lower() == 'ccd':
        scale_min = "4.5"
        scale_max = "5.5"
    else:
        scale_min = "3.5"
        scale_max = "4.5"

    # Get a list of all FITS images, exclude whatever has catalog name in
    all_fits = sorted(g.glob("*.fits"))
    print(f"Found {len(all_fits)} FITS files.")
    all_fits = [f for f in all_fits if "_cat" not in f]
    # exclude words in the fits file
    excluded_list = ["bias", "dark", "flat", "morning", "evening", "catalog", "phot"]
    all_fits = [f for f in all_fits if not any(word in f.lower() for word in excluded_list)]
    print(f"Found {len(all_fits)} FITS files after excluding not suitable files.")

    if not all_fits:
        print("No FITS files found.")
        exit(1)

    # Select the first image as the reference
    ref_image = all_fits[0]
    base_name = ref_image.split('.fits')[0]
    prefix = fits.getheader(ref_image)['OBJECT']
    cat_file = f"{prefix}_catalog.fits"

    # Get coordinates from the reference image header
    if args.camera.lower() == 'ccd':
        with fits.open(ref_image) as ff:
            ra = str(ff[0].header['CMD_RA'])
            dec = str(ff[0].header['CMD_DEC'])
            epoch = str(ff[0].header['DATE-OBS'])
            box_size = "3"  # Adjustable
    else:
        with fits.open(ref_image) as ff:
            ra = str(ff[0].header['TELRAD'])
            dec = str(ff[0].header['TELDECD'])
            epoch = str(ff[0].header['DATE-OBS'])
            box_size = "3"  # Adjustable

    # Create the catalog if it doesn't exist
    if not os.path.exists(cat_file):
        print(f'Did not find catalog file: {cat_file}')
        cmd_args = ["/Users/u5500483/Documents/GitHub/W1m_stuff/make_ref_catalog_W1m.py",
                    ra, dec, box_size, box_size, epoch, cat_file]
        cmd = " ".join(cmd_args)
        os.system(cmd)
        print("Catalog created for image {} with prefix: {}\n".format(ref_image, prefix))

    # Solve reference image with catalog file
    if os.path.exists(cat_file):
        print(f'Solving reference image: {ref_image}')
        cmd2_args = ["/Users/u5500483/Documents/GitHub/W1m_stuff/solve_ref_images_W1m.py",
                     cat_file, ref_image, "--scale_min", scale_min, "--scale_max", scale_max]

        if args.save_matched_cat:
            cmd2_args.append("--save_matched_cat")
        if args.defocus is not None:
            cmd2_args.append(f"--defocus {args.defocus:.2f}")
        if args.force3rd:
            cmd2_args.append("--force3rd")

        cmd2 = " ".join(cmd2_args)
        result = os.system(cmd2)

        if result != 0:  # Exit if the reference image fails to solve
            print(f"Failed to solve the reference image {ref_image}. Exiting the script.")
            exit(1)
        else:
            print(f"Successfully solved the reference image {ref_image}.\n")

    # Iterate and solve remaining FITS images
    for fits_file in all_fits:
        if fits_file == ref_image:
            continue

        with fits.open(fits_file) as hdulist:
            object_keyword = hdulist[0].header.get('OBJECT', '')
            if object_keyword.startswith(prefix):
                if "_cat" not in fits_file and fits_file != ref_image:
                    if 'CTYPE1' in hdulist[0].header and 'CTYPE2' in hdulist[0].header and 'ZP_ORDER' in hdulist[
                        0].header:
                        print(f"Image {fits_file} is already solved. Skipping..\n")
                        continue

                    print(f"Solving image {fits_file} for prefix: {prefix}\n")
                    cmd2_args = ["/Users/u5500483/Documents/GitHub/W1m_stuff/solve_ref_images_W1m.py",
                                 cat_file, fits_file, "--scale_min", scale_min, "--scale_max", scale_max]

                    if args.save_matched_cat:
                        cmd2_args.append("--save_matched_cat")
                    if args.defocus is not None:
                        cmd2_args.append(f"--defocus {args.defocus:.2f}")
                    if args.force3rd:
                        cmd2_args.append("--force3rd")

                    cmd2 = " ".join(cmd2_args)
                    result = os.system(cmd2)

                    if result != 0:
                        print(f"Failed to solve the image {fits_file}. Skipping to the next image.\n")
                        continue  # Skip this image and move to the next
                    else:
                        print(f"Successfully solved the image {fits_file}.\n")