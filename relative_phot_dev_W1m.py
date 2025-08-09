#!/usr/bin/env python
"""
- First, cut the table for stars 9.5-12 mags; these will be used as reference stars
- Exclude the tic_id you want to perform relative photometry (target_flux)
- Measure the RMS for each raw lightcurve for your reference stars
- Find the reference stars with the lowest RMS (2 sigma clipping threshold)
- Use these stars and sum their fluxes (sum_fluxes)
- Find the mean of the reference master flux (mean_ref_flux)
- Normalize the sum_fluxes by dividing by the mean_ref_flux (normalized_reference_flux)
- Normalize the target_flux by dividing by the mean of the target_flux (normalized_target_flux)
- Perform relative photometry by dividing the normalized_target_flux by the normalized_reference_flux (dt_flux)
- Apply a second-order polynomial to correct for color (dt_flux_poly)
"""
import argparse
import os
from collections import Counter

import numpy as np
import logging
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from astropy.table import Table
from utils import (plot_images, get_phot_files, read_phot_file,
                   bin_time_flux_error, expand_and_rename_table)

# Constants for filtering stars
COLOR_TOLERANCE = 0.1
MAGNITUDE_TOLERANCE = 1


# Set up the logger
logger = logging.getLogger("rel_phot_logger")
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create file handler which logs even debug messages
log_filename = "relative_photometry.log"
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)
logger.addHandler(fh)


def target_info(table, tic_id_to_plot, APERTURE):
    target_star = table[table['tic_id'] == tic_id_to_plot]  # Extract the target star data
    # Extract the TESS magnitude of the target star
    target_tmag = target_star['Tmag'][0]
    target_flux = target_star[f'flux_{APERTURE}']  # Extract the flux of the target star
    target_fluxerr = target_star[f'fluxerr_{APERTURE}']  # Extract the flux error of the target star
    target_time = target_star['jd_bary']  # Extract the time of the target star
    target_sky = target_star[f'flux_w_sky_{APERTURE}'] - target_star[f'flux_{APERTURE}']
    target_color_index = target_star['gaiabp'][0] - target_star['gaiarp'][0]  # Extract the color index
    airmass_list = target_star['airmass']  # Extract airmass_list from target star
    zp_list = target_star['ZP']  # Extract the zero point of the target star
    # Calculate mean flux for the target star (specific to the chosen aperture)
    target_flux_mean = target_star[f'flux_{APERTURE}'].mean()

    return (target_tmag, target_color_index, airmass_list, target_flux_mean,
            target_sky, target_flux, target_fluxerr, target_time, zp_list)


def limits_for_comps(table, tic_id_to_plot, APERTURE, dmb, dmf, crop_size):
    # Get target star info including the mean flux
    target_tmag, target_color, airmass_list, target_flux_mean, _, _, _, _, _ = (
        target_info(table, tic_id_to_plot, APERTURE))

    # Filter based on color index within the tolerance
    color_index = table['gaiabp'] - table['gaiarp']
    color_mask = np.abs(color_index - target_color) <= COLOR_TOLERANCE
    color_data = table[color_mask]

    # Filter stars brighter than the target within dmb and fainter than the target within dmf
    mag_mask = (color_data['Tmag'] >= target_tmag - dmb) & (color_data['Tmag'] <= target_tmag + dmf)
    valid_color_mag_table = color_data[mag_mask]

    # Exclude stars with Tmag less than 9.4 and remove the target star from the table
    valid_color_mag_table = valid_color_mag_table[valid_color_mag_table['Tmag'] > 9.7]
    filtered_table = valid_color_mag_table[valid_color_mag_table['tic_id'] != tic_id_to_plot]

    # # If the target star is 12 mags or fainter, limit the comparison stars to 1000
    # if target_tmag >= 12:
    #     filtered_table = filtered_table[:1000]  # Limit to the first 1000 rows

    # Get target star coordinates
    x_target = table[table['tic_id'] == tic_id_to_plot]['x'][0]
    y_target = table[table['tic_id'] == tic_id_to_plot]['y'][0]

    # Apply crop filter based on coordinates
    if crop_size:
        x_min, x_max = x_target - crop_size // 2, x_target + crop_size // 2
        y_min, y_max = y_target - crop_size // 2, y_target + crop_size // 2

        # Further filter the comparison stars based on the crop region
        filtered_table = filtered_table[
            (filtered_table['x'] >= x_min) & (filtered_table['x'] <= x_max) &
            (filtered_table['y'] >= y_min) & (filtered_table['y'] <= y_max)
        ]

    return filtered_table, airmass_list


def find_comp_star_rms(comp_fluxes, airmass):
    comp_star_rms = []
    for i, flux in enumerate(comp_fluxes):
        airmass_cs = np.polyfit(airmass, flux, 1)
        airmass_mod = np.polyval(airmass_cs, airmass)
        flux_corrected = flux / airmass_mod
        flux_norm = flux_corrected / np.median(flux_corrected)
        rms_val = np.std(flux_norm)
        comp_star_rms.append(rms_val)
    return np.array(comp_star_rms)


def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0, sig_level=2., dmag=0.5):
    comp_star_rms = find_comp_star_rms(comp_fluxes, airmass)
    # logger.info(f"Initial RMS values for comparison stars: {comp_star_rms}")
    logger.info(f"Initial number of comparison stars: {len(comp_star_rms)}")
    comp_star_mask = np.array([True for _ in comp_star_rms])
    i = 0
    while True:
        i += 1
        comp_mags = np.copy(comp_mags0[comp_star_mask])
        comp_rms = np.copy(comp_star_rms[comp_star_mask])
        N1 = len(comp_mags)

        if N1 == 0:
            logger.info("No valid comparison stars left. Exiting.")
            break

        edges = np.arange(comp_mags.min(), comp_mags.max() + dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1] + edges[1:]) / 2.

        # Initialize std_medians and populate it based on bins
        std_medians = []
        for j in range(1, len(edges)):
            in_bin = comp_rms[dig == j]
            if len(in_bin) == 0:
                std_medians.append(np.nan)  # No stars in this bin
            else:
                std_medians.append(np.median(in_bin))

        std_medians = np.array(std_medians)

        # Remove NaN entries from std_medians and mag_nodes
        valid_mask = ~np.isnan(std_medians)
        mag_nodes = mag_nodes[valid_mask]
        std_medians = std_medians[valid_mask]

        # Handle case with too few points for spline fitting
        if len(mag_nodes) < 4 or len(std_medians) < 4:  # Less than 4 points
            logger.info("Not enough data for spline fitting. Trying linear interpolation.")
            if len(mag_nodes) > 1:
                mod = np.interp(comp_mags, mag_nodes, std_medians)  # Use linear interpolation
                mod0 = np.interp(comp_mags0, mag_nodes, std_medians)
            else:
                logger.info("Only one point available. Exiting.")
                break
        else:
            # Fit a spline to the medians if enough data
            spl = Spline(mag_nodes, std_medians)
            mod = spl(comp_mags)
            mod0 = spl(comp_mags0)

        std = np.std(comp_rms - mod)
        comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        N2 = np.sum(comp_star_mask)

        # the number of stars included and excluded
        logger.info(f"Iteration {i}: Stars included: {N2}, Stars excluded: {N1 - N2}")

        logger.info(f'Final stars included: {N2}')

        # Exit condition: no further changes or too many iterations
        if N1 == N2 or i > 10:
            break

    return comp_star_mask, comp_star_rms, i


def find_best_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size):
    # Filter the table based on color/magnitude tolerance
    filtered_table, airmass = limits_for_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size)
    tic_ids = np.unique(filtered_table['tic_id'])
    logger.info(f'Number of comparison stars after filtering: {len(tic_ids)}')

    if len(tic_ids) == 0:
        logger.warning(f"No valid comparison stars found for TIC ID {tic_id_to_plot}.")
        return None  # Return None if no comparison stars found

    comp_fluxes = []
    comp_mags = []
    reference_shape = None

    # Get the flux for the target TIC ID to compare later
    target_flux = table[table['tic_id'] == tic_id_to_plot][f'flux_{APERTURE}']

    if target_flux.size == 0:
        logger.warning(f"No flux data available for target TIC ID {tic_id_to_plot}.")
        return None

    # Initialize a list to track the valid TIC IDs after initial filtering
    valid_tic_ids = []

    for tic_id in tic_ids:
        flux = filtered_table[filtered_table['tic_id'] == tic_id][f'flux_{APERTURE}']
        tmag = filtered_table[filtered_table['tic_id'] == tic_id]['Tmag'][0]

        # Check shape consistency
        if reference_shape is None:
            reference_shape = flux.shape
        elif flux.shape != reference_shape:
            logger.warning(f"Shape mismatch for TIC ID {tic_id}: expected {reference_shape}, "
                           f"got {flux.shape}, skipping this comparison star.")
            continue  # Skip this flux array if shape does not match

        # If checks pass, add to lists and valid_tic_ids
        comp_fluxes.append(flux)
        comp_mags.append(tmag)
        valid_tic_ids.append(tic_id)  # Track only valid TIC IDs here

    # Convert lists to arrays for further processing
    comp_fluxes = np.array(comp_fluxes)
    comp_mags = np.array(comp_mags)
    tic_ids = np.array(valid_tic_ids)  # Update tic_ids to contain only valid IDs

    # Log dimensions before calling find_bad_comp_stars
    logger.info(f'The dimensions of these two are: {comp_mags.shape}, {comp_fluxes.shape}')

    # Ensure we have valid comparison stars before calling the next function
    if len(comp_fluxes) == 0:
        logger.warning(f"No valid comparison stars remaining for TIC ID {tic_id_to_plot} after checks.")
        return None

    # Call the function to find bad comparison stars
    try:
        comp_star_mask, comp_star_rms, iterations = find_bad_comp_stars(comp_fluxes, airmass, comp_mags)
    except Exception as e:
        logger.error(f"Error in find_bad_comp_stars for TIC ID {tic_id_to_plot}: {str(e)}")
        return None

    # Now, apply comp_star_mask to the already-filtered `tic_ids`
    good_tic_ids = tic_ids[comp_star_mask]  # `tic_ids` is already filtered, so dimensions will match

    # Now filter the table based on these tic_ids
    good_comp_star_table = filtered_table[np.isin(filtered_table['tic_id'], good_tic_ids)]

    return good_comp_star_table  # Return the filtered table including only good comp stars


def relative_phot(table, tic_id_to_plot, bin_size, APERTURE, DM_BRIGHT, DM_FAINT, crop_size):
    try:
        filtered_table = find_best_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size)

        if filtered_table is None or len(filtered_table) == 0:
            logger.warning(f"No valid comparison stars found for TIC ID {tic_id_to_plot}. Skipping.")
            return None

        (target_tmag, target_color_index, airmass_list, target_flux_mean,
         target_sky, target_flux, target_fluxerr, target_time, zp_list) = target_info(table, tic_id_to_plot, APERTURE)

        # Calculate the median sky value for our star
        sky_median = np.median(target_sky)

        tic_ids = np.unique(filtered_table['tic_id'])

        # Collect the shapes of the flux data for comparison stars
        flux_shapes = [filtered_table[filtered_table['tic_id'] == tic_id][f'flux_{APERTURE}'].shape for tic_id in
                       tic_ids]

        # Count the shapes and find the most common shape
        shape_counter = Counter(flux_shapes)
        most_common_shape, most_common_count = shape_counter.most_common(1)[0]
        logger.info(f"Most common shape: {most_common_shape} with count: {most_common_count}")

        # Check if the target TIC ID has the same shape
        target_flux_shape = table[table['tic_id'] == tic_id_to_plot][f'flux_{APERTURE}'].shape

        if target_flux_shape != most_common_shape:
            logger.warning(
                f"TIC ID {tic_id_to_plot} has shape {target_flux_shape}, which does not match the most "
                f"common shape {most_common_shape}. Skipping.")
            return None

        reference_fluxes = np.sum([filtered_table[filtered_table['tic_id'] == tic_id][f'flux_{APERTURE}']
                                   for tic_id in tic_ids], axis=0)
        logger.info(f"reference_fluxes shape after sum: {reference_fluxes.shape}")

        flux_ratio = target_flux / reference_fluxes
        flux_ratio_mean = np.mean(flux_ratio)
        dt_flux = flux_ratio / flux_ratio_mean
        dt_fluxerr = dt_flux * np.sqrt(
            (target_fluxerr / target_flux) ** 2 + (target_fluxerr[0] / target_flux[0]) ** 2)

        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(target_time, dt_flux,
                                                                             dt_fluxerr, bin_size)

        return (target_tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median,
                airmass_list, zp_list, target_color_index)

    except Exception as e:
        logger.error(f"Error in relative photometry for TIC ID {tic_id_to_plot}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Perform relative photometry for a given night')
    parser.add_argument('--bin_size', type=int, default=1, help='Number of images to bin')
    parser.add_argument('--aper', type=int, default=5, help='Aperture radius for photometry')
    parser.add_argument('--exposure', type=float, default=10, help='Exposure time for the images')
    parser.add_argument('--crop_size', type=int, default=1500, help='Size of the crop region around the target star')
    parser.add_argument('--dmb', type=float, default=0.2, help='Magnitude difference for comparison stars')
    parser.add_argument('--dmf', type=float, default=3.5, help='Magnitude difference for comparison stars')
    args = parser.parse_args()
    bin_size = args.bin_size
    APERTURE = args.aper
    EXPOSURE = args.exposure
    crop_size = args.crop_size
    DM_BRIGHT = args.dmb
    DM_FAINT = args.dmf

    # Set plot parameters
    plot_images()

    # Get the current night directory
    current_night_directory = os.getcwd()

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    logger.info(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        logger.info(f"Photometry file: {phot_file}")

        # Check if the output file already exists
        base_filename = phot_file.split('.')[0]  # Remove the file extension
        fits_filename = f"rel_{base_filename}_{APERTURE}_{bin_size}.fits"
        if os.path.exists(fits_filename):
            logger.info(f"Data for {phot_file} already saved to {fits_filename}. Skipping analysis.")
            continue

        # Create an empty list to store data for all TIC IDs
        data_list = []

        # Loop through all tic_ids in the photometry file
        for tic_id in np.unique(phot_table['tic_id']):
            print(f'The TIC_IDS to that will run is for total stars: {len(np.unique(phot_table["tic_id"]))}')
            # Check if all the Tmag values for the tic_id are less than or equal to 14
            if np.all(phot_table['Tmag'][phot_table['tic_id'] == tic_id] < 16):  # Adjusted threshold
                logger.info("")
                logger.info(f"Performing relative photometry for TIC ID = {tic_id} with Tmag = "
                            f"{phot_table['Tmag'][phot_table['tic_id'] == tic_id][0]:.3f}")
                # Perform relative photometry
                result = relative_phot(phot_table, tic_id, bin_size, APERTURE, DM_BRIGHT, DM_FAINT, crop_size)

                # Check if result is None
                if result is None:
                    logger.info(f"TIC ID {tic_id} skipped due to an error.")
                    continue

                # Unpack the result if it's not None
                (target_tmag, time_binned, dt_flux_binned, dt_fluxerr_binned, sky_median, airmass, zp, color) = result

                # Calculate RMS
                rms = np.std(dt_flux_binned)
                logger.info(f"RMS for TIC ID {tic_id} = {rms:.4f}")

                # Append data to the list with a check
                data_list.append((tic_id, target_tmag, time_binned, dt_flux_binned, dt_fluxerr_binned,
                                  sky_median, rms, airmass, zp, color))

                # Check the length of each data row before table creation
                for index, row in enumerate(data_list):
                    if len(row) != 10:
                        print(f"Row {index} has {len(row)} columns: {row}")

                # Create the table if all rows have correct length
                try:
                    data_table = Table(rows=data_list, names=('TIC_ID', 'Tmag', 'Time_BJD', 'Relative_Flux',
                                                              'Relative_Flux_err', 'Sky', 'RMS', 'Airmass',
                                                              'ZP', 'COLOR'))
                except ValueError as e:
                    print("Error creating Astropy table:", e)
                    raise

        expanded_data_table = expand_and_rename_table(data_table)

        expanded_data_table.write(fits_filename, format='fits', overwrite=True)

        logger.info(f"Data for {phot_file} saved to {fits_filename}.")


if __name__ == "__main__":
    main()
