#!/usr/bin/env python
"""
Description: This script optimizes photometric parameters for a given target star by iterating through various
 combinations of parameters and calculating the resulting RMS values.
It uses comparison stars to improve the photometric precision. This is the updated photometry script
The main pipeline will run in run_photometry.sh file
"""


import itertools
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from itertools import product
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from utils_W1m import plot_images, read_phot_file, bin_time_flux_error, \
    remove_outliers, bin_by_time_interval, calc_noise, get_phot_files, target_info

plot_images()


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


def find_bad_comp_stars(comp_fluxes, airmass, comp_mags0, sig_level=1.5, dmag=0.5):
    # Calculate initial RMS of comparison stars
    comp_star_rms = find_comp_star_rms(comp_fluxes, airmass)
    print(f"Initial number of comparison stars: {len(comp_star_rms)}")

    comp_star_mask = np.ones(len(comp_star_rms), dtype=bool)
    i = 0

    while True:
        i += 1
        comp_mags = comp_mags0[comp_star_mask]
        comp_rms = comp_star_rms[comp_star_mask]
        N1 = len(comp_mags)

        if N1 == 0:
            print("No valid comparison stars left. Exiting.")
            break

        edges = np.arange(comp_mags.min(), comp_mags.max() + dmag, dmag)
        dig = np.digitize(comp_mags, edges)
        mag_nodes = (edges[:-1] + edges[1:]) / 2.

        # Calculate median RMS per bin
        std_medians = np.array([np.median(comp_rms[dig == j]) if len(comp_rms[dig == j]) > 0 else np.nan
                                for j in range(1, len(edges))])

        # Remove NaNs from std_medians and mag_nodes
        valid_mask = ~np.isnan(std_medians)
        mag_nodes = mag_nodes[valid_mask]
        std_medians = std_medians[valid_mask]

        # Handle too few points for fitting
        if len(mag_nodes) < 4:
            if len(mag_nodes) > 1:
                mod = np.interp(comp_mags, mag_nodes, std_medians)
                mod0 = np.interp(comp_mags0, mag_nodes, std_medians)
            else:
                print("Not enough points for fitting. Exiting.")
                break
        else:
            spl = Spline(mag_nodes, std_medians, k=3)
            mod = spl(comp_mags)
            mod0 = spl(comp_mags0)

        std = np.std(comp_rms - mod)
        comp_star_mask = (comp_star_rms <= mod0 + std * sig_level)
        N2 = np.sum(comp_star_mask)

        print(f"Iteration {i}: Stars included: {N2}, Stars excluded: {N1 - N2}")

        # Exit if the number of stars doesn't change or too many iterations
        if N1 == N2 or i > 11:
            break

    print(f'RMS of comparison stars after filtering: {len(comp_star_rms[comp_star_mask])}')
    # print(f'RMS values after filtering: {comp_star_rms[comp_star_mask]}')

    return comp_star_mask, comp_star_rms, i


def limits_for_comps(table, tic_id_to_plot, APERTURE, dmb, dmf, crop_size, color_lim):
    # Get target star info including the mean flux
    # Get target star info including the mean flux
    target_tmag, target_color, airmass_list, target_flux_mean, _, _, _, _, _ = (
        target_info(table, tic_id_to_plot, APERTURE))

    # Filter based on color index within the tolerance
    color_index = table['gaiabp'] - table['gaiarp']
    color_mask = np.abs(color_index - target_color) <= color_lim
    color_data = table[color_mask]

    # Filter stars brighter than the target within dmb and fainter than the target within dmf
    mag_mask = (color_data['Tmag'] >= target_tmag - dmb) & (color_data['Tmag'] <= target_tmag + dmf)
    valid_color_mag_table = color_data[mag_mask]

    # Exclude stars with Tmag less than 9.4 and remove the target star from the table
    valid_color_mag_table = valid_color_mag_table[valid_color_mag_table['Tmag'] > 9.4]
    filtered_table = valid_color_mag_table[valid_color_mag_table['tic_id'] != tic_id_to_plot]

    if crop_size:
        # Get target star coordinates
        x_target = table[table['tic_id'] == tic_id_to_plot]['x'][0]
        y_target = table[table['tic_id'] == tic_id_to_plot]['y'][0]

        # Apply crop filter based on coordinates
        x_min, x_max = x_target - crop_size // 2, x_target + crop_size // 2
        y_min, y_max = y_target - crop_size // 2, y_target + crop_size // 2

        # Further filter the comparison stars based on the crop region
        filtered_table = filtered_table[
            (filtered_table['x'] >= x_min) & (filtered_table['x'] <= x_max) &
            (filtered_table['y'] >= y_min) & (filtered_table['y'] <= y_max)
            ]

    return filtered_table, airmass_list


def find_best_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size, color_lim):
    # Filter the table based on color/magnitude tolerance
    filtered_table, airmass = limits_for_comps(table, tic_id_to_plot, APERTURE, DM_BRIGHT, DM_FAINT, crop_size,
                                               color_lim)
    # Remove bad comparison stars

    tic_ids = np.unique(filtered_table['tic_id'])
    print(f'Number of comparison stars after filtering: {len(tic_ids)}')

    comp_fluxes = []
    comp_mags = []
    valid_tic_ids = []

    for tic_id in tic_ids:
        flux = filtered_table[filtered_table['tic_id'] == tic_id][f'flux_{APERTURE}']
        tmag = filtered_table[filtered_table['tic_id'] == tic_id]['Tmag'][0]

        comp_fluxes.append(flux)
        comp_mags.append(tmag)
        valid_tic_ids.append(tic_id)

    # Find the maximum length of flux arrays
    max_length = max(len(f) for f in comp_fluxes)

    # Filter out entries where flux arrays are shorter than max_length
    filtered_data = [(f, m, t) for f, m, t in zip(comp_fluxes, comp_mags, valid_tic_ids) if len(f) == max_length]

    # Unzip filtered data back into separate lists
    comp_fluxes, comp_mags, tic_ids = zip(*filtered_data)

    # Convert lists to arrays for further processing
    comp_fluxes = np.array(comp_fluxes)
    comp_mags = np.array(comp_mags)
    tic_ids = np.array(tic_ids)

    # Check if comp_mags is non-empty before proceeding
    if len(comp_mags) == 0:
        raise ValueError("No valid comparison stars found after filtering for flux and magnitude.")

    # Call the function to find bad comparison stars
    print(f'The dimensions of these two are: {comp_mags.shape}, {comp_fluxes.shape}')
    comp_star_mask, comp_star_rms, iterations = find_bad_comp_stars(comp_fluxes, airmass, comp_mags)

    # Filter the table based on the mask
    # print(f'Star with the min rms: {np.min(comp_star_rms)} and tic_id: {tic_ids[np.argmin(comp_star_rms)]}')

    # Filter tic_ids based on the mask
    good_tic_ids = tic_ids[comp_star_mask]

    # Now filter the table based on these tic_ids
    good_comp_star_table = filtered_table[np.isin(filtered_table['tic_id'], good_tic_ids)]

    return good_comp_star_table, airmass  # Return the filtered table including only good comp stars


def run_photometry(tic_id, dmb, dmf, crop, color_lim, cam, APERTURE):
    try:
        # (All your main() logic moved here with 'args.dmb' -> dmb etc.)
        # Skip CLI parser and set values directly
        if cam == 'IMX571-HWC':
            DC = 0.001
            GAIN = 0.75
            EXPOSURE = 10.0
            RN = 3.16
        elif cam == 'IMX571-LN':
            GAIN = 0.25
            DC = 0.001
            EXPOSURE = 10.0
            RN = 1.28
        elif cam == 'IMX571-LN12':
            GAIN = 3.85
            DC = 0.001
            EXPOSURE = 10.0
            RN = 2.85

        directory = '.'
        phot_file = get_phot_files(directory)[0]
        phot_table = read_phot_file(os.path.join(directory, phot_file))

        if tic_id not in np.unique(phot_table['tic_id']):
            return np.inf  # or None

        best_comps_table, AIRMASS = find_best_comps(phot_table, tic_id, APERTURE, dmb, dmf, crop, color_lim)
        tic_ids = np.unique(best_comps_table['tic_id'])

        # Check if any comparison stars are left
        if len(best_comps_table) == 0:
            print(f"No comparison stars found for TIC ID {tic_id} with parameters "
                  f"dmb={dmb}, dmf={dmf}, crop={crop}, color_lim={color_lim}. Skipping to next combination.")
            return np.inf, np.inf

        time_list = []
        flux_list = []
        fluxerr_list = []
        sky_list = []

        for tic_id_ in tic_ids:
            comp_time = best_comps_table[best_comps_table['tic_id'] == tic_id_]['jd_bary']
            comp_fluxes = best_comps_table[best_comps_table['tic_id'] == tic_id_][f'flux_{APERTURE}']
            comp_fluxerrs = best_comps_table[best_comps_table['tic_id'] == tic_id_][f'fluxerr_{APERTURE}']
            comp_sky = (phot_table[phot_table['tic_id'] == tic_id_][f'flux_w_sky_{APERTURE}'] -
                        phot_table[phot_table['tic_id'] == tic_id_][f'flux_{APERTURE}'])

            time_list.append(np.array(comp_time))
            flux_list.append(np.array(comp_fluxes))
            fluxerr_list.append(np.array(comp_fluxerrs))
            sky_list.append(np.array(comp_sky))

        reference_fluxes = np.sum(flux_list, axis=0)
        comp_errs = np.vstack(([calc_noise(APERTURE, EXPOSURE, DC, GAIN, RN, AIRMASS, cfi + csi)
                                for cfi, csi in zip(flux_list, sky_list)]))
        reference_fluxerrs = np.sqrt(np.sum(comp_errs ** 2, axis=0))

        target_star = phot_table[phot_table['tic_id'] == tic_id]
        target_flux = target_star[f'flux_{APERTURE}']
        target_fluxerr = target_star[f'fluxerr_{APERTURE}']
        target_sky = target_star[f'flux_w_sky_{APERTURE}'] - target_star[f'flux_{APERTURE}']
        target_time = target_star['jd_bary']
        target_err = calc_noise(APERTURE, EXPOSURE, DC, GAIN, RN, AIRMASS, target_flux + target_sky)

        flux_ratio = target_flux / reference_fluxes
        flux_ratio_mean = np.median(flux_ratio)
        target_fluxes_dt = flux_ratio / flux_ratio_mean

        err_factor = np.sqrt((target_err / target_flux) ** 2 + (reference_fluxerrs / reference_fluxes) ** 2)
        flux_err = flux_ratio * err_factor
        target_flux_err_dt = flux_err / flux_ratio_mean

        target_time, target_fluxes_dt, target_flux_err_dt, _, _ = (
            remove_outliers(target_time, target_fluxes_dt, target_flux_err_dt))

        rms_unbinned = np.std(target_fluxes_dt) * 1e6  # Convert to ppm

        target_time_binned, target_fluxes_binned, target_fluxerrs_binned = (
            bin_by_time_interval(target_time, target_fluxes_dt, target_flux_err_dt, 30))

        RMS_binned = np.std(target_fluxes_binned) * 1e6
        return RMS_binned, rms_unbinned
    except Exception as e:
        print(f"Error for dmb={dmb}, dmf={dmf}, crop={crop}, color_lim={color_lim}: {e}")
        return np.inf, np.inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize photometric parameters.")
    parser.add_argument('--tic_id', type=int, required=True, help="Target TIC ID")
    parser.add_argument('--cam', type=str, default='IMX571-HWC',
                        help="Camera type (IMX571-HWC, IMX571-LN, IMX571-LN12)")
    parser.add_argument('--aper', nargs='+', type=int, default=[10, 20, 30, 40],
                        help="Apertures to include in parameter search (e.g. --aper 10 20 30 40)")
    args = parser.parse_args()

    tic_id = args.tic_id

    # Load the photometry table
    phot_file = get_phot_files('.')  # This returns a list of file names
    phot_data = read_phot_file(os.path.join('.', phot_file[0]))

    # Filter to get the Tmag for the specific TIC ID
    target_row = phot_data[phot_data['tic_id'] == tic_id]
    if len(target_row) == 0:
        raise ValueError(f"TIC ID {tic_id} not found in photometry file.")
    tmag = target_row['Tmag'][0]
    print(f'The Tmag for TIC ID {tic_id} is {tmag}')
    del phot_data

    # --- Parameter grids ---
    APERTURE = args.aper
    dmb_range = [0.1, 0.5, 1, 1.5]
    dmf_range = [0.1, 0.5, 1.5, 2, 2.5]
    crop_range = [None, 1000, 1500, 3000, 4000]
    color_lim_range = [0.1, 0.2, 0.4, 0.6]
    tolerance = 100

    # --- Target RMS based on brightness ---
    if 7.5 < tmag < 9.5:
        dmb_range = [0.0]
        target_rms = 200
    elif 9.5 < tmag < 10:
        dmb_range = [0.0]
        target_rms = 300
    elif 10 < tmag < 11:
        target_rms = 400
    elif 11 <= tmag < 12:
        target_rms = 500
    elif 12 <= tmag < 13:
        target_rms = 600
    elif 13 <= tmag < 14:
        target_rms = 800
    elif 14 <= tmag < 15:
        target_rms = 1000
    elif 15 <= tmag < 16:
        target_rms = 2000

    best_rms = np.inf
    best_params = None
    found_optimal = False

    # --- Unified parameter iteration including aperture ---
    for aper, dmb, dmf, crop, color_lim in itertools.product(APERTURE, dmb_range, dmf_range, crop_range, color_lim_range):
        rms, rms_unbinned = run_photometry(tic_id, dmb, dmf, crop, color_lim, args.cam, aper)
        print(f"[aper={aper}] dmb={dmb}, dmf={dmf}, crop={crop}, color_lim={color_lim} "
              f"=> RMS: {rms:.2f}, RMS_unbinned: {rms_unbinned:.2f}")

        if np.abs(rms - target_rms) <= tolerance:
            print(f"\n🎯 Found optimal config! RMS = {rms:.2f} for aperture {aper}")
            print(f"Params => aper: {aper}, dmb: {dmb}, dmf: {dmf}, crop: {crop}, color_lim: {color_lim}")
            found_optimal = True
            with open("best_params_log.txt", "a") as f:
                cmd = (f"/home/ops/Apergis/W1m_stuff/rel_dev_dev.py {tic_id} --aper {aper} "
                       f"--dmb {dmb} --dmf {dmf}")
                if crop is not None:
                    cmd += f" --crop {crop}"
                cmd += f" --color {color_lim} --cam {args.cam}  # Found RMS: {rms:.2e}, RMS_unbinned: {rms_unbinned:.2f}\n"
                f.write(cmd)
            break

        if rms < best_rms:
            best_rms = rms
            best_rms_unbinned = rms_unbinned
            best_params = (aper, dmb, dmf, crop, color_lim)

    # --- Final summary ---
    if not found_optimal and best_params is not None:
        print(f"\n🔍 Best RMS found (not within target tolerance): {best_rms:.2f}, RMS_unbinned: {best_rms_unbinned:.2f}")
        print(f"Best parameters: aper={best_params[0]}, dmb={best_params[1]}, dmf={best_params[2]}, "
              f"crop={best_params[3]}, color_lim={best_params[4]}")
        with open("best_params_log.txt", "a") as f:
            cmd = (f"/home/ops/Apergis/W1m_stuff/rel_dev_dev.py {tic_id} "
                   f"--aper {best_params[0]} --dmb {best_params[1]} --dmf {best_params[2]}")
            if best_params[3] is not None:
                cmd += f" --crop {best_params[3]}"
            cmd += (f" --color {best_params[4]} --cam {args.cam}  # Best RMS: {best_rms:.2e}, "
                    f"RMS_unbinned: {best_rms_unbinned:.2f}\n")
            f.write(cmd)
    elif not found_optimal:
        print("⚠️ No valid RMS found.")
