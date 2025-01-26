#!/usr/bin/env python
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_phot_files, read_phot_file, plot_images

plot_images()


class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def measure_zp(table, APERTURE, EXPOSURE):
    tic_ids = np.unique(table['TIC_ID'])
    print(f'Found {len(tic_ids)} unique TIC IDs')
    zp_list = []
    color_list = []
    avg_flux = []
    mags = []

    for tic_id in tic_ids:
        # Average flux for the current TIC ID
        tic_data = table[table['TIC_ID'] == tic_id]

        tic_flux = np.mean(tic_data[f'flux_{APERTURE}'])
        if APERTURE == 4:
            # multiply flux by CCD gain that is 2
            tic_flux *= 2
        else:
            # multiply flux by CMOS gain that is 1.131
            tic_flux *= 1.131
        # First Tmag value for the current TIC ID
        tic_Tmag = tic_data['Tmag'][0]
        target_color_index = tic_data['gaiabp'][0] - tic_data['gaiarp'][0]

        # Calculate zero point for the current TIC ID
        zp = tic_Tmag + 2.5 * np.log10(tic_flux / EXPOSURE)
        print(f'TIC ID: {tic_id}, Zero Point: {zp}, Color Index: {target_color_index}')
        zp_list.append(zp)
        color_list.append(target_color_index)
        avg_flux.append(tic_flux)
        mags.append(tic_Tmag)

    return zp_list, color_list, avg_flux, mags


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Read and organize TIC IDs with associated '
                    'RMS, Sky, Airmass, ZP, and Magnitude from FITS table.'
                    'Example usage if you have CMOS: RN=1.56, DC=1.6, Aper=4, Exp=10.0, Bin=1'
                    'Example usage if you have CCD: RN=12.6, DC=0.00515, Aper=4, Exp=10.0, Bin=1')
    parser.add_argument('--exp', type=float, default=10.0, help='Exposure time in seconds')
    parser.add_argument('--aper', type=str, default=6, help='Aperture size in meters')
    args = parser.parse_args()
    EXPOSURE = args.exp
    APERTURE = args.aper  # Aperture size for the telescope

    # Get the current night directory
    current_night_directory = os.getcwd()

    # Get photometry files with the pattern 'phot_*.fits'
    phot_files = get_phot_files(current_night_directory)
    print(f"Photometry files: {phot_files}")

    # Loop through photometry files
    for phot_file in phot_files:
        phot_table = read_phot_file(os.path.join(current_night_directory, phot_file))

        print(f"Photometry file: {phot_file}")

        # Measure zero point
        zp_list, color_list, flux_list, tmag_list = measure_zp(phot_table, APERTURE, EXPOSURE)

        # save the results to a json file
        with open(f'zp{APERTURE}.json', 'w') as json_file:
            json.dump(np.nanmean(zp_list), json_file, indent=4)

        print(f"Results saved to zp{APERTURE}.json")

        # Filter out entries where either zp or color is NaN
        valid_data = [{'zp': zp, 'color': color, 'flux': flux, 'tmag': tmag}
                      for zp, color, flux, tmag in zip(zp_list, color_list, flux_list, tmag_list)
                      if not np.isnan(zp) and not np.isnan(color) and not np.isnan(flux) and not np.isnan(tmag)]

        # Extract zp_list and color_list separately after filtering
        filtered_zp_list = [entry['zp'] for entry in valid_data]
        filtered_color_list = [entry['color'] for entry in valid_data]
        filtered_flux_list = [entry['flux'] for entry in valid_data]
        filtered_tmag_list = [entry['tmag'] for entry in valid_data]
        print(f"Zero point average: {np.nanmean(filtered_zp_list)}")

        # color-coded with color index
        plt.scatter(filtered_tmag_list, filtered_flux_list, c=filtered_color_list, cmap='coolwarm', vmin=0.5, vmax=1.5)
        plt.colorbar(label='Color Index')
        plt.yscale('log')
        plt.xlabel('Flux')
        plt.ylabel('Tmag')
        plt.title('Flux vs Tmag')
        # revert the x-axis
        plt.gca().invert_xaxis()
        plt.xlim(9, 16)
        plt.show()

        with open(f'zp{APERTURE}_list.json', 'w') as json_file:
            # Save the filtered lists to a JSON file
            json.dump({'zp_list': filtered_zp_list, 'color_list': filtered_color_list,
                       'flux_list': filtered_flux_list, 'tmag_list': filtered_tmag_list},
                      json_file,  # Pass the file pointer here
                      cls=NumpyEncoder)

        print(f"Results saved to zp{APERTURE}_list.json")


if __name__ == "__main__":
    main()
