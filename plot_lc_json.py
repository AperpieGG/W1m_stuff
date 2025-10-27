#!/usr/bin/env python
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils_W1m import plot_images, bin_time_flux_error, bin_by_time_interval


def load_json(file_path):
    """
    Load JSON file and return data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def main(target_file):
    # Display any necessary images
    plot_images()

    # Load the data from the target JSON file
    data = load_json(target_file)

    # Extract relevant data
    # 1561 for CCD, 2000 for CMOS for WASP-30
    time = np.array(data['Time_BJD'])
    flux = np.array(data['Relative_Flux'])
    flux_err = np.array(data['Relative_Flux_err'])
    print(f'Time length: {len(time)}, Flux length: {len(flux)}, Flux Error length: {len(flux_err)}')

    # Bin the data, for CMOS is 30 for 5 min and for CCD is 23 (10 + 3 seconds readout)
    time_binned, flux_binned, fluxerr_binned = bin_by_time_interval(time, flux, flux_err, 0.1677)
    print(f'Binned Time length: {len(time_binned)}, Binned Flux length: {len(flux_binned)}, '
          f'Binned Flux Error length: {len(fluxerr_binned)}')

    # Create a single figure and axis
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    # Plot the light curve
    ax.plot(time_binned, flux_binned, 'ro')

    # Labels and legend
    ax.set_xlabel('Time (BJD)')
    ax.set_ylabel('Relative Flux')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process Light Curve with Wotan")
    parser.add_argument('--target_file', type=str, required=True, help="Path to the target JSON file")
    args = parser.parse_args()

    # Execute the main function with parsed arguments
    main(args.target_file)

