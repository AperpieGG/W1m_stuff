#!/usr/bin/env python

import argparse
import os
import json
import numpy as np
from astropy.io import fits
from utils_W1m import noise_sources


# Constants for noise calculations


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


def read_data(filename):
    """
    Read data from a FITS file and return it as a structured array.

    Parameters:
    - filename: str, path to the FITS file.

    Returns:
    - data: structured array from the FITS file.
    """
    try:
        with fits.open(filename) as hdul:
            data = hdul[1].data  # Assuming data is in the first extension
    except Exception as e:
        print(f"Error reading FITS file: {e}")
        return None

    return data


def extract_zero_point(json_file_path):
    """
    Extracts the zero point value from a JSON file.

    Parameters:
        json_file_path (str): Path to the JSON file containing the zero point value.

    Returns:
        float: The extracted zero point value.
    """
    try:
        with open(json_file_path, 'r') as json_file:
            zero_point = json.load(json_file)
        print(f"Extracted zero point: {zero_point}")
        return zero_point
    except FileNotFoundError:
        print(f"File {json_file_path} not found.")
        return None
    except json.JSONDecodeError:
        print("Error decoding JSON from file.")
        return None


def main():
    """ Main function to parse arguments, read data, calculate noise sources, and save results to a JSON file """
    # Parse command-line arguments
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(
            description='Read and organize TIC IDs with associated '
                        'RMS, Sky, Airmass, ZP, and Magnitude from FITS table.'
                        'Example usage if you have CMOS: RN=1.56, DC=1.6, Aper=4, Exp=10.0, Bin=1'
                        'Example usage if you have CCD: RN=12.6, DC=0.00515, Aper=4, Exp=10.0, Bin=1')
    parser.add_argument('filename', type=str, help='Name of the FITS file containing photometry data')
    parser.add_argument('--bin_size', type=int, default=1, help='Bin size for noise calculations')
    parser.add_argument('--cam', type=str, default='CMOS', help='Camera type (CMOS or CCD)')
    args = parser.parse_args()
    filename = args.filename
    bin_size = args.bin_size
    if args.cam == 'CMOS':
        READ_NOISE = 1.56
        DARK_CURRENT = 1.6
        GAIN = 1.13
        APERTURE = 5
        EXPOSURE = 10.0
    else:
        READ_NOISE = 12.9
        DARK_CURRENT = 0.00515
        GAIN = 2
        APERTURE = 4
        EXPOSURE = 10.0

    # Get the current working directory
    current_dir = os.getcwd()

    # Construct full path to the FITS file
    file_path = os.path.join(current_dir, filename)

    # Read data from the FITS file
    data = read_data(file_path)
    if data is None:
        return

    # Extract unique TIC IDs
    unique_tic_ids = np.unique(data['TIC_ID'])

    # Initialize lists for each variable
    airmass_list, zp_list, color_list = [], [], []
    RMS_list, sky_list, Tmags_list = [], [], []

    # Iterate over each unique TIC ID
    for tic_id in unique_tic_ids:
        tic_data = data[data['TIC_ID'] == tic_id]
        airmass_list.extend(tic_data['Airmass'])
        zp_list.extend(tic_data['ZP'])

        # Append other data points as before
        if tic_data['RMS'][0] is not None:
            RMS_list.append(tic_data['RMS'][0] * 1000000)  # Convert RMS to ppm

        # Handle COLOR column
        if 'COLOR' in data.columns.names:  # Check if COLOR column exists
            try:
                color_list.append(tic_data['COLOR'][0])
            except (IndexError, TypeError):
                print(f"Missing or invalid COLOR value for TIC_ID {tic_id}. Defaulting to black.")
                color_list.append("black")  # Default to black if COLOR is missing
        else:
            print(f"'COLOR' column missing in FITS file. Defaulting all data to black.")
            color_list.append("black")  # Default to black for all if column is missing

        sky_list.append(tic_data['Sky'][0])
        Tmags_list.append(tic_data['Tmag'][0])

    # Convert lists to numpy arrays for calculations
    airmass_array = np.array(airmass_list)

    # the file has the form phot_prefix.fits, I want to extract only the prefix
    zp = extract_zero_point(f'zp{APERTURE}.json')
    print('Calculate zp and header zp avg is: ', np.mean(zp), np.mean(zp_list))
    print('The average sky brightness is: ', np.mean(sky_list))

    # Get noise sources
    synthetic_mag, photon_shot_noise, sky_noise, read_noise, dc_noise, N, RNS = (
        noise_sources(sky_list, bin_size, airmass_array, zp, APERTURE,
                      READ_NOISE, DARK_CURRENT, EXPOSURE, GAIN))

    # Convert lists to JSON serializable lists
    synthetic_mag_list = synthetic_mag.tolist()
    photon_shot_noise_list = photon_shot_noise.tolist()
    sky_noise_list = sky_noise.tolist()
    read_noise_list = read_noise.tolist()
    dc_noise_list = dc_noise.tolist()
    N_list = N.tolist()
    RNS_list = RNS.tolist()
    Tmags_list = [float(x) for x in Tmags_list]
    color_list = [float(x) for x in color_list]

    # Save RMS_list, mags_list, and other lists to a JSON file
    output_data = {
        "TIC_IDs": unique_tic_ids.tolist(),
        "RMS_list": RMS_list,
        "Tmag_list": Tmags_list,
        "synthetic_mag": synthetic_mag_list,
        "photon_shot_noise": photon_shot_noise_list,
        "sky_noise": sky_noise_list,
        "read_noise": read_noise_list,
        "dc_noise": dc_noise_list,
        "N": N_list,
        "RNS": RNS_list,
        "COLOR": color_list
    }

    # Construct output file name
    cwd_last_four = os.getcwd()[-4:]
    file_name = f"rms_mags_{filename.replace('.fits', '')}_{cwd_last_four}.json"
    save_path = os.getcwd()
    output_path = os.path.join(save_path, file_name)

    # Save JSON file using custom encoder
    with open(output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4, cls=NumpyEncoder)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
