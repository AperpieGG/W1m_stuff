#!/usr/bin/env python3
"""
Plot HFD statistics (Histogram, vs Time, vs Airmass, vs Zeropoint) from FITS headers.

Usage:
python plot_hfd_from_headers.py /path/to/fits_directory
"""
from utils_W1m import plot_images
import fitsio
from path import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import warnings
from tqdm import tqdm
import os

plot_images()


def find_images(path):
    """Find all FITS images excluding calibration frames."""
    image_files = sorted(Path(path).files('*.fits'))
    print(f"Found {len(image_files)} FITS files in {path}.")
    exclude_words = ['bias', 'flat', 'catalog', 'dark', 'phot', 'evening', 'morning']
    image_files = [f for f in image_files if not any(word in f.lower() for word in exclude_words)]
    print(f"{len(image_files)} FITS files remain after filtering.")
    return image_files


def read_header_values(file):
    """Read HFD, AIRMASS, DATE-OBS, and MAGZP_T (zeropoint) from FITS header."""
    try:
        header = fitsio.read_header(file)
    except Exception:
        return None, None, None, None

    if 'HFD' not in header:
        return None, None, None, None

    hfd = header['HFD']
    airmass = header.get('AIRMASS', np.nan)
    magzp_t = header.get('MAGZP_T', np.nan)
    date_obs = header.get('DATE-OBS', header.get('JD', header.get('MJD', None)))

    # Convert DATE-OBS or JD/MJD to datetime
    time = None
    if isinstance(date_obs, str):
        try:
            time = datetime.fromisoformat(date_obs.replace('Z', ''))
        except Exception:
            pass
    elif isinstance(date_obs, (float, int)):
        jd = float(date_obs)
        if jd > 2400000:  # JD
            jd -= 2400000.5  # convert to MJD
        mjd_ref = datetime(1858, 11, 17)
        time = mjd_ref + timedelta(days=jd)

    return hfd, airmass, time, magzp_t


def main():
    parser = argparse.ArgumentParser(description="Plot HFD distribution, HFD vs Time, "
                                                 "HFD vs Airmass, and HFD vs Zeropoint from FITS headers.")
    parser.add_argument("input_dir", type=str, help="Directory containing FITS images.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_files = find_images(input_dir)
    # Create output directory if it doesn’t exist
    output_dir = input_dir + "shifts_plots"
    os.makedirs(output_dir, exist_ok=True)

    hfds, airmasses, times, magzps = [], [], [], []

    print(f"\nReading HFDs, Airmass, Zeropoint, and Time from {len(image_files)} images...")
    for file in tqdm(image_files):
        hfd, airmass, time, magzp_t = read_header_values(file)
        if hfd is not None:
            hfds.append(hfd)
            airmasses.append(airmass)
            times.append(time)
            magzps.append(magzp_t)

    hfds = np.array(hfds)
    airmasses = np.array(airmasses)
    magzps = np.array(magzps)
    times = np.array(times, dtype=object)

    valid_mask = np.isfinite(hfds)
    hfds, airmasses, magzps, times = hfds[valid_mask], airmasses[valid_mask], magzps[valid_mask], times[valid_mask]

    print(f"\nValid HFDs found in {len(hfds)} images.")
    if len(hfds) == 0:
        print("No valid HFD values found in FITS headers.")
        return
    #
    # --- Plot 1: Histogram of all HFDs ---
    plt.figure(figsize=(8, 5))
    plt.hist(hfds, bins=40, color='blue')
    plt.axvline(np.median(hfds), color='r', linestyle='--', label=f"Median = {np.median(hfds):.2f}\"")
    plt.xlabel("HFD (arcsec)")
    plt.ylabel("Count")
    plt.title("HFD Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # Use the title of the left subplot as filename
    filename = "hfd_histogram.pdf"
    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, format='pdf')
    plt.close()

    # --- Plot 2: HFD vs Time ---
    valid_times = [t for t in times if t is not None]
    if len(valid_times) > 0:
        valid_times_str = [t.strftime("%H:%M:%S") for t in valid_times]

        # Decide how many ticks you want on the x-axis
        num_ticks = 6  # for example, 6 ticks
        tick_indices = np.linspace(0, len(valid_times_str) - 1, num_ticks, dtype=int)
        tick_labels = [valid_times_str[i] for i in tick_indices]

        plt.figure(figsize=(8, 5))
        plt.scatter(valid_times_str, hfds, color='blue', s=30)
        plt.xlabel("Time (UTC)")
        plt.ylabel("HFD (arcsec)")
        plt.title("HFD vs Time")
        plt.ylim(7, 11)
        plt.xticks(tick_indices, tick_labels, rotation=45)  # Set ticks with limited labels

        plt.grid(alpha=0.3)
        plt.tight_layout()
        # Use the title of the left subplot as filename
        filename = "hfd_vs_time.pdf"
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        print("⚠️ No valid DATE-OBS or JD values found — skipping HFD vs Time plot.")

    # --- Plot 3: HFD vs Airmass ---
    if np.any(np.isfinite(airmasses)):
        plt.figure(figsize=(8, 5))
        plt.scatter(valid_times_str, airmasses, color='blue', s=30)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Airmass")
        plt.title("Airmass vs Time")
        plt.grid(alpha=0.3)
        plt.xticks(tick_indices, tick_labels, rotation=45)  # Set ticks with limited labels
        plt.tight_layout()
        # Use the title of the left subplot as filename
        filename = "airmass_vs_time.pdf"
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        print("⚠️ No valid AIRMASS values found — skipping HFD vs Airmass plot.")

    # --- Plot 4: Zeropoint vs HFD ---
    if np.any(np.isfinite(magzps)):
        plt.figure(figsize=(8, 5))
        plt.scatter(hfds, magzps, color='blue', s=30)
        plt.ylabel("Zeropoint MAGZP_T")
        plt.xlabel("HFD (arcsec)")
        plt.title("HFD vs Zeropoint")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        # Use the title of the left subplot as filename
        filename = "hfd_vs_zp.pdf"
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        print("⚠️ No valid MAGZP_T values found — skipping HFD vs Zeropoint plot.")

    # --- Plot 5: Zeropoint vs Airmass ---
    if np.any(np.isfinite(magzps)) and np.any(np.isfinite(airmasses)):
        plt.figure(figsize=(8, 5))
        plt.scatter(airmasses, magzps, color='blue', s=30)
        plt.xlabel("Airmass")
        plt.ylabel("Zeropoint MAGZP_T")
        plt.title("Zeropoint vs Airmass")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        # Use the title of the left subplot as filename
        filename = "zp_vs_airmass.pdf"
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        print("⚠️ No valid MAGZP_T or AIRMASS values found — skipping Zeropoint vs Airmass plot.")

    # --- Plot 6: Zeropoint vs Time ---
    if np.any(np.isfinite(magzps)) and len(valid_times) > 0:
        plt.figure(figsize=(8, 5))
        plt.scatter(valid_times_str, magzps, color='blue', s=30)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Zeropoint MAGZP_T")
        plt.title("Zeropoint vs Time")
        plt.xticks(tick_indices, tick_labels, rotation=45)  # Set ticks with limited labels
        plt.grid(alpha=0.3)
        plt.tight_layout()
        # Use the title of the left subplot as filename
        filename = "zp_vs_time.pdf"
        save_path = os.path.join(output_dir, filename)

        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        print("⚠️ No valid MAGZP_T or DATE-OBS/JD values found — skipping Zeropoint vs Time plot.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
