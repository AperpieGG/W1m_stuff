#! /usr/bin/env python
import glob
import json
import numpy as np
from matplotlib import pyplot as plt, ticker
from astropy.table import Table, vstack
import argparse
from utils_W1m import bin_time_flux_error, plot_images

plot_images()


def load_all_jsons_as_table():
    """Load all JSON photometry files and return as a combined Astropy Table."""
    all_tables = []

    for json_file in glob.glob(f"target*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            data = data[0]  # assume list of dicts

        row_count = len(data["Time_BJD"])
        table = Table({
            "TIC_ID": [data["TIC_ID"]] * row_count,
            "Time_BJD": data["Time_BJD"],
            "Relative_Flux": data["Relative_Flux"],
            "Relative_Flux_err": data["Relative_Flux_err"],
            "RMS": [data["RMS"]] * row_count,
        })

        all_tables.append(table)

    return vstack(all_tables)


def compute_rms_for_star(star_data, max_binning, exp):
    """Compute RMS vs binning for one star."""
    if len(star_data) > 300:
        star_data = star_data[:-300]
    elif len(star_data) < 300:
        star_data = star_data[100:]

    jd_mid = star_data['Time_BJD']
    rel_flux = star_data['Relative_Flux']
    rel_fluxerr = star_data['Relative_Flux_err']

    RMS_values = []
    time_seconds = []

    for i in range(1, max_binning):
        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(jd_mid, rel_flux, rel_fluxerr, i)
        RMS = np.std(dt_flux_binned)
        RMS_values.append(RMS)
        time_seconds.append(i * exp)

    return time_seconds, np.array(RMS_values)


def compute_rms_values(phot_table, args):
    """Compute RMS vs binning and return the median curve over all stars."""
    tic_ids = np.unique(phot_table['TIC_ID'])
    print(f"Total stars in brightness range: {len(tic_ids)}")

    all_rms_values = []
    times_binned = []

    for tic_id in tic_ids:
        star_data = phot_table[phot_table['TIC_ID'] == tic_id]
        time_seconds, RMS_values = compute_rms_for_star(star_data, args.bin, args.exp)
        all_rms_values.append(RMS_values)
        times_binned.append(time_seconds)

    median_rms = np.median(all_rms_values, axis=0) * 1e6  # ppm
    times_binned = times_binned[0]
    RMS_model = median_rms[0] / np.sqrt(np.arange(1, args.bin))

    return times_binned, median_rms, RMS_model


def plot_timescale(times, rms_values, RMS_model, title):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(times, rms_values, 'o', label="RMS (data)", color="blue")
    ax.plot(times, RMS_model, '--', label="Model", color="blue")
    ax.axvline(x=900, color='black', linestyle='--', label='Reference (900s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Exposure Time (s)')
    ax.set_ylabel('RMS (ppm)')
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate RMS vs time binning from all JSON files")
    parser.add_argument('--bin', type=int, default=180, help='Max binning size')
    parser.add_argument('--exp', type=int, default=10, help='Exposure time in seconds')
    parser.add_argument('--tic_id', type=str, help='Optional TIC ID to plot individually')
    args = parser.parse_args()

    phot_table = load_all_jsons_as_table()

    if args.tic_id:
        # plot only for a specific TIC_ID
        tic_id = args.tic_id
        if tic_id not in np.unique(phot_table['TIC_ID']):
            print(f"TIC {tic_id} not found in JSON files.")
        else:
            star_data = phot_table[phot_table['TIC_ID'] == tic_id]
            times, rms_values = compute_rms_for_star(star_data, args.bin, args.exp)
            rms_values_ppm = rms_values * 1e6
            RMS_model = rms_values_ppm[0] / np.sqrt(np.arange(1, args.bin))
            plot_timescale(times, rms_values_ppm, RMS_model, f"TIC {tic_id} RMS vs Timescale")
    else:
        # default: median over all stars
        times, avg_rms, RMS_model = compute_rms_values(phot_table, args=args)

        # Save results
        output_data = {
            "Time_Binned": times,
            "Median_RMS_ppm": avg_rms.tolist(),
            "RMS_Model_ppm": RMS_model.tolist()
        }
        with open("rms_vs_timescale.json", "w") as outfile:
            json.dump(output_data, outfile, indent=2)

        print("Saved RMS vs timescale results to rms_vs_timescale.json")
        plot_timescale(times, avg_rms, RMS_model, "Median RMS vs Timescale")