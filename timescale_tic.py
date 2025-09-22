#! /usr/bin/env python
import argparse
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt, ticker
from utils_W1m import bin_time_flux_error, plot_images

plot_images()


def load_fits_data(fits_file, tic_id_to_plot):
    """Load and parse FITS file containing photometry data for a single TIC_ID."""
    with fits.open(fits_file) as hdul:
        table = hdul[1].data  # assuming extension 1 has the data

    tic_id_data = table[table['TIC_ID'] == tic_id_to_plot]

    if len(tic_id_data) == 0:
        raise ValueError(f"TIC ID {tic_id_to_plot} not found in {fits_file}")

    # Handle time column (some files use Time_JD, others Time_BJD)
    if 'Time_JD' in tic_id_data.names:
        time = tic_id_data['Time_JD'][400:-800]
    else:
        time = tic_id_data['Time_BJD'][400:-800]

    flux = tic_id_data['Relative_Flux'][400:-800]
    flux_err = tic_id_data['Relative_Flux_err'][400:-800]

    # Handle missing airmass
    if 'Airmass' in tic_id_data.names:
        airmass = tic_id_data['Airmass'][400:-800]
    else:
        airmass = np.zeros_like(time)

    return {
        'Time_BJD': np.array(time),
        'Relative_Flux': np.array(flux),
        'Relative_Flux_err': np.array(flux_err),
        'Airmass': np.array(airmass),
        'RMS': tic_id_data['RMS'][0],
        'TIC_ID': tic_id_data['TIC_ID'][0],
        'Tmag': tic_id_data['Tmag'][0],
    }


def compute_rms_values(data, exp, max_binning):
    jd_mid = data['Time_BJD']
    rel_flux = data['Relative_Flux']
    rel_fluxerr = data['Relative_Flux_err']
    print(f'Using exposure time: {exp}')

    RMS_values = []
    time_seconds = []

    for i in range(1, max_binning):
        time_binned, dt_flux_binned, dt_fluxerr_binned = bin_time_flux_error(
            jd_mid, rel_flux, rel_fluxerr, i
        )
        exposure_time_seconds = i * exp
        RMS = np.std(dt_flux_binned)
        RMS_values.append(RMS)
        time_seconds.append(exposure_time_seconds)

    average_rms_values = np.array(RMS_values) * 1e3  # Convert to ppt
    RMS_model = average_rms_values[0] / np.sqrt(np.arange(1, max_binning))

    return time_seconds, average_rms_values, RMS_model


def plot_timescale(times, avg_rms, RMS_model, label, label_color):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(times, avg_rms, 'o', label=f"{label} Data", color='cornflowerblue', alpha=0.8)
    ax.plot(times, RMS_model, '--', label=f"{label} Model", color='blue')
    ax.axvline(x=900, color='black', linestyle='-', label='Reference Line (x=900)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Exposure Time (s)')
    ax.set_ylabel('RMS (ppt)')
    ax.set_title('RMS vs Exposure Time')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.tick_params(axis='y', which='minor', length=4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run and plot RMS from relative photometry FITS file.')
    parser.add_argument('fits_file', type=str, help='Path to FITS photometry file')
    parser.add_argument('tic', type=int, help='TIC ID to plot')
    parser.add_argument('--bin', type=int, default=180, help='Maximum binning steps')

    args = parser.parse_args()

    phot_data = load_fits_data(args.fits_file, args.tic)
    times, avg_rms, RMS_model = compute_rms_values(phot_data, exp=10, max_binning=args.bin)

    plot_timescale(times, avg_rms, RMS_model, label=str(phot_data['TIC_ID']), label_color='blue')