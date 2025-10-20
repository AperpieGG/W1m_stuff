#!/usr/bin/env python
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils_W1m import plot_images


def load_rms_mags_data():
    """
    Load RMS and magnitude data from JSON files.
    """
    path = '/Users/u5500483/Downloads/Toran_Sky/'
    filenames = [
        path + '20250808/rms_mags_rel_phot_KELT-16_30_1_0808.json',
        path + '20250921/rms_mags_rel_phot_KELT-16_30_1_0921.json',
    ]
    data_list = []
    for filename in filenames:
        with open(filename, 'r') as file:
            data_list.append(json.load(file))
    return data_list


def plot_noise_model(ax, data, main_label, mode_label, position='bottom-left'):
    """
    Plot RMS noise model on a given axis.
    """
    RMS_list = data['RMS_list']
    Tmag_list = data['Tmag_list']
    color_list = data['COLOR']
    synthetic_mag = data['synthetic_mag']
    RNS = data['RNS']
    photon_shot_noise = data['photon_shot_noise']
    read_noise = data['read_noise']
    dc_noise = data['dc_noise']
    sky_noise = data['sky_noise']
    N = data['N']

    # Filter stars with valid color
    total_mags, total_RMS, total_colors = [], [], []
    for i in range(len(Tmag_list)):
        if color_list[i] is not None:
            total_mags.append(Tmag_list[i])
            total_RMS.append(RMS_list[i])
            total_colors.append(color_list[i])

    # Scatter plot
    scatter = ax.scatter(total_mags, total_RMS, c=total_colors, cmap='coolwarm', vmin=0.5, vmax=1.5)

    # Plot noise components
    ax.plot(synthetic_mag, RNS, color='black', label='total noise')
    ax.plot(synthetic_mag, photon_shot_noise, color='green', label='photon shot', linestyle='--')
    ax.plot(synthetic_mag, read_noise, color='red', label='read noise', linestyle='--')
    ax.plot(synthetic_mag, dc_noise, color='purple', label='dark noise', linestyle='--')
    ax.plot(synthetic_mag, sky_noise, color='blue', label='sky bkg', linestyle='--')
    ax.plot(synthetic_mag, np.ones(len(synthetic_mag)) * N, color='orange', label='scintillation noise', linestyle='--')

    ax.set_yscale('log')
    ax.set_xlim(9, 16)
    ax.set_ylim(1e3, 1e6)
    ax.invert_xaxis()

    # Main label
    # Main label
    if position == 'bottom-left':
        ax.text(0.02, 0.93, main_label, transform=ax.transAxes,
                fontsize=15, ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        # Mode label at top-right
        ax.text(0.98, 0.97, mode_label, transform=ax.transAxes,
                fontsize=15, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    elif position == 'top-right':
        ax.text(0.98, 0.97, main_label, transform=ax.transAxes,
                fontsize=15, ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        # Mode label at bottom-left
        ax.text(0.02, 0.93, mode_label, transform=ax.transAxes,
                fontsize=15, ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    return scatter


def main():
    # Initialize plotting
    plot_images()
    data_list = load_rms_mags_data()

    # 1 row × 2 columns
    fig, axes = plt.subplots(1, 2, sharey=True, constrained_layout=True, figsize=(12, 5))

    main_labels = ["Moon", "LN"]
    mode_labels = ["LN", "No Moon"]

    for i, ax in enumerate(axes):
        pos = 'bottom-left' if i == 0 else 'top-right'
        plot_noise_model(ax, data_list[i], main_labels[i], mode_labels[i], position=pos)
        if i == 0:
            ax.set_ylabel('RMS per 10 seconds (ppm)')
        ax.set_xlabel('TESS Magnitude')

    # Single colorbar
    cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation='vertical', fraction=0.1, pad=0.05)
    cbar.set_label(label='$\mathdefault{G_{BP}-G_{RP}}$')
    path_save = '/Users/u5500483/Downloads/Toran_Sky/'
    plt.savefig(path_save + 'Noise_LN_NEW_FULL.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()