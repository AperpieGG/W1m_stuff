import sys
import os
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from astropy.io import fits
from donuts import Donuts

warnings.simplefilter('ignore', category=UserWarning)

log_file = 'check_cmos.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_shifts(x_shifts, y_shifts, save_path, prefix, time):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x_shifts, y_shifts, c=time, cmap='viridis',
                         label=f'Shifts for field: {prefix}', marker='o')
    plt.xlabel('X Shift (pixels)')
    plt.ylabel('Y Shift (pixels)')
    plt.title('Shifts to ref image')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.legend()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Colorbar
    plt.colorbar(scatter, label='Time')

    # Yesterday’s timestamp
    timestamp_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    # File path in "shifts_plots"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pdf_file_path = os.path.join(save_path, f"donuts_{prefix}_{timestamp_yesterday}.pdf")

    fig.savefig(pdf_file_path, bbox_inches='tight')
    plt.close(fig)
    print(f"PDF plot saved to: {pdf_file_path}\n")


def crop_center(data, crop_size=1000):
    """Crop the image data to the central crop_size x crop_size region."""
    y, x = data.shape
    startx = x // 2 - (crop_size // 2)
    starty = y // 2 - (crop_size // 2)
    return data[starty:starty + crop_size, startx:startx + crop_size]


def check_donuts(directory, file_groups, crop_size=1000):
    for file_group in file_groups:
        reference_image = os.path.join(directory, file_group[0])
        logger.info(f"Reference image: {reference_image}")
        with fits.open(reference_image) as hdul:
            ref_data = crop_center(hdul[0].data, crop_size=crop_size)
            ref_hdu = fits.PrimaryHDU(ref_data, header=hdul[0].header)
        d = Donuts(ref_hdu)  # Donuts accepts HDU objects too

        x_shifts, y_shifts, times = [], [], []
        prefix = os.path.basename(file_group[0]).split('.')[0]

        for filename in file_group[1:]:
            filepath = os.path.join(directory, filename)
            shift = d.measure_shift(filepath)
            sx = round(shift.x.value, 2)
            sy = round(shift.y.value, 2)
            logger.info(f'{filename} shift X: {sx} Y: {sy}')

            x_shifts.append(sx)
            y_shifts.append(sy)

            # Grab time from FITS header if available
            try:
                with fits.open(filepath) as hdul:
                    date_obs = hdul[0].header.get('DATE-OBS')
                    if date_obs:
                        times.append(date_obs)
                    else:
                        times.append(len(times))  # fallback numeric sequence
            except Exception:
                times.append(len(times))

            # Check for big shifts
            if np.any(np.array([abs(sx), abs(sy)]) > 2):
                logger.warning(f'{filename} image shift too big X: {sx} Y: {sy}')
                failed_dir = os.path.join(directory, 'failed_donuts')
                if not os.path.exists(failed_dir):
                    os.mkdir(failed_dir)
                os.rename(filepath, os.path.join(failed_dir, filename))

        # After all files in the group, make the plot
        if x_shifts and y_shifts:
            save_path = os.path.join(directory, "shifts_plots")
            # If DATE-OBS values are strings, you might want to convert them to floats or indices
            try:
                time_values = np.arange(len(times)) if isinstance(times[0], str) else times
            except Exception:
                time_values = np.arange(len(times))
            plot_shifts(x_shifts, y_shifts, save_path, prefix, time_values)