#!/usr/bin/env python3
"""
Measure HFD for a single target (given x,y pixels) across a directory of FITS images.

Follows the method used in the provided script: SEP background, SEP extraction, photutils aperture
(optional), sep.kron_radius, sep.sum_ellipse and sep.flux_radius to compute the half-flux diameter.
Outputs a plot of HFD (arcsec) vs time (BJD or DATE-OBS fallback to file index).

Example:
    python hfd_target_timeseries.py ./images 1 4 1.131 6 --x 1026.25 --y 939.39
"""
import os
import glob
import numpy as np
from utils_W1m import plot_images
import fitsio
from astropy.table import Table
import sep
import tqdm
from path import Path
import warnings
import matplotlib.pyplot as plt
from astropy.time import Time
import argparse

# silence annoying warnings from WCS / fits if any
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plot_images()


def find_images(path):
    path = Path(path)
    image_files = sorted(path.files('*.fits'))
    # exclude calibration/catalog files
    exclude_words = ['bias', 'flat', 'catalog', 'dark', 'phot', 'evening', 'morning', 'txt', 'xml']
    image_files = [f for f in image_files if not any(word in f.lower() for word in exclude_words)]
    return image_files


def measure_target_hfd(files, x_target, y_target, binning, plate_scale, GAIN, RADIUS,
                       sep_threshold=1.3, verbose=False):
    """
    Measure HFD for the target pixel across all files.

    Parameters
    ----------
    files : list of Path or str
    x_target, y_target : float
        pixel coordinates of the target (image coordinates)
    binning : int
        detector binning factor
    plate_scale : float
        plate scale in arcsec/pixel (unbinned)
    GAIN : float
        camera gain (e-/ADU) used when needed by SEP
    RADIUS : float
        radius parameter used for aperture/kron steps (units of 'a' in the original method)

    Returns
    -------
    times : list
    hfds_arcsec : list
    flags : list (0==ok, >0 indicates problems)
    filenames : list
    """
    hfds_arcsec = []
    times = []
    flags = []
    filenames = []

    for file in tqdm.tqdm(files, file=None):
        fname = str(file)
        filenames.append(os.path.basename(fname))
        try:
            # read image and header using fitsio (fast)
            header = fitsio.read_header(fname)
            data = fitsio.read(fname).astype(np.float32)

            # background subtraction using SEP
            bkg = sep.Background(data)
            data_sub = data - bkg.back()

            # try source extraction (low threshold first then higher if fails)
            try:
                objs = Table(sep.extract(data_sub, sep_threshold, err=bkg.globalrms, gain=GAIN))
            except Exception:
                try:
                    objs = Table(sep.extract(data_sub, sep_threshold + 2.0, err=bkg.globalrms, gain=GAIN))
                except Exception:
                    # extraction failed — record NaN and flag
                    hfds_arcsec.append(np.nan)
                    times.append(np.nan)
                    flags.append(3)  # extraction failed
                    continue

            if len(objs) == 0:
                hfds_arcsec.append(np.nan)
                times.append(np.nan)
                flags.append(4)  # no objects detected
                continue

            # adjust theta like original code
            if 'theta' in objs.colnames:
                objs['theta'][objs['theta'] > np.pi / 2] -= np.pi / 2

            # compute distances to target and pick the closest detection
            # objs['x'], objs['y'] are in image pixel coords
            d2 = (objs['x'] - x_target) ** 2 + (objs['y'] - y_target) ** 2
            idx_closest = int(np.argmin(d2))
            obj = objs[idx_closest]

            # apply some basic filters (optional) similar to original:
            # require some minimum SNR / area if those columns exist
            if 'tnpix' in objs.colnames and obj['tnpix'] <= 5:
                hfds_arcsec.append(np.nan);
                times.append(np.nan);
                flags.append(5);
                continue

            # For HFD estimation we follow the original approach:
            # 1) compute kron radius for the chosen object (returns array, so wrap single values)
            try:
                kronrad, kron_flag = sep.kron_radius(data_sub,
                                                     np.array([obj['x']]), np.array([obj['y']]),
                                                     np.array([obj['a']]), np.array([obj['b']]),
                                                     np.array([obj['theta']]), RADIUS)
            except Exception:
                hfds_arcsec.append(np.nan);
                times.append(np.nan);
                flags.append(6);
                continue

            # 2) compute elliptical flux using sum_ellipse (use 2.5*kron as in original)
            try:
                flux, _, flux_flag = sep.sum_ellipse(data_sub,
                                                     np.array([obj['x']]), np.array([obj['y']]),
                                                     np.array([obj['a']]), np.array([obj['b']]),
                                                     np.array([obj['theta']]), 2.5 * kronrad, subpix=0)
            except Exception:
                hfds_arcsec.append(np.nan);
                times.append(np.nan);
                flags.append(7);
                continue

            # 3) compute flux_radius for 50% flux fraction: note sep.flux_radius expects arrays
            try:
                # The radius search limit passed in the original code was RADIUS * a
                r, radius_flag = sep.flux_radius(data_sub,
                                                 np.array([obj['x']]), np.array([obj['y']]),
                                                 RADIUS * np.array([obj['a']]),
                                                 0.5, normflux=flux, subpix=5)
            except Exception:
                hfds_arcsec.append(np.nan);
                times.append(np.nan);
                flags.append(8);
                continue

            # apply flag filter: only keep if kron_flag, flux_flag, radius_flag are zero
            ok_mask = (kron_flag == 0) & (flux_flag == 0) & (radius_flag == 0)
            if not ok_mask[0]:
                hfds_arcsec.append(np.nan);
                times.append(np.nan);
                flags.append(9);
                continue

            # diameter in pixels = 2 * r
            hfd_pixels = 2.0 * r[0]

            # convert to arcsec using plate scale and binning (plate_scale is arcsec/pixel unbinned)
            hfd_arcsec = hfd_pixels * plate_scale * binning

            # extract time from header: prefer BJD, then 'DATE-OBS' (ISO), then 'MJD-OBS', else use file index
            if 'BJD' in header:
                time_val = header['BJD']
            elif 'MJD-OBS' in header:
                time_val = header['MJD-OBS']
            elif 'DATE-OBS' in header:
                # try safe parsing to ISO via astropy Time
                try:
                    time_val = Time(header['DATE-OBS'], format='isot', scale='utc').jd  # store as JD
                except Exception:
                    time_val = np.nan
            else:
                time_val = np.nan

            hfds_arcsec.append(hfd_arcsec)
            times.append(time_val)
            flags.append(0)  # success

            if verbose:
                print(f"{os.path.basename(str(file))}: HFD={hfd_arcsec:.3f}\" (pixels={hfd_pixels:.2f})")

        except Exception as e:
            # unexpected error on this file
            hfds_arcsec.append(np.nan)
            times.append(np.nan)
            flags.append(2)
            if verbose:
                print(f"Error processing {file}: {e}")
            continue

    return np.array(times), np.array(hfds_arcsec), np.array(flags), [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Measure HFD over a series of FITS images for one target pixel")
    parser.add_argument('input_dir', type=str, help='Directory with FITS images')
    parser.add_argument('binning', type=int, help='Binning factor (integer)')
    parser.add_argument('plate_scale', type=float, help='Plate scale in arcsec/pixel (unbinned)')
    parser.add_argument('gain', type=float, help='Camera gain in e-/ADU (used by SEP)')
    parser.add_argument('radius', type=float, help='RADIUS parameter (units of a) used in measurements')
    parser.add_argument('--x', type=float, required=True, help='Target X pixel coordinate')
    parser.add_argument('--y', type=float, required=True, help='Target Y pixel coordinate')
    parser.add_argument('--sep_thresh', type=float, default=1.3, help='SEP detection threshold (sigma)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    files = find_images(args.input_dir)
    if len(files) == 0:
        raise SystemExit("No FITS files found after filtering.")

    times, hfds_arcsec, flags, filenames = measure_target_hfd(files, args.x, args.y,
                                                              args.binning, args.plate_scale,
                                                              args.gain, args.radius,
                                                              sep_threshold=args.sep_thresh,
                                                              verbose=args.verbose)

    # Convert times array to a plottable form
    # Some times are JD-like (if DATE-OBS converted to JD earlier), some may be 'MJD-OBS' or BJD numeric.
    # We'll try to plot only valid numeric times; for non-numeric we fallback to index.
    numeric_mask = np.isfinite(times)
    if numeric_mask.sum() >= 2:
        # prefer to plot vs numeric time (e.g. JD). If times are MJD, user can interpret accordingly.
        x_plot = times[numeric_mask]
        y_plot = hfds_arcsec[numeric_mask]
        x_labels = filenames  # we will annotate if needed
        plt.figure(figsize=(8, 5))
        plt.scatter(x_plot, y_plot, color='blue', s=30)
        plt.xlabel("Time (UTC)")
        plt.ylabel("HFD (arcsec)")
        plt.title("HFD vs Time")
        plt.title(f'HFD time series for target (x={args.x:.2f}, y={args.y:.2f})')
        plt.ylim(7, 11)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        # fallback to index-based plot
        idx = np.arange(len(hfds_arcsec))
        plt.figure(figsize=(10, 5))
        plt.plot(idx, hfds_arcsec, 'o-', color='C0')
        plt.xlabel('Frame index')
        plt.ylabel('HFD (arcsec)')
        plt.title(f'HFD time series (index) for target (x={args.x:.2f}, y={args.y:.2f})')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # print summary statistics
    ok = np.isfinite(hfds_arcsec)
    if ok.sum() > 0:
        print(f"\nMeasured HFD for {ok.sum()}/{len(hfds_arcsec)} frames.")
        print(f"Mean HFD = {np.nanmean(hfds_arcsec):.3f} arcsec")
        print(f"Median HFD = {np.nanmedian(hfds_arcsec):.3f} arcsec")
        print(f"Std HFD = {np.nanstd(hfds_arcsec):.3f} arcsec")
    else:
        print("\nNo valid HFD measurements obtained.")


if __name__ == '__main__':
    main()
