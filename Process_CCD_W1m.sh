#!/bin/bash

# Record the start time
start_time=$(date +%s)

echo "Starting processing..."

# Loop through each subdirectory that matches "action*_observeField"
# shellcheck disable=SC2044
for observe_dir in $(find . -maxdepth 1 -type d -name "action*_observeField"); do
    echo "Found observeField directory: $observe_dir"

    # Change to the directory
    cd "$observe_dir" || continue
    echo "Changed to directory: $(pwd)"

    # Run the initial Python scripts
    python /home/ops/ngcmos/unzip_fits.py     # unzips the FITS files and deletes the bz2 extension
    python /home/ops/ngcmos/trim_ccd.py
    # add this point you have to sent the catalog input file for phot.
    python /home/ops/refcatpipe2/cmos/simple_wrapper.py --camera ccd
    python /home/ops/ngcmos/check_ccd.py
    python /home/ops/ngcmos/adding_headers.py
    python /home/ops/ngcmos/calibration_images_ccd.py
    python /home/ops/ngcmos/process_ccd.py

#    # Run the additional Python scripts within this subdirectory
    python /home/ops/fwhm_stars/fwhm_batches.py --size 13.5 --cam CCD   # make plot and save to fwhm_results.json
    python /home/ops/ngcmos/relative_phot_dev.py --aper 4
    python /home/ops/ngcmos/measure_zp.py --aper 4
    python /home/ops/fwhm_stars/best_fwhm.py --size 13.5                # save to fwhm_positions.json
    python /home/ops/ngcmos/zip_fits.py                                 # zip the FITS files to bz2 and delete .fits files

    # Return to the parent directory
    cd - || exit
done

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"