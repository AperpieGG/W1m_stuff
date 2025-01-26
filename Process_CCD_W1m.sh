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
    python /Users/u5500483/Documents/GitHub/W1m_stuff/unzip_fits_W1m.py     # unzips the FITS files and deletes the bz2 extension
    python /Users/u5500483/Documents/GitHub/W1m_stuff/trim_ccd_W1m.py
    # add this point you have to sent the catalog input file for phot.
    python /Users/u5500483/Documents/GitHub/W1m_stuff/simple_wrapper_W1m.py --camera ccd
    python /Users/u5500483/Documents/GitHub/W1m_stuff/check_ccd_W1m.py
    python /Users/u5500483/Documents/GitHub/W1m_stuff/adding_headers_W1M.py
    python /Users/u5500483/Documents/GitHub/W1m_stuff/calibration_images_ccd_W1m.py
    python /Users/u5500483/Documents/GitHub/W1m_stuff/process_ccd_W1m.py

#    # Run the additional Python scripts within this subdirectory
    python /Users/u5500483/Documents/GitHub/W1m_stuff/relative_phot_dev_W1m.py --aper 4
    python /Users/u5500483/Documents/GitHub/W1m_stuff/measure_zp_W1m.py --aper 4
    python /Users/u5500483/Documents/GitHub/W1m_stuff/zip_fits_W1m.py                                 # zip the FITS files to bz2 and delete .fits files

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