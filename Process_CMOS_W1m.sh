#!/bin/bash

# Record the start time
start_time=$(date +%s)

echo "Starting processing..."

# Create directories.json with the specified paths
cat <<EOF > directories.json
{
  "base_paths": [
    "/Users/u5500483/Downloads/DATA_MAC/CMOS/"
  ],
  "out_paths": [
    "/Users/u5500483/Downloads/DATA_MAC/CMOS/calibration_images/"
  ]
}
EOF

# Run the Python scripts
python /Users/u5500483/Documents/GitHub/W1m_stuff/simple_wrapper_W1m.py --camera cmos
python /Users/u5500483/Documents/GitHub/W1m_stuff/check_cmos_W1m.py
python /Users/u5500483/Documents/GitHub/W1m_stuff/adding_headers_W1m.py
python /Users/u5500483/Documents/GitHub/W1m_stuff/create_flats_W1m.py
python /Users/u5500483/Documents/GitHub/W1m_stuff/process_cmos_W1m.py
#python /home/ops/fwhm_stars/fwhm_batches.py --size 11 --cam CMOS # make plot and save to fwhm_results.json
#python /Users/u5500483/Documents/GitHub/W1m_stuff/relative_phot_dev_W1m.py --aper 5
#python /Users/u5500483/Documents/GitHub/W1m_stuff/measure_zp_W1m.py --aper 5
#python /home/ops/fwhm_stars/best_fwhm.py --size 11 # save to fwhm_positions.json
#python /Users/u5500483/Documents/GitHub/W1m_stuff/remove_fits_files_W1m.py

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"