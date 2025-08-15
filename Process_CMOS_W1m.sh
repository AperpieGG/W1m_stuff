#!/bin/bash
#You will run this on the observing night directory (i.e. 20250806)
# Record the start time
start_time=$(date +%s)

echo "Starting processing..."

# Run the Python scripts
python /home/ops/Apergis/W1m_stuff/W1m_stuff/simple_wrapper_W1m.py --camera IMX571
python /home/ops/Apergis/W1m_stuff/W1m_stuff/check_cmos_W1m.py
python /home/ops/Apergis/W1m_stuff/W1m_stuff/adding_headers_W1m.py
#python /Users/u5500483/Documents/GitHub/W1m_stuff/create_flats_W1m.py
python /home/ops/Apergis/W1m_stuff/process_cmos_W1m.py --gain 0.75
#python /home/ops/fwhm_stars/fwhm_batches.py --size 11 --cam cmos # make plot and save to fwhm_results.json
python /Users/u5500483/Documents/GitHub/W1m_stuff/relative_phot_dev_W1m.py --aper 30
#python /Users/u5500483/Documents/GitHub/W1m_stuff/measure_zp_W1m.py --aper 20 --exp 10 --gain 0.75
#python /home/ops/fwhm_stars/best_fwhm.py --size 11 # save to fwhm_positions.json
#python /Users/u5500483/Documents/GitHub/W1m_stuff/remove_fits_files_W1m.py
python /home/ops/Apergis/W1m_stuff/zip_fits_files_W1m.py

echo "Finishing processing!"

# Record the end time
end_time=$(date +%s)

# Calculate the total time taken
elapsed_time=$((end_time - start_time))

# Print the total time taken
echo "Total time taken: $elapsed_time seconds"