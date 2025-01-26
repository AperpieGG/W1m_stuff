# W1m_stuff

Process of data from W1m telescope.
Cameras Used: Ikon-L 936 Blue, Ikon-L 936 Red

The steps are the following:
1) Make reference catalog from Vizier and fixing from proper motion
2) Solve reference image using the catalog, outputs photometry catalog for later
3) Process ccd for W1m for image reduction and photometry usung SEP, outputs a .fits table with the photometry
4) Relative photometry script, outputs relative photometry .fits table
5) Noise model, outputs the noise model for the system.
