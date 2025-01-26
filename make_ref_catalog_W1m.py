#!/usr/bin/env python3

from astroquery.vizier import Vizier
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import argparse
import sys
import traceback
import argparse as ap
import numpy as np
from astropy.time import Time
import warnings
from astropy.units import UnitsWarning
from astroquery.vizier import Vizier

# Suppress Astropy UnitsWarnings
warnings.simplefilter('ignore', UnitsWarning)


def arg_parse():
    """
    Parse the command line arguments for querying Vizier catalogs.
    """
    parser = ap.ArgumentParser()
    parser.add_argument('ra',
                        type=str,
                        help='Field center RA in degrees.')
    parser.add_argument('dec',
                        type=str,
                        help='Field center Dec in degrees.')
    parser.add_argument('box_width',
                        type=float,
                        help='Box width in tangent-plane degrees.')
    parser.add_argument('box_height',
                        type=float,
                        help='Box height in degrees.')
    parser.add_argument('epoch',
                        type=str,
                        help='Reference datetime string for proper-motion calculations.')
    parser.add_argument('output',
                        type=str,
                        default='catalog.fits',
                        help='Path to save the output catalog.')
    parser.add_argument('--blend-delta',
                        type=float,
                        default=1,
                        help='Maximum magnitude delta for a star to be considered as blended.')
    parser.add_argument('--catalog',
                        type=str,
                        default='IV/39/tic82',
                        help='Vizier catalog ID to query (default: TIC8).')
    return parser.parse_args()


def generate_sampling_grid(center, width, height, max_query_radius):
    """
    Generates a grid of RA, Dec coordinates that cover the box defined by
    the center, width, and height, at a given tessellation stride (distance between points).

    Parameters
    ----------
    center : SkyCoord
        Coordinates of the field center
    width : float
        Field width (tangent plane) in degrees
    height : float
        Field height in degrees
    max_query_radius : float
        Maximum query radius in degrees for tessellation

    Returns
    -------
    coords : SkyCoord
        Coordinates of sampling grid points
    """
    # Size of square box that fits inside max_query_radius
    box_width = max_query_radius * np.sqrt(2)
    box_inset = max_query_radius * (1 - np.sqrt(2))

    # Initialise list of coordinates
    coords = []

    # Define declination boundaries
    dec = center.dec + (height + box_inset) / 2
    end_dec = center.dec - (height + box_inset) / 2

    # Clip points below the South Pole (not observable)
    south_pole = -90 * u.degree
    if end_dec < south_pole:
        coords.append(SkyCoord(0 * u.degree, south_pole))
        end_dec = south_pole + box_width

    # Iterate over declination and right ascension
    while dec > end_dec:
        ra = center.ra - (width + box_inset) / (2 * np.cos(dec))
        end_ra = center.ra + (width + box_inset) / (2 * np.cos(dec))
        while ra < end_ra:
            coords.append(SkyCoord(ra, dec))
            ra += box_width

        coords.append(SkyCoord(end_ra, dec))
        dec -= box_width

    # Add coordinates for the bottom declination stripe
    ra = center.ra - (width + box_inset) / (2 * np.cos(end_dec))
    end_ra = center.ra + (width + box_inset) / (2 * np.cos(end_dec))
    while ra < end_ra:
        coords.append(SkyCoord(ra, end_dec))
        ra += box_width
    coords.append(SkyCoord(end_ra, end_dec))

    return SkyCoord(coords)


def grab_table_description_vizier(catalog_id):
    """
    Fetches the column names and data types from a Vizier catalog.

    Parameters
    ----------
    catalog_id : str
        Vizier catalog ID (e.g., "I/345/gaia2" for Gaia DR2).

    Returns
    -------
    col_ids : list
        Names of the columns in the catalog.
    col_types : list
        Data types of the columns in the catalog.

    Raises
    ------
    Exception
        If the catalog is not found.
    """
    from astroquery.vizier import Vizier

    # Fetch catalog metadata
    Vizier.ROW_LIMIT = 1  # Fetch only metadata without heavy data
    catalog = Vizier.get_catalogs(catalog_id)

    if len(catalog) == 0:
        raise Exception(f"Catalog {catalog_id} not found in Vizier.")

    # Extract column names and data types
    result_table = catalog[0]  # Use the first table in the catalog
    col_ids = result_table.colnames
    col_types = [result_table[name].dtype for name in col_ids]

    return col_ids, col_types


def build_query_vizier(sample_coords, sample_radius, catalog_id, htm_depth=None):
    """
    Builds and executes a query to fetch data from Vizier for a given region.

    Parameters
    ----------
    sample_coords : SkyCoord
        Sample coordinates to query around.
    sample_radius : Angle
        Radius to sample around (in degrees).
    catalog_id : str
        Vizier catalog ID (e.g., "I/345/gaia2" for Gaia DR2).
    htm_depth : int, optional
        Depth of the triangular mesh (not used for Vizier queries).

    Returns
    -------
    query_result : Table
        The fetched query results.
    column_names : list
        Names of the columns in the result.
    column_dtypes : list
        Data types for each column.

    Raises
    ------
    Exception
        If no data is returned from the Vizier query.
    """
    from astroquery.vizier import Vizier

    # Query Vizier catalog
    Vizier.ROW_LIMIT = -1  # No row limit for the query
    result = Vizier.query_region(sample_coords, radius=30 * u.arcmin, catalog=catalog_id)

    if len(result) == 0:
        raise Exception(f"No data found for catalog {catalog_id} in the specified region.")

    # Select the first table in the result
    result_table = result[0]

    # Extract column names and data types
    column_names = result_table.colnames
    column_dtypes = [result_table[name].dtype for name in column_names]

    return result_table, column_names, column_dtypes


def query_vizier(center, width, height, catalog='IV/39/tic82'):
    """
    Query a Vizier catalog for a specified region.

    Parameters
    ----------
    center : SkyCoord
        Field center coordinates.
    width : float
        Width of the region in degrees.
    height : float
        Height of the region in degrees.
    catalog : str
        Vizier catalog ID (default: Gaia DR2).

    Returns
    -------
    results : Table
        The query results.
    """
    Vizier.ROW_LIMIT = -1  # No limit on the number of rows
    region = f"{width}x{height} deg"
    results = Vizier.query_region(center, width=width * u.deg, height=height * u.deg, catalog=catalog)
    return results


def fetch_catalog_vizier(ra_center, dec_center, box_width, box_height,
                         reference_epoch, output_path, blend_delta, catalog_id="IV/38/tic"):
    """
    Fetch catalog from Vizier, apply proper motion corrections, perform blending checks,
    and output the catalog to a FITS file. Returns the processed catalog and metadata.
    """

    # Query Vizier for data with additional columns
    print("Fetching catalog from Vizier...")
    Vizier.ROW_LIMIT = -1
    vizier_query = Vizier(
        columns=["TIC", "GAIA", "RAJ2000", "DEJ2000", "Tmag", "Gmag", "BPmag", "RPmag", "pmRA", "pmDE"],
        column_filters={"Gmag": "<18"}  # Filter for Gaia magnitude < 16
    )
    vizier_query.ROW_LIMIT = -1  # Set the row limit after creating the Vizier instance
    try:
        catalog = vizier_query.query_region(SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg)),
                                            width=box_width * u.deg, height=box_height * u.deg,
                                            catalog=catalog_id)
        if len(catalog) == 0:
            print(f"No data found in catalog {catalog_id}.")
            return None, None, None

        catalog = catalog[0]  # Use the first table
    except Exception as e:
        print(f"Failed to fetch data from Vizier: {e}")
        return None, None, None

    print(f"Fetched {len(catalog)} rows from Vizier.")

    # Fetch column names and types
    print("Fetching column descriptions...")
    try:
        col_ids, col_types = grab_table_description_vizier(catalog_id)
        print(f"Column names: {col_ids}")
    except Exception as e:
        print(f"Error fetching column descriptions: {e}")
        return None, None, None

    # Apply proper motion corrections
    reference_epoch = Time(reference_epoch, scale='utc')
    delta_years = (reference_epoch - Time('2000-01-01T00:00:00')).to(u.year).value

    if "pmRA" in catalog.colnames and "pmDE" in catalog.colnames:
        print("Applying proper motion corrections...")
        ra = catalog["RAJ2000"].data
        dec = catalog["DEJ2000"].data
        pmra = np.nan_to_num(catalog["pmRA"].data)
        pmdec = np.nan_to_num(catalog["pmDE"].data)

        ra_corr = ra + pmra * delta_years / (3.6e6 * np.cos(np.radians(dec)))
        dec_corr = dec + pmdec * delta_years / 3.6e6
    else:
        print("Proper motion data not available; skipping corrections.")
        ra_corr = catalog["RAJ2000"].data
        dec_corr = catalog["DEJ2000"].data

    # Perform blending checks
    print("Performing blending checks...")
    start = Time.now()
    n_stars = len(ra_corr)
    blended = np.zeros(n_stars, dtype=bool)
    exclusion_radius = 6 * 4. / 3600.  # Radius in degrees for blending exclusion

    for i in range(n_stars):
        if blended[i]:
            continue
        distances = np.sqrt((ra_corr - ra_corr[i]) ** 2 +
                            (dec_corr - dec_corr[i]) ** 2)
        magnitudes = np.abs(catalog["Gmag"].data - catalog["Gmag"].data[i])
        blended[i] = np.any((distances < exclusion_radius) & (magnitudes < blend_delta) & (distances > 0))

    print(f"Blending check complete. {np.sum(blended)} stars flagged as blended.")
    print(f"Elapsed time: {Time.now() - start}")

    # Add corrected RA/Dec to the catalog
    catalog["RA_CORR"] = ra_corr
    catalog["DEC_CORR"] = dec_corr
    # Add a column for the blending flag
    catalog["BLENDED"] = blended

    # Check for missing columns and warn
    for column in ["Tmag", "Gmag", "BPmag", "RPmag"]:
        if column not in catalog.colnames:
            print(f"Warning: Column {column} not found in catalog {catalog_id}.")

    # Fix metadata before saving to FITS
    print("Cleaning metadata for FITS compatibility...")
    if "description" in catalog.meta:
        catalog.meta["description"] = catalog.meta["description"][:70]

    for key, value in catalog.meta.items():
        if isinstance(value, str) and "log" in value:
            catalog.meta[key] = value.replace("log", "")

    # Output catalog to FITS
    print("Saving catalog to FITS...")
    try:
        # Clean metadata for FITS compliance
        if "description" in catalog.meta:
            catalog.meta["description"] = catalog.meta["description"][:70]  # Truncate to 70 characters

        # Collect keys that need modification or removal
        keys_to_remove = []
        for key, value in catalog.meta.items():
            if isinstance(value, str) and len(value) > 70:
                catalog.meta[key] = value[:70]  # Truncate long strings
            if isinstance(key, str) and len(key) > 8:
                keys_to_remove.append(key)  # Collect keys longer than 8 characters for removal

        # Remove problematic keys after iteration
        for key in keys_to_remove:
            del catalog.meta[key]

        # Write the catalog to a FITS file
        catalog.write(output_path, format="fits", overwrite=True)
        print(f"Catalog successfully saved to {output_path}.")
    except Exception as e:
        print(f"Error saving catalog to FITS: {e}")
        traceback.print_exc()
        return None, None, None

    return catalog, col_ids, col_types


if __name__ == "__main__":
    args = arg_parse()
    cat, cols, types = fetch_catalog_vizier(args.ra, args.dec, args.box_width, args.box_height,
                                            args.epoch, args.output, args.blend_delta)


