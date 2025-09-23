import pandas as pd
import sys
import os
from tabulate import tabulate   # make sure tabulate is installed

# Path to your Excel file
file_path = "/Users/u5500483/OneDrive - University of Warwick/CMOS_NGTS_DATA.xlsx"

# Make sure the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

# Load only the 5th sheet (index 4)
df = pd.read_excel(file_path, sheet_name=4)

if len(sys.argv) > 1:
    query = sys.argv[1].lower()
    results = df[df["TARGET"].astype(str).str.lower().str.contains(query)]
    
    if results.empty:
        print(f"No results found for '{query}' in TARGET column.")
    else:
        # Pretty print as a table with borders
        print(tabulate(results, headers="keys", tablefmt="psql", showindex=False))
else:
    print("Usage: python query_excel.py <target_name>")
