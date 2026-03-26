from pathlib import Path
import pandas as pd
import xarray as xr
import re

# Folder containing NetCDF files
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data/raw"
OUT_FILE = BASE_DIR / "data/processed/na_pm25_cells_clean.csv"

LAT_MIN, LAT_MAX = 7.0, 84.0
LON_MIN, LON_MAX = -168.0, -52.0

# Downsample grid
LAT_STRIDE = 10
LON_STRIDE = 10


def parse_date(filename):
    match = re.search(r'_(\d{6})_', filename)
    if match is None:
        raise ValueError(f"Cannot extract YYYYMM from filename: {filename}")
    return pd.to_datetime(match.group(1), format="%Y%m")


def extract_values(nc_file):

    ds = xr.open_dataset(nc_file)

    # Automatically detect PM2.5 variable
    var = list(ds.data_vars)[0]
    da = ds[var]

    # Downsample grid
    da = da.isel(
        lat=slice(None, None, LAT_STRIDE),
        lon=slice(None, None, LON_STRIDE)
    )

    # Subset North America
    da = da.where(
        (da.lat >= LAT_MIN) &
        (da.lat <= LAT_MAX) &
        (da.lon >= LON_MIN) &
        (da.lon <= LON_MAX),
        drop=True
    )

    df = da.to_dataframe().reset_index()

    df.rename(columns={var: "pm25"}, inplace=True)

    df["date"] = parse_date(nc_file.name)

    # Remove rows without PM2.5 estimates
    df = df.dropna(subset=["pm25"])

    return df


def main():

    # Only monthly files
    files = sorted(DATA_DIR.glob("GHAP_PM2.5_M1K_*.nc"))

    print("Monthly files found:", len(files))

    all_dfs = []

    for f in files:
        print("Processing:", f.name)
        df = extract_values(f)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    full_df.to_csv(OUT_FILE, index=False)

    print("\nSaved cleaned dataset:", OUT_FILE)
    print("Total rows:", len(full_df))
    print(full_df.head())


if __name__ == "__main__":
    main()