# Separating Data into Training, Testing, and Validation.py
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = Path("Data/raw_nc")   # <-- point this at your selected .nc folder
PATTERN = "GHAP_PM2.5_M1K_*.nc"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20  # most recent goes to test

OUT_TRAIN = "train_files.csv"
OUT_VAL   = "val_files.csv"
OUT_TEST  = "test_files.csv"


def parse_monthly_filename(path: str):
    """
    Expected filename style:
      GHAP_PM2.5_M1K_201701_V1.nc
    date_str = '201701' (YYYYMM)
    """
    fname = os.path.basename(path)
    parts = fname.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {fname}")

    date_str = parts[3]  # YYYYMM
    if len(date_str) != 6 or not date_str.isdigit():
        raise ValueError(f"Could not parse YYYYMM from: {fname}")

    year = date_str[:4]
    month = date_str[4:6]
    ym = int(date_str)  # for sorting
    return ym, year, month, path


def save_split_table(rows, out_name: str):
    with open(out_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Year", "Month", "YYYYMM", "File Path"])
        for idx, (ym, year, month, path) in enumerate(rows, start=1):
            writer.writerow([idx, year, month, ym, path])


def main():
    files = sorted(DATA_DIR.glob(PATTERN))
    if not files:
        # fallback: allow running from within raw_nc directory
        files = [Path(p) for p in glob.glob(PATTERN)]
    files = [str(p) for p in files]

    print("Total monthly files:", len(files))
    if len(files) == 0:
        raise RuntimeError(f"No monthly .nc files found in {DATA_DIR.resolve()} matching {PATTERN}")

    rows = [parse_monthly_filename(p) for p in files]
    rows.sort(key=lambda r: r[0])  # chronological by YYYYMM

    n = len(rows)
    n_train = int(TRAIN_FRAC * n)
    n_val   = int(VAL_FRAC * n)
    n_test  = n - n_train - n_val  # remainder goes to test (most recent)

    train_rows = rows[:n_train]
    val_rows   = rows[n_train:n_train + n_val]
    test_rows  = rows[n_train + n_val:]

    print(f"Train: {len(train_rows)} (earliest)")
    print(f"Val:   {len(val_rows)} (middle)")
    print(f"Test:  {len(test_rows)} (most recent)")

    save_split_table(train_rows, OUT_TRAIN)
    save_split_table(val_rows, OUT_VAL)
    save_split_table(test_rows, OUT_TEST)

    # sanity check print ranges
    df = pd.DataFrame(rows, columns=["YYYYMM", "Year", "Month", "Path"])
    print("Full date span:", df["YYYYMM"].min(), "->", df["YYYYMM"].max())
    print("Train span:", train_rows[0][0], "->", train_rows[-1][0])
    print("Val span:", val_rows[0][0], "->", val_rows[-1][0] if val_rows else "N/A")
    print("Test span:", test_rows[0][0], "->", test_rows[-1][0])

    print("Split tables saved successfully.")


if __name__ == "__main__":
    main()