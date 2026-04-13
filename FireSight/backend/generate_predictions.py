"""
Generate predictions.json for the FireSight backend.

Usage:
  python generate_predictions.py --demo
  python generate_predictions.py --from-csv ../data/processed/projection_2026_jun_oct.csv
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "predictions.json"

LAT_MIN, LAT_MAX = 32.0, 42.0
LON_MIN, LON_MAX = -124.0, -114.0
GRID_STEP = 0.25


def frange(start, stop, step):
    vals = []
    v = start
    while v < stop:
        vals.append(round(v, 2))
        v += step
    return vals


def generate_demo_predictions():
    lats = frange(LAT_MIN, LAT_MAX, GRID_STEP)
    lons = frange(LON_MIN, LON_MAX, GRID_STEP)
    rng = random.Random(42)

    hotspots = [
        (34.5, -118.5, 0.8),
        (36.5, -118.8, 0.7),
        (39.0, -121.5, 0.75),
        (38.5, -122.5, 0.65),
        (34.0, -117.0, 0.7),
        (40.5, -122.5, 0.6),
        (37.0, -120.0, 0.55),
    ]

    predictions = []
    for lat in lats:
        for lon in lons:
            base = 0.02 + 0.03 * (1 - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN))
            hotspot_boost = 0.0
            for hlat, hlon, hstrength in hotspots:
                dist = math.sqrt((lat - hlat) ** 2 + (lon - hlon) ** 2)
                hotspot_boost += hstrength * math.exp(-dist ** 2 / 0.8)
            risk = max(min(base + hotspot_boost + rng.gauss(0, 0.01), 1.0), 0.0)
            predictions.append({"lat": round(lat, 2), "lon": round(lon, 2), "risk": round(risk, 4)})
    return predictions


def from_csv(csv_path):
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. pip install pandas")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    lat_col = "lat_cell" if "lat_cell" in df.columns else "latitude"
    lon_col = "lon_cell" if "lon_cell" in df.columns else "longitude"
    prob_col = "predicted_fire_probability"

    if prob_col not in df.columns:
        print(f"ERROR: '{prob_col}' not found. Available: {df.columns.tolist()}")
        sys.exit(1)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        latest = df["date"].max()
        df = df[df["date"] == latest]
        print(f"Using predictions for date: {latest}")

    grid = df.groupby([lat_col, lon_col])[prob_col].mean().reset_index()
    predictions = []
    for _, row in grid.iterrows():
        predictions.append({"lat": round(float(row[lat_col]), 2), "lon": round(float(row[lon_col]), 2), "risk": round(float(row[prob_col]), 4)})
    return predictions


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true")
    group.add_argument("--from-csv", type=str)
    args = parser.parse_args()

    if args.demo:
        predictions = generate_demo_predictions()
    else:
        predictions = from_csv(args.from_csv)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
