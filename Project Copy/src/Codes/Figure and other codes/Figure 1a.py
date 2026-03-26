import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":

    df = pd.read_excel(os.path.join(DATA_DIR, "Figure 1.xlsx"))

    lon = pd.to_numeric(df["Lon"], errors="coerce")
    lat = pd.to_numeric(df["Lat"], errors="coerce")
    r2  = pd.to_numeric(df["R2"], errors="coerce")

    mask = (~lon.isna()) & (~lat.isna()) & (~r2.isna())
    lon = lon[mask]
    lat = lat[mask]
    r2  = r2[mask]

    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="black", linewidth=0.2)

    sc = ax.scatter(
        lon,
        lat,
        c=r2,
        s=12,
        cmap="RdYlBu_r",
        vmin=0.3,
        vmax=0.75,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(
    sc,
    ax=ax,
    orientation="horizontal",
    pad=0.05,
    fraction=0.05,
    ticks=[0.3, 0.4, 0.5, 0.6, 0.7]
)
    cbar.set_label("CV R²", fontsize=14)

    cbar.ax.tick_params(labelsize=14, width=2)
    cbar.outline.set_linewidth(2)

    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight("bold")

    ax.text(-175, 75, "(a) CV-R²", fontsize=18, fontweight="bold",
            transform=ccrs.PlateCarree())

    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1a_R2_map.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
