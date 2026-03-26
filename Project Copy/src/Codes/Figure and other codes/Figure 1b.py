import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'Data')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PATH = os.path.join(DATA_DIR, 'Figure 1.xlsx')

df = pd.read_excel(FILE_PATH, sheet_name='Figure 1b')

lon = df['Lon'].values
lat = df['Lat'].values
nrmse = df['NRMSE'].values

fig = plt.figure(figsize=(12, 4))
ax = plt.axes(projection=ccrs.PlateCarree())

sc = ax.scatter(
    lon,
    lat,
    c=nrmse,
    cmap='RdYlBu_r',
    vmin=0.2,
    vmax=1.8,
    s=6,
    alpha=0.85,
    transform=ccrs.PlateCarree()
)

ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
ax.add_feature(cfeature.BORDERS, linewidth=0.25)
ax.set_global()

ax.gridlines(draw_labels=False, linewidth=0.2, linestyle='--')

bounds = [0.2, 0.6, 1.0, 1.4, 1.8]

divider = make_axes_locatable(ax)
cax = divider.append_axes(
    "bottom",
    size="6%",
    pad=0.25,
    axes_class=plt.Axes
)

cbar = plt.colorbar(
    sc,
    cax=cax,
    orientation='horizontal',
    ticks=bounds,
    boundaries=bounds,
    spacing='uniform',
    drawedges=True
)

ax.text(
    0.01, 0.05,
    '(b) NRMSE',
    transform=ax.transAxes,
    fontsize=12,
    fontweight='bold'
)

output_path = os.path.join(OUTPUT_DIR, 'Figure_1b_NRMSE.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
