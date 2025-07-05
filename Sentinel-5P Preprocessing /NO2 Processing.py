# ============================================
# Sentinel-5P NO2 Monthly Processing Workflow
# ============================================

"""
Workflow Steps:
1. Load TROPOMI NO2 data from Google Earth Engine (GEE).
2. Convert units from mol/m² to µg/m³.
3. Export monthly rasters to Google Drive.
4. Fill missing values using land mask and spatial interpolation.
5. Downscale rasters to 1 km resolution.
6. Evaluate downscaling performance.
7. Stack all rasters into a single multi-band file.

Author: Muhammad Hilal Arrizqon & Balqis Meiliana
"""

import ee
import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.interpolate import griddata
from skimage.transform import resize
from scipy.ndimage import zoom
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='...')

# Define AOI
Map = geemap.Map()
Map

AOI = ee.Geometry.Polygon([
    [
        [106.3984, -5.8962],
        [106.3984, -6.6415],
        [107.2663, -6.6415],
        [107.2663, -5.8962]
    ]
])

Map.addLayer(AOI, {}, "AOI")
Map.centerObject(AOI, 8)

# --------------------------
# Earth Engine NO₂ Acquisition
# --------------------------

# Load TROPOMI NO₂ data
NO2_collection = (
    ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
    .select('tropospheric_NO2_column_number_density')
    .filterDate('2022-01-01', '2024-12-31')
    .filterBounds(AOI)
)

# Conversion constants
MOLAR_MASS_NO2 = 46.0055   # g/mol
HEIGHT_TROPOSPHERE = 10_000  # meters
A = 1e6  # g/m³ to µg/m³

CONVERSION_FACTOR = (MOLAR_MASS_NO2 * A) / HEIGHT_TROPOSPHERE

# Generate monthly images
months = list(range(1, 13))
years = list(range(2022, 2025))

NO2_monthly_images = []

for y in years:
    for m in months:
        filtered = (
            NO2_collection
            .filter(ee.Filter.calendarRange(y, y, 'year'))
            .filter(ee.Filter.calendarRange(m, m, 'month'))
        )

        monthly_mean_mol = filtered.mean()

        monthly_mean_ug = (
            monthly_mean_mol.multiply(CONVERSION_FACTOR)
            .clip(AOI)
            .set('year', y)
            .set('month', m)
            .set('system:time_start', ee.Date.fromYMD(y, m, 1))
            .set('label', f"NO2_{y:04d}_{m:02d}")
        )

        NO2_monthly_images.append(monthly_mean_ug)

# Convert list to ImageCollection
NO2_monthly_collection = ee.ImageCollection(NO2_monthly_images)

# Export each image to Google Drive
for image in NO2_monthly_images:
    label = image.get('label').getInfo()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=label,
        folder='Tropomi_NO2_Monthly',
        fileNamePrefix=label,
        region=AOI,
        crs='EPSG:32748',
        scale=1113.2,
        maxPixels=1e13
    )
    task.start()
    print(f"Export task '{label}' started.")

# -----------------------------
# Fill Missing Values by Interpolation
# -----------------------------

import os
import numpy as np
import rasterio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import glob

# Paths
input_folder = "/content/drive/MyDrive/PROCESSING/DATA/PrecursorO3-Sentinel5P_22_24/Tropomi_NO2_Monthly"
output_folder = os.path.join(input_folder, "output_interpolated_no2")
os.makedirs(output_folder, exist_ok=True)

# Process each raster
raster_files = glob.glob(os.path.join(input_folder, "*.tif"))

for raster_file in raster_files:
    print(f"Processing: {os.path.basename(raster_file)}")

    with rasterio.open(raster_file) as src:
        no2 = src.read(1).astype(float)
        no2[no2 < 0] = np.nan
        transform = src.transform
        crs = src.crs
        shape = src.shape

        # Interpolation
        x = np.arange(shape[1])
        y = np.arange(shape[0])
        xv, yv = np.meshgrid(x, y)

        x_flat, y_flat, z_flat = xv.flatten(), yv.flatten(), no2.flatten()
        valid = ~np.isnan(z_flat)

        z_interp = griddata(
            (x_flat[valid], y_flat[valid]),
            z_flat[valid],
            (xv, yv),
            method='linear'
        )

        no2_filled = np.where(np.isnan(no2), z_interp, no2)

        # Save result
        output_path = os.path.join(output_folder, os.path.basename(raster_file))
        with rasterio.open(
            output_path, "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(no2_filled.astype("float32"), 1)

        # Visualize
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axs[0].imshow(no2, cmap='viridis')
        axs[0].set_title("Before Interpolation")
        plt.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(no2_filled, cmap='viridis')
        axs[1].set_title("After Interpolation")
        plt.colorbar(im2, ax=axs[1])

        plt.suptitle(os.path.basename(raster_file))
        plt.tight_layout()
        plt.show()

print("\n✅ All NO₂ rasters filled and visualized.")

# -----------------------------
# Downscale to 1 km Resolution
# -----------------------------

from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage.transform import resize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def visualize_comparison(before, after, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    vmin, vmax = np.nanmin([before, after]), np.nanmax([before, after])

    im1 = axs[0].imshow(before, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title("Before Downscaling")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(after, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title("After Downscaling (1 km)")
    plt.colorbar(im2, ax=axs[1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def evaluate_stats(before, after, filename):
    after_resized = resize(after, before.shape, order=1, preserve_range=True, anti_aliasing=False)

    mask = ~np.isnan(before) & ~np.isnan(after_resized)
    if np.count_nonzero(mask) == 0:
        print(f"⚠️ {filename} not evaluable (all NaN).")
        return None

    y_true, y_pred = before[mask], after_resized[mask]
    mse, rmse = mean_squared_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
    mae, r2 = mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)

    print(f"📊 Evaluation for {filename} — RMSE: {rmse:.2e}, MSE: {mse:.2e}, MAE: {mae:.2e}, R²: {r2:.4f}")

    return {"Filename": filename, "RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2}

def resample_raster_to_1km(input_path, output_path, visualize=True):
    with rasterio.open(input_path) as src:
        src_array = src.read(1)
        dst_transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds, resolution=1000
        )

        kwargs = src.meta.copy()
        kwargs.update({'transform': dst_transform, 'width': width, 'height': height})

        dst_array = np.empty((height, width), dtype=src.dtypes[0])
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
            src_nodata=np.nan
        )

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(dst_array, 1)

    print(f"✅ {os.path.basename(input_path)} downscaled to 1 km.")

    if visualize:
        visualize_comparison(src_array, dst_array, os.path.basename(input_path))

    return src_array, dst_array

input_folder = output_folder
output_folder_resample = os.path.join(input_folder, "no2_1km")
os.makedirs(output_folder_resample, exist_ok=True)

results = []
for raster_file in glob.glob(os.path.join(input_folder, "*.tif")):
    filename = os.path.basename(raster_file)
    output_path = os.path.join(output_folder_resample, f"{os.path.splitext(filename)[0]}_resampled.tif")

    before, after = resample_raster_to_1km(raster_file, output_path)
    stats = evaluate_stats(before, after, filename)
    if stats:
        results.append(stats)

# -----------------------------
# Stack All Layers
# -----------------------------

import rioxarray as rxr
import xarray as xr

input_dir = output_folder_resample

# Update this list based on your files
raster_files = sorted(glob.glob(os.path.join(input_dir, "*_resampled.tif")))

raster_list = [rxr.open_rasterio(fp) for fp in raster_files]

stacked_raster = xr.concat(raster_list, dim='band')

stacked_output_path = os.path.join(input_dir, "Stacking_Layer", "NO2_Jakarta.tif")
os.makedirs(os.path.dirname(stacked_output_path), exist_ok=True)

stacked_raster.rio.to_raster(stacked_output_path)

print(f"✅ Stacking complete — saved to {stacked_output_path}")
