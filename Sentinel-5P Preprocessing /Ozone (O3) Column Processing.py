# ============================================
# Sentinel-5P O3 Column Monthly Processing Workflow
# ============================================

"""
Workflow Steps:
1. Load TROPOMI O3 Column data from Google Earth Engine (GEE).
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

# ===================================
# Earth Engine O3 Acquisition
# ===================================

O3_collection = (
    ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_O3")
    .select('O3_column_number_density')
    .filterDate('2022-01-01', '2024-12-31')
    .filterBounds(AOI)
)

# === Conversion constants ===
MOLAR_MASS_O3 = 48  # grams/mol
HEIGHT_TROPOSPHERE = 10000  # meters
GRAM_TO_MICROGRAM = 1e6

# mol/m² to µg/m³ conversion factor
CONVERSION_FACTOR = (MOLAR_MASS_O3 * GRAM_TO_MICROGRAM) / HEIGHT_TROPOSPHERE

# === Generate monthly mean images ===
months = list(range(1, 13))
years = list(range(2022, 2025))

O3_monthly_images = []

for year in years:
    for month in months:
        filtered = (
            O3_collection
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .filter(ee.Filter.calendarRange(month, month, 'month'))
        )
        monthly_mean_mol = filtered.mean()
        monthly_mean_ug = (
            monthly_mean_mol.multiply(CONVERSION_FACTOR)
            .clip(AOI)
            .set('year', year)
            .set('month', month)
            .set('system:time_start', ee.Date.fromYMD(year, month, 1))
            .set('label', f"O3_{year:04d}_{month:02d}")
        )
        O3_monthly_images.append(monthly_mean_ug)

# === Convert to ImageCollection ===
O3_monthly_collection = ee.ImageCollection(O3_monthly_images)

# === Export each monthly image to Google Drive ===
for image in O3_monthly_images:
    label = image.get('label').getInfo()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=label,
        folder='O3_Monthly',
        fileNamePrefix=label,
        scale=1113.2,
        region=AOI,
        crs='EPSG:32748',
        maxPixels=1e13
    )
    task.start()
    print(f"Export task '{label}' started.")

# ===================================
# Missing Value Filling (Interpolation)
# ===================================

input_folder = "/content/drive/MyDrive/O3_Monthly"
output_folder = os.path.join(input_folder, "output_interpolated")
os.makedirs(output_folder, exist_ok=True)

raster_files = glob(os.path.join(input_folder, "*.tif"))

for raster_file in raster_files:
    print(f"Processing: {os.path.basename(raster_file)}")

    with rasterio.open(raster_file) as src:
        o3 = src.read(1).astype(float)
        o3[o3 < 0] = np.nan  # Remove invalid values
        transform = src.transform
        crs = src.crs
        shape = src.shape

        # Interpolation for the whole raster area
        x = np.arange(shape[1])
        y = np.arange(shape[0])
        xv, yv = np.meshgrid(x, y)

        z_flat = o3.flatten()
        valid = ~np.isnan(z_flat)
        points = np.column_stack((xv.flatten()[valid], yv.flatten()[valid]))
        values = z_flat[valid]

        z_interp = griddata(points, values, (xv, yv), method='linear')
        o3_filled = np.where(np.isnan(o3), z_interp, o3)

        # Save output
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
            dst.write(o3_filled.astype("float32"), 1)

        # Visualize
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        cmap = 'viridis'

        im1 = axs[0].imshow(o3, cmap=cmap)
        axs[0].set_title("Before Interpolation")
        plt.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(o3_filled, cmap=cmap)
        axs[1].set_title("After Interpolation")
        plt.colorbar(im2, ax=axs[1])

        plt.suptitle(os.path.basename(raster_file))
        plt.tight_layout()
        plt.show()

print("\n✅ All Sentinel-5P O3 rasters interpolated and visualized.")

# ===================================
# Downscaling
# ===================================

def visualize_comparison(before_array, after_array, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    cmap = 'viridis'
    vmin = min(np.nanmin(before_array), np.nanmin(after_array))
    vmax = max(np.nanmax(before_array), np.nanmax(after_array))

    im1 = axs[0].imshow(before_array, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title("Before Resampling")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(after_array, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title("After Resampling (1 km)")
    plt.colorbar(im2, ax=axs[1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def resample_raster(input_path, output_path, target_resolution=1000, visualize=True):
    with rasterio.open(input_path) as src:
        src_array = src.read(1)
        dst_transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds, resolution=target_resolution
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'transform': dst_transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst_array = np.empty((height, width), dtype=src.dtypes[0])
            reproject(
                source=src_array,
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan
            )
            dst.write(dst_array, 1)

    print(f"✅ {os.path.basename(input_path)} resampled to 1 km")

    if visualize:
        visualize_comparison(src_array, dst_array, os.path.basename(input_path))

input_folder = output_folder
resampled_folder = os.path.join(input_folder, "../output_resampled")
os.makedirs(resampled_folder, exist_ok=True)

for raster_file in glob(os.path.join(input_folder, "*.tif")):
    filename = os.path.basename(raster_file)
    output_path = os.path.join(resampled_folder, filename)
    resample_raster(raster_file, output_path, visualize=True)

# ===================================
# Downscaling Evaluation
# ===================================

def evaluate_resampling(original_path, resampled_path):
    with rasterio.open(original_path) as src_orig, rasterio.open(resampled_path) as src_resamp:
        original = src_orig.read(1)
        resampled_reproj = np.empty_like(original)
        reproject(
            source=src_resamp.read(1),
            destination=resampled_reproj,
            src_transform=src_resamp.transform,
            src_crs=src_resamp.crs,
            dst_transform=src_orig.transform,
            dst_crs=src_orig.crs,
            resampling=Resampling.bilinear
        )

        mask = ~np.isnan(original) & ~np.isnan(resampled_reproj)
        y_true = original[mask]
        y_pred = resampled_reproj[mask]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "filename": os.path.basename(original_path),
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        }

original_files = sorted(glob(os.path.join(input_folder, "*.tif")))
resampled_files = sorted(glob(os.path.join(resampled_folder, "*.tif")))

for orig, resamp in zip(original_files, resampled_files):
    stats = evaluate_resampling(orig, resamp)
    print(f"📊 {stats['filename']}: RMSE={stats['RMSE']:.2e}, MAE={stats['MAE']:.2e}, "
          f"MSE={stats['MSE']:.2e}, R²={stats['R2']:.4f}")

# ===================================
# Layer Stacking
# ===================================

stack_input_dir = resampled_folder
raster_files = sorted(glob(os.path.join(stack_input_dir, "*.tif")))
raster_list = [rxr.open_rasterio(f) for f in raster_files]
stacked_raster = xr.concat(raster_list, dim='band')

stacked_output = os.path.join(stack_input_dir, "../O3_Jakarta_Stacked.tif")
stacked_raster.rio.to_raster(stacked_output)

print(f"\n✅ Layer stacking complete: {stacked_output}")
