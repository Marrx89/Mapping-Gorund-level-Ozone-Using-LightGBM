# ============================================
# Sentinel-5P CO Monthly Processing Workflow
# ============================================

"""
Workflow Steps:
1. Load TROPOMI CO data from Google Earth Engine (GEE).
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
# Earth Engine CO Acquisition
# ===================================

CO_collection = (
    ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
    .select('CO_column_number_density')
    .filterDate('2022-01-01', '2024-12-31')
    .filterBounds(AOI)
)

# === Conversion constants ===
MOLAR_MASS_CO = 28  # grams/mol
HEIGHT_TROPOSPHERE = 10_000  # meters
GRAM_TO_MICROGRAM = 1e6

CONVERSION_FACTOR = (MOLAR_MASS_CO * GRAM_TO_MICROGRAM) / HEIGHT_TROPOSPHERE

# === Generate monthly mean images ===
months = list(range(1, 13))
years = list(range(2022, 2025))

CO_monthly_images = []

for year in years:
    for month in months:
        filtered = (
            CO_collection
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
            .set('label', f"CO_{year:04d}_{month:02d}")
        )
        CO_monthly_images.append(monthly_mean_ug)

CO_monthly_collection = ee.ImageCollection(CO_monthly_images)

# === Export each monthly image to Google Drive ===
for image in CO_monthly_images:
    label = image.get('label').getInfo()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=label,
        folder='Tropomi_CO_Monthly',
        fileNamePrefix=label,
        scale=1113.2,
        region=AOI,
        crs='EPSG:32748',
        maxPixels=1e13
    )
    task.start()
    print(f"Export task '{label}' started.")

# ===================================
# Missing Value Filling (Land Mask Interpolation)
# ===================================

input_folder = "/content/drive/MyDrive/Tropomi_CO_Monthly"
shapefile_path = "/content/drive/MyDrive/JABODETABEK/JABODETABEK.shp"
output_folder = os.path.join(input_folder, "output_interpolated_CO")
os.makedirs(output_folder, exist_ok=True)

gdf = gpd.read_file(shapefile_path)
raster_files = glob(os.path.join(input_folder, "*.tif"))

for raster_file in raster_files:
    print(f"Processing: {os.path.basename(raster_file)}")

    with rasterio.open(raster_file) as src:
        co = src.read(1).astype(float)
        co[co < 0] = np.nan
        transform = src.transform
        crs = src.crs
        shape = src.shape

        gdf_raster_crs = gdf.to_crs(crs)

        mask = features.rasterize(
            ((geom, 1) for geom in gdf_raster_crs.geometry),
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        )

        x = np.arange(shape[1])
        y = np.arange(shape[0])
        xv, yv = np.meshgrid(x, y)

        z_flat = co.flatten()
        mask_flat = mask.flatten()

        valid = ~np.isnan(z_flat) & (mask_flat == 1)
        points = np.column_stack((xv.flatten()[valid], yv.flatten()[valid]))
        values = z_flat[valid]

        z_interp = griddata(points, values, (xv, yv), method='linear')
        co_filled = np.where((np.isnan(co)) & (mask == 1), z_interp, co)

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
            dst.write(co_filled.astype("float32"), 1)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        cmap = 'viridis'

        im1 = axs[0].imshow(co, cmap=cmap)
        axs[0].set_title("Before Interpolation")
        plt.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(co_filled, cmap=cmap)
        axs[1].set_title("After Interpolation (Land Only)")
        plt.colorbar(im2, ax=axs[1])

        plt.suptitle(os.path.basename(raster_file))
        plt.tight_layout()
        plt.show()

print("\n✅ All Sentinel-5P CO rasters interpolated and visualized.")

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

resampled_folder = os.path.join(input_folder, "output_resampled_CO")
os.makedirs(resampled_folder, exist_ok=True)

for raster_file in glob(os.path.join(output_folder, "*.tif")):
    filename = os.path.basename(raster_file)
    output_path = os.path.join(resampled_folder, filename)
    resample_raster(raster_file, output_path, visualize=True)

# ===================================
# Downscaling Evaluation
# ===================================

def evaluate_resampling(original_path, resampled_path):
    with rasterio.open(original_path) as src_orig, rasterio.open(resampled_path) as src_resamp:
        original = src_orig.read(1)
        resampled = src_resamp.read(1)

        zoom_factors = (
            original.shape[0] / resampled.shape[0],
            original.shape[1] / resampled.shape[1]
        )
        resampled_upscaled = zoom(resampled, zoom_factors, order=1)

        mask = ~np.isnan(original) & ~np.isnan(resampled_upscaled)
        y_true = original[mask]
        y_pred = resampled_upscaled[mask]

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

original_files = sorted(glob(os.path.join(output_folder, "*.tif")))
resampled_files = sorted(glob(os.path.join(resampled_folder, "*.tif")))

for orig, resamp in zip(original_files, resampled_files):
    stats = evaluate_resampling(orig, resamp)
    print(f"📊 {stats['filename']}: RMSE={stats['RMSE']:.2e}, MAE={stats['MAE']:.2e}, "
          f"MSE={stats['MSE']:.2e}, R²={stats['R2']:.4f}")

# ===================================
# Layer Stacking
# ===================================

raster_files = sorted(glob(os.path.join(resampled_folder, "*.tif")))
raster_list = [rxr.open_rasterio(f) for f in raster_files]
stacked_raster = xr.concat(raster_list, dim='band')

stacked_output = os.path.join(input_folder, "CO_Jakarta_Stacked.tif")
stacked_raster.rio.to_raster(stacked_output)

print(f"\n✅ Layer stacking complete: {stacked_output}")
