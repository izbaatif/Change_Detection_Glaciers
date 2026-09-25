# ============================================================
# IMPORT LIBRARIES
# ============================================================

# File and system handling
import os
import sys
import glob
import gc
import random

# Numerical processing
import numpy as np

# Raster processing and geospatial operations
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.mask
from rasterio.mask import mask
from rasterio import features
from rasterio.warp import reproject, Resampling

# Vector geospatial processing
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

# Date handling
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import patches

# Statistical analysis
import scipy.stats as stats
from scipy.stats import mannwhitneyu

# Image processing operations
from scipy.ndimage import (
    uniform_filter,
    maximum_filter,
    label,
    center_of_mass,
    sobel
)

# Image similarity and feature extraction
from skimage.filters import scharr
from skimage.metrics import structural_similarity as ssim

# Machine learning
from sklearn.metrics import mutual_info_score

# Linear algebra
from numpy.linalg import norm

# Coordinate transformations
from pyproj import Transformer



# ============================================================
# TIFF CROPPING FUNCTIONS
# ============================================================

def crop_tiff(tiff_path, shapefile_path, output_path=None):
    """
    Crop a raster TIFF file using a vector shapefile boundary.

    Parameters
    ----------
    tiff_path : str
        Input raster file path.
    
    shapefile_path : str
        Glacier boundary shapefile.

    output_path : str, optional
        Path where cropped raster will be saved.

    Returns
    -------
    output_path : str
        Location of cropped raster.
    """

    # Load glacier boundary shapefile
    gdf = gpd.read_file(shapefile_path)
    
    with rasterio.open(tiff_path) as src:

        # Reproject glacier boundary to raster CRS
        # because rasterio.mask requires matching coordinate systems
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        
        # Convert geometries into rasterio compatible format
        shapes = [
            feature["geometry"]
            for feature in gdf.__geo_interface__["features"]
        ]
        
        # Crop raster using glacier boundary
        out_image, out_transform = mask(
            src,
            shapes,
            crop=True
        )

        # Copy original raster metadata
        out_meta = src.meta.copy()
        
        # Update metadata after cropping
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # If output path is not provided,
        # automatically generate cropped filename
        if output_path is None:
            base, ext = os.path.splitext(tiff_path)
            output_path = f"{base}_cropped{ext}"
        
        # Save cropped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
    print(f"   Cropped TIFF saved at: {output_path}")

    return output_path



def crop_tiff_to_shapefile(base_dir, shapefile_path):
    """
    Crop all VV and VH Sentinel-1 TIFF files
    from each acquisition date using glacier shapefile.

    Expected folder structure:

    base_dir/
        YYYY_MM_DD/
            VV.tif
            VH.tif

    Output:

        YYYY_MM_DD/
            VV_shape.tif
            VH_shape.tif

    Returns
    -------
    time_series_data : dict
        Dictionary containing temporal SAR file paths.
    """

    time_series_data = {}

    # Check if input directory exists
    if not os.path.exists(base_dir):
        print(
            f" Error: Base directory '{base_dir}' does not exist."
        )
        return time_series_data


    # Iterate through acquisition folders
    for burst_name in sorted(os.listdir(base_dir)):

        burst_path = os.path.join(
            base_dir,
            burst_name
        )

        # Ignore files and keep only folders
        if not os.path.isdir(burst_path):
            continue


        # Extract acquisition date from folder name
        try:
            date_obj = datetime.strptime(
                burst_name,
                "%Y_%m_%d"
            ).date()

        except ValueError:
            print(
                f"Skipping {burst_name}: cannot parse date"
            )
            continue


        print(
            f"\nProcessing burst folder: {burst_name}"
        )


        # Input VV and VH paths
        vv_tif = os.path.join(
            burst_path,
            "VV.tif"
        )

        vh_tif = os.path.join(
            burst_path,
            "VH.tif"
        )


        # Output cropped paths
        vv_output = os.path.join(
            burst_path,
            "VV_shape.tif"
        )

        vh_output = os.path.join(
            burst_path,
            "VH_shape.tif"
        )


        # -----------------------------
        # Crop VV polarization
        # -----------------------------
        vv_done = False

        if (
            os.path.exists(vv_tif)
            and os.path.exists(shapefile_path)
        ):

            try:
                crop_tiff(
                    vv_tif,
                    shapefile_path,
                    vv_output
                )

                print(
                    f"   VV processed: {vv_output}"
                )

                vv_done = True

            except Exception as e:
                print(
                    f"   VV error ({burst_name}): {e}"
                )


        # -----------------------------
        # Crop VH polarization
        # -----------------------------
        vh_done = False

        if (
            os.path.exists(vh_tif)
            and os.path.exists(shapefile_path)
        ):

            try:
                crop_tiff(
                    vh_tif,
                    shapefile_path,
                    vh_output
                )

                print(
                    f"   VH processed: {vh_output}"
                )

                vh_done = True

            except Exception as e:
                print(
                    f"   VH error ({burst_name}): {e}"
                )


        # Store successfully processed dates
        if vv_done or vh_done:

            time_series_data[date_obj] = {

                "burst_name": burst_name,

                "VV_path": vv_tif,

                "VH_path": vh_tif,

                "VV_shape_path":
                    vv_output if vv_done else None,

                "VH_shape_path":
                    vh_output if vh_done else None,
            }


    # Sort acquisitions chronologically
    time_series_data = dict(
        sorted(time_series_data.items())
    )
    
    print(
        f"\n Loaded {len(time_series_data)} "
        "bursts successfully into time-series data framework."
    )

    return time_series_data


# ============================================================
# VV / VH CROPPED IMAGE VISUALIZATION
# ============================================================

def save_cropped_polarizations_plot(
    time_series_data,
    shapefile_path,
    output_images
):
    """
    Loads cropped VV and VH images, normalizes them,
    overlays glacier boundaries, and saves temporal plots.

    Each acquisition date is saved separately as:

        output_images/
            VV_VH_cropped/
                YYYY_MM_DD.png

    Parameters
    ----------
    time_series_data : dict
        Dictionary containing cropped VV/VH raster paths.

    shapefile_path : str
        Glacier boundary shapefile.

    output_images : str
        Directory for saving plots.
    """

    if not time_series_data:
        print(
            " Error: The provided time_series_data dictionary "
            "is empty or None."
        )
        return



    # ----------------------------------------------------
    # Internal helper:
    # Load raster, crop to glacier geometry,
    # and normalize pixel values between 0 and 1
    # ----------------------------------------------------
    def _load_crop_normalize(tiff_path, geoms):

        with rasterio.open(tiff_path) as src:

            cropped, transform = mask(
                src,
                geoms,
                crop=True,
                filled=True,
                nodata=np.nan
            )

            # Extract single raster band
            array = cropped[0]


            # Ignore invalid pixels outside glacier
            valid = array[
                ~np.isnan(array)
            ]


            if valid.size == 0:
                raise ValueError(
                    f"No valid pixels found within geometry for: {tiff_path}"
                )


            # Min-max normalization
            arr_min = valid.min()
            arr_max = valid.max()


            if arr_max == arr_min:

                norm = np.zeros_like(array)

            else:

                norm = (
                    (array - arr_min)
                    /
                    (arr_max - arr_min)
                )


            return norm, transform, src.crs




    # ----------------------------------------------------
    # Internal helper:
    # Convert raster dimensions and transform into
    # plotting coordinates
    # ----------------------------------------------------
    def _get_extent(arr, transform):

        h, w = arr.shape

        xmin = transform[2]

        xmax = (
            xmin
            +
            w * transform[0]
        )

        ymax = transform[5]

        ymin = (
            ymax
            +
            h * transform[4]
        )

        return xmin, xmax, ymin, ymax




    # Load glacier boundary
    shape = gpd.read_file(
        shapefile_path
    )


    # Use first raster to determine required CRS
    example_raster_path = next(
        iter(time_series_data.values())
    )["VV_shape_path"]


    with rasterio.open(example_raster_path) as src:

        raster_crs = src.crs



    # Match shapefile projection with SAR raster
    shape = shape.to_crs(
        raster_crs
    )


    # Merge glacier polygons into single geometry
    geom = [
        unary_union(
            shape.geometry
        )
    ]



    # ----------------------------------------------------
    # Process every acquisition date
    # ----------------------------------------------------
    for date, data in time_series_data.items():

        if hasattr(date, "strftime"):

            date_str = date.strftime(
                "%Y_%m_%d"
            )

        else:

            date_str = str(date).replace(
                "-",
                "_"
            )



        vv_file = data["VV_shape_path"]

        vh_file = data["VH_shape_path"]


        # Skip dates without both polarizations
        if not vv_file or not vh_file:

            print(
                f"⚠️ Skipping date {date_str}: "
                "Missing cropped TIFF file paths."
            )

            continue



        try:

            vv_norm, vv_transform, _ = (
                _load_crop_normalize(
                    vv_file,
                    geom
                )
            )


            vh_norm, vh_transform, _ = (
                _load_crop_normalize(
                    vh_file,
                    geom
                )
            )


        except Exception as e:

            print(
                f"⚠️ Skipping date {date_str} "
                f"due to processing error: {e}"
            )

            continue



        # Calculate raster geographic extents
        vv_xmin, vv_xmax, vv_ymin, vv_ymax = (
            _get_extent(
                vv_norm,
                vv_transform
            )
        )

        vh_xmin, vh_xmax, vh_ymin, vh_ymax = (
            _get_extent(
                vh_norm,
                vh_transform
            )
        )



        # Clip glacier outline to raster area
        vv_shape_crop = shape.clip(
            box(
                vv_xmin,
                vv_ymin,
                vv_xmax,
                vv_ymax
            )
        )


        vh_shape_crop = shape.clip(
            box(
                vh_xmin,
                vh_ymin,
                vh_xmax,
                vh_ymax
            )
        )



        # Set invalid pixels to black
        cmap = plt.cm.gray.copy()

        cmap.set_bad(
            color="black"
        )



        # Create VV and VH comparison figure
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 6)
        )


        fig.suptitle(
            f"{date_str} — VV & VH Cropped (0–1 normalized)"
        )



        # -----------------------------
        # VV visualization
        # -----------------------------
        im1 = axes[0].imshow(
            vv_norm,
            cmap=cmap,
            extent=[
                vv_xmin,
                vv_xmax,
                vv_ymin,
                vv_ymax
            ],
            origin="upper"
        )


        axes[0].set_title(
            "VV (0–1 normalized)"
        )


        if len(vv_shape_crop) > 0:

            vv_shape_crop.boundary.plot(
                ax=axes[0],
                color="red",
                linewidth=1
            )


        fig.colorbar(
            im1,
            ax=axes[0]
        )



        # -----------------------------
        # VH visualization
        # -----------------------------
        im2 = axes[1].imshow(
            vh_norm,
            cmap=cmap,
            extent=[
                vh_xmin,
                vh_xmax,
                vh_ymin,
                vh_ymax
            ],
            origin="upper"
        )


        axes[1].set_title(
            "VH (0–1 normalized)"
        )


        if len(vh_shape_crop) > 0:

            vh_shape_crop.boundary.plot(
                ax=axes[1],
                color="red",
                linewidth=1
            )


        fig.colorbar(
            im2,
            ax=axes[1]
        )



        plt.tight_layout()



        # Save plot
        target_dir = os.path.join(
            output_images,
            "VV_VH_cropped"
        )


        os.makedirs(
            target_dir,
            exist_ok=True
        )


        final_save_path = os.path.join(
            target_dir,
            f"{date_str}.png"
        )


        plt.savefig(
            final_save_path,
            bbox_inches="tight",
            dpi=300,
            facecolor="white"
        )


        print(
            f" Plot saved at: {final_save_path}"
        )


        plt.close()




# ============================================================
# RASTER / SHAPEFILE OVERLAP HELPER
# ============================================================

def crop_shapefile_to_raster(
    shape_path,
    arr,
    transform
):
    """
    Crop glacier boundary to raster spatial footprint.

    Parameters
    ----------
    shape_path : str
        Glacier shapefile.

    arr : ndarray
        Raster array.

    transform : affine transform
        Raster georeferencing transform.

    Returns
    -------
    gdf_crop : GeoDataFrame
        Cropped glacier boundary.

    extent : tuple
        Raster plotting extent.
    """

    # Load glacier vector data
    gdf = gpd.read_file(
        shape_path
    )


    # Raster dimensions
    h, w = arr.shape


    # Extract raster boundaries
    xmin = transform.c

    ymax = transform.f

    xmax = (
        xmin
        +
        transform.a * w
    )

    ymin = (
        ymax
        +
        transform.e * h
    )


    # Create raster footprint polygon
    raster_box = box(
        xmin,
        ymin,
        xmax,
        ymax
    )


    crop_area = gpd.GeoDataFrame(
        geometry=[raster_box],
        crs=gdf.crs
    )


    # Clip glacier boundary to raster coverage
    gdf_crop = gpd.clip(
        gdf,
        crop_area
    )


    return gdf_crop, (
        xmin,
        xmax,
        ymin,
        ymax
    )


# ============================================================
# TEMPORAL STACK COMPILATION
# ============================================================

def compile_temporal_stacks(time_series_data):
    """
    Load all cropped VV and VH rasters and combine them into
    temporal 3D arrays.

    Output format:

        vv_stack:
            (number_of_dates, rows, columns)

        vh_stack:
            (number_of_dates, rows, columns)

    Since cropped images may have slightly different dimensions,
    all images are trimmed to the smallest common spatial size.
    """

    vv_stack = []
    vh_stack = []


    # Load each acquisition date in chronological order
    for date, data in sorted(time_series_data.items()):

        # Read VV polarization
        with rasterio.open(
            data["VV_shape_path"]
        ) as src:

            vv_stack.append(
                src.read(1)
            )


        # Read VH polarization
        with rasterio.open(
            data["VH_shape_path"]
        ) as src:

            vh_stack.append(
                src.read(1)
            )


    # Find common spatial dimensions
    # This avoids stacking errors caused by different raster sizes
    min_rows = min(
        arr.shape[0]
        for arr in vv_stack
    )

    min_cols = min(
        arr.shape[1]
        for arr in vv_stack
    )


    # Crop every image to common dimensions
    vv_stack = np.array(
        [
            arr[:min_rows, :min_cols]
            for arr in vv_stack
        ]
    )


    vh_stack = np.array(
        [
            arr[:min_rows, :min_cols]
            for arr in vh_stack
        ]
    )


    return vv_stack, vh_stack




# ============================================================
# GENERAL GEOSPATIAL PLOT FUNCTION
# ============================================================

def plot_and_save(
    arr,
    profile,
    title,
    filename,
    output_images_dir,
    glacier_shp,
    cmap='gray',
    color_label=None
):
    """
    Plot raster layer with glacier boundary overlay
    and save as image.

    Used for:
        - Mean maps
        - Standard deviation maps
        - CV maps
        - Ratio maps
        - Statistical outputs
    """


    # Create output directory if missing
    os.makedirs(
        output_images_dir,
        exist_ok=True
    )


    # Crop glacier boundary to raster extent
    gdf_crop, extent = crop_shapefile_to_raster(
        glacier_shp,
        arr,
        profile["transform"]
    )


    # Create figure
    plt.figure(
        figsize=(8, 6)
    )


    # Display raster
    im = plt.imshow(
        arr,
        cmap=cmap,
        extent=extent
    )


    # Add color scale
    plt.colorbar(
        im,
        fraction=0.046,
        pad=0.04,
        label=color_label
    )


    # Overlay glacier outline
    if len(gdf_crop) > 0:

        gdf_crop.boundary.plot(
            ax=plt.gca(),
            color='red',
            linewidth=1
        )


    plt.title(
        title
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )


    plt.axis(
        "on"
    )


    # Save figure
    out_path = os.path.join(
        output_images_dir,
        filename
    )


    plt.savefig(
        out_path,
        bbox_inches='tight',
        dpi=300
    )


    plt.close()


    print(
        f" Plot saved: {out_path}"
    )





# ============================================================
# TEMPORAL STATISTICAL FEATURE GENERATION
# ============================================================

def compute_and_plot_pixel_stats(
    time_series_data,
    vv_stack,
    vh_stack,
    output_dir,
    output_images_dir,
    glacier_shp
):
    """
    Calculate temporal statistics from Sentinel-1
    VV, VH, and VV/VH ratio stacks.

    Generated statistics:

        Mean
        Standard deviation
        Maximum value

    Both .npy and GeoTIFF outputs are generated.
    """


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # Container for statistical layers
    stats_dict = {

        "VV_mean":
            np.mean(vv_stack, axis=0),

        "VV_std":
            np.std(vv_stack, axis=0),

        "VV_max":
            np.max(vv_stack, axis=0),


        "VH_mean":
            np.mean(vh_stack, axis=0),

        "VH_std":
            np.std(vh_stack, axis=0),

        "VH_max":
            np.max(vh_stack, axis=0),
    }



    # --------------------------------------------------------
    # Calculate VV/VH ratio stack
    #
    # Small epsilon avoids division by zero
    # --------------------------------------------------------
    eps = 1e-12

    ratio_stack = np.divide(
        vv_stack,
        vh_stack,
        out=np.full_like(
            vv_stack,
            np.nan
        ),
        where=vh_stack > eps
    )



    # Ratio temporal statistics
    stats_dict["VV_VH_mean"] = np.nanmean(
        ratio_stack,
        axis=0
    )

    stats_dict["VV_VH_std"] = np.nanstd(
        ratio_stack,
        axis=0
    )

    stats_dict["VV_VH_max"] = np.nanmax(
        ratio_stack,
        axis=0
    )



    # Save numpy arrays
    for key, arr in stats_dict.items():

        np.save(
            os.path.join(
                output_dir,
                f"{key}.npy"
            ),
            arr
        )



    # --------------------------------------------------------
    # Export GeoTIFF outputs using first acquisition
    # as spatial reference
    # --------------------------------------------------------
    template_file = sorted(
        time_series_data.items()
    )[0][1]["VV_shape_path"]


    with rasterio.open(
        template_file
    ) as src:

        meta = src.meta.copy()

        meta.update(
            dtype=rasterio.float32,
            count=1
        )



        for key, arr in stats_dict.items():

            out_path = os.path.join(
                output_dir,
                f"{key}.tif"
            )


            with rasterio.open(
                out_path,
                'w',
                **meta
            ) as dst:

                dst.write(
                    arr.astype(np.float32),
                    1
                )



    # --------------------------------------------------------
    # Generate visualization maps
    # --------------------------------------------------------

    # Mean maps
    plot_and_save(
        stats_dict["VV_mean"],
        meta,
        "VV Mean",
        "VV_mean.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude"
    )


    plot_and_save(
        stats_dict["VH_mean"],
        meta,
        "VH Mean",
        "VH_mean.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude"
    )


    plot_and_save(
        np.nan_to_num(
            stats_dict["VV_VH_mean"],
            nan=0.0
        ),
        meta,
        "VV/VH Ratio Mean",
        "VV_VH_mean.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude Ratio"
    )



    # Standard deviation maps
    plot_and_save(
        stats_dict["VV_std"],
        meta,
        "VV Standard Deviation",
        "VV_std.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude"
    )


    plot_and_save(
        stats_dict["VH_std"],
        meta,
        "VH Standard Deviation",
        "VH_std.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude"
    )


    plot_and_save(
        np.nan_to_num(
            stats_dict["VV_VH_std"],
            nan=0.0
        ),
        meta,
        "VV/VH Ratio Standard Deviation",
        "VV_VH_std.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude Ratio"
    )



    # Maximum value maps
    plot_and_save(
        stats_dict["VV_max"],
        meta,
        "VV Maximum",
        "VV_max.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude"
    )


    plot_and_save(
        stats_dict["VH_max"],
        meta,
        "VH Maximum",
        "VH_max.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude"
    )


    plot_and_save(
        np.nan_to_num(
            stats_dict["VV_VH_max"],
            nan=0.0
        ),
        meta,
        "VV/VH Ratio Maximum",
        "VV_VH_max.png",
        output_images_dir,
        glacier_shp,
        cmap='gray',
        color_label="Amplitude Ratio"
    )


    return ratio_stack


# ============================================================
# COEFFICIENT OF VARIATION CALCULATION
# ============================================================

def compute_cv_and_plot(
    output_dir,
    output_images_dir,
    glacier_shp
):
    """
    Calculate coefficient of variation (CV) maps.

    CV measures relative temporal variability:

        CV = Standard Deviation / Mean

    The calculation is performed for:

        VV
        VH
        VV/VH ratio

    Outputs:
        - NumPy arrays (.npy)
        - GeoTIFF files (.tif)
        - Visualization plots (.png)
    """



    # Process each available temporal feature layer
    for var in [
        "VV",
        "VH",
        "VV_VH"
    ]:


        # Input mean and standard deviation maps
        mean_tif = os.path.join(
            output_dir,
            f"{var}_mean.tif"
        )

        std_tif = os.path.join(
            output_dir,
            f"{var}_std.tif"
        )



        # Skip variables where required maps do not exist
        if (
            not os.path.exists(mean_tif)
            or
            not os.path.exists(std_tif)
        ):
            continue



        # Load mean raster
        with rasterio.open(mean_tif) as src:

            mean_arr = src.read(1)

            profile = src.profile



        # Load standard deviation raster
        std_arr = rasterio.open(
            std_tif
        ).read(1)



        # Calculate coefficient of variation
        # Small epsilon prevents division by zero
        cv_arr = (
            std_arr
            /
            (mean_arr + 1e-12)
        )



        # Save NumPy version
        np.save(
            os.path.join(
                output_dir,
                f"{var}_CV.npy"
            ),
            cv_arr
        )



        # Prepare GeoTIFF metadata
        cv_profile = profile.copy()

        cv_profile.update(
            dtype=rasterio.float32,
            count=1
        )



        # Save CV GeoTIFF
        cv_tif_path = os.path.join(
            output_dir,
            f"{var}_CV.tif"
        )


        with rasterio.open(
            cv_tif_path,
            'w',
            **cv_profile
        ) as dst:

            dst.write(
                cv_arr.astype(np.float32),
                1
            )



        # Mask invalid/negative values before visualization
        cv_tif_arr = np.ma.masked_where(
            cv_arr <= 0,
            cv_arr
        )



        # Plot CV map
        plot_and_save(
            cv_tif_arr,
            cv_profile,
            f"{var} – Coefficient of Variation",
            f"{var}_CV.png",
            output_images_dir,
            glacier_shp,
            cmap="rainbow",
            color_label="CV"
        )





# ============================================================
# BASELINE STATISTICS EXCLUDING MAXIMUM VV EVENT
# ============================================================

def compute_excluding_max_vv(
    time_series_data,
    vv_stack,
    vh_stack,
    ratio_stack,
    output_dir,
    output_images_dir,
    glacier_shp
):
    """
    Compute temporal statistics after removing the
    acquisition date with maximum VV response.

    Purpose:
        Reduce influence of extreme surge/collapse events
        and estimate stable glacier background behaviour.

    Steps:
        1. Identify maximum VV acquisition per pixel.
        2. Remove that acquisition from temporal stack.
        3. Calculate remaining temporal statistics.
    """



    os.makedirs(
        output_dir,
        exist_ok=True
    )



    # Sorted acquisition dates
    dates = sorted(
        time_series_data.keys()
    )



    # --------------------------------------------------------
    # Find acquisition index containing maximum VV
    # for every pixel
    # --------------------------------------------------------
    max_idx = np.argmax(
        vv_stack,
        axis=0
    )



    # Extract maximum VV values
    vv_max = np.take_along_axis(
        vv_stack,
        max_idx[None, ...],
        axis=0
    )[0]



    # Extract corresponding VH values
    vh_at_vvmax = np.take_along_axis(
        vh_stack,
        max_idx[None, ...],
        axis=0
    )[0]



    # Calculate VV/VH ratio at maximum VV event
    ratio_at_vvmax = (
        vv_max
        /
        (vh_at_vvmax + 1e-6)
    )



    # Use first acquisition as spatial reference
    with rasterio.open(
        time_series_data[dates[0]]["VV_shape_path"]
    ) as src:

        profile = src.profile



    # Save maximum VV ratio and event index
    profile.update(
        dtype=rasterio.float32,
        count=2
    )



    with rasterio.open(
        os.path.join(
            output_dir,
            "VV_VH_ratio_at_VVmax_with_date.tif"
        ),
        'w',
        **profile
    ) as dst:


        # Band 1:
        # Ratio value during maximum VV event
        dst.write(
            ratio_at_vvmax.astype(np.float32),
            1
        )


        # Band 2:
        # Acquisition index containing maximum VV
        dst.write(
            max_idx.astype(np.float32),
            2
        )



    # --------------------------------------------------------
    # Create mask excluding maximum VV acquisition
    # --------------------------------------------------------
    mask_other = np.ones_like(
        vv_stack,
        dtype=bool
    )



    for t in range(
        vv_stack.shape[0]
    ):

        # Remove maximum VV date for each pixel
        mask_other[t] = (
            max_idx != t
        )



    # Replace removed values with NaN
    vv_other = np.where(
        mask_other,
        vv_stack,
        np.nan
    )


    vh_other = np.where(
        mask_other,
        vh_stack,
        np.nan
    )


    ratio_other = np.where(
        mask_other,
        ratio_stack,
        np.nan
    )



    # Output rasters contain one band
    profile.update(
        count=1
    )



    # --------------------------------------------------------
    # Calculate statistics from remaining dates
    # --------------------------------------------------------
    stats_map = {


        "VV_mean_excluding_maxVV.tif":
            np.nanmean(
                vv_other,
                axis=0
            ),


        "VV_std_excluding_maxVV.tif":
            np.nanstd(
                vv_other,
                axis=0
            ),


        "VH_mean_excluding_maxVV.tif":
            np.nanmean(
                vh_other,
                axis=0
            ),


        "VH_std_excluding_maxVV.tif":
            np.nanstd(
                vh_other,
                axis=0
            ),


        "VV_VH_ratio_mean_excluding_maxVV.tif":
            np.nanmean(
                ratio_other,
                axis=0
            ),


        "VV_VH_ratio_std_excluding_maxVV.tif":
            np.nanstd(
                ratio_other,
                axis=0
            )
    }



    # Save statistical layers
    for name, arr in stats_map.items():

        with rasterio.open(
            os.path.join(
                output_dir,
                name
            ),
            'w',
            **profile
        ) as dst:


            dst.write(
                arr.astype(np.float32),
                1
            )



    # --------------------------------------------------------
    # Visualization of excluded maximum-VV statistics
    # --------------------------------------------------------

    plot_and_save(
        np.nan_to_num(
            stats_map[
                "VV_mean_excluding_maxVV.tif"
            ]
        ),
        profile,
        "VV Mean Excluding Maximum VV",
        "VV_excluding_max_mean.png",
        output_images_dir,
        glacier_shp,
        cmap='Reds',
        color_label="Amplitude"
    )



    plot_and_save(
        np.nan_to_num(
            stats_map[
                "VV_VH_ratio_mean_excluding_maxVV.tif"
            ]
        ),
        profile,
        "VV/VH Ratio Mean Excluding Maximum VV",
        "VV_VH_excluding_max_mean.png",
        output_images_dir,
        glacier_shp,
        cmap='Reds',
        color_label="Amplitude Ratio"
    )



    plot_and_save(
        np.nan_to_num(
            stats_map[
                "VV_std_excluding_maxVV.tif"
            ]
        ),
        profile,
        "VV Std Excluding Maximum VV",
        "VV_excluding_max_std.png",
        output_images_dir,
        glacier_shp,
        cmap='Reds',
        color_label="Amplitude"
    )



    plot_and_save(
        np.nan_to_num(
            stats_map[
                "VV_VH_ratio_std_excluding_maxVV.tif"
            ]
        ),
        profile,
        "VV/VH Ratio Std Excluding Maximum VV",
        "VV_VH_excluding_max_std.png",
        output_images_dir,
        glacier_shp,
        cmap='Reds',
        color_label="Amplitude Ratio"
    )



# ============================================================
# Z-SCORE BASED STATISTICAL ANOMALY DETECTION
# ============================================================

def detect_statistical_zscore_anomalies_per_date(
    shp_path,
    output_images_dir,
    time_series_data,
    confidence,
    output_path,
    tif="pixel_stats"
):
    """
    Detect statistically significant pixels using
    spatial Z-score analysis.

    The method:
        1. Loads VV/VH ratio at maximum VV response.
        2. Calculates glacier-wide mean and standard deviation.
        3. Computes pixel-wise Z-score.
        4. Selects pixels exceeding confidence threshold.
        5. Saves anomaly rasters by acquisition date.

    Parameters
    ----------
    confidence :
        Statistical confidence level used to derive
        critical Z threshold.
    """


    # Input raster containing:
    # Band 1 -> VV/VH ratio at maximum VV
    # Band 2 -> acquisition index
    tif_path = os.path.join(
        tif,
        "VV_VH_ratio_at_VVmax_with_date.tif"
    )


    if not time_series_data:
        print(
            " Error: The provided time_series_data dictionary is empty or None."
        )
        return



    # Convert confidence value:
    # Example:
    # 99 -> 0.99
    conf_decimal = (
        confidence / 100.0
        if confidence > 1
        else confidence
    )


    # Calculate significance level
    # for two-tailed statistical test
    alpha = 1 - conf_decimal


    # Convert confidence level into critical Z value
    z_thresh = abs(
        stats.norm.ppf(
            alpha / 2
        )
    )



    # Output folders
    folder_label = str(
        confidence
    ).replace(
        ".",
        "_"
    )


    output_dir = output_path


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    os.makedirs(
        output_images_dir,
        exist_ok=True
    )



    # --------------------------------------------------------
    # Load ratio raster and acquisition date index
    # --------------------------------------------------------
    with rasterio.open(
        tif_path
    ) as src:


        ratio_arr = src.read(
            1
        ).astype(float)


        date_idx = src.read(
            2
        ).astype(int)


        transform = src.transform


        profile = src.profile



        extent = [
            transform.c,
            transform.c + transform.a * src.width,
            transform.f + transform.e * src.height,
            transform.f
        ]



    # --------------------------------------------------------
    # Create glacier mask
    # --------------------------------------------------------
    gdf = gpd.read_file(
        shp_path
    )


    gdf_crop = gdf.clip(
        box(
            extent[0],
            extent[2],
            extent[1],
            extent[3]
        )
    )



    mask_grid = rasterio.features.rasterize(
        [
            (geom, 1)
            for geom in gdf_crop.geometry
        ],
        out_shape=ratio_arr.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )



    # Keep only glacier pixels
    ratio_glacier = np.where(
        mask_grid == 1,
        ratio_arr,
        np.nan
    )



    # Glacier-wide statistics
    mean_val = np.nanmean(
        ratio_glacier
    )

    std_val = np.nanstd(
        ratio_glacier
    )



    # Pixel-wise Z-score
    z = (
        ratio_arr - mean_val
    ) / std_val



    # Significant deviations
    sig_mask = (
        np.abs(z)
        >=
        z_thresh
    )



    # --------------------------------------------------------
    # Prepare date labels
    # --------------------------------------------------------
    dates = np.unique(
        date_idx[
            date_idx >= 0
        ]
    )


    sorted_dates = sorted(
        time_series_data.keys()
    )


    date_labels = [
        sorted_dates[i].strftime("%d/%m/%Y")
        for i in dates
    ]



    # Generate colors based on acquisition dates
    cmap_base = plt.get_cmap(
        "turbo",
        len(dates)
    )


    date_colors = np.zeros(
        (
            ratio_arr.shape[0],
            ratio_arr.shape[1],
            4
        )
    )


    valid = (
        date_idx >= 0
    )


    date_colors[valid] = cmap_base(
        date_idx[valid] /
        (len(dates)-1)
    )



    # Use ratio magnitude as transparency
    ratio_norm = (
        ratio_arr - np.nanmin(ratio_arr)
    ) / (
        np.nanmax(ratio_arr)
        -
        np.nanmin(ratio_arr)
    )


    date_colors[..., 3] = np.where(
        sig_mask,
        ratio_norm,
        0
    )



    # --------------------------------------------------------
    # Plot significant anomaly locations
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(12,10)
    )


    ax.set_facecolor(
        "white"
    )


    ax.imshow(
        date_colors,
        extent=extent
    )


    gdf_crop.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1
    )


    ax.set_xlabel(
        "Longitude"
    )


    ax.set_ylabel(
        "Latitude"
    )


    ax.set_title(
        f"Statistically Significant Ratio at Max VV "
        f"({conf_decimal*100:.1f}% Confidence Interval)"
    )



    # Date colorbar
    cmap_dates = ListedColormap(
        [
            cmap_base(i/(len(dates)-1))
            for i in range(len(dates))
        ]
    )


    bounds = np.arange(
        -0.5,
        len(dates)+0.5,
        1
    )


    norm = BoundaryNorm(
        bounds,
        cmap_dates.N
    )


    mappable = plt.cm.ScalarMappable(
        cmap=cmap_dates,
        norm=norm
    )


    mappable.set_array([])


    cbar = fig.colorbar(
        mappable,
        ax=ax,
        ticks=np.arange(len(dates))
    )


    cbar.ax.set_yticklabels(
        date_labels
    )


    cbar.set_label(
        "Date of Maximum VV"
    )



    out_fig_path = os.path.join(
        output_images_dir,
        f"VV_VH_ratio_significant_{folder_label}.png"
    )


    plt.savefig(
        out_fig_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )


    plt.close()



    print(
        f" Saved to: {out_fig_path}"
    )



    # --------------------------------------------------------
    # Save individual anomaly rasters
    # --------------------------------------------------------

    profile.update(
        dtype=rasterio.float32,
        count=1
    )


    for t in dates:


        date_mask = (
            (date_idx == t)
            &
            sig_mask
        )


        out_arr = np.where(
            date_mask,
            ratio_arr,
            np.nan
        ).astype(np.float32)



        num_saved_points = np.sum(
            date_mask
        )


        date_str = sorted_dates[t].strftime(
            "%Y-%m-%d"
        )


        print(
            f" Date: {date_str} → "
            f"{num_saved_points} active anomalous points."
        )



        out_path = os.path.join(
            output_dir,
            f"ratio_significant_{folder_label}_{date_str}.tif"
        )


        with rasterio.open(
            out_path,
            "w",
            **profile
        ) as dst:

            dst.write(
                out_arr,
                1
            )


    print(
        f" All anomalous GeoTIFF tiles saved to folder: '{output_dir}/'\n"
    )



# ============================================================
# SINGLE-DATE STATISTICAL THRESHOLD ANOMALY DETECTION
# ============================================================

def detect_anomalies_statistical_threshold(
    vv_stack,
    vh_stack,
    time_series_data,
    output_dir,
    filter_size,
    k
):
    """
    Detect anomalies independently for each acquisition date
    using spatial statistical thresholding.

    For each date:

        1. Calculate VV/VH ratio in dB.
        2. Apply spatial smoothing.
        3. Calculate scene mean and standard deviation.
        4. Mark pixels above:

                Mean + (k × Standard Deviation)

    This method identifies pixels that are unusually high
    compared with the surrounding scene statistics.
    """


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    ratio_stack = []



    # --------------------------------------------------------
    # Generate VV/VH ratio time-series
    # --------------------------------------------------------
    for i in range(
        vv_stack.shape[0]
    ):

        # Replace invalid SAR values
        vv = np.where(
            vv_stack[i] <= 0,
            1e-6,
            vv_stack[i]
        ).astype(float)


        vh = np.where(
            vh_stack[i] <= 0,
            1e-6,
            vh_stack[i]
        ).astype(float)



        # Convert SAR amplitudes to dB ratio
        #
        # Spatial filtering reduces speckle noise
        ratio_stack.append(
            uniform_filter(
                10 * np.log10(vv),
                size=filter_size
            )
            -
            uniform_filter(
                10 * np.log10(vh),
                size=filter_size
            )
        )



    ratio_stack = np.array(
        ratio_stack
    )



    # Use first raster as output reference
    template_file = sorted(
        time_series_data.items()
    )[0][1]["VV_shape_path"]


    with rasterio.open(
        template_file
    ) as src:

        meta = src.meta.copy()


        meta.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )



    # --------------------------------------------------------
    # Process each acquisition independently
    # --------------------------------------------------------
    for t in range(
        ratio_stack.shape[0]
    ):


        # Calculate spatial statistics
        scene_mean = np.mean(
            ratio_stack[t]
        )


        scene_std = np.std(
            ratio_stack[t]
        )



        # Pixels exceeding statistical threshold
        anomaly_mask = (
            ratio_stack[t]
            >
            (
                scene_mean
                +
                k * scene_std
            )
        )



        # Count detected anomaly pixels
        num_saved_points = np.sum(
            anomaly_mask
        )


        date_str = sorted(
            time_series_data.keys()
        )[t].strftime(
            "%Y-%m-%d"
        )



        print(
            f"  Date: {date_str} → "
            f"Isolated {num_saved_points} "
            "active anomalous points."
        )



        # Save anomaly raster
        with rasterio.open(
            os.path.join(
                output_dir,
                f"VV_VH_anomaly_{date_str}.tif"
            ),
            "w",
            **meta
        ) as dst:

            dst.write(
                anomaly_mask.astype(np.uint8),
                1
            )





# ============================================================
# TEMPORAL ROLLING STATISTICAL THRESHOLD DETECTION
# ============================================================

def detect_anomalies_rolling_threshold(
    vv_stack,
    vh_stack,
    time_series_data,
    output_dir,
    filter_size,
    std_threshold
):
    """
    Detect anomalies using temporal statistics.

    Unlike single-date thresholding, this method compares
    each acquisition against the complete temporal behaviour
    of each pixel.

    Threshold:

        temporal mean + (threshold × temporal standard deviation)

    """



    os.makedirs(
        output_dir,
        exist_ok=True
    )


    ratio_stack = []



    # --------------------------------------------------------
    # Calculate VV/VH ratio time-series
    # --------------------------------------------------------
    for i in range(
        vv_stack.shape[0]
    ):


        vv = np.where(
            vv_stack[i] <= 0,
            1e-6,
            vv_stack[i]
        ).astype(float)


        vh = np.where(
            vh_stack[i] <= 0,
            1e-6,
            vh_stack[i]
        ).astype(float)



        # Convert to logarithmic VV/VH ratio
        ratio_stack.append(
            uniform_filter(
                10 * np.log10(vv),
                size=filter_size
            )
            -
            uniform_filter(
                10 * np.log10(vh),
                size=filter_size
            )
        )



    ratio_stack = np.array(
        ratio_stack
    )



    # Calculate temporal baseline statistics
    overall_mean = np.mean(
        ratio_stack,
        axis=0
    )


    overall_std = np.std(
        ratio_stack,
        axis=0
    )



    # Output raster reference
    template_file = sorted(
        time_series_data.items()
    )[0][1]["VV_shape_path"]



    with rasterio.open(
        template_file
    ) as src:

        meta = src.meta.copy()


        meta.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )



    # --------------------------------------------------------
    # Compare every date against temporal baseline
    # --------------------------------------------------------
    for t in range(
        ratio_stack.shape[0]
    ):


        anomaly_mask = (
            ratio_stack[t]
            >
            (
                overall_mean
                +
                std_threshold * overall_std
            )
        )



        num_saved_points = np.sum(
            anomaly_mask
        )


        date_str = sorted(
            time_series_data.keys()
        )[t].strftime(
            "%Y-%m-%d"
        )



        print(
            f"  Date: {date_str} → "
            f"Isolated {num_saved_points} "
            "active anomalous points."
        )



        # Save binary anomaly mask
        with rasterio.open(
            os.path.join(
                output_dir,
                f"VV_VH_anomaly_{date_str}.tif"
            ),
            "w",
            **meta
        ) as dst:


            dst.write(
                anomaly_mask.astype(np.uint8),
                1
            )



# ============================================================
# WILKS' LAMBDA STATISTICAL CHANGE DETECTION
# ============================================================

def detect_change_wilks_lambda(
    vv_stack,
    vh_stack,
    time_series_data,
    output_dir,
    L,
    lower_alpha=1e-6,
    upper_alpha=1-1e-6,
    dominance_ratio=0.65,
    no_change_band=0.03,
    min_pixels=6,
    max_points_per_image=40
):
    """
    Detect temporal changes using Wilks' Lambda based
    statistical hypothesis testing.

    The method compares covariance-related measurements
    between consecutive SAR acquisitions.

    Processing steps:

        1. Calculate determinant-like quantity from VV and VH.
        2. Compute Lambda values for previous and current dates.
        3. Estimate statistical significance using beta distribution.
        4. Remove insignificant and unstable regions.
        5. Extract spatial change centroids.
        6. Save detected change locations as raster masks.


    Parameters
    ----------
    L :
        Number of looks parameter used to define beta distribution.

    dominance_ratio :
        Controls how strongly one acquisition must dominate
        before being considered a directional change.

    min_pixels :
        Minimum connected component size.

    max_points_per_image :
        Maximum number of saved change locations per image.
    """



    os.makedirs(
        output_dir,
        exist_ok=True
    )



    # Beta distribution parameters derived from number of looks
    alpha_val, beta_val = (
        0.750 * L,
        2.250 * L
    )



    # Initial SAR observation
    det_X = (
        vv_stack[0]
        *
        vh_stack[0]
    )



    # Reference raster metadata
    template_file = sorted(
        time_series_data.items()
    )[0][1]["VV_shape_path"]



    with rasterio.open(
        template_file
    ) as src:


        meta = src.meta.copy()


        meta.update(
            dtype=rasterio.uint8,
            count=1
        )



    # --------------------------------------------------------
    # Compare every acquisition with previous state
    # --------------------------------------------------------
    for i in range(
        1,
        vv_stack.shape[0]
    ):


        # Current acquisition measurement
        det_Y = (
            vv_stack[i]
            *
            vh_stack[i]
        )


        det_sum = (
            det_X
            +
            det_Y
        )



        # Wilks' Lambda components
        lambda_X = np.divide(
            det_X,
            det_sum,
            out=np.zeros_like(det_X),
            where=det_sum != 0
        )


        lambda_Y = np.divide(
            det_Y,
            det_sum,
            out=np.zeros_like(det_Y),
            where=det_sum != 0
        )



        # Ignore pixels close to theoretical no-change value
        stable = (
            np.abs(lambda_X - 0.25)
            <
            no_change_band
        )



        # Statistical probability estimation
        p_X = stats.beta.cdf(
            lambda_X,
            alpha_val,
            beta_val
        )


        p_Y = stats.beta.cdf(
            lambda_Y,
            alpha_val,
            beta_val
        )



        # Significant changes
        sig_X = (
            (
                (p_X < lower_alpha)
                |
                (p_X > upper_alpha)
            )
            &
            (~stable)
        )


        sig_Y = (
            (
                (p_Y < lower_alpha)
                |
                (p_Y > upper_alpha)
            )
            &
            (~stable)
        )



        # ----------------------------------------------------
        # Connected component filtering
        # ----------------------------------------------------
        def _filter(m):

            lbl, num = label(m)


            out = np.zeros_like(
                m,
                dtype=bool
            )


            for r in range(
                1,
                num + 1
            ):

                if np.sum(lbl == r) >= min_pixels:

                    out |= (
                        lbl == r
                    )



            # Convert regions to centroids
            lbl, num = label(out)


            pt_out = np.zeros_like(
                m,
                dtype=bool
            )


            for r in range(
                1,
                num + 1
            ):

                if np.any(
                    lbl == r
                ):

                    y, x = center_of_mass(
                        lbl == r
                    )


                    pt_out[
                        int(round(y)),
                        int(round(x))
                    ] = True


            return pt_out



        # Categorize detected changes
        rem = _filter(
            sig_X
            &
            (~sig_Y)
            &
            (lambda_X > dominance_ratio)
        )


        add = _filter(
            sig_Y
            &
            (~sig_X)
            &
            (lambda_Y > dominance_ratio)
        )


        indef = _filter(
            sig_X
            &
            sig_Y
        )



        # ----------------------------------------------------
        # Limit number of output points
        # ----------------------------------------------------
        def _limit(
            m,
            max_p
        ):

            ys, xs = np.where(m)


            if len(ys) > max_p:

                idx = np.random.choice(
                    len(ys),
                    size=max_p,
                    replace=False
                )


                nm = np.zeros_like(
                    m,
                    dtype=bool
                )


                nm[
                    ys[idx],
                    xs[idx]
                ] = True


                return nm


            return m



        rem = _limit(
            rem,
            max_points_per_image // 3
        )


        add = _limit(
            add,
            max_points_per_image // 3
        )


        indef = _limit(
            indef,
            max_points_per_image // 3
        )



        # Final binary change map
        points_mask = np.zeros_like(
            lambda_X,
            dtype=np.uint8
        )


        points_mask[
            rem
            |
            add
            |
            indef
        ] = 1



        num_saved_points = np.sum(
            points_mask == 1
        )


        date_str = sorted(
            time_series_data.keys()
        )[i].strftime(
            "%Y-%m-%d"
        )


        print(
            f"  Date: {date_str} → "
            f"Extracted {num_saved_points} "
            "active statistical target centroids."
        )



        # Save output raster
        with rasterio.open(
            os.path.join(
                output_dir,
                f"wilks_points_{date_str}.tif"
            ),
            "w",
            **meta
        ) as dst:


            dst.write(
                points_mask,
                1
            )





# ============================================================
# ISOLATION FOREST ANOMALY DETECTION
# ============================================================

def detect_anomalies_isolation_forest(
    vv_stack,
    vh_stack,
    time_series_data,
    output_dir,
    filter_size,
    contamination
):
    """
    Detect anomalous SAR pixels using Isolation Forest.

    Isolation Forest is an unsupervised machine learning
    algorithm that isolates unusual observations by creating
    random decision trees.

    Processing:

        1. Calculate VV/VH ratio in dB.
        2. Apply spatial smoothing.
        3. Flatten each acquisition into feature vectors.
        4. Train Isolation Forest independently for each date.
        5. Convert detected outliers back into raster format.


    Parameters
    ----------
    contamination :
        Expected proportion of anomalies in the scene.
    """



    from sklearn.ensemble import IsolationForest



    os.makedirs(
        output_dir,
        exist_ok=True
    )



    ratio_stack = []



    # --------------------------------------------------------
    # Generate SAR ratio stack
    # --------------------------------------------------------
    for i in range(
        vv_stack.shape[0]
    ):


        vv = np.where(
            vv_stack[i] <= 0,
            1e-6,
            vv_stack[i]
        ).astype(
            np.float32
        )


        vh = np.where(
            vh_stack[i] <= 0,
            1e-6,
            vh_stack[i]
        ).astype(
            np.float32
        )



        ratio_stack.append(
            uniform_filter(
                10.0 * np.log10(vv),
                size=filter_size
            )
            -
            uniform_filter(
                10.0 * np.log10(vh),
                size=filter_size
            )
        )



    ratio_stack = np.array(
        ratio_stack
    )



    n_dates, rows, cols = ratio_stack.shape



    # Raster metadata reference
    template_file = sorted(
        time_series_data.items()
    )[0][1]["VV_shape_path"]



    with rasterio.open(
        template_file
    ) as src:


        meta = src.meta.copy()


        meta.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )



    # --------------------------------------------------------
    # Run Isolation Forest per acquisition
    # --------------------------------------------------------
    for t in range(
        n_dates
    ):


        # Convert image into pixel feature vector
        X = ratio_stack[t].reshape(
            -1,
            1
        )



        # Remove invalid values
        valid_mask = np.isfinite(
            X[:, 0]
        )


        X_valid = X[
            valid_mask
        ]



        # Train unsupervised anomaly detector
        iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )


        labels = iso.fit_predict(
            X_valid
        )



        # Isolation Forest labels:
        #
        #   -1 → anomaly
        #    1 → normal
        final_anomalies_valid = (
            (labels == -1)
            &
            (X_valid[:, 0] > 0)
        )



        # Restore original raster shape
        anomaly_mask = np.zeros(
            X.shape[0],
            dtype=np.uint8
        )


        anomaly_mask[
            valid_mask
        ] = (
            final_anomalies_valid.astype(
                np.uint8
            )
        )


        anomaly_grid = anomaly_mask.reshape(
            rows,
            cols
        )



        num_saved_points = np.sum(
            anomaly_grid == 1
        )


        date_str = sorted(
            time_series_data.keys()
        )[t].strftime(
            "%Y-%m-%d"
        )


        print(
            f"  Date: {date_str} → "
            f"Isolated {num_saved_points} "
            "active anomalous points."
        )



        # Save anomaly raster
        with rasterio.open(
            os.path.join(
                output_dir,
                f"VV_VH_anomaly_{date_str}.tif"
            ),
            "w",
            **meta
        ) as dst:


            dst.write(
                anomaly_grid,
                1
            )


# ============================================================
# IMAGE SIMILARITY METRICS
# ============================================================
#
# These functions compare image patches extracted from:
#
#   - Before-event optical image
#   - After-event optical image
#
# Each metric evaluates structural or statistical similarity
# between anomaly locations and randomly sampled glacier areas.
#
# The functions operate only on pixels selected by final_mask
# to ensure that only valid glacier pixels are considered.
#
# ============================================================



def _calculate_gms(w1, w2):
    """
    Calculate Gradient Magnitude Similarity (GMS).

    GMS compares edge structures between two image patches.

    Steps:
        1. Compute Scharr gradients.
        2. Calculate gradient magnitude.
        3. Compare gradient strength using similarity equation.

    Higher values indicate more similar structures.
    """

    # Compute horizontal and vertical gradients
    dx = scharr(
        w1,
        axis=0
    )

    dy = scharr(
        w1,
        axis=1
    )


    # Gradient magnitude of first image
    g1 = np.sqrt(
        dx**2 +
        dy**2
    )


    # Compute gradients for second image
    dx = scharr(
        w2,
        axis=0
    )

    dy = scharr(
        w2,
        axis=1
    )


    # Gradient magnitude of second image
    g2 = np.sqrt(
        dx**2 +
        dy**2
    )


    # Stabilizing constant
    c = 0.002


    # Gradient magnitude similarity equation
    return (
        (2 * g1 * g2 + c)
        /
        (g1**2 + g2**2 + c)
    )





def _get_masked_fsim_metrics(
    img1,
    img2,
    final_mask,
    r,
    c,
    rad
):
    """
    Calculate FSIM-like similarity using
    Gradient Magnitude Similarity.

    Only pixels inside final_mask are used.
    """


    # Extract local window around point
    r0, r1 = (
        r - rad,
        r + rad + 1
    )

    c0, c1 = (
        c - rad,
        c + rad + 1
    )


    w1 = img1[
        r0:r1,
        c0:c1
    ]


    w2 = img2[
        r0:r1,
        c0:c1
    ]



    # Compute gradient similarity map
    gms_map = _calculate_gms(
        w1,
        w2
    )


    # Return mean similarity only
    # from valid glacier pixels
    return gms_map[
        final_mask
    ].mean()





def _get_masked_ssim_metrics(
    img1,
    img2,
    final_mask,
    r,
    c,
    rad
):
    """
    Calculate Structural Similarity Index (SSIM).

    SSIM evaluates:
        - luminance
        - contrast
        - structural information

    Higher values indicate more similar patches.
    """


    r0, r1 = (
        r - rad,
        r + rad + 1
    )

    c0, c1 = (
        c - rad,
        c + rad + 1
    )


    w1 = img1[
        r0:r1,
        c0:c1
    ]

    w2 = img2[
        r0:r1,
        c0:c1
    ]



    # Select only valid pixels
    w1_masked = w1[
        final_mask
    ]

    w2_masked = w2[
        final_mask
    ]



    # Required dynamic range for SSIM
    data_range = (
        w1_masked.max()
        -
        w1_masked.min()
    )


    # Avoid undefined SSIM
    if data_range == 0:
        return None



    return ssim(
        w1_masked,
        w2_masked,
        data_range=data_range
    )





def _get_masked_zncc(
    img1,
    img2,
    final_mask,
    r,
    c,
    rad
):
    """
    Calculate Zero Normalized Cross Correlation (ZNCC).

    Measures correlation between two patches after
    removing brightness differences.

    Range:
        +1  -> identical structure
         0  -> no correlation
        -1  -> opposite structure
    """


    r0, r1 = (
        r - rad,
        r + rad + 1
    )

    c0, c1 = (
        c - rad,
        c + rad + 1
    )



    # Extract valid pixels only
    w1 = img1[
        r0:r1,
        c0:c1
    ][final_mask].flatten()


    w2 = img2[
        r0:r1,
        c0:c1
    ][final_mask].flatten()



    # Require enough valid samples
    if (
        len(w1) < 20
        or np.std(w1) == 0
        or np.std(w2) == 0
    ):
        return None



    return np.corrcoef(
        w1,
        w2
    )[0, 1]





def _get_masked_lesh(
    img1,
    img2,
    final_mask,
    r,
    c,
    rad,
    n_bins=8
):
    """
    Calculate Local Edge Similarity Histogram (LESH).

    LESH compares distributions of local edge
    orientations between image patches.

    Steps:
        1. Compute Sobel gradients.
        2. Extract edge magnitude and orientation.
        3. Create normalized orientation histogram.
        4. Compare histograms using cosine similarity.
    """



    r0, r1 = (
        r - rad,
        r + rad + 1
    )

    c0, c1 = (
        c - rad,
        c + rad + 1
    )


    w1 = img1[
        r0:r1,
        c0:c1
    ]

    w2 = img2[
        r0:r1,
        c0:c1
    ]



    def calc_lesh(
        patch
    ):

        # Calculate image gradients
        gx = sobel(
            patch,
            axis=0
        )

        gy = sobel(
            patch,
            axis=1
        )


        magnitude = np.sqrt(
            gx**2 +
            gy**2
        )[final_mask]


        orientation = np.arctan2(
            gy,
            gx
        )[final_mask]



        if len(magnitude) < 20:
            return None



        # Convert orientation to [0, pi]
        orientation = np.mod(
            orientation,
            np.pi
        )



        # Weighted orientation histogram
        hist, _ = np.histogram(
            orientation,
            bins=n_bins,
            range=(0, np.pi),
            weights=magnitude
        )


        if np.sum(hist) == 0:
            return None



        return hist / np.sum(hist)



    d1 = calc_lesh(
        w1
    )

    d2 = calc_lesh(
        w2
    )



    if d1 is None or d2 is None:
        return None



    # Histogram similarity
    return np.dot(
        d1,
        d2
    ) / (
        norm(d1)
        *
        norm(d2)
    )





def _get_masked_mi(
    img1,
    img2,
    final_mask,
    r,
    c,
    rad,
    n_bins=32
):
    """
    Calculate Mutual Information (MI).

    MI measures statistical dependency between
    two image patches.

    Higher values indicate stronger relationship.
    """


    r0, r1 = (
        r - rad,
        r + rad + 1
    )

    c0, c1 = (
        c - rad,
        c + rad + 1
    )


    w1 = img1[
        r0:r1,
        c0:c1
    ][final_mask]


    w2 = img2[
        r0:r1,
        c0:c1
    ][final_mask]



    if len(w1) < 20:
        return None



    # Convert continuous values into histogram bins
    vals1 = np.digitize(
        w1,
        np.histogram(
            w1,
            bins=n_bins
        )[1]
    )


    vals2 = np.digitize(
        w2,
        np.histogram(
            w2,
            bins=n_bins
        )[1]
    )



    return mutual_info_score(
        vals1,
        vals2
    )





def _get_masked_hog(
    img1,
    img2,
    final_mask,
    r,
    c,
    rad,
    n_bins=9
):
    """
    Calculate Histogram of Oriented Gradients (HOG)
    difference between two patches.

    Lower values indicate more similar edge structures.
    """



    r0, r1 = (
        r - rad,
        r + rad + 1
    )

    c0, c1 = (
        c - rad,
        c + rad + 1
    )



    # Keep invalid pixels masked
    w1_masked = np.where(
        final_mask,
        img1[r0:r1, c0:c1],
        np.nan
    )


    w2_masked = np.where(
        final_mask,
        img2[r0:r1, c0:c1],
        np.nan
    )



    def calc_hog(
        patch_masked
    ):

        dx = scharr(
            patch_masked,
            axis=0
        )


        dy = scharr(
            patch_masked,
            axis=1
        )



        magnitude = np.sqrt(
            dx**2 +
            dy**2
        ).flatten()


        orientation = (
            np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )
            %
            180
        ).flatten()



        valid = ~np.isnan(
            magnitude
        )


        magnitude = magnitude[
            valid
        ]


        orientation = orientation[
            valid
        ]



        if len(magnitude) < 20:
            return None



        hist, _ = np.histogram(
            orientation,
            bins=n_bins,
            range=(0,180),
            weights=magnitude
        )


        return (
            hist.astype(np.float32)
            /
            (
                np.linalg.norm(hist)
                +
                1e-10
            )
        )



    h1 = calc_hog(
        w1_masked
    )

    h2 = calc_hog(
        w2_masked
    )



    if h1 is None or h2 is None:
        return None



    return np.linalg.norm(
        h1 - h2
    )





def _get_masked_sad(
    img1,
    img2,
    final_mask,
    r,
    c,
    rad
):
    """
    Calculate Sum of Absolute Differences (SAD).

    SAD is a direct pixel intensity difference metric.

    Higher values indicate stronger change.
    """



    r0, r1 = (
        r - rad,
        r + rad + 1
    )

    c0, c1 = (
        c - rad,
        c + rad + 1
    )



    w1 = img1[
        r0:r1,
        c0:c1
    ][final_mask]


    w2 = img2[
        r0:r1,
        c0:c1
    ][final_mask]



    if len(w1) < 20:
        return None



    return np.sum(
        np.abs(
            w1 - w2
        )
    )


# ============================================================
# GEOSPATIAL DATA PROCESSING HELPERS
# ============================================================
#
# These functions handle:
#
#   1. Raster reprojection between SAR and optical grids.
#   2. Mapping SAR anomaly locations to PlanetScope pixels.
#   3. Random glacier point sampling for statistical comparison.
#
# ============================================================



def _reproject_to_target(
    src_path,
    target_shape,
    target_transform,
    target_crs
):
    """
    Reproject an input raster to a target raster grid.

    Used to align Sentinel-1 derived masks with
    PlanetScope image geometry.

    Parameters
    ----------
    src_path :
        Source raster file.

    target_shape :
        Output raster dimensions.

    target_transform :
        Output affine transformation.

    target_crs :
        Output coordinate reference system.


    Returns
    -------
    dst_array :
        Reprojected raster array.
    """



    # Open source raster
    with rasterio.open(
        src_path
    ) as src:


        src_array = src.read(
            1
        )



        # Create empty output raster
        dst_array = np.zeros(
            target_shape,
            dtype=src_array.dtype
        )



        # Perform reprojection
        reproject(
            source=src_array,
            destination=dst_array,

            src_transform=src.transform,
            src_crs=src.crs,

            dst_transform=target_transform,
            dst_crs=target_crs,

            # Nearest neighbour preserves
            # binary anomaly masks
            resampling=Resampling.nearest
        )


    return dst_array





def _map_sar_points(
    sar_file,
    valid_mask,
    transform_ps,
    row_limit,
    W,
    radius
):
    """
    Convert SAR anomaly pixels into corresponding
    PlanetScope image coordinates.

    Steps:

        1. Read SAR anomaly raster.
        2. Extract anomaly pixel coordinates.
        3. Convert SAR pixel coordinates to map coordinates.
        4. Transform coordinates into PlanetScope pixels.
        5. Keep only valid glacier locations.


    Returns
    -------
    list of tuples:
        PlanetScope row-column coordinates.
    """



    sig_coords = []



    # Read SAR anomaly raster
    with rasterio.open(
        sar_file
    ) as ds_sar:


        mask = (
            ds_sar.read(1)
            >
            0
        )


        # Extract anomaly pixels
        r_sar, c_sar = np.where(
            mask
        )



        # Convert SAR pixels to geographic coordinates
        xs, ys = rasterio.transform.xy(
            ds_sar.transform,
            r_sar,
            c_sar,
            offset="center"
        )



        # Convert geographic coordinates
        # into PlanetScope raster coordinates
        cols_p, rows_p = (
            ~transform_ps
            *
            (
                xs,
                ys
            )
        )



        rows_p = np.round(
            rows_p
        ).astype(int)


        cols_p = np.round(
            cols_p
        ).astype(int)



        # Validate mapped locations
        for r, c in zip(
            rows_p,
            cols_p
        ):


            # Ensure enough surrounding pixels
            # exist for patch extraction
            if (
                radius <= r < row_limit-radius
                and
                radius <= c < W-radius
            ):


                # Keep only valid glacier pixels
                if valid_mask[r, c]:

                    sig_coords.append(
                        (
                            r,
                            c
                        )
                    )



    # Remove duplicated coordinates
    return list(
        set(sig_coords)
    )





def _sample_random_points(
    valid_mask,
    sig_coords_arr,
    num_pts,
    radius,
    row_limit,
    W,
    min_dist
):
    """
    Randomly sample glacier locations.

    Random points are used as a control group
    against SAR detected anomaly locations.

    Constraints:

        - Must lie inside glacier.
        - Must have valid SAR coverage.
        - Must remain separated from detected anomalies.


    Returns
    -------
    list :
        Random row-column coordinates.
    """



    rand_coords = []

    attempts = 0



    # Rejection sampling loop
    while (
        len(rand_coords) < num_pts
        and
        attempts < 20000
    ):


        # Generate random pixel
        r = random.randint(
            radius,
            row_limit-radius-1
        )


        c = random.randint(
            radius,
            W-radius-1
        )



        # Check glacier validity
        if valid_mask[r, c]:


            # Ensure random samples are not
            # too close to SAR anomalies
            if np.all(
                np.linalg.norm(
                    sig_coords_arr
                    -
                    np.array([r, c]),
                    axis=1
                )
                >= min_dist
            ):


                rand_coords.append(
                    (
                        r,
                        c
                    )
                )



        attempts += 1



    return rand_coords


# ============================================================
# UNIVERSAL IMAGE SIMILARITY VALIDATION PIPELINE
# ============================================================
#
# This function evaluates whether SAR detected anomaly
# locations correspond to actual surface changes visible
# in optical imagery.
#
# Workflow:
#
#   Sentinel-1 anomaly points
#          |
#          v
#   Coordinate transformation
#          |
#          v
#   PlanetScope patch extraction
#          |
#          v
#   Compare anomaly locations against random glacier points
#          |
#          v
#   Statistical validation
#
# ============================================================


def run_similarity_pipeline(
    window_size,
    output_dir,
    band_idx,
    date_pairs,
    shapefile_path,
    vv_mask_path,
    vh_mask_path,
    n_iterations=100,
    metrics_to_run=['fsim']
):
    """
    Executes structural change analysis over SAR detected
    anomaly locations using optical imagery comparison.

    For each SAR anomaly date pair:

        1. Extract anomaly locations.
        2. Extract corresponding optical patches.
        3. Generate random glacier control points.
        4. Compute similarity/change metrics.
        5. Repeat using bootstrap iterations.
        6. Evaluate statistical significance.


    Parameters
    ----------
    window_size :
        Optical patch diameter in pixels.

    band_idx :
        PlanetScope band used for comparison.

    date_pairs :
        List containing before image,
        after image, and SAR anomaly raster.

    n_iterations :
        Number of random sampling repetitions.

    metrics_to_run :
        Similarity metrics to evaluate.
    """



    os.makedirs(
        output_dir,
        exist_ok=True
    )



    # Radius of circular patch around each point
    radius = window_size // 2


    # Minimum distance between random samples
    # and detected anomaly locations
    min_dist = 2 * radius



    # Latitude cutoff used to remove invalid
    # lower image regions
    lat_cutoff = None



    # --------------------------------------------------------
    # Available metric functions
    # --------------------------------------------------------
    #
    # Similarity metrics:
    #
    #   FSIM  -> structural similarity
    #   SSIM  -> image structural similarity
    #   ZNCC  -> correlation similarity
    #   LESH  -> edge orientation similarity
    #   MI    -> information similarity
    #   HOG   -> gradient structure difference
    #   SAD   -> absolute pixel difference
    #

    metric_funcs = {

        'fsim':
            _get_masked_fsim_metrics,

        'ssim':
            _get_masked_ssim_metrics,

        'zncc':
            _get_masked_zncc,

        'lesh':
            _get_masked_lesh,

        'mi':
            _get_masked_mi,

        'hog':
            _get_masked_hog,

        'sad':
            _get_masked_sad
    }



    # --------------------------------------------------------
    # Initialize PlanetScope reference grid
    # --------------------------------------------------------

    with rasterio.open(
        date_pairs[0][1]
    ) as ds:


        transform_ps = ds.transform

        crs_ps = ds.crs

        H, W = (
            ds.height,
            ds.width
        )



    # Convert latitude cutoff into raster row
    row_limit = int(
        (
            transform_ps.f
            -
            lat_cutoff
        )
        /
        abs(transform_ps.e)
    )


    row_limit = np.clip(
        row_limit,
        0,
        H
    )



    # --------------------------------------------------------
    # Create glacier mask
    # --------------------------------------------------------

    gdf = gpd.read_file(
        shapefile_path
    ).to_crs(
        crs_ps
    )



    glacier_mask = features.rasterize(
        shapes=gdf.geometry,

        out_shape=(
            H,
            W
        ),

        transform=transform_ps,

        fill=0,

        all_touched=True,

        default_value=1
    ).astype(
        bool
    )



    # --------------------------------------------------------
    # Reproject SAR validity masks
    # into optical image grid
    # --------------------------------------------------------

    vv_data = _reproject_to_target(
        vv_mask_path,
        (
            H,
            W
        ),
        transform_ps,
        crs_ps
    )


    vh_data = _reproject_to_target(
        vh_mask_path,
        (
            H,
            W
        ),
        transform_ps,
        crs_ps
    )



    # Valid analysis area:
    #
    # glacier pixels
    # +
    # valid SAR coverage
    #

    valid_mask = (
        glacier_mask
        &
        (
            (vv_data > 0)
            |
            (vh_data > 0)
        )
    )



    # --------------------------------------------------------
    # Circular patch mask
    # --------------------------------------------------------

    y, x = np.ogrid[
        -radius:radius+1,
        -radius:radius+1
    ]


    circ_mask = (
        x**2
        +
        y**2
        <=
        radius**2
    )



    # ========================================================
    # Run selected similarity metrics
    # ========================================================

    for metric in metrics_to_run:


        metric = metric.lower()



        if metric not in metric_funcs:

            print(
                f"Skipping unregistered metric parameter: {metric}"
            )

            continue



        calc_func = metric_funcs[
            metric
        ]



        dates_labels = []

        diff_matrix = []

        sig_counts = []



        print(
            "\n"
            +
            "="*60
        )


        print(
            f"RUNNING PIPELINE METRIC: {metric.upper()}"
        )


        print(
            "="*60
        )



        # ----------------------------------------------------
        # Process every before-after image pair
        # ----------------------------------------------------

        for (
            label,
            before_file,
            after_file,
            sar_file
        ) in date_pairs:


            # Load selected optical band
            before_img = rasterio.open(
                before_file
            ).read(
                band_idx
            ).astype(
                np.float32
            )


            after_img = rasterio.open(
                after_file
            ).read(
                band_idx
            ).astype(
                np.float32
            )



            # Map SAR anomalies into
            # PlanetScope coordinates
            sig_coords = _map_sar_points(
                sar_file,
                valid_mask,
                transform_ps,
                row_limit,
                W,
                radius
            )


            sig_coords_arr = np.array(
                sig_coords
            )


            n_sig = len(
                sig_coords
            )



            if n_sig == 0:

                print(
                    f"{label} — No verified sample points falling in current window boundary mask."
                )

                continue



            sig_counts.append(
                n_sig
            )


            dates_labels.append(
                label
            )



            iter_diffs = []

            sig_avgs = []

            rand_avgs = []



            # ------------------------------------------------
            # Bootstrap random sampling iterations
            # ------------------------------------------------

            for _ in range(
                n_iterations
            ):


                rand_coords = _sample_random_points(
                    valid_mask,
                    sig_coords_arr,
                    n_sig,
                    radius,
                    row_limit,
                    W,
                    min_dist
                )



                s_vals = []

                r_vals = []


                # ------------------------------------------------
                # Calculate metric values for SAR anomaly points
                # ------------------------------------------------
                for r, c in sig_coords:

                    # Create valid circular patch mask
                    # restricted to glacier pixels
                    f_mask = (
                        circ_mask
                        &
                        valid_mask[
                            r-radius:r+radius+1,
                            c-radius:c+radius+1
                        ]
                    )


                    # Ignore patches with insufficient
                    # valid glacier pixels
                    if np.sum(f_mask) >= 20:


                        v = calc_func(
                            before_img,
                            after_img,
                            f_mask,
                            r,
                            c,
                            radius
                        )


                        if v is not None:

                            s_vals.append(
                                v
                            )



                # ------------------------------------------------
                # Calculate metric values for random control points
                # ------------------------------------------------
                for r, c in rand_coords:


                    f_mask = (
                        circ_mask
                        &
                        valid_mask[
                            r-radius:r+radius+1,
                            c-radius:c+radius+1
                        ]
                    )


                    if np.sum(f_mask) >= 20:


                        v = calc_func(
                            before_img,
                            after_img,
                            f_mask,
                            r,
                            c,
                            radius
                        )


                        if v is not None:

                            r_vals.append(
                                v
                            )



                # Mean metric value for anomaly locations
                s_mean = (
                    np.mean(s_vals)
                    if s_vals
                    else np.nan
                )


                # Mean metric value for random locations
                r_mean = (
                    np.mean(r_vals)
                    if r_vals
                    else np.nan
                )



                # Store valid bootstrap results
                if (
                    not np.isnan(s_mean)
                    and
                    not np.isnan(r_mean)
                ):


                    sig_avgs.append(
                        s_mean
                    )


                    rand_avgs.append(
                        r_mean
                    )


                    # Difference between control
                    # and detected anomaly regions
                    iter_diffs.append(
                        r_mean - s_mean
                    )



            # Save bootstrap differences
            # for distribution plotting
            diff_matrix.append(
                iter_diffs
            )



            # ====================================================
            # Statistical evaluation
            # ====================================================


            # Metric interpretation differs:
            #
            # Similarity metrics:
            #   FSIM, SSIM, ZNCC, LESH, MI
            #
            #   Higher value = more similar
            #
            #
            # Difference metrics:
            #   HOG, SAD
            #
            #   Lower value = more similar
            #
            alt_side = (
                'less'
                if metric in [
                    'hog',
                    'sad'
                ]
                else
                'greater'
            )



            # Mann-Whitney U test:
            #
            # Tests whether anomaly and
            # random distributions differ
            stat, p_val = mannwhitneyu(
                rand_avgs,
                sig_avgs,
                alternative=alt_side
            )



            # ----------------------------------------------------
            # Cohen's d effect size
            # ----------------------------------------------------

            diffs_arr = (
                np.array(rand_avgs)
                -
                np.array(sig_avgs)
            )


            mean_diff = np.mean(
                diffs_arr
            )


            std_diff = np.std(
                diffs_arr,
                ddof=1
            )


            cohens_d = (
                mean_diff / std_diff
                if std_diff != 0
                else 0.0
            )



            print(
                f"{label} — n={n_sig} pts"
            )


            print(
                f"   Mean Difference "
                f"(Random - Significant): "
                f"{mean_diff:.6f}"
            )


            print(
                f"   Mann-Whitney U "
                f"statistical value: "
                f"{stat:.2f} | "
                f"p-value: {p_val:.2e}"
            )


            print(
                f"   Cohen's d: "
                f"{cohens_d:.4f}"
            )



        # ========================================================
        # Distribution visualization
        # ========================================================


        plt.figure(
            figsize=(14, 6)
        )


        # Add number of detected
        # anomaly points to labels
        plot_labels = [
            f"{lbl}\n(n={cnt})"
            for lbl, cnt
            in zip(
                dates_labels,
                sig_counts
            )
        ]



        plt.boxplot(
            diff_matrix,
            labels=plot_labels,
            showfliers=False
        )



        # Zero indicates no difference
        # between anomaly and random regions
        plt.axhline(
            0,
            color='red',
            linestyle='--',
            label='Zero Reference Deviation'
        )



        plt.xticks(
            rotation=45
        )


        plt.ylabel(
            f"{metric.upper()} Difference "
            "(Random Baseline - Anomalies)"
        )


        plt.title(
            f"Distribution of Spatial "
            f"{metric.upper()} Variation "
            f"Across Acquisition Cycles "
            f"({n_iterations} Iterations)"
        )


        plt.grid(
            True,
            linestyle=':',
            alpha=0.6
        )



        plt.tight_layout()



        save_path = os.path.join(
            output_dir,
            f"{metric}_distribution.png"
        )


        plt.savefig(
            save_path,
            dpi=300
        )


        print(
            "\nSaved distribution visualization "
            f"chart template to: {save_path}"
        )


        plt.show()


def plot_dsm_difference(
    before_dsm,
    after_dsm,
    shapefile,
    output_dir,
    tif_name="dsm_difference.tif",
    png_name="dsm_difference.png",
    show=True
):
    """
    Create DSM elevation difference raster and visualization.

    The function:
    1. Loads before and after DSM products.
    2. Clips both DSMs using the glacier boundary shapefile.
    3. Reprojects the after DSM onto the before DSM grid.
    4. Computes elevation difference:
       
            DSM difference = After DSM - Before DSM

    5. Saves the difference raster as GeoTIFF.
    6. Generates a visualization.

    Returns
    -------
    dsm_diff_path : str
        Path to saved DSM difference GeoTIFF.
    """



    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------
    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # Convert output path to absolute path
    # This avoids notebook working-directory issues
    output_dir = os.path.abspath(
        output_dir
    )



    tif_path = os.path.join(
        output_dir,
        tif_name
    )


    png_path = os.path.join(
        output_dir,
        png_name
    )



    print(
        f"Saving DSM TIFF to: {tif_path}"
    )


    print(
        f"Saving DSM plot to: {png_path}"
    )



    # Load glacier boundary shapefile
    gdf = gpd.read_file(
        shapefile
    )



    # =========================================================
    # BEFORE DSM PROCESSING
    # =========================================================

    with rasterio.open(
        before_dsm
    ) as src_before:


        # Convert glacier boundary CRS
        # to match DSM projection
        gdf_before = gdf.to_crs(
            src_before.crs
        )



        # Crop DSM using glacier boundary
        before_clip, before_transform = mask(
            src_before,
            gdf_before.geometry,
            crop=True
        )



        # Extract first raster band
        before_clip = before_clip[0].astype(
            np.float32
        )



        # Convert nodata pixels to NaN
        # so they are ignored during calculations
        if src_before.nodata is not None:

            before_clip[
                before_clip == src_before.nodata
            ] = np.nan



        before_crs = src_before.crs



    # =========================================================
    # AFTER DSM PROCESSING
    # =========================================================

    with rasterio.open(
        after_dsm
    ) as src_after:


        # Transform glacier boundary
        # to after DSM coordinate system
        gdf_after = gdf.to_crs(
            src_after.crs
        )



        # Crop after DSM
        after_clip, after_transform = mask(
            src_after,
            gdf_after.geometry,
            crop=True
        )



        after_clip = after_clip[0].astype(
            np.float32
        )



        # Replace nodata values
        # with NaN
        if src_after.nodata is not None:

            after_clip[
                after_clip == src_after.nodata
            ] = np.nan



        after_crs = src_after.crs



    # =========================================================
    # ALIGN AFTER DSM WITH BEFORE DSM GRID
    # =========================================================
    #
    # DSM products may have:
    # - different resolution
    # - different coordinate grids
    # - different extents
    #
    # Therefore the after DSM is resampled
    # to exactly match the before DSM.
    #

    after_resampled = np.empty_like(
        before_clip
    )


    reproject(
        source=after_clip,
        destination=after_resampled,
        src_transform=after_transform,
        src_crs=after_crs,
        dst_transform=before_transform,
        dst_crs=before_crs,
        resampling=Resampling.bilinear
    )



    # =========================================================
    # COMPUTE ELEVATION CHANGE
    # =========================================================
    #
    # Positive values:
    #   Surface elevation increase
    #
    # Negative values:
    #   Surface elevation loss
    #

    dsm_diff = (
        after_resampled
        -
        before_clip
    )



    # Output raster metadata
    meta = {
        "driver": "GTiff",
        "height": dsm_diff.shape[0],
        "width": dsm_diff.shape[1],
        "count": 1,
        "dtype": rasterio.float32,
        "crs": before_crs,
        "transform": before_transform,
        "nodata": -9999.0
    }



    # Replace NaN values before saving
    dsm_diff_save = np.where(
        np.isnan(dsm_diff),
        -9999.0,
        dsm_diff
    ).astype(
        np.float32
    )



    # Save DSM difference GeoTIFF
    with rasterio.open(
        tif_path,
        "w",
        **meta
    ) as dst:

        dst.write(
            dsm_diff.astype("float32"),
            1
        )



    # =========================================================
    # VISUALIZATION
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )



    # Limit visualization range using
    # 95th percentile to reduce extreme outliers
    vmax = np.nanpercentile(
        np.abs(dsm_diff),
        95
    )



    im = ax.imshow(
        dsm_diff,
        cmap="viridis",
        vmin=-vmax,
        vmax=vmax
    )



    ax.set_title(
        "DSM Elevation Difference (After - Before)"
    )


    ax.axis(
        "off"
    )



    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )


    cbar.set_label(
        "Elevation Change (m)"
    )



    plt.tight_layout()



    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )



    if show:

        plt.show()

    else:

        plt.close()



    return tif_path


def plot_anomaly_points_on_dsm(
    dsm_tif,
    anomaly_tifs,
    output_dir,
    shapefile_path,
    patch_diameter_m=75,
    png_name="anomaly_points.png",
    downsample=10,
    point_radius=2,
    show=True
):
    """
    Plot SAR anomaly locations over DSM elevation difference raster.

    Parameters
    ----------
    dsm_tif : str
        DSM difference GeoTIFF generated from before/after DSM comparison.

    anomaly_tifs : list
        List containing anomaly dates and corresponding raster paths:
        
        [
            ("2017-08-20", "anomaly_file.tif"),
            ("2017-09-01", "anomaly_file.tif")
        ]

    output_dir : str
        Directory where output visualization is saved.

    shapefile_path : str
        Glacier boundary shapefile used for masking.

    patch_diameter_m : float
        Diameter of anomaly validation region in meters.

    downsample : int
        Factor used to reduce DSM resolution for visualization.

    Returns
    -------
    png_path : str
        Path of saved anomaly visualization.
    """



    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )



    # Convert output path to absolute path
    # to avoid notebook working-directory issues

    output_dir = os.path.abspath(
        output_dir
    )


    png_path = os.path.join(
        output_dir,
        png_name
    )



    print(
        f"Saving DSM plot to: {png_path}"
    )



    # =========================================================
    # LOAD DSM DIFFERENCE RASTER
    # =========================================================

    with rasterio.open(
        dsm_tif
    ) as ds:


        # Read DSM elevation change layer
        dsm = ds.read(1).astype(
            np.float32
        )


        transform_dsm = ds.transform
        crs_dsm = ds.crs

        H, W = ds.height, ds.width



    # =========================================================
    # CREATE GLACIER MASK
    # =========================================================

    # Load glacier boundary
    # and convert to DSM coordinate system

    gdf = gpd.read_file(
        shapefile_path
    ).to_crs(
        crs_dsm
    )



    # Rasterize glacier polygon
    # so only glacier pixels are considered

    glacier_mask = features.rasterize(
        shapes=gdf.geometry,
        out_shape=(H, W),
        transform=transform_dsm,
        fill=0,
        all_touched=True,
        default_value=1
    ).astype(bool)



    valid_mask = glacier_mask



    # =========================================================
    # CONVERT PATCH SIZE FROM METERS TO PIXELS
    # =========================================================

    # DSM spatial resolution
    res = (
        transform_dsm.a
        -
        transform_dsm.e
    ) / 2



    # Convert physical patch radius
    # into raster pixel units

    patch_radius_pixels = (
        patch_diameter_m / 2 / res
    )



    # =========================================================
    # MAP SAR ANOMALIES TO DSM GRID
    # =========================================================

    sig_coords = []



    for date, sar_file in anomaly_tifs:


        with rasterio.open(
            sar_file
        ) as ds_sar:


            # Extract binary anomaly mask

            mask = ds_sar.read(1) > 0



            # Obtain SAR pixel locations

            r_sar, c_sar = np.where(
                mask
            )



            # Convert SAR pixels to geographic coordinates

            xs, ys = rasterio.transform.xy(
                ds_sar.transform,
                r_sar,
                c_sar,
                offset="center"
            )



            # -------------------------------------------------
            # Coordinate transformation
            #
            # SAR and DSM products may use different CRS.
            # Transform anomaly coordinates before overlaying.
            # -------------------------------------------------

            if ds_sar.crs != crs_dsm:


                transformer = Transformer.from_crs(
                    ds_sar.crs,
                    crs_dsm,
                    always_xy=True
                )


                xs, ys = transformer.transform(
                    xs,
                    ys
                )



            # Convert geographic coordinates
            # into DSM raster indices

            rows, cols = rasterio.transform.rowcol(
                transform_dsm,
                xs,
                ys
            )



            # Keep only anomalies located
            # inside DSM extent and glacier area

            for r, c in zip(rows, cols):

                if (
                    0 <= r < H
                    and
                    0 <= c < W
                    and
                    valid_mask[r, c]
                ):

                    sig_coords.append(
                        (r, c)
                    )



    # Remove duplicate anomaly locations

    sig_coords = list(
        set(sig_coords)
    )



    # =========================================================
    # DOWNSAMPLE DSM FOR VISUALIZATION
    # =========================================================

    # Large DSM rasters are reduced
    # only for plotting efficiency.

    dsm_ds = dsm[
        ::downsample,
        ::downsample
    ]



    # Convert anomaly coordinates
    # to downsampled coordinate system

    sig_coords_ds = [
        (
            r // downsample,
            c // downsample
        )
        for r, c in sig_coords
    ]



    # Convert patch radius accordingly

    radius_ds = max(
        1,
        int(
            patch_radius_pixels /
            downsample
        )
    )



    # =========================================================
    # CREATE FINAL VISUALIZATION
    # =========================================================

    fig = plt.figure(
        figsize=(10, 10)
    )



    # Main image axis

    ax = fig.add_axes(
        [
            0.05,
            0.08,
            0.78,
            0.82
        ]
    )



    # Use 95th percentile clipping
    # to improve visualization contrast

    vmax = np.nanpercentile(
        np.abs(dsm_ds),
        95
    )



    im = ax.imshow(
        dsm_ds,
        cmap="viridis",
        vmin=-vmax,
        vmax=vmax
    )



    # ---------------------------------------------------------
    # Draw anomaly validation patches
    # ---------------------------------------------------------

    for r, c in sig_coords_ds:


        ax.add_patch(
            patches.Circle(
                (c, r),
                radius_ds,
                edgecolor="red",
                facecolor="none",
                lw=0.8
            )
        )



    ax.axis(
        "off"
    )



    # =========================================================
    # COLORBAR
    # =========================================================

    cax = fig.add_axes(
        [
            0.85,
            0.12,
            0.025,
            0.75
        ]
    )


    cbar = fig.colorbar(
        im,
        cax=cax
    )


    cbar.set_label(
        "Elevation Change (m)"
    )



    plt.tight_layout()



    # Save final figure

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )



    if show:

        plt.show()

    else:

        plt.close()



    return png_path

    
