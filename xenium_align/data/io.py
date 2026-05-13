import logging
from pathlib import Path
import tifffile
from ome_types import from_xml

import json
import snappy 
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

def load_downsampled_image(path, level_index=0):
    """
    Example: Load a specific pyramidal level as numpy array.
    """
    # Load image
    logger.info(f"Loading {Path(path).name}...")
    with tifffile.TiffFile(path, is_ome=False) as tif:
        levels = tif.series[0].levels 
        full_res = levels[0]
        target_res = levels[level_index]
        # Load downscaled image
        lowres_array = target_res.asarray()
        logger.info(f"Level {level_index}:  - Shape: {target_res.shape} ({target_res.axes})")
        # Scale factor
        x_idx, y_idx = target_res.axes.find('X'), target_res.axes.find('Y')
        scale_x = full_res.shape[x_idx] / target_res.shape[x_idx]
        scale_y = full_res.shape[y_idx] / target_res.shape[y_idx]
    # Original spacing from OME-XML
    meta_info = get_ome_metadata(path)
    # New spacing for spatial consistency
    meta = {
        "orig_spacing_x": meta_info["spacing_x"],
        "scale_x": scale_x,
        "orig_spacing_y": meta_info["spacing_y"],
        "scale_y": scale_y,
        "spacing": (meta_info["spacing_x"] * scale_x, meta_info["spacing_y"] * scale_y)
    }
    return lowres_array, meta


def get_ome_metadata(path):
    """Extract physical spacing, axes, and shape from OME-XML metadata."""
    with tifffile.TiffFile(path) as tif:
        ome = from_xml(tif.ome_metadata)
        return {
            "spacing_x": float(ome.images[0].pixels.physical_size_x),
            "spacing_y": float(ome.images[0].pixels.physical_size_y)
        }


def calculate_pyramidal_offset(path, level_idx, meta_xe):
    """
    Calculates the spatial offset between a pyramid level and the full-resolution image.
    
    This accounts for both:
    1. The 'Pixel Center Shift' inherent to downsampling: 0.5 * (scale - 1)
    2. The 'Canvas Padding' added by the scanner to fit tile boundaries.
    
    Args:
        path (str): Path to the TIFF file.
        level_idx (int): The pyramid level to calibrate (e.g., 4 or 5).
        pixel_size_um (float): Resolution at Level 0 (default 0.2125 for Xenium).
        
    Returns:
        tuple: (offset_x_um, offset_y_um) to be added to transformed coordinates.
    """
    with tifffile.TiffFile(path, is_ome=False) as tif:
        series = tif.series[0]
        l0 = series.levels[0]
        ln = series.levels[level_idx]
        
        # Get dimensions (handles C-style YX or CYX shapes)
        h0, w0 = l0.shape[-2:]
        hn, wn = ln.shape[-2:]
        
        # Theoretical downsampling scale (power of 2)
        scale = 2**level_idx
        
        # Component 1: Geometric Padding (difference between theoretical and actual grid)
        pad_x_pixels = (wn * scale - w0) / 2
        pad_y_pixels = (hn * scale - h0) / 2
        
        # Component 2: Pixel Center Shift (0.5 pixel correction)
        center_shift_pixels = 0.5 * (scale - 1)
        
        # Combine and convert to microns
        # We use absolute padding as it represents the offset from the original origin
        ps_x, ps_y = meta_xe['orig_spacing_x'], meta_xe['orig_spacing_y']
        offset_x = (abs(pad_x_pixels) + center_shift_pixels) * ps_x
        offset_y = (abs(pad_y_pixels) + center_shift_pixels) * ps_y
        
        return offset_x, offset_y


def list_resolutions(path):
    """
    Show all available resolution levels in a Xenium OME-TIFF file.
    Note: Xenium files store pyramids in SubIFDs, accessed via series[0].levels.
    """
    with tifffile.TiffFile(path, is_ome=False) as tif:
        # Xenium main series containing multiple sub-levels
        main_series = tif.series[0]
        levels = main_series.levels
        logger.info(f"File: {path}")
        logger.info(f"Total resolution levels found: {len(levels)}")
        logger.info("-" * 30)
        
        for i, lv in enumerate(levels):
            shape = lv.shape
            axes = lv.axes
            logger.info(f"Level {i}:")
            logger.info(f"  - Shape: {shape} ({axes})")


def uncompress_snappy_to_geojson(input_snappy, output_geojson):
    # Read and uncompress .geosjon.snappy
    with open(input_snappy, 'rb') as f:
        compressed_data = f.read()
        decompressed_data = snappy.uncompress(compressed_data)
        data = json.loads(decompressed_data)
    gdf = gpd.GeoDataFrame.from_features(data).explode(index_parts=False)
    gdf_fixed = _fix_geom(gdf)
    gdf_fixed = gdf_fixed.dissolve(by=gdf_fixed.index, as_index=False)
    # Save as .geojson
    gdf_fixed.to_file(output_geojson)


def export_xenium_to_pixel_geojson(parquet_path, meta_xe, output_path):
    """
    Convert Xenium nucleus_boundaries.parquet (µm) to .geojson (pixel)
    """
    # Load nucleus_boundaries.parquet
    df = pd.read_parquet(parquet_path)
    # Transform coords to pixel
    coords = np.ascontiguousarray(df[['vertex_x', 'vertex_y']].values, dtype=np.float64)
    coords[:, 0] /= meta_xe['orig_spacing_x']
    coords[:, 1] /= meta_xe['orig_spacing_y']
    # Transform pixel vertices to polygons (cells)
    ids = df['cell_id'].values
    changes = np.where(ids[1:] != ids[:-1])[0] + 1
    ring_offsets = np.concatenate(([0], changes, [len(df)])).astype(np.int64)
    poly_offsets = np.arange(len(ring_offsets), dtype=np.int64)
    geoms = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON, 
        coords, 
        (ring_offsets, poly_offsets)
    )
    unique_ids = ids[ring_offsets[:-1]]
    # Create final geodataframe
    gdf = gpd.GeoDataFrame({'cell_id': unique_ids, 'class': 'Nucleus'}, geometry=geoms)
    gdf_fixed = _fix_geom(gdf)
    gdf_fixed.to_file(output_path)

def _fix_geom(gdf):
    # Repair and filter geometries
    gdf['geometry'] = gdf.geometry.make_valid()
    gdf['geometry'] = gdf.geometry.apply(_clean_geom)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf

def _clean_geom(geom):
    if geom.geom_type == 'Polygon':
        return geom
    if geom.geom_type == 'MultiPolygon':
        # Keep biggest shape
        return max(geom.geoms, key=lambda a: a.area)
    if geom.geom_type == 'GeometryCollection':
        # Keep biggest shape
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        return max(polys, key=lambda a: a.area) if polys else None
    return None

def load_gdf_pixel_to_microns(input_path, meta, index_col_name):
    gdf = gpd.read_file(input_path)
    spacing_x, spacing_y = meta['orig_spacing_x'], meta['orig_spacing_y']
    # Convert to microns
    gdf.geometry = gdf.geometry.scale(xfact=spacing_x, yfact=spacing_y, origin=(0,0))
    gdf.crs = None
    # Preparation of indices: drop existing, reset, and rename
    gdf = gdf.reset_index().rename(columns={'index': index_col_name})
    
    return gdf