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
from shapely.geometry import Polygon, shape

import anndata as ad
import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel


logger = logging.getLogger(__name__)



_CHANNELS: dict[str, dict[str, str]] = {
    "DAPI":  {"v4_prefix": "ch0000_", "legacy": "morphology_focus_0000.ome.tif"},
    "ATP1A": {"v4_prefix": "ch0001_", "legacy": "morphology_focus_0001.ome.tif"},
    "18S":   {"v4_prefix": "ch0002_", "legacy": "morphology_focus_0003.ome.tif"},
}


def get_xenium_image_paths(xe_dir: Path | str) -> dict[str, Path]:
    """
    Return morphology image paths for a Xenium run directory.
    
    Supports XOA v4.0+ dynamic naming (``ch0000_dapi.ome.tif``)
    and legacy numeric filenames (``morphology_focus_0000.ome.tif``).
    
    Parameters
    ----------
    xe_dir
        Root directory of the Xenium run output.
        
    Returns
    -------
    dict[str, Path]
        Mapping of channel name → image path.
        Keys: ``"DAPI"``, ``"ATP1A"``, ``"18S"``.
    """
    morphology_dir = (Path(xe_dir) / "morphology_focus").resolve(strict=True)
    
    # XOA v4.0+: ch0000_<name>.ome.tif
    paths = {
        name: matches[0]
        for name, cfg in _CHANNELS.items()
        if (matches := sorted(morphology_dir.glob(f"{cfg['v4_prefix']}*.ome.tif")))
    }
    
    # Fallback to legacy filenames if any channel is missing
    if len(paths) < len(_CHANNELS):
        paths = {
            name: morphology_dir / cfg["legacy"]
            for name, cfg in _CHANNELS.items()
        }
        
    return paths


def choose_level_for_target_spacing(path, target_spacing_um=2.0):
    """
    Return the pyramid level with spacing closest to target_spacing_um.

    Level index != physical resolution across files: native spacing depends
    on the acquisition system, not the index (e.g. multi-IF scanner ~0.5
    um/px vs Xenium morphology ~0.2 um/px, both IF). Fixing a level number
    is therefore inconsistent across files; targeting spacing directly is not.

    target_spacing_um=2.0: empirical middle ground. Coarser -> not enough
    signal for SimpleITK to lock onto during registration. Finer -> larger
    image, slower registration, no real gain in alignment quality.

    Parameters
    ----------
    path : str or Path
        Pyramidal OME-TIFF path.
    target_spacing_um : float, default 2.0
        Target resolution, microns/pixel.

    Returns
    -------
    int
        Closest-matching level index. Feed to `load_downsampled_image` as `level_index`.
    """
    with tifffile.TiffFile(path, is_ome=False) as tif:
        levels = tif.series[0].levels
        full_res = levels[0]
        x_idx = full_res.axes.find('X')
        meta_info = get_ome_metadata(path)

        # Scan every level, keep the one closest to target_spacing_um
        best_level, best_diff, best_spacing = 0, None, None
        for i, lvl in enumerate(levels):
            scale = full_res.shape[x_idx] / lvl.shape[x_idx]
            spacing = meta_info['spacing_x'] * scale
            diff = abs(spacing - target_spacing_um)
            if best_diff is None or diff < best_diff:
                best_diff, best_level, best_spacing = diff, i, spacing

    logger.info(
        f"{Path(path).name}: native spacing = {meta_info['spacing_x']:.4f} um/px, "
        f"level {best_level} selected (spacing={best_spacing:.4f} um/px, target={target_spacing_um})"
    )
    return best_level


def load_downsampled_image(path, level_index=3):
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
    spacing_x = meta_info["spacing_x"] * scale_x
    spacing_y = meta_info["spacing_y"] * scale_y
    meta = {
        "orig_spacing_x": meta_info["spacing_x"],
        "scale_x": scale_x,
        "orig_spacing_y": meta_info["spacing_y"],
        "scale_y": scale_y,
        "spacing": (spacing_x, spacing_y),
        "extent": np.array([target_res.shape[x_idx] * spacing_x, target_res.shape[y_idx] * spacing_y]),
    }
    return lowres_array, meta


def get_ome_metadata(path):
    """Extract physical spacing, axes, and shape from OME-XML metadata."""
    FALLBACK_SPACINGS = {
        "xenium": 0.2125,
        "he": 0.3273,
        "mif": 0.4968,
    }
    
    with tifffile.TiffFile(path) as tif:
        ome = from_xml(tif.ome_metadata)
        px = ome.images[0].pixels

        # Missing spacing
        if px.physical_size_x is None or px.physical_size_y is None:
            ref_spacings = ", ".join(f"{v} um/px ({k})" for k, v in FALLBACK_SPACINGS.items())
            logger.error(
                f"{Path(path).name}: missing physical_size_x/y in OME metadata, spacing cannot be determined. "
                f"Reference native spacing: {ref_spacings}."
            )

            try:
                choice = input("Missing spacing — enter which one to use (xenium / he / mif): ").strip().lower()
            except EOFError:
                choice = ""
            
            if choice in FALLBACK_SPACINGS:
                spacing_x = spacing_y = FALLBACK_SPACINGS[choice]
                logger.warning(f"{Path(path).name}: using fallback spacing {spacing_x} um/px ({choice}).")
            else:
                raise ValueError(f"Missing pixel size metadata in {path}")
        else:
            spacing_x = float(px.physical_size_x)
            spacing_y = float(px.physical_size_y)

        # flag when spacting = 1.0 (missing metadata)
        if spacing_x == 1.0 or spacing_y == 1.0:
            ref_spacings = ", ".join(f"{v} um/px ({k})" for k, v in FALLBACK_SPACINGS.items())
            logger.warning(
                f"{Path(path).name}: spacing = 1.0 um/px, likely uncalibrated metadata. "
                f"Reference native spacing: {ref_spacings}."
            )

            try:
                choice = input("Uncalibrated spacing (1.0) — enter which one to use, or leave empty to keep 1.0 (xenium / he / mif): ").strip().lower()
            except EOFError:
                choice = ""
            
            if choice in FALLBACK_SPACINGS:
                spacing_x = spacing_y = FALLBACK_SPACINGS[choice]
                logger.warning(f"{Path(path).name}: using fallback spacing {spacing_x} um/px ({choice}).")
        
        return {
            "spacing_x": spacing_x,
            "spacing_y": spacing_y
        }


def calculate_pyramidal_offset(path, meta_xe, level_index = 3):
    """
    Calculates the spatial offset between a pyramid level and the full-resolution image.
    
    This accounts for both:
    1. The 'Pixel Center Shift' inherent to downsampling: 0.5 * (scale - 1)
    2. The 'Canvas Padding' added by the scanner to fit tile boundaries.
    
    Args:
        path (str): Path to the TIFF file.
        level_index (int): The pyramid level to calibrate (e.g., 3).
        pixel_size_um (float): Resolution at Level 0 (default 0.2125 for Xenium).
        
    Returns:
        tuple: (offset_x_um, offset_y_um) to be added to transformed coordinates.
    """
    with tifffile.TiffFile(path, is_ome=False) as tif:
        series = tif.series[0]
        l0 = series.levels[0]
        ln = series.levels[level_index]
        
        # Get dimensions (handles C-style YX or CYX shapes)
        h0, w0 = l0.shape[-2:]
        hn, wn = ln.shape[-2:]
        
        # Theoretical downsampling scale (power of 2)
        scale = 2**level_index
        
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





def load_gdf_pixel_to_microns(input_path, meta):
    gdf = gpd.read_file(input_path)
    gdf = _fix_geodataframe(gdf)
    # Keep only detected nuclei (remove QuPath's ROI annotation from the segmentation run)
    if 'objectType' in gdf.columns:
        gdf = gdf[gdf['objectType'] == 'detection'].copy()
    spacing_x, spacing_y = meta['orig_spacing_x'], meta['orig_spacing_y']
    # Convert to microns
    gdf.geometry = gdf.geometry.scale(xfact=spacing_x, yfact=spacing_y, origin=(0,0))
    gdf.crs = None

    return gdf



















def _fix_geodataframe(gdf):
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

def _load_snappy_as_gdf(path: Path, output_geojson: Path | None = None) -> gpd.GeoDataFrame:
    with open(path, 'rb') as f:
        decompressed_data = snappy.uncompress(f.read())
    data_dict = json.loads(decompressed_data)
    gdf = gpd.GeoDataFrame.from_features(data_dict).explode(index_parts=False)
    gdf = _fix_geodataframe(gdf)
    gdf.index = pd.Index([f"cell_{i}" for i in range(len(gdf))], name="cell_id")
    gdf["objectType"] = "annotation"
    gdf["name"] = gdf.index 
    if output_geojson is not None:
        gdf.to_file(output_geojson, driver="GeoJSON")
    return gdf

def _to_anndata(gdf: gpd.GeoDataFrame) -> ad.AnnData:
    centroids = shapely.get_coordinates(gdf.geometry.centroid)
    obs = gdf.drop(columns="geometry").copy()
    for col in obs.columns:
        if obs[col].apply(lambda x: isinstance(x, (dict, list))).any():
            obs[col] = obs[col].apply(json.dumps)
    obs["region"] = "cells"
    obs["instance_id"] = np.arange(len(gdf))
    adata = ad.AnnData(obs=obs)
    adata.obsm["spatial"] = centroids
    return adata


def _to_spatialdata(gdf: gpd.GeoDataFrame, adata: ad.AnnData) -> sd.SpatialData:
    return sd.SpatialData(
        shapes={"cells": ShapesModel.parse(gdf)},
        tables={"table": TableModel.parse(adata)},
    )


def read_cellvit(path: str | Path, output_geojson: str | Path | None = None) -> sd.SpatialData:
    """Load a CellViT++ cells.geojson.snappy file and return a SpatialData object."""
    gdf = _load_snappy_as_gdf(Path(path), Path(output_geojson) if output_geojson else None)
    adata = _to_anndata(gdf)
    return _to_spatialdata(gdf, adata)




def export_xenium_to_pixel_geojson(
    xenium_dir, 
    meta_xe, 
    output_dir, 
    export_cells: bool = True, 
    export_nucleus: bool = True
):
    """
    Convert Xenium nucleus_boundaries.parquet (µm) and / or cell_boundaries.parquet (µm) to .geojson (pixel)
    """
    targets = []
    if export_cells:
        targets.append(("cell_boundaries.parquet", "XENIUM_cells.geojson", "Cell"))
    if export_nucleus:
        targets.append(("nucleus_boundaries.parquet", "XENIUM_nucleus.geojson", "Nucleus"))
        
    for filename, output_name, feature_class in targets:
        parquet_path = Path(xenium_dir) / filename
        output_path = Path(output_dir) / output_name
        
        if not parquet_path.exists():
            continue
            
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
        gdf = gpd.GeoDataFrame({'name': unique_ids, 'type': 'detection'}, geometry=geoms)
        gdf_fixed = _fix_geodataframe(gdf)
        gdf_fixed["objectType"] = "detection"
        gdf_fixed.to_file(output_path)



from shapely.geometry import Point
import anndata as ad
import os

def load_xenium_adata(
    h5ad_path: str, meta_xe: dict, combo_dir: str
) -> gpd.GeoDataFrame:
    """Read Xenium h5ad, convert micron coordinates to native pixels using meta_xe, and save to GeoJSON for QuPath."""
    adata = ad.read_h5ad(h5ad_path)
    phys_xe = adata.obsm["spatial"]
    px_xe = phys_xe / [meta_xe["orig_spacing_x"], meta_xe["orig_spacing_y"]]
    transformed_points = [Point(pt[0], pt[1]) for pt in px_xe]
    data = {}
    if "cell_type" in adata.obs.columns:
        data["name"] = adata.obs["cell_type"].values
        data["classification"] = adata.obs["cell_type"].values
    else:
        data["name"] = adata.obs_names
        data["classification"] = "Cell"
    gdf_pixels = gpd.GeoDataFrame(data, geometry=transformed_points)
    output_path = os.path.join(combo_dir, f"xenium_cells_pixels.geojson")
    gdf_pixels.to_file(output_path, driver="GeoJSON")
    print(f"Exported {len(gdf_pixels)} cells to {output_path}")
    return gdf_pixels