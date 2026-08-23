import os
import numpy as np
import pandas as pd
import geopandas as gpd
import SimpleITK as sitk
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely import set_precision

import logging
logger = logging.getLogger(__name__)


def apply_transform(gdf, target_cell_type, transform_func, **kwargs):
    # Filter specific cell type if given
    if target_cell_type is not None:
        mask = gdf['classification'].apply(lambda x: x.get('name') == target_cell_type)
        gdf = gdf.loc[mask].copy()
    else:
        gdf = gdf.copy()
    # Apply transform
    gdf['geometry'] = gdf['geometry'].apply(lambda g: transform_func(g, **kwargs))
    
    return gdf

def sitk_transform(cell, m_he, m_xe, tx, ox, oy):
    pts = np.array(cell.exterior.coords)
    # Convert to microns
    phys_he = pts * [m_he['orig_spacing_x'], m_he['orig_spacing_y']]
    # Apply transform
    phys_xe = np.array([tx.TransformPoint(p) for p in phys_he])
    # Flip coordinates if the target image has been flipped
    if m_xe.get('flipped', False):
        phys_xe = m_xe['flip_extent'] - phys_xe
    # Convert to pixel and apply offset if necessary
    px_xe = (phys_xe - [ox, oy]) / [m_xe['orig_spacing_x'], m_xe['orig_spacing_y']]
    return Polygon(px_xe)


def sitk_inverse_transform(cell, m_he, m_xe, tx, ox, oy):
    pts = np.array(cell.exterior.coords)
    # Pixels Xenium -> Microns Xenium
    phys_xe = (pts * [m_xe['orig_spacing_x'], m_xe['orig_spacing_y']]) + [ox, oy]
    # Flip coordinates if the target image has been flipped
    if m_xe.get('flipped', False):
        phys_xe = m_xe['flip_extent'] - phys_xe
    # Apply inverse transform (Microns Xenium -> Microns HE)
    phys_he = np.array([tx.TransformPoint(p) for p in phys_xe])
    # Microns HE -> Pixels HE
    px_he = phys_he / [m_he['orig_spacing_x'], m_he['orig_spacing_y']]
    return Polygon(px_he)


import os
from pathlib import Path
from typing import Union
import geopandas as gpd
import spatialdata as sd
import SimpleITK as sitk

# Type alias lisible
SpatialInput = Union[str, Path, gpd.GeoDataFrame, sd.SpatialData]



def _resolve_gdf(source: SpatialInput, sdata_shapes_key: str | None = None) -> gpd.GeoDataFrame:
    if isinstance(source, (str, Path)):
        return gpd.read_file(source)
    if isinstance(source, gpd.GeoDataFrame):
        return source.copy()
    if isinstance(source, sd.SpatialData):
        if sdata_shapes_key is None:
            available = list(source.shapes.keys())
            raise ValueError(f"`sdata_shapes_key` est requis pour un SpatialData. Couches disponibles : {available}")
        if sdata_shapes_key not in source.shapes:
            available = list(source.shapes.keys())
            raise KeyError(f"Couche '{sdata_shapes_key}' introuvable dans le SpatialData. Couches disponibles : {available}")
        return source.shapes[sdata_shapes_key].copy()
    raise TypeError(f"Type de source non supporté : {type(source)}.")


def _get_inverse_composite_transform_polygons(composite_tx, meta_source, spacing=10.0, margin=100.0):
    tx_rigid = composite_tx.GetNthTransform(0)
    tx_bspline = composite_tx.GetNthTransform(1)

    width_phys, height_phys = meta_source['extent']
    min_x, min_y = -margin, -margin
    max_x, max_y = width_phys + margin, height_phys + margin

    ref_spacing = [spacing, spacing]
    size = [int((max_x - min_x) / ref_spacing[0]), int((max_y - min_y) / ref_spacing[1])]
    ref_image = sitk.Image(size, sitk.sitkUInt8)
    ref_image.SetSpacing(ref_spacing)
    ref_image.SetOrigin((float(min_x), float(min_y)))

    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_image)
    displacement_field = displacement_filter.Execute(tx_bspline)
    inverse_displacement = sitk.InvertDisplacementField(displacement_field, maximumNumberOfIterations=100)
    tx_bspline_inverse = sitk.DisplacementFieldTransform(inverse_displacement)
    tx_rigid_inverse = tx_rigid.GetInverse()
    full_inverse_transform = sitk.CompositeTransform(tx_bspline_inverse.GetDimension())
    full_inverse_transform.AddTransform(tx_bspline_inverse)
    full_inverse_transform.AddTransform(tx_rigid_inverse)
    return full_inverse_transform



def _fix_geodataframe(gdf):
    # Repair and filter geometries
    gdf['geometry'] = gdf.geometry.make_valid()
    gdf['geometry'] = gdf.geometry.apply(_clean_geom)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    # Fusionne les points quasi-identiques (pics/quasi-auto-intersections issus du transform sitk)
    gdf['geometry'] = set_precision(gdf.geometry.values, grid_size=0.0005)
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



def apply_sitk_transform(
    source: SpatialInput,
    combo_dir: str | Path,
    meta_source: dict,
    meta_target: dict,
    ms: int = 10,
    offset_x: float = 0,
    offset_y: float = 0,
    target_cell_type: str | None = None,
    sdata_shapes_key: str | None = None,
    inverse: bool = False,
) -> gpd.GeoDataFrame:
    rigid_path = os.path.join(combo_dir, f"transformation_rigid_{ms}.tfm")
    bspline_path = os.path.join(combo_dir, f"transformation_bspline_{ms}.tfm")
    tx_rigid = sitk.ReadTransform(rigid_path)
    tx_bspline = sitk.ReadTransform(bspline_path)
    composite_tx_ori = sitk.CompositeTransform([tx_rigid, tx_bspline])
    gdf = _resolve_gdf(source, sdata_shapes_key=sdata_shapes_key)
    if inverse:
        output_path = os.path.join(combo_dir, f"transformed_inverse_{ms}.geojson")
        print("Computing inverse transform field...")
        composite_tx = _get_inverse_composite_transform_polygons(composite_tx_ori, meta_source)
        transform_func = sitk_inverse_transform
    else:
        output_path = os.path.join(combo_dir, f"transformed_{ms}.geojson")
        composite_tx = composite_tx_ori
        transform_func = sitk_transform
    gdf_transformed = apply_transform(
        gdf,
        target_cell_type,
        transform_func,
        m_he=meta_source,
        m_xe=meta_target,
        tx=composite_tx,
        ox=offset_x,
        oy=offset_y,
    )
    gdf_transformed = _fix_geodataframe(gdf_transformed)
    gdf_transformed.to_file(output_path)
    logger.info(f"Transformed cells (sitk) exported to {output_path}")





from scipy.signal import fftconvolve
import tifffile as tifi

def find_crop_origin_large_wsi(original_path, crop_path):
    with tifi.TiffFile(original_path) as tif_orig, tifi.TiffFile(
        crop_path
    ) as tif_crop:
        size_lvl0 = tif_orig.series[0].levels[0].shape[1]
        size_lvln = tif_orig.series[0].levels[0].shape[1]
        downsample = size_lvl0 / size_lvln
        print(f"Load image to get the offset from crop...")
        img_orig = tif_orig.series[0].levels[0].asarray()
        img_crop = tif_crop.series[0].levels[0].asarray()
        if len(img_orig.shape) == 3:
            img_orig = np.mean(img_orig, axis=2)
        if len(img_crop.shape) == 3:
            img_crop = np.mean(img_crop, axis=2)
    img_orig = img_orig - np.mean(img_orig)
    img_crop = img_crop - np.mean(img_crop)
    corr = fftconvolve(img_orig, img_crop[::-1, ::-1], mode="same")
    y_mid, x_mid = np.unravel_index(np.argmax(corr), corr.shape)
    y_loc = y_mid - img_crop.shape[0] // 2
    x_loc = x_mid - img_crop.shape[1] // 2
    x_origin_pixel = int(round(x_loc * downsample))
    y_origin_pixel = int(round(y_loc * downsample))
    print(f"\n--- Coordonnées d'origine trouvées (Niveau 0) ---")
    print(f"X: {x_origin_pixel} px")
    print(f"Y: {y_origin_pixel} px")
    return x_origin_pixel, y_origin_pixel



def get_affine_coeffs(matrix_affine, x_crop_offset, y_crop_offset):
    # Shapely expects (a, b, d, e, xoff, yoff) where:
    # x' = ax + by + xoff
    # y' = dx + ey + yoff
    # Map from :
    # [[ a, b, xoff ],
    #  [ d, e, yoff ],
    #  [ 0, 0, 1    ]]
    a = matrix_affine[0, 0]
    b = matrix_affine[0, 1]
    xoff = matrix_affine[0, 2] - x_crop_offset
    d = matrix_affine[1, 0]
    e = matrix_affine[1, 1]
    yoff = matrix_affine[1, 2] - y_crop_offset
    
    return (a, b, d, e, xoff, yoff)

def apply_affine_transform(input_path, combo_dir, matrix_path, inverse = False, original_path = None, crop_path = None, target_cell_type=None):
    # Load matrix
    mat = pd.read_csv(matrix_path, header=None).values
    # Define inverse matrix if we want the inverse
    if inverse:
        output_path = os.path.join(combo_dir, f"affine_transformed_inverse.geojson")
        mat = _inverse_matrix(mat)
    else:
        output_path = os.path.join(combo_dir, f"affine_transformed.geojson")
    # Define offset if xenium explorer alignment has been done on a bigger image than sitk alignment
    if original_path and crop_path is not None:
        x_crop_offset, y_crop_offset = find_crop_origin_large_wsi(original_path, crop_path)
    else:
        x_crop_offset, y_crop_offset = 0, 0
    # Coefficients for affine transform
    matrix = get_affine_coeffs(mat, x_crop_offset, y_crop_offset)
    # Load .geojson (pixel)
    gdf = gpd.read_file(input_path)
    # Apply transformation
    gdf_transformed = apply_transform(gdf, target_cell_type, affine_transform, matrix=matrix)
    # Save transformed cells
    gdf_transformed.to_file(output_path)
    logger.info(f"Transformed cells (affine) exported to {output_path}")

def _inverse_matrix(mat):
    # Inverse matrix if needed
    m_inv = np.linalg.inv(mat)
    
    return m_inv
