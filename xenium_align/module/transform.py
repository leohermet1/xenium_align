import os
import numpy as np
import pandas as pd
import geopandas as gpd
import SimpleITK as sitk
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

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
    # Convert to pixel and apply offset if necessary
    px_xe = (phys_xe - [ox, oy]) / [m_xe['orig_spacing_x'], m_xe['orig_spacing_y']]
    return Polygon(px_xe)


def apply_sitk_transform(input_path, combo_dir, output_path, ms, meta_he, meta_xe, offset_x=0, offset_y=0, target_cell_type=None):
    rigid_path=os.path.join(combo_dir, f"transformation_rigid_{ms}.tfm")
    bspline_path=os.path.join(combo_dir, f"transformation_bspline_{ms}.tfm")
    # Load transforms
    tx_rigid = sitk.ReadTransform(rigid_path)
    tx_bspline = sitk.ReadTransform(bspline_path)
    # Rigid + BSpline as composite
    composite_tx = sitk.CompositeTransform([tx_rigid, tx_bspline])
    # Load .geojson (pixel)
    gdf = gpd.read_file(input_path)
    # Apply transformation
    gdf_transformed = apply_transform(gdf, target_cell_type, sitk_transform, 
        m_he=meta_he, 
        m_xe=meta_xe, 
        tx=composite_tx, 
        ox=offset_x, 
        oy=offset_y
    )
    # Save transformed cells
    gdf_transformed.to_file(output_path)
    logger.info(f"Transformed cells (sitk) exported to {output_path}")


def get_affine_coeffs(matrix_affine):
    # Shapely expects (a, b, d, e, xoff, yoff) where:
    # x' = ax + by + xoff
    # y' = dx + ey + yoff
    # Map from :
    # [[ a, b, xoff ],
    #  [ d, e, yoff ],
    #  [ 0, 0, 1    ]]
    a = matrix_affine[0, 0]
    b = matrix_affine[0, 1]
    xoff = matrix_affine[0, 2]
    d = matrix_affine[1, 0]
    e = matrix_affine[1, 1]
    yoff = matrix_affine[1, 2]
    
    return (a, b, d, e, xoff, yoff)

def apply_affine_transform(input_path, output_path, matrix_path, target_cell_type=None):
    # Load matrix
    mat = pd.read_csv(matrix_path, header=None).values
    # Coefficients for affine transform
    matrix = get_affine_coeffs(mat)
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