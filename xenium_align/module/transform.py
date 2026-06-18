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





import os
from pathlib import Path
from typing import Union
import geopandas as gpd
import spatialdata as sd
import SimpleITK as sitk

# Type alias lisible
SpatialInput = Union[str, Path, gpd.GeoDataFrame, sd.SpatialData]


def _resolve_gdf(
    source: SpatialInput,
    sdata_shapes_key: str | None = None,
) -> gpd.GeoDataFrame:
    """Résout n'importe quelle source spatiale en GeoDataFrame.
    Parameters
    ----------
    source:
        - str / Path  → chemin vers un fichier GeoJSON (ou tout format lisible par GeoPandas)
        - GeoDataFrame → utilisé tel quel
        - SpatialData  → ``sdata_shapes_key`` est obligatoire ; renvoie ``source.shapes[sdata_shapes_key]``
    sdata_shapes_key:
        Nom de la couche Shapes dans le SpatialData (ex. ``"cells"``, ``"cell_boundaries"``).
        Ignoré si ``source`` n'est pas un SpatialData.
    Returns
    -------
    GeoDataFrame prêt à l'emploi (copie légère pour éviter les mutations).
    """
    if isinstance(source, (str, Path)):
        return gpd.read_file(source)
    if isinstance(source, gpd.GeoDataFrame):
        return source.copy()
    if isinstance(source, sd.SpatialData):
        if sdata_shapes_key is None:
            available = list(source.shapes.keys())
            raise ValueError(
                f"`sdata_shapes_key` est requis pour un SpatialData. "
                f"Couches disponibles : {available}"
            )
        if sdata_shapes_key not in source.shapes:
            available = list(source.shapes.keys())
            raise KeyError(
                f"Couche '{sdata_shapes_key}' introuvable dans le SpatialData. "
                f"Couches disponibles : {available}"
            )
        return source.shapes[sdata_shapes_key].copy()
    raise TypeError(
        f"Type de source non supporté : {type(source)}. "
        f"Attendu : str | Path | GeoDataFrame | SpatialData."
    )


def apply_sitk_transform(
    source: SpatialInput,
    combo_dir: str | Path,
    meta_he: dict,
    meta_xe: dict,
    ms: int = 10,
    offset_x: float = 0,
    offset_y: float = 0,
    target_cell_type: str | None = None,
    sdata_shapes_key: str | None = None,
) -> gpd.GeoDataFrame:
    """Applique une transformation SimpleITK (rigid + bspline) à des cellules spatiales.
    Parameters
    ----------
    source:
        Chemin GeoJSON, GeoDataFrame, ou SpatialData.
    sdata_shapes_key:
        Requis uniquement si ``source`` est un SpatialData
        (ex. ``"cells"`` pour CellViT, ``"cell_boundaries"`` pour Xenium).
    Returns
    -------
    GeoDataFrame transformé (également sauvegardé dans ``output_path``).
    """
    # 1. Charger les transforms
    rigid_path = os.path.join(combo_dir, f"transformation_rigid_{ms}.tfm")
    bspline_path = os.path.join(combo_dir, f"transformation_bspline_{ms}.tfm")
    tx_rigid = sitk.ReadTransform(rigid_path)
    tx_bspline = sitk.ReadTransform(bspline_path)
    composite_tx = sitk.CompositeTransform([tx_rigid, tx_bspline])
    # 2. Résoudre le GeoDataFrame depuis n'importe quelle source
    gdf = _resolve_gdf(source, sdata_shapes_key=sdata_shapes_key)
    # 3. Appliquer la transformation
    gdf_transformed = apply_transform(
        gdf,
        target_cell_type,
        sitk_transform,
        m_he=meta_he,
        m_xe=meta_xe,
        tx=composite_tx,
        ox=offset_x,
        oy=offset_y,
    )
    output_path=os.path.join(combo_dir, f"cellvit_transformed_{ms}.geojson")
    # 4. Sauvegarder
    gdf_transformed.to_file(output_path)
    logger.info(f"Transformed cells (sitk) exported to {output_path}")
    return gdf_transformed







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




import os
import geopandas as gpd
import numpy as np
import SimpleITK as sitk
from shapely.geometry import Point


def _get_inverse_composite_transform(composite_tx, gdf_xe_pixels, meta_xe):
    """Extract and invert a SimpleITK composite transform (Rigid + BSpline) using dynamic reference grid."""
    tx_rigid = composite_tx.GetNthTransform(0)
    tx_bspline = composite_tx.GetNthTransform(1)
    coords = np.array([[g.x, g.y] for g in gdf_xe_pixels.geometry])
    phys_coords = coords * [
        float(meta_xe["orig_spacing_x"]),
        float(meta_xe["orig_spacing_y"]),
    ]
    min_x, min_y = np.min(phys_coords, axis=0) - 100.0
    max_x, max_y = np.max(phys_coords, axis=0) + 100.0
    spacing = [10.0, 10.0]
    size = [int((max_x - min_x) / spacing[0]), int((max_y - min_y) / spacing[1])]
    ref_image = sitk.Image(size, sitk.sitkUInt8)
    ref_image.SetSpacing(spacing)
    ref_image.SetOrigin((float(min_x), float(min_y)))
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_image)
    displacement_field = displacement_filter.Execute(tx_bspline)
    inverse_displacement = sitk.InvertDisplacementField(
        displacement_field, maximumNumberOfIterations=100
    )
    tx_bspline_inverse = sitk.DisplacementFieldTransform(inverse_displacement)
    tx_rigid_inverse = tx_rigid.GetInverse()
    full_inverse_transform = sitk.CompositeTransform(
        tx_bspline_inverse.GetDimension()
    )
    full_inverse_transform.AddTransform(tx_bspline_inverse)
    full_inverse_transform.AddTransform(tx_rigid_inverse)
    return full_inverse_transform


def transform_xenium_adata_pixels_to_he(
    gdf_xe_pixels: gpd.GeoDataFrame,
    meta_he: dict,
    meta_xe: dict,
    combo_dir: str,
    ox: float = 0,
    oy: float = 0,
    ms: int = 10,
) -> gpd.GeoDataFrame:
    """Transform an Xenium pixel GeoDataFrame to HE pixels using the inverted displacement field and save to GeoJSON."""
    rigid_path = os.path.join(combo_dir, f"transformation_rigid_{ms}.tfm")
    bspline_path = os.path.join(combo_dir, f"transformation_bspline_{ms}.tfm")
    tx_rigid = sitk.ReadTransform(rigid_path)
    tx_bspline = sitk.ReadTransform(bspline_path)
    composite_tx = sitk.CompositeTransform([tx_rigid, tx_bspline])
    print("Computing inverse transform field...")
    tx_inverse = _get_inverse_composite_transform(
        composite_tx, gdf_xe_pixels, meta_xe
    )
    transformed_points = []
    print(f"Applying inverse transform to {len(gdf_xe_pixels)} cells...")
    for geom in gdf_xe_pixels.geometry:
        px = (
            float(geom.x) * float(meta_xe["orig_spacing_x"]) + float(ox)
        )
        py = float(geom.y) * float(meta_xe["orig_spacing_y"]) + float(oy)
        phys_he_full = tx_inverse.TransformPoint((px, py))
        px_he = phys_he_full[0] / float(meta_he["orig_spacing_x"])
        py_he = phys_he_full[1] / float(meta_he["orig_spacing_y"])
        transformed_points.append(Point(px_he, py_he))
    gdf_full = gdf_xe_pixels.copy()
    gdf_full["geometry"] = transformed_points
    gdf_full["objectType"] = "annotation"
    output_path = os.path.join(combo_dir, f"xenium_to_he_{ms}.geojson")
    gdf_full.to_file(output_path, driver="GeoJSON")
    print(f"Exported transformed cells to {output_path}")
    return gdf_full