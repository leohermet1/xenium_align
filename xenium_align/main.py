import os
from pathlib import Path

import xenium_align as xa
from ._constants import (
    level_index, 
    ms, 
    spline_order, 
    combo_name,
    output_dir,
    gt_idx,
    pred_idx 
)

def main():
    # Base directory
    HOME = Path("/home/hermet/Registration/Pierre_Xenium/")
    # Specific project folders
    HE_DIR = HOME / "ADK_screen_250226_neutro"
    XE_DIR = HOME / "output_XETG00373_0059858_colon_IM_02_20250811_171708_ADK"
    # Files
    HE_IMG_PATH = HE_DIR / "ADK_explorer.ome.tif"
    XE_IMG_PATH = {
        "DAPI": XE_DIR / "morphology_focus" / "morphology_focus_0000.ome.tif",
        "ATP1A": XE_DIR / "morphology_focus" / "morphology_focus_0001.ome.tif",
        "18S": XE_DIR / "morphology_focus" / "morphology_focus_0002.ome.tif"
    }
    matrix_path = HOME / "ADK_screen_020326_alignment_files/matrix.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # --- SOURCE (H&E) ---
    # Load hematoxylin
    raw_he, meta_he = xa.data.load_downsampled_image(HE_IMG_PATH, level_index)
    he_proc = xa.data.prepare_he(raw_he)
    fixed = xa.data.get_sitk_image(he_proc, meta_he)
    # CELLVIT SEGMENTATION
    ## Convert to QuPath-readable format
    input_cellvit_snappy = HE_DIR / "cells.geojson.snappy"
    output_cellvit_geojson = output_dir / "nucleus_cellvit_fixed.geojson"
    xa.data.uncompress_snappy_to_geojson(input_cellvit_snappy, output_cellvit_geojson)

    # --- TARGET (Xenium) ---
    # Load DAPI
    XE_IMG_PATH_DAPI = XE_IMG_PATH.pop("DAPI")
    dapi, meta_xe = xa.data.load_downsampled_image(XE_IMG_PATH_DAPI, level_index)
    channels_raw = {"DAPI": dapi}
    offset_x, offset_y = xa.data.calculate_pyramidal_offset(XE_IMG_PATH_DAPI, level_index, meta_xe)
    # Load other channels
    channels_raw.update({
        name: xa.data.load_downsampled_image(path, level_index)[0] 
        for name, path in XE_IMG_PATH.items()
    })
    channels_proc, combos = xa.data.prepare_xe_generate_combination(channels_raw)
    # Define the composite image
    channels_to_combine = combos[combo_name]
    xe_proc = xa.data.combine_xenium_channels(channels_proc, channels_to_combine)
    moving = xa.data.get_sitk_image(xe_proc, meta_xe)
    # XENIUM SEGMENTATION
    ## Convert to QuPath-readable format
    ### Cells
    input_cells_xenium_parquet = XE_DIR / "cell_boundaries.parquet"
    output_cells_xenium_geojson = output_dir / "cells_xenium_fixed.geojson"
    xa.data.export_xenium_to_pixel_geojson(input_cells_xenium_parquet, meta_xe, output_cells_xenium_geojson)
    ### Nucleus
    input_nucleus_xenium_parquet = XE_DIR / "nucleus_boundaries.parquet"
    output_nucleus_xenium_geojson = output_dir / "nucleus_xenium_fixed.geojson"
    xa.data.export_xenium_to_pixel_geojson(input_nucleus_xenium_parquet, meta_xe, output_nucleus_xenium_geojson)

    # --- REGISTRATION ---
    combo_dir = os.path.join(output_dir, combo_name)
    if not os.path.exists(combo_dir): os.makedirs(combo_dir)
    # Visualize initial overlap
    xa.plot.single_overlay(fixed, moving, combo_dir, "initial.png")
    # Run registration
    moving_rigid, tx_rigid, moving_bspline, tx_bspline = run_registration(fixed, moving, combo_dir, ms)
    # Visualize results
    xa.plot.registration_summary(fixed, moving, moving_rigid, moving_bspline, combo_dir, ms)
    xa.plot.local_deformations(fixed, tx_bspline, ms, spline_order, combo_dir)

    # --- TRANSFORM ---
    ## Affine
    output_cellvit_affine_transformed_geojson=os.path.join(combo_dir, f"cellvit_transformed_affine.geojson")
    apply_affine_transform(output_cellvit_geojson, output_cellvit_affine_transformed_geojson, matrix_path)
    ## sITK
    output_cellvit_sitk_transformed_geojson=os.path.join(combo_dir, f"cellvit_transformed_{ms}.geojson")
    apply_sitk_transform(output_cellvit_geojson, combo_dir, output_cellvit_sitk_transformed_geojson, ms, meta_he, meta_xe, offset_x, offset_y)

    # --- METRICS ---
    ## Affine
    gdf_gt = xa.data.load_gdf_pixel_to_microns(output_nucleus_xenium_geojson, meta_xe, gt_idx)
    gdf_pred = xa.data.load_gdf_pixel_to_microns(output_cellvit_affine_transformed_geojson, meta_xe, pred_idx)
    best_matches_affine = xa.metrics.match_and_compute_iou(gdf_pred, gdf_gt)
    output_iou_distr_affine = os.path.join(combo_dir, "iou_distribution_affine.png")
    xa.plot.plot_iou_distribution(best_matches_affine, output_iou_distr_affine)
    output_iou_spatial_affine = os.path.join(combo_dir, "iou_spatial_affine.png")
    xa.plot.plot_spatial_alignment(best_matches_affine, output_iou_spatial_affine)
    ## sITK
    gdf_gt = xa.data.load_gdf_pixel_to_microns(output_nucleus_xenium_geojson, meta_xe, gt_idx)
    gdf_pred = xa.data.load_gdf_pixel_to_microns(output_cellvit_sitk_transformed_geojson, meta_xe, pred_idx)
    best_matches_sitk = xa.metrics.match_and_compute_iou(gdf_pred, gdf_gt)
    output_iou_distr_sitk = os.path.join(combo_dir, "iou_distribution_sitk.png")
    xa.plot.plot_iou_distribution(best_matches_sitk, output_iou_distr_sitk)
    output_iou_spatial_sitk = os.path.join(combo_dir, "iou_spatial_sitk.png")
    xa.plot.plot_spatial_alignment(best_matches_sitk, output_iou_spatial_sitk)
    ## Comparison
    output_iou_comp = os.path.join(combo_dir, "iou_benchmark.png")
    xa.plot.plot_iou_distribution_comp(best_matches_sitk, "sITK", best_matches_affine, "Affine", output_iou_comp)


if __name__ == "__main__":
    main()