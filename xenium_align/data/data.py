# xenium_align/data/data.py
from pathlib import Path
import os
from typing import Dict, Any, Tuple

# Import des fonctions depuis io.py et preprocess.py
from .io import (
    choose_level_for_target_spacing,
    load_downsampled_image,
    get_xenium_image_paths,
    calculate_pyramidal_offset,
)
from .preprocess import (
    prepare_he,
    prepare_xe_generate_combination,
    combine_xenium_channels,
    flip_if_needed,
)

def load_images_and_metadata(
    HE_IMG_PATH: Path,
    XE_DIR: Path,
    output_dir: Path,
    combo_name: str = "DAPI_ATP1A_18S",
) -> Dict[str, Any]:
    """
    Loads and preprocesses source (H&E) and target (Xenium) images for alignment.
    This function combines I/O and preprocessing steps to prepare images and metadata
    required for registration and transformation.

    Args:
        HE_IMG_PATH (Path): Path to the H&E image file (e.g., OME-TIFF).
        XE_DIR (Path): Path to the Xenium data directory.
        output_dir (Path): Output directory for saving results.
        combo_name (str, optional): Name of the channel combination to use. Defaults to "DAPI_ATP1A_18S".

    Returns:
        Dict[str, Any]: A dictionary containing:
            - meta_source: Metadata for the source image.
            - meta_target: Metadata for the target image.
            - offset_x (int): X-offset for pyramid level.
            - offset_y (int): Y-offset for pyramid level.
            - proc_source: Processed source image.
            - proc_target: Processed target image.
            - combo_dir (str): Path to the output directory for this channel combination.
            - combo_name (str): Name of the channel combination used.
    """
    # --- SOURCE (H&E) ---
    # Define the image resolution
    level_idx = choose_level_for_target_spacing(HE_IMG_PATH, target_spacing_um=2)
    # Load hematoxylin
    raw_source, meta_source = load_downsampled_image(HE_IMG_PATH, level_index=level_idx)
    proc_source = prepare_he(raw_source, signal=100)

    # --- TARGET (Xenium) ---
    # Get all Xenium image paths
    XE_IMG_PATH = get_xenium_image_paths(XE_DIR)
    # Define the image resolution for DAPI
    XE_IMG_PATH_DAPI = XE_IMG_PATH.pop("DAPI")
    level_idx = choose_level_for_target_spacing(XE_IMG_PATH_DAPI, target_spacing_um=2)
    # Load DAPI
    dapi, meta_target = load_downsampled_image(XE_IMG_PATH_DAPI, level_index=level_idx)
    # Initialize channels_raw with DAPI
    channels_raw = {"DAPI": dapi}
    # Calculate pyramidal offset
    offset_x, offset_y = calculate_pyramidal_offset(
        XE_IMG_PATH_DAPI, meta_target, level_index=level_idx
    )
    # Load other channels
    channels_raw.update({
        name: load_downsampled_image(path, level_index=level_idx)[0]
        for name, path in XE_IMG_PATH.items()
    })
    # Prepare Xenium channels and generate combinations
    channels_proc, combos = prepare_xe_generate_combination(channels_raw)
    # Define the composite image to match
    channels_to_combine = combos[combo_name]
    proc_target = combine_xenium_channels(channels_proc, channels_to_combine)
    # Rotate the target image if needed
    proc_target = flip_if_needed(proc_source, proc_target, meta_target, name="proc_target")
    # Create output directory based on the combo of channels
    combo_dir = os.path.join(output_dir, combo_name)
    if not os.path.exists(combo_dir):
        os.makedirs(combo_dir)

    # Return all necessary data and metadata
    return (
        meta_source,
        meta_target,
        offset_x,
        offset_y,
        proc_source,
        proc_target,
        combo_dir,
        combo_name,
    )
