import numpy as np
from skimage.color import separate_stains, hed_from_rgb
from skimage.exposure import rescale_intensity
from itertools import combinations
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)

def prepare_he(image_rgb):
    # Deconvolve RGB image to HED space
    stains = separate_stains(image_rgb, hed_from_rgb)
    hematoxylin = stains[:, :, 0]
    # Remove noise by saturating the top 1% pixels
    threshold = np.percentile(hematoxylin, 99)
    hematoxylin[hematoxylin >= threshold] = 0
    # Normalize intensity
    array_he = rescale_intensity(hematoxylin, out_range=(0, 255))

    return array_he

def prepare_xe(image_data):
    # Normalize single channel intensity
    array_xe = rescale_intensity(image_data, out_range=(0, 255))

    return array_xe

def combine_xenium_channels(channels_arrays, channels_to_combine):
    logger.info(f"Combine {channels_to_combine}...")
    # Sum selected channels into a single array
    array_xe = np.sum([channels_arrays[c] for c in channels_to_combine], axis=0)

    return array_xe

def prepare_xe_generate_combination(channels_raw, reference_key="DAPI"):
    """
    Get the different image combinations with DAPI channel.
    The combo that visually aligns the most with the Hematoxylin channel is 'DAPI_ATP1A'.
    """
    # Apply individual preprocessing to each raw channel
    channels_proc = {k: prepare_xe(v) for k, v in channels_raw.items()}
    # Generate all possible combinations containing the reference key
    keys = list(channels_proc.keys())
    combos = {}
    for r in range(1, len(keys) + 1):
        for combo_keys in combinations(keys, r):
            if reference_key in combo_keys:
                combo_name = "_".join(combo_keys)
                combos[combo_name] = list(combo_keys)
    
    return channels_proc, combos

def get_sitk_image(lowres_array, meta):
    # Set as SimpleITK image
    img_sitk = sitk.GetImageFromArray(lowres_array)
    # Set new spacing to the image
    img_sitk.SetSpacing(meta['spacing'])

    return img_sitk