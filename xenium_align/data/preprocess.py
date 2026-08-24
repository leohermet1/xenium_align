import numpy as np
from skimage.color import separate_stains, hed_from_rgb
from skimage.exposure import rescale_intensity
from itertools import combinations
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)

def prepare_he(image_rgb, signal = 99):
    # Deconvolve RGB image to HED space
    ## HED = closest match to HES (no exact matrix available); works fine
    ## here since we only keep the hematoxylin channel, shared with HE/HES
    stains = separate_stains(image_rgb, hed_from_rgb)
    hematoxylin = stains[:, :, 0]
    if signal:
        # Remove noise by saturating the top 1% pixels by default
        threshold = np.percentile(hematoxylin, signal)
        hematoxylin[hematoxylin >= threshold] = 0
    # Normalize intensity
    array_he = rescale_intensity(hematoxylin, out_range=(0, 255))
    return array_he

def prepare_mif(image_data):
    # Normalize single channel intensity
    array_if = rescale_intensity(image_data, out_range=(0, 255))

    return array_if

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
    channels_proc = {k: prepare_mif(v) for k, v in channels_raw.items()}
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

def flip_if_needed(fixed, moving, meta, name="image", size=128):
    meta['flip_extent'] = np.array([moving.shape[1] * meta['spacing'][0], moving.shape[0] * meta['spacing'][1]])
    meta['flipped'] = False

    def thumbnail(arr):
        step_y = max(1, arr.shape[0] // size)
        step_x = max(1, arr.shape[1] // size)
        t = arr[::step_y, ::step_x].astype(np.float32)
        t -= t.mean()
        std = t.std()
        return t / std if std > 0 else t

    t_fixed = thumbnail(fixed)
    t_moving = thumbnail(moving)

    h = min(t_fixed.shape[0], t_moving.shape[0])
    w = min(t_fixed.shape[1], t_moving.shape[1])
    t_fixed, t_moving = t_fixed[:h, :w], t_moving[:h, :w]
    t_moving_180 = t_moving[::-1, ::-1]

    corr_0 = np.corrcoef(t_fixed.ravel(), t_moving.ravel())[0, 1]
    corr_180 = np.corrcoef(t_fixed.ravel(), t_moving_180.ravel())[0, 1]
    
    if corr_180 > corr_0:
        logger.info(f"Flipping {name}")
        meta["flipped"] = True
        moving = moving[::-1, ::-1]

    return moving