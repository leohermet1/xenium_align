import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ._utils import boost_intensity

def create_rgb_composite(fixed, moving):
    """Create an RGB array from two sitk images (Fixed=Red, Moving=Green)."""
    # Preprocess both images for visualization
    f_8 = sitk.Cast(boost_intensity(fixed), sitk.sitkUInt8)
    m_8 = sitk.Cast(boost_intensity(moving), sitk.sitkUInt8)
    
    # Convert to numpy and stack into RGB
    arr_f = sitk.GetArrayFromImage(f_8)
    arr_m = sitk.GetArrayFromImage(m_8)
    rgb = np.zeros((*arr_f.shape, 3), dtype=np.uint8)
    rgb[..., 0] = arr_f # Red
    rgb[..., 1] = arr_m # Green
    
    return rgb

def registration_summary(fixed, moving, moving_rigid, moving_bspline, combo_dir, mesh_size):
    """Generate and save a 1x3 panel comparing registration steps."""
    # Resample initial moving to fixed space for the "Initial" plot
    moving_initial = sitk.Resample(moving, fixed, sitk.Transform(), sitk.sitkLinear, 0.0)
    
    # Generate RGB composites for each stage
    rgb_initial = create_rgb_composite(fixed, moving_initial)
    rgb_rigid = create_rgb_composite(fixed, moving_rigid)
    rgb_final = create_rgb_composite(fixed, moving_bspline)
    
    # Create 1x3 diagnostic figure
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))
    
    axes[0].imshow(rgb_initial)
    axes[0].set_title("Initial")
    
    axes[1].imshow(rgb_rigid)
    axes[1].set_title("Rigid")
    
    axes[2].imshow(rgb_final)
    axes[2].set_title("BSpline")
    
    # Finalize formatting and save
    for ax in axes: 
        ax.axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(combo_dir, f"registration_{mesh_size}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def single_overlay(fixed, moving, combo_dir, filename):
    """Generate and save a single RGB composite overlay from two images."""
    # Ensure moving image is resampled to fixed space
    moving_resampled = sitk.Resample(
        moving, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    
    # Create the RGB composite (Fixed=Red/Blue, Moving=Green)
    rgb_overlay = create_rgb_composite(fixed, moving_resampled)
    
    # Render and save without axes or margins
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb_overlay)
    plt.axis('off')
    output_path = os.path.join(combo_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()