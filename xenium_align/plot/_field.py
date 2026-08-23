import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ._utils import boost_intensity, get_jacobian_cmap

def _get_bspline_grid(fixed_img, outTx_Bspline, mesh_size, spline_order):
    # Extract grid from BSpline coefficients
    x_coeff, _ = outTx_Bspline.GetCoefficientImages()
    grid_origin = x_coeff.GetOrigin()
    grid_spacing = x_coeff.GetSpacing()
    
    ctrl_pts = mesh_size + spline_order
    
    # Create meshgrid in physical space
    x = np.linspace(grid_origin[0], grid_origin[0] + (ctrl_pts - 1) * grid_spacing[0], ctrl_pts)
    y = np.linspace(grid_origin[1], grid_origin[1] + (ctrl_pts - 1) * grid_spacing[1], ctrl_pts)
    xx_um, yy_um = np.meshgrid(x, y)
    
    # Map physical coordinates to fixed image pixel space
    xx_px = (xx_um - fixed_img.GetOrigin()[0]) / fixed_img.GetSpacing()[0]
    yy_px = (yy_um - fixed_img.GetOrigin()[1]) / fixed_img.GetSpacing()[1]
    
    # Reshape transformation parameters to 2D displacement vectors
    params = np.array(outTx_Bspline.GetParameters())
    num_params = len(params) // 2
    u = params[:num_params].reshape((ctrl_pts, ctrl_pts))
    v = params[num_params:].reshape((ctrl_pts, ctrl_pts))
    
    return xx_px, yy_px, u, v

def _compute_jacobian_map(fixed_img, outTx_Bspline):
    # Generate displacement field
    displacement_field = sitk.TransformToDisplacementField(
        outTx_Bspline, sitk.sitkVectorFloat64, fixed_img.GetSize(), 
        fixed_img.GetOrigin(), fixed_img.GetSpacing(), fixed_img.GetDirection()
    )
    # Calculate local determinant of the Jacobian matrix
    jacobian_det = sitk.DisplacementFieldJacobianDeterminant(displacement_field)
    return sitk.GetArrayFromImage(jacobian_det)

def _compute_displacement_magnitude_map(fixed_img, outTx_Bspline):
    # Generate displacement field (physical units = microns, same as fixed_img spacing)
    displacement_field = sitk.TransformToDisplacementField(
        outTx_Bspline, sitk.sitkVectorFloat64, fixed_img.GetSize(),
        fixed_img.GetOrigin(), fixed_img.GetSpacing(), fixed_img.GetDirection()
    )
    disp_arr = sitk.GetArrayFromImage(displacement_field)  # (H, W, 2) -> [dx, dy] in microns
    return np.sqrt(disp_arr[..., 0]**2 + disp_arr[..., 1]**2)

def local_deformations(fixed_img, outTx_Bspline, output_dir, mesh_size = 10, spline_order = 3, inverse = False):
    # Prepare background image
    fixed_boosted = sitk.GetArrayFromImage(boost_intensity(fixed_img))
    # Prepare local deformation visualizations
    xx_px, yy_px, u, v = _get_bspline_grid(fixed_img, outTx_Bspline, mesh_size, spline_order)
    jac_arr = _compute_jacobian_map(fixed_img, outTx_Bspline)
    disp_arr = _compute_displacement_magnitude_map(fixed_img, outTx_Bspline)
    
    fig, axs = plt.subplots(1, 3, figsize=(37, 12))
    # Quiver plot of displacements
    axs[0].imshow(fixed_boosted, cmap='Reds', alpha=0.8)
    axs[0].quiver(xx_px, yy_px, -u, -v, color='black', units='xy', angles='xy', 
                  scale_units='xy', scale=1, width=8, headwidth=4, headlength=5)
    axs[0].scatter(xx_px, yy_px, color='blue', s=5, alpha=0.7)
    axs[0].set_title(f"Displacement field (Mesh: {mesh_size})")
    axs[0].axis('off')
    # Displacement magnitude heatmap (microns)
    im = axs[1].imshow(disp_arr, cmap='inferno', vmin=0, vmax=np.percentile(disp_arr, 99))
    plt.colorbar(im, ax=axs[1], shrink=0.6, label='Displacement (µm)')
    axs[1].imshow(fixed_boosted, alpha=0.2, cmap='gray')
    axs[1].set_title("Displacement Magnitude")
    axs[1].axis('off')
    # Jacobian determinant heatmap
    im = axs[2].imshow(jac_arr, cmap=get_jacobian_cmap(), vmin=0.9, vmax=1.1)
    cbar = plt.colorbar(im, ax=axs[2], shrink=0.6)
    cbar.set_ticks([0.9, 1.0, 1.1])
    cbar.set_ticklabels(['Expansion', '', 'Compression'])
    axs[2].imshow(fixed_boosted, alpha=0.2, cmap='gray')
    axs[2].set_title("Jacobian Determinant")
    axs[2].axis('off')
    # Save report
    plt.tight_layout()
    if inverse:
        output_path = os.path.join(output_dir, f"local_deformations_{mesh_size}_inverse.png")
    else:
        output_path = os.path.join(output_dir, f"local_deformations_{mesh_size}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()