import os
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)

def run_registration(fixed, moving, combo_dir, ms):
    # Step 1: Execute global alignment (Rigid)
    logger.info(f"Running rigid registration...")
    tx_rigid, moving_rigid = rigid_registration(fixed, moving)
    sitk.WriteTransform(tx_rigid, os.path.join(combo_dir, f"transformation_rigid_{ms}.tfm"))
    # Step 2: Execute local refinement (BSpline)
    logger.info(f"Running bspline registration...")
    tx_bspline, moving_bspline = bspline_registration(fixed, moving_rigid, ms)
    sitk.WriteTransform(tx_bspline, os.path.join(combo_dir, f"transformation_bspline_{ms}.tfm"))
    
    return moving_rigid, tx_rigid, moving_bspline, tx_bspline

def rigid_registration(fixed, moving):
    ## Rigid reg
    R_rigid = sitk.ImageRegistrationMethod()
    # Correlation similarity metric
    R_rigid.SetMetricAsCorrelation()
    # Gradient descent optimizer
    R_rigid.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4, numberOfIterations=1000, gradientMagnitudeTolerance=1e-8)
    R_rigid.SetOptimizerScalesFromIndexShift()
    # Initialize transform
    tx_rigid_init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R_rigid.SetInitialTransform(tx_rigid_init)
    R_rigid.SetInterpolator(sitk.sitkLinear)
    # Run registration
    out_tx_rigid = R_rigid.Execute(fixed, moving)
    moving_rigid = sitk.Resample(moving, fixed, out_tx_rigid, sitk.sitkLinear, 0.0)
    
    return out_tx_rigid, moving_rigid
 
def bspline_registration(fixed, moving_rigid, ms):
    ## Bspline reg
    R_bspline = sitk.ImageRegistrationMethod()
    # Correlation similarity metric
    R_bspline.SetMetricAsCorrelation()
    # Configure L-BFGS-B optimizer
    R_bspline.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-10, numberOfIterations=1000, maximumNumberOfCorrections=100, maximumNumberOfFunctionEvaluations=1000)
    # Initialize transform with specific mesh size
    tx_bspline_init = sitk.BSplineTransformInitializer(fixed, [ms, ms], 3)
    R_bspline.SetInitialTransform(tx_bspline_init, True)
    R_bspline.SetInterpolator(sitk.sitkLinear)
    # Run registration
    out_tx_bspline = R_bspline.Execute(fixed, moving_rigid)
    moving_final = sitk.Resample(moving_rigid, fixed, out_tx_bspline, sitk.sitkLinear, 0.0)
    
    return out_tx_bspline, moving_final