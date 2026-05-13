import numpy as np
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap

def boost_intensity(img, percentile=99.5):
    """Normalize and boost image intensity for visualization."""
    arr = sitk.GetArrayFromImage(img)
    upper = np.percentile(arr, percentile) 
    img_boosted = sitk.Clamp(img, lowerBound=0, upperBound=upper)
    return sitk.RescaleIntensity(img_boosted, 0, 255)

def get_jacobian_cmap():
    """Returns the custom colormap for Jacobian determinant visualization."""
    colors = [[0.0, '#2390ff'], [0.5, '#000000'], [1.0, '#ff1b37']]
    return LinearSegmentedColormap.from_list('jac_cmap', colors)