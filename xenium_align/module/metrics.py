import json
import geopandas as gpd
from shapely.geometry import shape

import logging
logger = logging.getLogger(__name__)




def match_and_compute_iou(gdf_pred, gdf_gt):
    """
    Perform spatial join and calculate IoU/Distances.
    """
    # Spatial Join
    matched = gpd.sjoin(gdf_pred, gdf_gt[['gt_idx', 'cell_id', 'class', 'geometry']], how="inner", predicate="intersects")
    # Strict alignment via Merge
    matched = matched.merge(
        gdf_gt[['gt_idx', 'cell_id', 'class', 'geometry']],
        on='gt_idx',
        suffixes=('_pred', '_gt')
    )
    s_pred = gpd.GeoSeries(matched['geometry_pred'])
    s_gt = gpd.GeoSeries(matched['geometry_gt'])
    # Compute IoU (Intersection and Union area)
    intersection_area = s_pred.intersection(s_gt).area
    union_area = s_pred.union(s_gt).area
    matched['intersection_area'] = intersection_area
    matched['iou'] = intersection_area / union_area
    # Centroid Distance (Physical Shift)
    matched['dist_error'] = s_pred.centroid.distance(s_gt.centroid)
    best_matches = resolve_matches(matched)
    summary_statistics(best_matches, gdf_pred, gdf_gt)
    
    return best_matches

def resolve_matches(matched):
    """
    Keep only the best match for each pred_idx and gt_idx
    (Based on the intersection area).
    """
    # Sort by intersection_area descending (best matches at the top)
    matched_sorted = matched.sort_values('intersection_area', ascending=False)
    # Keep only the best match for each pred_idx
    best_matched_pred = matched_sorted.drop_duplicates(subset=['pred_idx'], keep='first')
    # Keep only the best match for each gt_idx
    best_matches = best_matched_pred.drop_duplicates(subset=['gt_idx'], keep='first')
    
    return best_matches

def summary_statistics(best_matches, gdf_pred, gdf_gt):
    """
    Show IoU metrics
    """
    mean_iou = best_matches['iou'].mean()
    success_rate = (best_matches['iou'] > 0.5).sum() / len(gdf_pred) * 100
    print(mean_iou)
    logger.info(f"Mean IoU: {mean_iou:.4f}")
    logger.info(f"Success Rate (IoU > 0.5): {success_rate:.2f}%")
    logger.info(f"Average GT Area: {gdf_gt.geometry.area.mean():.2f}")
    logger.info(f"Average Pred Area: {gdf_pred.geometry.area.mean():.2f}")
