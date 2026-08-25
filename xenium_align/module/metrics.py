import json
import geopandas as gpd
from shapely.geometry import shape

import logging
logger = logging.getLogger(__name__)




def match_and_compute_iogt(gdf_pred, gdf_gt):
    """
    Spatial join pred/gt geometries and compute intersection-over-gt-area / centroid distance,
    keeping only the best-matching pred for each gt cell.
    """
    pairs = gdf_pred.sjoin(gdf_gt, how="inner", predicate="intersects")['index_right']

    matched = gpd.GeoDataFrame({
        'pred_idx': pairs.index,
        'gt_idx': pairs.values,
        'geometry_pred': gdf_pred.geometry.loc[pairs.index].values,
        'geometry_gt': gdf_gt.geometry.loc[pairs.values].values,
    })

    s_pred = gpd.GeoSeries(matched['geometry_pred'])
    s_gt = gpd.GeoSeries(matched['geometry_gt'])

    matched['intersection_area'] = s_pred.intersection(s_gt).area
    # Intersection over GT area
    matched['iogt'] = matched['intersection_area'] / s_gt.area
    matched['dist_error'] = s_pred.centroid.distance(s_gt.centroid)

    # Keep only the best-matching pred per gt cell
    best_matches = matched.sort_values('iogt', ascending=False).drop_duplicates(subset='gt_idx', keep='first')

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
    Show IoGT metrics
    """
    mean_iogt = best_matches['iogt'].mean()
    median_iogt = best_matches['iogt'].median()
    success_rate = (best_matches['iogt'] > 0.5).sum() / len(gdf_pred) * 100
    logger.info(f"Mean IoGT: {mean_iogt:.4f}")
    logger.info(f"Median IoGT: {median_iogt:.4f}")
    logger.info(f"Success Rate (IoGT > 0.5): {success_rate:.2f}%")
    logger.info(f"Average GT Area: {gdf_gt.geometry.area.mean():.2f}")
    logger.info(f"Average Pred Area: {gdf_pred.geometry.area.mean():.2f}")
