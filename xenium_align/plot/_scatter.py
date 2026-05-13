import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd

def plot_spatial_alignment(best_matches, output_path="alignment_report_scatter.png", dpi=300):
    """
    Generates a scatter plot of cells colored by their IoU value 
    to visualize spatial alignment quality.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get centroids
    centroids = gpd.GeoSeries(best_matches['geometry_pred']).centroid
    
    # Create scatter plot: color represents IoU quality
    sc = ax.scatter(
        centroids.x, 
        centroids.y, 
        c=best_matches['iou'], 
        cmap='RdYlGn', 
        s=0.05, 
        alpha=0.6
    )
    plt.colorbar(sc, ax=ax, label='IoU Value')
    ax.set_title('Spatial Alignment Quality (Centroids)')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)