import matplotlib
try:
    get_ipython()  # existe seulement dans un contexte Jupyter/IPython
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd

def plot_spatial_alignment(best_matches, output_path="alignment_report_scatter.png", dpi=300, title=None, ax=None):
    """
    Generates a scatter plot of cells colored by their IoGT value 
    to visualize spatial alignment quality.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get centroids
    centroids = gpd.GeoSeries(best_matches['geometry_pred']).centroid
    
    # Create scatter plot: color represents IoGT quality
    sc = ax.scatter(
        centroids.x, 
        centroids.y, 
        c=best_matches['iogt'], 
        cmap='RdYlGn', 
        s=0.05, 
        alpha=0.6
    )
    plt.colorbar(sc, ax=ax, label='IoGT Value')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    ax.invert_yaxis()

    if title:
        ax.set_title(title)
    
    if standalone:
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)