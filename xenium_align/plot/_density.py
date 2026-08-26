import matplotlib
try:
    get_ipython()  # existe seulement dans un contexte Jupyter/IPython
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_iogt_distribution(best_matches, output_path="iogt_distribution_report.png", bins=100, dpi=300):
    """
    Generates a histogram with a KDE curve to visualize the distribution 
    of Intersection over Union (IoGT) values across all matches.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot IoGT Distribution using Seaborn
    sns.histplot(
        best_matches['iogt'], 
        bins=bins, 
        kde=True, 
        ax=ax, 
        color='blue'
    )
    ax.set_title('Global Similarity Distribution (IoGT)')
    ax.set_xlabel('Intersection over Union')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)

def plot_iogt_distribution_comp(best_matches_1, label_1, best_matches_2, label_2, output_path="iogt_comparison.png", dpi=300):
    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: IoGT Comparison
    sns.kdeplot(best_matches_1['iogt'], fill=True, color="blue", label=f"{label_1} (Median: {best_matches_1['iogt'].median():.3f})", ax=axes[0])
    sns.kdeplot(best_matches_2['iogt'], fill=True, color="red", label=f"{label_2} (Median: {best_matches_2['iogt'].median():.3f})", ax=axes[0])
    axes[0].set_title("IoGT Distribution")
    axes[0].set_xlabel("Intersection over Union")
    axes[0].legend()
    
    # Plot 2: Centroid Distance Comparison (The "True" Alignment Metric)
    sns.kdeplot(best_matches_1['dist_error'], fill=True, color="blue", label=f"{label_1} (Median Shift: {best_matches_1['dist_error'].median():.2f})", ax=axes[1])
    sns.kdeplot(best_matches_2['dist_error'], fill=True, color="red", label=f"{label_2} (Median Shift: {best_matches_2['dist_error'].median():.2f})", ax=axes[1])
    axes[1].set_xlim(0, 30)
    axes[1].set_title("Centroid Distance Distribution")
    axes[1].set_xlabel("Distance between Nuclei Centers (µm)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)