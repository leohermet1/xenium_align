import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_iou_distribution(best_matches, output_path="iou_distribution_report.png", bins=100, dpi=300):
    """
    Generates a histogram with a KDE curve to visualize the distribution 
    of Intersection over Union (IoU) values across all matches.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot IoU Distribution using Seaborn
    sns.histplot(
        best_matches['iou'], 
        bins=bins, 
        kde=True, 
        ax=ax, 
        color='blue'
    )
    ax.set_title('Global Similarity Distribution (IoU)')
    ax.set_xlabel('Intersection over Union')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

def plot_iou_distribution_comp(best_matches_1, label_1, best_matches_2, label_2, output_path="iou_comparison.png", dpi=300):
    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: IoU Comparison
    sns.kdeplot(best_matches_1['iou'], fill=True, color="blue", label=f"{label_1} (Median: {best_matches_1['iou'].median():.3f})", ax=axes[0])
    sns.kdeplot(best_matches_2['iou'], fill=True, color="red", label=f"{label_2} (Median: {best_matches_2['iou'].median():.3f})", ax=axes[0])
    axes[0].set_title("IoU Distribution")
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