#  Aim:
#  Apply agglomerative clustering approach to create a
#  dendrogram for the given points using single linkage.
#
#  Dataset:
#  Point | X | Y
#    A   | 1 | 1
#    B   | 2 | 1
#    C   | 4 | 3
#    D   | 5 | 4
#
#  Objective:
#  - Group data points into clusters by iteratively merging
#    the closest cluster pair (single linkage)
#  - Visualize the process through a dendrogram
#
#  Linkage Method: Single (minimum pairwise distance)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Dataset ──────────────────────────────────────────────────
points_label = ['A', 'B', 'C', 'D']
points_value = np.array([[1, 1],
                          [2, 1],
                          [4, 3],
                          [5, 4]])

print("=" * 45)
print("  AGGLOMERATIVE (HIERARCHICAL) CLUSTERING")
print("=" * 45)
print("\n  Data Points and Values:")
for label, val in zip(points_label, points_value):
    print(f"    {label}: {val.tolist()}")

# ── Agglomerative Model ──────────────────────────────────────
model = AgglomerativeClustering(n_clusters=2, linkage='single')
model.fit(points_value)

cluster_labels = model.labels_
num_clusters   = len(np.unique(cluster_labels))

print("\n" + "=" * 45)
print("  CLUSTERING RESULTS")
print("=" * 45)
print(f"\n  Cluster Labels : {cluster_labels.tolist()}")

for i in range(num_clusters):
    pts  = [points_label[j] for j, lbl in enumerate(cluster_labels) if lbl == i]
    vals = [points_value[j].tolist() for j, lbl in enumerate(cluster_labels) if lbl == i]
    print(f"\n  Cluster {i+1}:")
    print(f"    Points : {pts}")
    print(f"    Values : {vals}")

print("=" * 45)

# ── Pairwise Distance Matrix ─────────────────────────────────
from scipy.spatial.distance import cdist
dist_matrix = cdist(points_value, points_value, metric='euclidean')
print("\n  Pairwise Distance Matrix:")
header = "       " + "  ".join(f"{l:>5}" for l in points_label)
print(header)
for i, row in enumerate(dist_matrix):
    print(f"  {points_label[i]}  " + "  ".join(f"{v:5.2f}" for v in row))

# ── Visualization ────────────────────────────────────────────
cluster_colors = ['#3498db', '#e74c3c']   # Blue=Cluster1, Red=Cluster2
point_colors   = [cluster_colors[lbl] for lbl in cluster_labels]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1 — Scatter with cluster coloring
ax = axes[0]
for i, (pt, c, name) in enumerate(zip(points_value, point_colors, points_label)):
    ax.scatter(*pt, color=c, s=250, zorder=5,
               edgecolors='black', linewidths=1.2)
    ax.annotate(f'  {name}({pt[0]},{pt[1]})', pt, fontsize=11, fontweight='bold')

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=cluster_colors[i], label=f'Cluster {i+1}')
              for i in range(num_clusters)]
ax.legend(handles=legend_els)
ax.set_title('Agglomerative Clustering — Final Clusters')
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(0, 7); ax.set_ylim(0, 6)

# Plot 2 — Dendrogram
Z = linkage(points_value, method='single')
ax2 = axes[1]
dendrogram(Z, labels=points_label, ax=ax2,
           color_threshold=2.0,
           link_color_func=lambda k: '#555555')
ax2.set_title('Dendrogram (Single Linkage)')
ax2.set_xlabel('Points')
ax2.set_ylabel('Distance (Euclidean)')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Experiment 8 - Agglomerative (Hierarchical) Clustering', fontsize=13)
plt.tight_layout()
plt.show()
