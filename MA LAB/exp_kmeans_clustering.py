
#
#  Aim:
#  Implement K-Means Clustering in Python using Scikit-learn
#  on the given dataset with K=2 clusters and initial
#  centroids A(2,3) and C(6,6).
#
#  Dataset:
#  Datapoint | X | Y
#     A      | 2 | 3
#     B      | 3 | 4
#     C      | 6 | 6
#     D      | 7 | 7
#
#  K = 2
#  Initial Centroids: A(2,3) and C(6,6)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── Dataset ──────────────────────────────────────────────────
X      = np.array([[2, 3], [3, 4], [6, 6], [7, 7]])
labels = ['A', 'B', 'C', 'D']

initial_centroids = np.array([[2, 3], [6, 6]])

# ── K-Means Model ────────────────────────────────────────────
kmeans = KMeans(n_clusters=2,
                init=initial_centroids,
                n_init=1,
                max_iter=300,
                random_state=42)
kmeans.fit(X)

cluster_labels  = kmeans.labels_
final_centroids = kmeans.cluster_centers_

# ── Results ──────────────────────────────────────────────────
print("=" * 45)
print("      K-MEANS CLUSTERING  (K=2)")
print("=" * 45)
print("\n  Cluster Assignments:")
for i, (name, cl) in enumerate(zip(labels, cluster_labels)):
    print(f"    Point {name} {X[i]} → Cluster {cl + 1}")

print("\n  Initial Centroids:")
for i, c in enumerate(initial_centroids):
    print(f"    C{i+1}: ({c[0]}, {c[1]})")

print("\n  Final Centroids:")
for i, c in enumerate(final_centroids):
    print(f"    C{i+1}: ({c[0]:.2f}, {c[1]:.2f})")

print(f"\n  Inertia (WCSS) : {kmeans.inertia_:.4f}")
print(f"  Iterations     : {kmeans.n_iter_}")
print("=" * 45)

# ── Visualization ────────────────────────────────────────────
cluster_colors = ['#e74c3c', '#3498db']   # Red=Cluster1, Blue=Cluster2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1 — Final clustering
for i, (pt, cl, name) in enumerate(zip(X, cluster_labels, labels)):
    axes[0].scatter(*pt, color=cluster_colors[cl], s=200, zorder=5,
                    edgecolors='black', linewidths=1)
    axes[0].annotate(f'  {name}', pt, fontsize=12, fontweight='bold')

axes[0].scatter(final_centroids[:, 0], final_centroids[:, 1],
                marker='X', s=350, c='black', zorder=6, label='Final Centroids')
axes[0].scatter(initial_centroids[:, 0], initial_centroids[:, 1],
                marker='o', s=300, facecolors='none',
                edgecolors='green', linewidths=2.5,
                zorder=6, label='Initial Centroids')
axes[0].set_title('K-Means Clustering — Final Result')
axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot 2 — Cluster membership pie chart
cluster_counts = np.bincount(cluster_labels)
axes[1].pie(cluster_counts,
            labels=[f'Cluster {i+1}\n({cluster_counts[i]} pts)' for i in range(2)],
            colors=cluster_colors, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Cluster Distribution')

plt.suptitle('Experiment 7 - K-Means Clustering (K=2)', fontsize=13)
plt.tight_layout()
plt.show()
