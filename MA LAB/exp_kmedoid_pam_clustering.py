#  Aim:
#  Implement the K-Medoid (PAM) clustering on a dataset of
#  5 points with K=2 and initial medoids m1=Point A (2)
#  and m2=Point D (10).
#
#  Dataset:
#  Point | Value
#    A   |   2
#    B   |   4
#    C   |   5
#    D   |  10
#    E   |  12
#
#  K = 2
#  Initial Medoids: m1 = Point A (index 0), m2 = Point D (index 3)
#
#  Algorithm (PAM):
#  1. Assign each point to nearest medoid (by |distance|)
#  2. For each cluster, pick the point minimizing total
#     intra-cluster absolute distance as the new medoid
#  3. Repeat until medoids don't change
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ── Dataset ──────────────────────────────────────────────────
data  = np.array([2, 4, 5, 10, 12])
names = ['A', 'B', 'C', 'D', 'E']
k     = 2
medoid = [0, 3]   # Initial medoid indices: A=0, D=3

print("=" * 45)
print("     K-MEDOID (PAM) CLUSTERING  (K=2)")
print("=" * 45)
print("\n  Dataset:")
for name, val in zip(names, data):
    print(f"    Point {name} : {val}")

print(f"\n  Initial Medoids: m1=Point A ({data[0]}), m2=Point D ({data[3]})")

# ── PAM Algorithm ────────────────────────────────────────────
iteration = 0
for _ in range(100):
    iteration += 1
    clusters = {i: [] for i in range(k)}
    for i, p in enumerate(data):
        nearest = np.argmin([abs(p - data[m]) for m in medoid])
        clusters[nearest].append(i)

    new_medoids = [
        min(V, key=lambda m: sum(abs(data[m] - data[j]) for j in V))
        for V in clusters.values()
    ]

    if new_medoids == medoid:
        break
    medoid = new_medoids

# ── Results ──────────────────────────────────────────────────
print(f"\n  Converged in {iteration} iteration(s)")
print("\n" + "=" * 45)
print("  CLUSTER ASSIGNMENTS")
print("=" * 45)
for i, indices in clusters.items():
    pts    = [names[j] for j in indices]
    vals   = [data[j]  for j in indices]
    med_pt = names[medoid[i]]
    med_val= data[medoid[i]]
    print(f"\n  Cluster {i+1}:")
    print(f"    Points  : {pts}")
    print(f"    Values  : {vals}")
    print(f"    Medoid  : Point {med_pt} = {med_val}")

# ── Total Cost (sum of absolute distances) ───────────────────
total_cost = sum(
    abs(data[j] - data[medoid[i]])
    for i, indices in clusters.items()
    for j in indices
)
print(f"\n  Total Cost (Σ|dist|) : {total_cost}")
print("=" * 45)

# ── Visualization ────────────────────────────────────────────
colors       = ['steelblue', 'tomato']
cluster_map  = {j: i for i, idx in clusters.items() for j in idx}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Plot 1 — 1-D scatter on number line
ax = axes[0]
for i, (v, name) in enumerate(zip(data, names)):
    c = colors[cluster_map[i]]
    ax.scatter(v, 0, color=c, s=250, zorder=3, edgecolors='black', linewidths=1)
    ax.annotate(f' {name}({v})', (v, 0), textcoords='offset points',
                xytext=(0, 12), ha='center', fontsize=11, fontweight='bold')

for ci, m in enumerate(medoid):
    ax.scatter(data[m], 0, color=colors[ci], marker='*',
               s=600, edgecolors='black', linewidths=1.5,
               zorder=5, label=f'Medoid C{ci+1} = {names[m]}({data[m]})')

ax.axhline(0, color='gray', linewidth=1)
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xlabel('Value')
ax.set_title('K-Medoid Clustering — Number Line View')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Plot 2 — Bar chart colored by cluster
ax2 = axes[1]
bar_colors = [colors[cluster_map[i]] for i in range(len(data))]
bars = ax2.bar(names, data, color=bar_colors, edgecolor='black', alpha=0.85)
for ci, m in enumerate(medoid):
    bars[m].set_edgecolor('black')
    bars[m].set_linewidth(3)
    bars[m].set_hatch('*')
    ax2.text(m, data[m] + 0.2, f'Medoid\nC{ci+1}', ha='center',
             fontsize=9, fontweight='bold', color=colors[ci])
ax2.set_xlabel('Point')
ax2.set_ylabel('Value')
ax2.set_title('Cluster Distribution (★ = Medoid)')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=f'Cluster {i+1}') for i in range(k)]
ax2.legend(handles=legend_elements)

plt.suptitle('Experiment 7B - K-Medoid (PAM) Clustering (K=2)', fontsize=13)
plt.tight_layout()
plt.show()
