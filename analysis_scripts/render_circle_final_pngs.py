"""
render_circle_final_pngs.py
===========================
Renders final high-resolution snapshot PNG images for all circle DLCA sweep runs
(N = 100, 150, 200, 250, 300, 500, 1000, 2000, 3000, 4000, 5000, 7000, 9000).
"""

import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

runs = [
    (r"sweeps/sweep_circles_20261010_230233", 100),
    (r"sweeps/sweep_circles_20261010_230233", 150),
    (r"sweeps/sweep_circles_20261010_230233", 200),
    (r"sweeps/sweep_circles_20261010_230233", 250),
    (r"sweeps/sweep_circles_20261010_230233", 300),
    (r"sweeps/sweep_circles_20261010_230233", 500),
    (r"sweeps/sweep_circles_20261010_230233", 1000),
    (r"sweeps/sweep_circles_20261010_230233", 2000),
    (r"sweeps/sweep_circles_20261010_230233", 3000),
    (r"sweeps/sweep_circles_20261010_230233", 4000),
    (r"sweeps/sweep_circles_20261010_230233", 5000),
    (r"sweeps/sweep_circles_20261011_021023", 7000),
    (r"sweeps/sweep_circles_20261011_021023", 9000),
]

generated = []
for base, n in runs:
    sub = os.path.join(base, f"N_{n}")
    if not os.path.exists(sub):
        continue
    traj_f = glob.glob(os.path.join(sub, "traj_*.npy"))
    rad_f = glob.glob(os.path.join(sub, "radii_*.npy"))
    meta_f = glob.glob(os.path.join(sub, "meta_*.json"))
    cid_f = glob.glob(os.path.join(sub, "clusterids_*.npy"))
    if not (traj_f and rad_f and meta_f):
        continue

    with open(meta_f[0], "r") as f:
        meta = json.load(f)
    traj = np.load(traj_f[0])
    radii = np.load(rad_f[0])

    width = meta.get("width", 3200)
    height = meta.get("height", 2400)
    final_coords = traj[-1]  # shape (N, 2)

    total_area = float(np.sum(np.pi * (radii.astype(np.float64)**2)))
    domain_area = float(width * height)
    phi = total_area / domain_area
    n_clusters = meta.get("final_clusters", 1)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    patches_a = []
    patches_b = []
    for i in range(len(radii)):
        c = Circle((final_coords[i, 0], final_coords[i, 1]), radii[i])
        if radii[i] <= 8.5:
            patches_a.append(c)
        else:
            patches_b.append(c)

    p_a = PatchCollection(patches_a, facecolors="cyan", edgecolors="black", linewidths=0.3, alpha=0.85)
    p_b = PatchCollection(patches_b, facecolors="coral", edgecolors="black", linewidths=0.3, alpha=0.85)
    ax.add_collection(p_a)
    ax.add_collection(p_b)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    plt.title(f"Circle DLCA Final: N={len(radii)} | phi={phi:.5f} | Final Clusters={n_clusters}")

    out_png = os.path.join(sub, f"final_circle_N{n}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    generated.append(out_png)
    print(f"Rendered: {out_png}")

print(f"\nAll {len(generated)} final circle PNGs successfully rendered and saved!")
