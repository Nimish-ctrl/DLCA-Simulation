"""
generate_analysis_plots.py
==========================
Generates 4 publication-quality analysis plots:
1. Circles: 3D Surface Mesh Plot (Area Fraction, Number of Steps, Number of Clusters)
2. Circles: 2D Plot (Area Fraction vs Number of Final Clusters)
3. Triangles: 3D Surface Mesh Plot (Area Fraction, Number of Steps, Number of Clusters)
4. Triangles: 2D Plot (Area Fraction vs Number of Final Clusters)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0


def plot_circles():
    csv_path = os.path.join('Data points', 'circles', 'manifest.csv')
    df = pd.read_csv(csv_path).sort_values(by='area_fraction').reset_index(drop=True)

    checkpoints = [500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]
    cp_cols = [f'clusters_at_step_{cp}' for cp in checkpoints]

    # -------------------------------------------------------------
    # 1. Circles 3D Surface Mesh Plot
    # -------------------------------------------------------------
    raw_pts = []
    for i, row in df.iterrows():
        phi = row['area_fraction']
        for cp, col in zip(checkpoints, cp_cols):
            z = float(row[col])
            raw_pts.append((phi, np.log10(cp), z))

    pts = np.array(raw_pts)

    grid_phi = np.linspace(df['area_fraction'].min(), df['area_fraction'].max(), 75)
    grid_log_steps = np.linspace(np.log10(checkpoints[0]), np.log10(checkpoints[-1]), 75)
    P_grid, S_grid = np.meshgrid(grid_phi, grid_log_steps)

    Z_grid = griddata(pts[:, :2], pts[:, 2], (P_grid, S_grid), method='cubic')
    Z_nan = np.isnan(Z_grid)
    if np.any(Z_nan):
        Z_linear = griddata(pts[:, :2], pts[:, 2], (P_grid, S_grid), method='linear')
        Z_grid[Z_nan] = Z_linear[Z_nan]
    Z_grid = np.clip(Z_grid, 1.0, None)

    fig = plt.figure(figsize=(13.5, 9.5), dpi=250)
    ax = fig.add_subplot(111, projection='3d')

    # Aesthetic backdrop panes
    ax.xaxis.pane.fill = True; ax.yaxis.pane.fill = True; ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.98, 0.85))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.98, 0.85))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.96, 0.85))
    ax.xaxis.pane.set_edgecolor('#dcdde1')
    ax.yaxis.pane.set_edgecolor('#dcdde1')
    ax.zaxis.pane.set_edgecolor('#dcdde1')

    # 3D Surface Mesh
    surf = ax.plot_surface(
        P_grid, S_grid, Z_grid,
        cmap=cm.turbo,
        edgecolor='#2c3e50',
        linewidth=0.22,
        alpha=0.88,
        antialiased=True,
        rstride=2,
        cstride=2
    )

    # Floor Contour Projections
    ax.contour(
        P_grid, S_grid, Z_grid,
        zdir='z', offset=0,
        cmap=cm.turbo,
        levels=15,
        linewidths=1.2,
        alpha=0.65
    )

    # Empirical data lines and scatter points
    for i, row in df.iterrows():
        phi = row['area_fraction']
        z_vals = [float(row[col]) for col in cp_cols]
        log_s = [np.log10(cp) for cp in checkpoints]
        ax.plot(np.full_like(log_s, phi), log_s, z_vals, color='#1e272e', linewidth=1.4, alpha=0.85, zorder=10)
        ax.scatter(np.full_like(log_s, phi), log_s, z_vals, color='#eb4d4b', s=15, alpha=0.9, zorder=11)

    tick_steps = [500, 1000, 5000, 20000, 50000, 100000, 200000]
    tick_log = [np.log10(s) for s in tick_steps]
    tick_labels = ['500', '1k', '5k', '20k', '50k', '100k', '200k']
    ax.set_yticks(tick_log)
    ax.set_yticklabels(tick_labels, fontsize=10)

    ax.set_xlabel(r'Area Fraction ($\phi$)', fontsize=11, fontweight='bold', labelpad=14)
    ax.set_ylabel('Simulation Steps', fontsize=11, fontweight='bold', labelpad=14)
    ax.set_zlabel('Number of Clusters', fontsize=11, fontweight='bold', labelpad=12)
    ax.set_title('Circle DLCA: 3D Surface Mesh of Cluster Aggregation Dynamics\nArea Fraction vs Steps vs Cluster Count', fontsize=14, fontweight='bold', pad=25)

    ax.set_zlim(0, Z_grid.max() * 1.05)
    ax.view_init(elev=24, azim=-62)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=16, pad=0.1)
    cbar.set_label('Cluster Count', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    ax.grid(True, linestyle=':', alpha=0.45)
    out_3d = os.path.join('Data points', 'circles', 'plot_3d_circles_phi_steps_clusters.png')
    plt.savefig(out_3d, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_3d}")

    # -------------------------------------------------------------
    # 2. Circles 2D Plot: Area Fraction vs Final Clusters
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=250)
    final_clusters = df['final_clusters'].values
    req_n = df['requested_particles'].values
    phis = df['area_fraction'].values

    ax.axhspan(0.8, 1.2, color='#2ecc71', alpha=0.22, label=r'Fully Coalesced Single Cluster ($N_{c} = 1$)')
    ax.plot(phis, final_clusters, marker='o', markersize=7.5, color='#1f77b4', linewidth=2.4, label='Final Cluster Count', zorder=4)

    y_offsets = [8, 14, -14, 10, -12, 10, 8, 8, -12, 8, 8, 8, 8]
    x_offsets = [0, 2, -2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]

    for i, (p, fc, n) in enumerate(zip(phis, final_clusters, req_n)):
        yo = y_offsets[i] if i < len(y_offsets) else 8
        xo = x_offsets[i] if i < len(x_offsets) else 0
        ax.annotate(
            f"N={n}\n({fc})", (p, fc),
            textcoords="offset points",
            xytext=(xo, yo),
            ha='center',
            fontsize=8.5,
            fontweight='semibold',
            color='#1a252f',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#bdc3c7', alpha=0.8, linewidth=0.6)
        )

    ax.set_xlabel(r'Area Fraction ($\phi$)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Number of Final Clusters', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Circle DLCA: Area Fraction vs Number of Final Clusters\n($N = 100$ to $9000$ | Box $3200 \\times 2400$)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(-1, max(final_clusters) + 12)
    ax.set_xlim(-0.015, max(phis) + 0.025)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper right')

    out_2d = os.path.join('Data points', 'circles', 'plot_2d_circles_phi_vs_final_clusters.png')
    plt.tight_layout()
    plt.savefig(out_2d, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_2d}")


def plot_triangles():
    csv_path = os.path.join('Data points', 'triangles', 'manifest.csv')
    df = pd.read_csv(csv_path).sort_values(by='area_fraction').reset_index(drop=True)

    checkpoints = [500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]
    cp_cols = [f'clusters_at_step_{cp}' for cp in checkpoints]

    # -------------------------------------------------------------
    # 3. Triangles 3D Surface Mesh Plot
    # -------------------------------------------------------------
    raw_pts = []
    for i, row in df.iterrows():
        phi = row['area_fraction']
        for cp, col in zip(checkpoints, cp_cols):
            val = row[col]
            z = float(val) if pd.notnull(val) and val != '' else 1.0
            raw_pts.append((phi, np.log10(cp), z))

    pts = np.array(raw_pts)

    grid_phi = np.linspace(df['area_fraction'].min(), df['area_fraction'].max(), 75)
    grid_log_steps = np.linspace(np.log10(checkpoints[0]), np.log10(checkpoints[-1]), 75)
    P_grid, S_grid = np.meshgrid(grid_phi, grid_log_steps)

    Z_grid = griddata(pts[:, :2], pts[:, 2], (P_grid, S_grid), method='cubic')
    Z_nan = np.isnan(Z_grid)
    if np.any(Z_nan):
        Z_linear = griddata(pts[:, :2], pts[:, 2], (P_grid, S_grid), method='linear')
        Z_grid[Z_nan] = Z_linear[Z_nan]
    Z_grid = np.clip(Z_grid, 1.0, None)

    fig = plt.figure(figsize=(13.5, 9.5), dpi=250)
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.pane.fill = True; ax.yaxis.pane.fill = True; ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.98, 0.85))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.98, 0.85))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.96, 0.85))
    ax.xaxis.pane.set_edgecolor('#dcdde1')
    ax.yaxis.pane.set_edgecolor('#dcdde1')
    ax.zaxis.pane.set_edgecolor('#dcdde1')

    surf = ax.plot_surface(
        P_grid, S_grid, Z_grid,
        cmap=cm.viridis,
        edgecolor='#2c3e50',
        linewidth=0.22,
        alpha=0.88,
        antialiased=True,
        rstride=2,
        cstride=2
    )

    ax.contour(
        P_grid, S_grid, Z_grid,
        zdir='z', offset=0,
        cmap=cm.viridis,
        levels=15,
        linewidths=1.2,
        alpha=0.65
    )

    for i, row in df.iterrows():
        phi = row['area_fraction']
        z_vals = [float(row[col]) if pd.notnull(row[col]) and row[col] != '' else 1.0 for col in cp_cols]
        log_s = [np.log10(cp) for cp in checkpoints]
        ax.plot(np.full_like(log_s, phi), log_s, z_vals, color='#1e272e', linewidth=1.4, alpha=0.85, zorder=10)
        ax.scatter(np.full_like(log_s, phi), log_s, z_vals, color='#e67e22', s=15, alpha=0.9, zorder=11)

    tick_steps = [500, 1000, 5000, 20000, 50000, 100000, 200000]
    tick_log = [np.log10(s) for s in tick_steps]
    tick_labels = ['500', '1k', '5k', '20k', '50k', '100k', '200k']
    ax.set_yticks(tick_log)
    ax.set_yticklabels(tick_labels, fontsize=10)

    ax.set_xlabel(r'Area Fraction ($\phi$)', fontsize=11, fontweight='bold', labelpad=14)
    ax.set_ylabel('Simulation Steps', fontsize=11, fontweight='bold', labelpad=14)
    ax.set_zlabel('Number of Clusters', fontsize=11, fontweight='bold', labelpad=12)
    ax.set_title('Equilateral Triangles DLCA: 3D Surface Mesh of Aggregation Dynamics\nArea Fraction vs Steps vs Cluster Count', fontsize=14, fontweight='bold', pad=25)

    ax.set_zlim(0, Z_grid.max() * 1.05)
    ax.view_init(elev=24, azim=-62)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=16, pad=0.1)
    cbar.set_label('Cluster Count', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    ax.grid(True, linestyle=':', alpha=0.45)
    out_3d = os.path.join('Data points', 'triangles', 'plot_3d_triangles_phi_steps_clusters.png')
    plt.savefig(out_3d, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_3d}")

    # -------------------------------------------------------------
    # 4. Triangles 2D Plot: Area Fraction vs Final Clusters
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=250)
    final_clusters = df['final_clusters'].values
    req_n = df['requested_particles'].values
    phis = df['area_fraction'].values

    ax.axhspan(0.8, 1.2, color='#27ae60', alpha=0.22, label=r'Fully Coalesced Single Cluster ($N_{c} = 1$)')
    ax.plot(phis, final_clusters, marker='s', markersize=7.5, color='#d35400', linewidth=2.4, label='Final Cluster Count', zorder=4)

    y_offsets_tri = [10, -18, 10, -16, 10, -14, 10, 10, -14, 10, 10]
    x_offsets_tri = [-2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0]

    for i, (p, fc, n) in enumerate(zip(phis, final_clusters, req_n)):
        yo = y_offsets_tri[i] if i < len(y_offsets_tri) else 10
        xo = x_offsets_tri[i] if i < len(x_offsets_tri) else 0
        ax.annotate(
            f"N={n}\n({fc})", (p, fc),
            textcoords="offset points",
            xytext=(xo, yo),
            ha='center',
            fontsize=8.5,
            fontweight='semibold',
            color='#1a252f',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#bdc3c7', alpha=0.8, linewidth=0.6)
        )

    ax.set_xlabel(r'Area Fraction ($\phi$)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Number of Final Clusters', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Equilateral Triangles DLCA: Area Fraction vs Number of Final Clusters\n($N = 100$ to $5000$ | Box $1150 \\times 870$)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(-0.5, max(final_clusters) + 2.5)
    ax.set_xlim(-0.015, max(phis) + 0.025)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper right')

    out_2d = os.path.join('Data points', 'triangles', 'plot_2d_triangles_phi_vs_final_clusters.png')
    plt.tight_layout()
    plt.savefig(out_2d, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_2d}")


if __name__ == '__main__':
    plot_circles()
    plot_triangles()
    print("\nAll 4 analysis plots successfully updated!")
