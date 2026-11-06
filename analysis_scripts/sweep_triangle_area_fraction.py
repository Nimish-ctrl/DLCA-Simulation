"""
sweep_triangle_area_fraction.py — Equilateral Triangle DLCA Sweep Orchestrator
==============================================================================
Runs an automated, multi-process parameter sweep over particle counts / area fractions
for identical equilateral triangles (r = 8.0) in a 1150 x 870 box.
Logs fine-grained cluster counts, extracts step checkpoints, validates placed area fraction,
and maintains live manifest CSV/JSON.
"""

import os
import sys
import time
import json
import csv
import math
import random
import argparse
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collisions.collisions import (
    create_equilateral_triangle_vertices,
    get_triangle_bounding_radius,
    get_triangle_area,
    get_triangle_self_inertia_ratio,
    get_triangle_corners,
    get_mtv_triangle_triangle
)

# ==============================================================================
# Global Configuration
# ==============================================================================
DEFAULT_COUNTS = [100, 150, 200, 250, 300, 500, 1000, 2000, 3000, 4000, 5000]
DEFAULT_CHECKPOINTS = [500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]
DEFAULT_MAX_TIME = 200000
DEFAULT_STORE_EVERY_K = 100
DEFAULT_WIDTH = 1150
DEFAULT_HEIGHT = 870
DEFAULT_RADIUS = 8.0
DEFAULT_VELOCITY = 5.0
DEFAULT_OUTPUT_BASE = "sweeps"

EPS_CONTACT = 1.0
EPS_PUSH = 0.2
ROT_OMEGA_BASE = 0.0
ROT_BROWNIAN = 0.03


def _create_triangle_particle(r, x, y, orientation, cluster_id, particle_id):
    verts = create_equilateral_triangle_vertices(r)
    r_bound = get_triangle_bounding_radius(verts)
    area = get_triangle_area(verts)
    self_inertia = get_triangle_self_inertia_ratio(verts)
    return {
        'x': float(x),
        'y': float(y),
        'orientation': float(orientation),
        'local_vertices': verts,
        'r_bound': float(r_bound),
        'r': float(r_bound),
        'area': float(area),
        'self_inertia_ratio': float(self_inertia),
        'shape': 'triangle',
        'triangle_type': 'equilateral',
        'cluster_id': int(cluster_id),
        'cluster_members': [],
        'angle': random.uniform(0, 2 * math.pi),
        'id': int(particle_id)
    }


def _update_spatial_grid(particles, grid, cell_size):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j].clear()
    for p in particles:
        cx = int(p['x'] / cell_size)
        cy = int(p['y'] / cell_size)
        if 0 <= cx < len(grid) and 0 <= cy < len(grid[0]):
            grid[cx][cy].append(p)


def _keep_cluster_inside_bounds(cluster, width, height):
    if not cluster:
        return
    min_x = min(p['x'] - p['r_bound'] for p in cluster)
    max_x = max(p['x'] + p['r_bound'] for p in cluster)
    min_y = min(p['y'] - p['r_bound'] for p in cluster)
    max_y = max(p['y'] + p['r_bound'] for p in cluster)

    dx = dy = 0.0
    if min_x < 0.0:      dx = -min_x
    elif max_x > width:  dx = width - max_x
    if min_y < 0.0:      dy = -min_y
    elif max_y > height: dy = height - max_y

    if dx != 0.0 or dy != 0.0:
        for p in cluster:
            p['x'] += dx
            p['y'] += dy


def _enforce_world_bounds_all(particles, width, height):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)
    for cluster in clusters.values():
        _keep_cluster_inside_bounds(cluster, width, height)


def _get_cluster_properties(cluster_particles, velocity):
    total_mass = sum(p['area'] for p in cluster_particles)
    if total_mass == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    com_x = sum(p['x'] * p['area'] for p in cluster_particles) / total_mass
    com_y = sum(p['y'] * p['area'] for p in cluster_particles) / total_mass
    avg_vx = sum(velocity / p['r_bound'] * math.cos(p.get('angle', 0)) for p in cluster_particles) / len(cluster_particles)
    avg_vy = sum(velocity / p['r_bound'] * math.sin(p.get('angle', 0)) for p in cluster_particles) / len(cluster_particles)
    velocity_scale = 1.0 / (1.0 + math.log(1.0 + total_mass / 1000.0))
    avg_vx *= velocity_scale
    avg_vy *= velocity_scale
    return com_x, com_y, avg_vx, avg_vy, total_mass


def _radius_of_gyration(cluster):
    if not cluster:
        return 0.0
    masses = [p['area'] for p in cluster]
    M = sum(masses)
    if M == 0:
        return 0.0
    com_x = sum(p['x'] * m for p, m in zip(cluster, masses)) / M
    com_y = sum(p['y'] * m for p, m in zip(cluster, masses)) / M
    rg2 = sum(
        m * ((p['x'] - com_x)**2 + (p['y'] - com_y)**2 + p.get('self_inertia_ratio', 0.0))
        for p, m in zip(cluster, masses)
    ) / M
    return math.sqrt(max(0.0, rg2))


def _resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    safe_dist = (p1['r_bound'] + p2['r_bound']) + eps
    if dist_sq > safe_dist**2:
        return False

    res = get_mtv_triangle_triangle(p1, p2, eps=1e-9)
    if res is None:
        return False

    mtv, _ = res
    push_dist = math.hypot(mtv[0], mtv[1])
    if push_dist == 0:
        return False

    ux, uy = mtv[0] / push_dist, mtv[1] / push_dist
    overlap = push_dist + eps

    c1 = p1['cluster_members']
    c2 = p2['cluster_members']
    m1 = max(sum(pp['area'] for pp in c1), 1e-12)
    m2 = max(sum(pp['area'] for pp in c2), 1e-12)
    tm = m1 + m2

    s1 = (m2 / tm) * overlap
    s2 = (m1 / tm) * overlap

    for q in c1:
        q['x'] += ux * s1
        q['y'] += uy * s1
    for q in c2:
        q['x'] -= ux * s2
        q['y'] -= uy * s2

    return True


def _resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2, eps=EPS_CONTACT):
    for _ in range(passes):
        moved_any = False
        _update_spatial_grid(particles, grid, cell_size)
        processed_pairs = set()

        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x'] / cell_size), int(p1['y'] / cell_size)

            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for p2 in grid[gx][gy]:
                        if p1 is p2 or p1['cluster_id'] == p2['cluster_id']:
                            continue
                        c_pair = (min(p1['cluster_id'], p2['cluster_id']),
                                  max(p1['cluster_id'], p2['cluster_id']))
                        if c_pair in processed_pairs:
                            continue

                        rs = p1['r_bound'] + p2['r_bound']
                        dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
                        if dist_sq < (rs + eps)**2:
                            if _resolve_overlap_pair_as_clusters(p1, p2, eps=eps):
                                moved_any = True
                                processed_pairs.add(c_pair)
        if not moved_any:
            break


def _rotate_all_clusters(particles, grid, cell_size, omega_base=0.0, omega_brownian=0.03, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cluster in clusters.values():
        if not cluster:
            continue
        M = sum(p['area'] for p in cluster)
        if M <= 0.0:
            continue
        cx = sum(p['x'] * p['area'] for p in cluster) / M
        cy = sum(p['y'] * p['area'] for p in cluster) / M

        mass_damp = 1.0 / (1.0 + math.log(1.0 + M / 1000.0))
        d_theta = (omega_base + random.gauss(0.0, omega_brownian)) * mass_damp * dt

        if d_theta != 0.0:
            cos_t = math.cos(d_theta)
            sin_t = math.sin(d_theta)
            for p in cluster:
                dx = p['x'] - cx
                dy = p['y'] - cy
                p['x'] = cx + dx * cos_t - dy * sin_t
                p['y'] = cy + dx * sin_t + dy * cos_t
                p['orientation'] = (p.get('orientation', 0.0) + d_theta) % (2 * math.pi)


def _merge_clusters(particles, cluster_id1, cluster_id2, velocity):
    cluster1 = [p for p in particles if p['cluster_id'] == cluster_id1]
    cluster2 = [p for p in particles if p['cluster_id'] == cluster_id2]
    new_cid = min(cluster_id1, cluster_id2)
    other_cid = max(cluster_id1, cluster_id2)

    for p in cluster2 if new_cid == cluster_id1 else cluster1:
        p['cluster_id'] = new_cid

    merged_members = cluster1 + cluster2
    for p in merged_members:
        p['cluster_members'] = merged_members

    mass1 = sum(p['area'] for p in cluster1)
    mass2 = sum(p['area'] for p in cluster2)
    total_mass = mass1 + mass2
    _, _, vx1, vy1, _ = _get_cluster_properties(cluster1, velocity)
    _, _, vx2, vy2, _ = _get_cluster_properties(cluster2, velocity)
    denom = total_mass if total_mass > 0 else 1.0
    new_vx = (vx1 * mass1 + vx2 * mass2) / denom
    new_vy = (vy1 * mass1 + vy2 * mass2) / denom
    vel_reduction = 1.0 / (1.0 + math.log(1.0 + total_mass / 1000.0))
    new_vx *= vel_reduction
    new_vy *= vel_reduction
    new_angle = math.atan2(new_vy, new_vx)
    for p in merged_members:
        p['angle'] = new_angle


def _update_cluster_movement(particles, width, height, velocity):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cluster_id, cluster in clusters.items():
        if not cluster:
            continue
        total_mass = sum(p['area'] for p in cluster)
        com_x = sum(p['x'] * p['area'] for p in cluster) / total_mass
        com_y = sum(p['y'] * p['area'] for p in cluster) / total_mass
        avg_vx = sum(velocity / p['r_bound'] * math.cos(p.get('angle', 0)) for p in cluster) / len(cluster)
        avg_vy = sum(velocity / p['r_bound'] * math.sin(p.get('angle', 0)) for p in cluster) / len(cluster)
        vel_scale = 1.0 / (1.0 + math.log(1.0 + total_mass / 1000.0))
        avg_vx *= vel_scale
        avg_vy *= vel_scale

        br_angle = random.uniform(0, 2 * math.pi)
        mf = 1.0 / max(total_mass, 1e-12) ** 0.3
        dx = avg_vx + velocity * mf * math.cos(br_angle)
        dy = avg_vy + velocity * mf * math.sin(br_angle)
        final_angle = br_angle

        offsets = [(p['x'] - com_x, p['y'] - com_y, p['r_bound']) for p in cluster]
        xs_left = [com_x + dx + ox - r for ox, oy, r in offsets]
        xs_right = [com_x + dx + ox + r for ox, oy, r in offsets]
        ys_top = [com_y + dy + oy - r for ox, oy, r in offsets]
        ys_bottom = [com_y + dy + oy + r for ox, oy, r in offsets]

        shift_x = shift_y = 0.0
        bounced_x = bounced_y = False
        if min(xs_left) < 0: shift_x, bounced_x = -min(xs_left), True
        elif max(xs_right) > width: shift_x, bounced_x = width - max(xs_right), True
        if min(ys_top) < 0: shift_y, bounced_y = -min(ys_top), True
        elif max(ys_bottom) > height: shift_y, bounced_y = height - max(ys_bottom), True

        dx += shift_x
        dy += shift_y
        if bounced_x: final_angle = math.pi - final_angle
        if bounced_y: final_angle = -final_angle

        for idx, p in enumerate(cluster):
            ox, oy, _ = offsets[idx]
            p['x'] = com_x + dx + ox
            p['y'] = com_y + dy + oy
            p['angle'] = final_angle

        _keep_cluster_inside_bounds(cluster, width, height)


def run_triangle_simulation(n_req, run_save_dir, max_time=DEFAULT_MAX_TIME, store_every_k=DEFAULT_STORE_EVERY_K,
                            radius=DEFAULT_RADIUS, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                            seed=None, max_attempts=15000):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(run_save_dir, exist_ok=True)
    cell_size = max(18.0, radius * 2.2)
    grid_w = math.ceil(width / cell_size)
    grid_h = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_h)] for _ in range(grid_w)]

    particles = []
    cluster_id_counter = 0

    # 1. Placement with fast SAT collision validation
    for i in range(n_req):
        placed = False
        for attempt in range(max_attempts):
            x = random.uniform(radius + 2, width - radius - 2)
            y = random.uniform(radius + 2, height - radius - 2)
            theta = random.uniform(0, 2 * math.pi)

            cand = _create_triangle_particle(radius, x, y, theta, cluster_id_counter, i)
            cx = int(x / cell_size)
            cy = int(y / cell_size)

            overlap = False
            for gx in range(max(0, cx - 1), min(grid_w, cx + 2)):
                for gy in range(max(0, cy - 1), min(grid_h, cy + 2)):
                    for other in grid[gx][gy]:
                        d2 = (x - other['x'])**2 + (y - other['y'])**2
                        if d2 <= (radius + other['r_bound'])**2:
                            if get_mtv_triangle_triangle(cand, other, eps=1e-6) is not None:
                                overlap = True
                                break
                    if overlap:
                        break
                if overlap:
                    break

            if not overlap:
                cand['cluster_members'] = [cand]
                particles.append(cand)
                grid[cx][cy].append(cand)
                cluster_id_counter += 1
                placed = True
                break

        if not placed:
            print(f"Warning: Could only place {len(particles)} / {n_req} triangles.")
            break

    for p in particles:
        p['cluster_members'] = [p]

    num_saved_frames = (max_time // store_every_k) + 60
    trajectories = np.zeros((num_saved_frames, len(particles), 2), dtype=np.float32)
    orientations = np.zeros((num_saved_frames, len(particles)), dtype=np.float32)
    local_verts_arr = np.array([p['local_vertices'] for p in particles], dtype=np.float32)

    cluster_count_evolution = []
    current_cluster_count = len(particles)
    global_pairs = []
    t_save_idx = 0
    start_time = time.time()

    # Initial overlap relaxation
    _resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
    _enforce_world_bounds_all(particles, width, height)

    for time_step in range(max_time):
        _update_spatial_grid(particles, grid, cell_size)

        # Brownian rotation
        _rotate_all_clusters(particles, grid, cell_size,
                             omega_base=ROT_OMEGA_BASE,
                             omega_brownian=ROT_BROWNIAN,
                             dt=1.0)

        checked_pairs = set()
        collision_happened = False

        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x'] / cell_size), int(p1['y'] / cell_size)
            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for p2 in grid[gx][gy]:
                        if p1 is p2 or p1['cluster_id'] == p2['cluster_id']:
                            continue
                        cluster_pair = (min(p1['cluster_id'], p2['cluster_id']),
                                        max(p1['cluster_id'], p2['cluster_id']))
                        if cluster_pair in checked_pairs:
                            continue

                        dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
                        if dist_sq > (p1['r_bound'] + p2['r_bound'] + EPS_CONTACT)**2:
                            continue

                        if _resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT * 1.1):
                            _merge_clusters(particles, p1['cluster_id'], p2['cluster_id'], DEFAULT_VELOCITY)
                            _update_spatial_grid(particles, grid, cell_size)
                            checked_pairs.add(cluster_pair)
                            collision_happened = True
                            current_cluster_count -= 1

        if collision_happened:
            _resolve_all_overlaps_rigid(particles, grid, cell_size, passes=1)

        _update_cluster_movement(particles, width, height, DEFAULT_VELOCITY)

        if collision_happened:
            _resolve_all_overlaps_rigid(particles, grid, cell_size, passes=1)

        _enforce_world_bounds_all(particles, width, height)

        cluster_count_evolution.append((time_step, current_cluster_count))
        is_save_step = (time_step % store_every_k) == 0

        # Save trajectory frame
        if is_save_step and t_save_idx < num_saved_frames:
            for idx, p in enumerate(particles):
                trajectories[t_save_idx, idx, 0] = p['x']
                trajectories[t_save_idx, idx, 1] = p['y']
                orientations[t_save_idx, idx] = p.get('orientation', 0.0)
            t_save_idx += 1

        # Check early completion
        if current_cluster_count == 1:
            print(f"All particles merged into one cluster at time step {time_step}")
            if not is_save_step and t_save_idx < num_saved_frames:
                for idx, p in enumerate(particles):
                    trajectories[t_save_idx, idx, 0] = p['x']
                    trajectories[t_save_idx, idx, 1] = p['y']
                    orientations[t_save_idx, idx] = p.get('orientation', 0.0)
                t_save_idx += 1
            break

    elapsed = time.time() - start_time
    trajectories = trajectories[:t_save_idx]
    orientations = orientations[:t_save_idx]
    clustercount_data = np.array(cluster_count_evolution, dtype=np.int32)

    # Compute final fractal dimension Df from clusters
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cid, members in clusters.items():
        if len(members) >= 2:
            rg = _radius_of_gyration(members)
            m = sum(pp['area'] for pp in members)
            if m >= 10.0 and rg > 1e-8:
                global_pairs.append((rg, m))

    if len(global_pairs) >= 6:
        unique = {}
        for rg, m in global_pairs:
            key = (round(rg, 8), round(m, 8))
            if key not in unique:
                unique[key] = (rg, m)
        Rg_vals, M_vals = zip(*unique.values())
        logRg = np.log(np.array(Rg_vals))
        logM = np.log(np.array(M_vals))
        slope, intercept = np.polyfit(logRg, logM, 1)
        Df = float(slope)
    else:
        Df = float('nan')

    tag = f"equilateral_N{len(particles)}_t{max_time}"
    traj_path = os.path.join(run_save_dir, f"traj_{tag}.npy")
    orient_path = os.path.join(run_save_dir, f"orient_{tag}.npy")
    verts_path = os.path.join(run_save_dir, f"vertices_{tag}.npy")
    radii_path = os.path.join(run_save_dir, f"radii_{tag}.npy")
    clusterids_path = os.path.join(run_save_dir, f"clusterids_{tag}.npy")
    clustercount_path = os.path.join(run_save_dir, f"clustercount_{tag}.npy")
    meta_path = os.path.join(run_save_dir, f"meta_{tag}.json")
    plot_path = os.path.join(run_save_dir, f"final_{tag}.png")

    np.save(traj_path, trajectories)
    np.save(orient_path, orientations)
    np.save(verts_path, local_verts_arr)
    np.save(radii_path, np.array([p['r_bound'] for p in particles], dtype=np.float32))
    np.save(clusterids_path, np.array([p['cluster_id'] for p in particles], dtype=np.int32))
    np.save(clustercount_path, clustercount_data)

    meta = {
        "triangle_type": "equilateral",
        "radius": radius,
        "N_total": len(particles),
        "width": width,
        "height": height,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "saved_frames": int(trajectories.shape[0]),
        "final_clusters": int(current_cluster_count),
        "final_step": int(clustercount_data[-1, 0]) if len(clustercount_data) > 0 else 0,
        "Df": float(Df),
        "elapsed_seconds": float(elapsed)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Static plot
    plt.figure(figsize=(10, 7.5))
    ax = plt.gca()
    for p in particles:
        corners = get_triangle_corners(p)
        poly = patches.Polygon(corners, closed=True, facecolor='dodgerblue', alpha=0.85)
        ax.add_patch(poly)
        outline = patches.Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=0.5)
        ax.add_patch(outline)
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Equilateral Triangles | N={len(particles)} | Clusters: {current_cluster_count} | Steps: {meta['final_step']}")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "requested_particles": int(n_req),
        "placed_particles": len(particles),
        "final_clusters": int(current_cluster_count),
        "final_step": meta['final_step'],
        "elapsed_seconds": elapsed,
        "traj_path": traj_path,
        "orient_path": orient_path,
        "verts_path": verts_path,
        "radii_path": radii_path,
        "clusterids_path": clusterids_path,
        "clustercount_path": clustercount_path,
        "meta_path": meta_path,
        "plot_path": plot_path,
        "Df": Df
    }


def _worker_wrapper(args):
    (n_req, run_save_dir, max_time, store_every_k, radius, width, height, checkpoints, seed, max_attempts) = args
    res = run_triangle_simulation(
        n_req=n_req,
        run_save_dir=run_save_dir,
        max_time=max_time,
        store_every_k=store_every_k,
        radius=radius,
        width=width,
        height=height,
        seed=seed,
        max_attempts=max_attempts
    )

    actual_placed = res['placed_particles']
    # Area of 1 equilateral triangle: (3 * sqrt(3) / 4) * r^2
    tri_area = (3.0 * math.sqrt(3.0) / 4.0) * (radius ** 2)
    total_particle_area = float(actual_placed * tri_area)
    domain_area = float(width * height)
    area_fraction = float(total_particle_area / domain_area)

    clustercount_data = np.load(res['clustercount_path'])
    final_clusters = res['final_clusters']
    final_step = res['final_step']
    fully_coalesced = bool(final_clusters == 1)

    checkpoint_results = {}
    for cp in checkpoints:
        if len(clustercount_data) == 0:
            checkpoint_results[cp] = actual_placed
        else:
            valid_rows = clustercount_data[clustercount_data[:, 0] <= cp]
            if len(valid_rows) > 0:
                checkpoint_results[cp] = int(valid_rows[-1, 1])
            else:
                checkpoint_results[cp] = int(clustercount_data[0, 1])

    res.update({
        "area_fraction": area_fraction,
        "total_particle_area": total_particle_area,
        "domain_area": domain_area,
        "fully_coalesced": fully_coalesced,
        "subfolder": os.path.abspath(run_save_dir),
        "checkpoints": checkpoint_results
    })
    return res


def run_sweep(counts=None, checkpoints=None, max_time=DEFAULT_MAX_TIME,
              store_every_k=DEFAULT_STORE_EVERY_K, radius=DEFAULT_RADIUS,
              width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
              num_workers=4, output_dir=None, seed_base=42):
    if counts is None:
        counts = DEFAULT_COUNTS
    if checkpoints is None:
        checkpoints = DEFAULT_CHECKPOINTS

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(DEFAULT_OUTPUT_BASE, f"sweep_equilateral_triangles_{ts}")

    os.makedirs(output_dir, exist_ok=True)
    manifest_csv = os.path.join(output_dir, "manifest.csv")
    manifest_json = os.path.join(output_dir, "manifest.json")
    status_log = os.path.join(output_dir, "status.log")

    print(f"=== Starting Equilateral Triangle Sweep ===")
    print(f"Output Directory : {output_dir}")
    print(f"Arena Box        : {width} x {height} (Area = {width * height})")
    print(f"Triangle Radius  : {radius} (Single Triangle Area = {(3*math.sqrt(3)/4)*radius*radius:.2f})")
    print(f"Target Counts    : {counts}")
    print(f"Workers          : {num_workers}")
    print(f"Max Steps        : {max_time}")

    tasks = []
    for idx, n in enumerate(counts):
        run_dir = os.path.join(output_dir, f"N_{n}")
        seed = (seed_base + idx * 7919) if seed_base is not None else None
        tasks.append((n, run_dir, max_time, store_every_k, radius, width, height, checkpoints, seed, 25000))

    results = []
    t_start = time.time()

    def update_manifest():
        results_sorted = sorted(results, key=lambda r: r['requested_particles'])
        with open(manifest_json, "w") as f:
            json.dump({
                "sweep_info": {
                    "triangle_type": "equilateral",
                    "radius": radius,
                    "width": width,
                    "height": height,
                    "domain_area": width * height,
                    "triangle_area": (3 * math.sqrt(3) / 4) * radius**2,
                    "counts": counts,
                    "checkpoints": checkpoints,
                    "max_time": max_time,
                    "store_every_k": store_every_k,
                    "num_workers": num_workers
                },
                "runs": results_sorted
            }, f, indent=2)

        base_fields = [
            "requested_particles", "placed_particles", "placement_ratio",
            "area_fraction", "final_clusters", "fully_coalesced",
            "final_step", "elapsed_seconds", "Df", "subfolder"
        ]
        cp_fields = [f"clusters_at_step_{cp}" for cp in checkpoints]
        fieldnames = base_fields + cp_fields

        with open(manifest_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_sorted:
                row = {
                    "requested_particles": r["requested_particles"],
                    "placed_particles": r["placed_particles"],
                    "placement_ratio": f"{r['placed_particles'] / r['requested_particles']:.4f}",
                    "area_fraction": f"{r['area_fraction']:.5f}",
                    "final_clusters": r["final_clusters"],
                    "fully_coalesced": r["fully_coalesced"],
                    "final_step": r["final_step"],
                    "elapsed_seconds": f"{r['elapsed_seconds']:.2f}",
                    "Df": f"{r['Df']:.4f}" if not math.isnan(r.get('Df', float('nan'))) else "NaN",
                    "subfolder": r["subfolder"]
                }
                for cp in checkpoints:
                    row[f"clusters_at_step_{cp}"] = r["checkpoints"].get(cp, "")
                writer.writerow(row)

    with mp.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(_worker_wrapper, tasks):
            results.append(res)
            msg = (f"[COMPLETED] N={res['requested_particles']} | Placed={res['placed_particles']} "
                   f"| phi={res['area_fraction']:.4f} | Final Clusters={res['final_clusters']} "
                   f"| Steps={res['final_step']} | Time={res['elapsed_seconds']:.1f}s")
            print(msg)
            with open(status_log, "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
            update_manifest()

    total_time = time.time() - t_start
    print(f"\nAll {len(results)} runs finished in {total_time:.1f}s ({total_time/60:.2f} mins).")
    print(f"Manifest written to: {manifest_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equilateral Triangle DLCA Area-Fraction Sweep")
    parser.add_argument("--counts", nargs="+", type=int, default=DEFAULT_COUNTS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS if 'DEFAULT_WORKERS' in locals() else 4)
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_sweep(
        counts=args.counts,
        max_time=args.max_time,
        radius=args.radius,
        width=args.width,
        height=args.height,
        num_workers=args.workers,
        output_dir=args.output_dir
    )
