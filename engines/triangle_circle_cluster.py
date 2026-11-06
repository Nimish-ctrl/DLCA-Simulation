"""
triangle_circle_cluster.py — Mixed Triangle-Circle DLCA Simulation
==================================================================
Simulates diffusion-limited cluster aggregation for a mixed system of 
Circles and Triangles using SAT/MTV collision physics and spatial grid acceleration.
"""

import sys
import argparse
import random
import math
import time
import json
import os
import threading
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk, messagebox

from collisions.collisions import (
    create_equilateral_triangle_vertices,
    get_polygon_corners,
    get_triangle_corners,
    get_mtv_circle_circle,
    get_mtv_triangle_triangle,
    get_mtv_triangle_circle,
    get_mtv_circle_triangle
)

# ---------------------------
# Global Configuration
# ---------------------------
width = 1600
height = 1200

spawn_width = 800
spawn_height = 600

spawn_x0 = (width - spawn_width) / 2.0
spawn_y0 = (height - spawn_height) / 2.0

velocity = 5.0
max_time = 40000
min_mass = 10.0
min_Rg = 1e-8
min_pts_for_fit = 6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(SCRIPT_DIR, "sim_npy_out_triangle_circle")
save_plots = os.path.join(SCRIPT_DIR, "plots_triangle_circle")
store_every_k = 1

# Contact gap and safety push
EPS_CONTACT = 1.0
EPS_PUSH    = 0.2

# Unbiased Brownian rotation parameters
ROT_OMEGA_BASE = 0.0
ROT_BROWNIAN   = 0.03

# Optimization parameters
MAX_SEPARATION_PASSES = 20
CLEANUP_OUTER_REPEATS = 1
ENFORCE_GAP_ROUNDS = 2
EARLY_TERMINATION_THRESHOLD = 0.1

# ---------------------------
# Collision & Geometry Dispatcher
# ---------------------------
def get_bounding_radius(p):
    return p.get('r_bound', p['r'])

def get_mtv_mixed(p1, p2, eps=1e-9):
    """
    Dispatches to appropriate collision routine based on the pair of shapes:
    circle-circle, triangle-triangle, triangle-circle, or circle-triangle.
    """
    s1 = p1['shape']
    s2 = p2['shape']

    if s1 == 'circle' and s2 == 'circle':
        return get_mtv_circle_circle(p1, p2, eps)
    elif s1 == 'triangle' and s2 == 'triangle':
        return get_mtv_triangle_triangle(p1, p2, eps)
    elif s1 == 'triangle' and s2 == 'circle':
        return get_mtv_triangle_circle(p1, p2, eps)
    elif s1 == 'circle' and s2 == 'triangle':
        res = get_mtv_triangle_circle(p2, p1, eps)
        if res is not None:
            return (res[1], res[0])
        return None
    return None

# ---------------------------
# Grid & Spatial Helpers
# ---------------------------
def distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

def update_spatial_grid(particles, grid, cell_size):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j].clear()
    for p in particles:
        cx = int(p['x'] / cell_size)
        cy = int(p['y'] / cell_size)
        if 0 <= cx < len(grid) and 0 <= cy < len(grid[0]):
            grid[cx][cy].append(p)

def insert_into_grid(p, grid, cell_size):
    i = int(p['x'] / cell_size)
    j = int(p['y'] / cell_size)
    if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
        grid[i][j].append(p)

def check_spawn_overlap(particles, x, y, r_bound, grid, cell_size):
    cell_x, cell_y = int(x / cell_size), int(y / cell_size)
    for i in range(max(0, cell_x - 1), min(len(grid), cell_x + 2)):
        for j in range(max(0, cell_y - 1), min(len(grid[0]), cell_y + 2)):
            for p in grid[i][j]:
                if distance(x, y, p['x'], p['y']) <= (r_bound + p['r_bound'])**2:
                    return False
    return True

# ---------------------------
# Cluster Management & Physics
# ---------------------------
def _move_cluster(cluster, dx, dy):
    for q in cluster:
        q['x'] += dx
        q['y'] += dy

def rebuild_cluster_members(particles):
    by_id = defaultdict(list)
    for p in particles:
        by_id[p['cluster_id']].append(p)
    for p in particles:
        p['cluster_members'] = by_id[p['cluster_id']]

def cluster_mass(cluster):
    return sum(p['r']**2 for p in cluster)

def radius_of_gyration(cluster):
    if len(cluster) <= 1:
        return 0.0
    M = cluster_mass(cluster)
    if M <= 0:
        return 0.0
    com_x = sum(p['x'] * p['r']**2 for p in cluster) / M
    com_y = sum(p['y'] * p['r']**2 for p in cluster) / M
    rg_sq = sum(p['r']**2 * ((p['x'] - com_x)**2 + (p['y'] - com_y)**2) for p in cluster) / M
    return math.sqrt(max(0.0, rg_sq))

def resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    res = get_mtv_mixed(p1, p2, eps=0.0)
    if res is None:
        return False

    mtv1, _ = res
    mtv_len = math.hypot(mtv1[0], mtv1[1])
    if mtv_len <= 1e-12:
        return False

    desired_push = mtv_len + eps
    push_x = (mtv1[0] / mtv_len) * desired_push
    push_y = (mtv1[1] / mtv_len) * desired_push

    c1 = p1['cluster_members']
    c2 = p2['cluster_members']
    m1 = cluster_mass(c1)
    m2 = cluster_mass(c2)
    tot_m = m1 + m2
    if tot_m <= 0:
        return False

    w1 = m2 / tot_m
    w2 = m1 / tot_m

    _move_cluster(c1,  push_x * w1,  push_y * w1)
    _move_cluster(c2, -push_x * w2, -push_y * w2)
    return True

def separate_all_overlaps_grid(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=20):
    for _ in range(max_rounds):
        moved = False
        update_spatial_grid(particles, grid, cell_size)
        processed_pairs = set()

        for i in range(len(particles)):
            pi = particles[i]
            cx, cy = int(pi['x'] / cell_size), int(pi['y'] / cell_size)
            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for pj in grid[gx][gy]:
                        if pi is pj or pi['cluster_id'] == pj['cluster_id']:
                            continue
                        pair_key = (min(pi['id'], pj['id']), max(pi['id'], pj['id']))
                        if pair_key in processed_pairs:
                            continue
                        if distance(pi['x'], pi['y'], pj['x'], pj['y']) <= (pi['r_bound'] + pj['r_bound'])**2:
                            if resolve_overlap_pair_as_clusters(pi, pj, eps=eps):
                                moved = True
                                processed_pairs.add(pair_key)
        if not moved:
            break

def enforce_eps_gap_grid(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=2):
    for _ in range(max_rounds):
        moved = False
        update_spatial_grid(particles, grid, cell_size)
        for i in range(len(particles)):
            pi = particles[i]
            cx, cy = int(pi['x'] / cell_size), int(pi['y'] / cell_size)
            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for pj in grid[gx][gy]:
                        if pi is pj or pi['cluster_id'] == pj['cluster_id']:
                            continue
                        if distance(pi['x'], pi['y'], pj['x'], pj['y']) <= (pi['r_bound'] + pj['r_bound'] + eps)**2:
                            if resolve_overlap_pair_as_clusters(pi, pj, eps=eps):
                                moved = True
        if not moved:
            break

def resolve_all_overlaps_rigid(particles, grid, cell_size, passes=1):
    for _ in range(passes):
        separate_all_overlaps_grid(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=MAX_SEPARATION_PASSES)

def fully_separate_two_clusters(particles, grid, cell_size, c1_id, c2_id, eps=EPS_CONTACT, max_passes=20):
    for _ in range(max_passes):
        overlapping = False
        for p1 in particles:
            if p1['cluster_id'] != c1_id:
                continue
            for p2 in particles:
                if p2['cluster_id'] != c2_id:
                    continue
                if distance(p1['x'], p1['y'], p2['x'], p2['y']) <= (p1['r_bound'] + p2['r_bound'])**2:
                    if resolve_overlap_pair_as_clusters(p1, p2, eps=eps):
                        overlapping = True
        if not overlapping:
            break

def merge_clusters(particles, c1_id, c2_id):
    c1_members = [p for p in particles if p['cluster_id'] == c1_id]
    c2_members = [p for p in particles if p['cluster_id'] == c2_id]
    if not c1_members or not c2_members:
        return

    m1 = cluster_mass(c1_members)
    m2 = cluster_mass(c2_members)
    denom = m1 + m2
    if denom <= 0:
        return

    vx1 = sum(velocity / p['r'] * math.cos(p['angle']) for p in c1_members) / len(c1_members)
    vy1 = sum(velocity / p['r'] * math.sin(p['angle']) for p in c1_members) / len(c1_members)
    vx2 = sum(velocity / p['r'] * math.cos(p['angle']) for p in c2_members) / len(c2_members)
    vy2 = sum(velocity / p['r'] * math.sin(p['angle']) for p in c2_members) / len(c2_members)

    merged_id = min(c1_id, c2_id)
    merged_members = c1_members + c2_members
    for p in merged_members:
        p['cluster_id'] = merged_id

    rebuild_cluster_members(particles)

    new_vx = (vx1 * m1 + vx2 * m2) / denom
    new_vy = (vy1 * m1 + vy2 * m2) / denom
    vel_reduction = 1.0 / (1.0 + denom / 10000.0)
    new_vx *= vel_reduction
    new_vy *= vel_reduction
    new_angle = math.atan2(new_vy, new_vx)

    for p in merged_members:
        p['angle'] = new_angle

def mixed_collisions(p1, p2, grid, cell_size):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    if distance(p1['x'], p1['y'], p2['x'], p2['y']) > (p1['r_bound'] + p2['r_bound'])**2:
        return False

    res = get_mtv_mixed(p1, p2, eps=0.0)
    if res is None:
        return False

    global particles_ref
    fully_separate_two_clusters(particles_ref, grid, cell_size, p1['cluster_id'], p2['cluster_id'], eps=0.001)
    update_spatial_grid(particles_ref, grid, cell_size)
    merge_clusters(particles_ref, p1['cluster_id'], p2['cluster_id'])
    update_spatial_grid(particles_ref, grid, cell_size)
    return True

# ---------------------------
# Rigid Cluster Movement & Bounds
# ---------------------------
def keep_cluster_inside_bounds(cluster, width, height):
    if not cluster:
        return
    min_x = min(p['x'] - p['r_bound'] for p in cluster)
    max_x = max(p['x'] + p['r_bound'] for p in cluster)
    min_y = min(p['y'] - p['r_bound'] for p in cluster)
    max_y = max(p['y'] + p['r_bound'] for p in cluster)

    dx = 0.0
    dy = 0.0
    if min_x < 0.0:     dx = -min_x
    elif max_x > width: dx = width - max_x
    if min_y < 0.0:      dy = -min_y
    elif max_y > height: dy = height - max_y

    if dx != 0.0 or dy != 0.0:
        for p in cluster:
            p['x'] += dx
            p['y'] += dy

def enforce_world_bounds_all(particles, width, height):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)
    for cluster in clusters.values():
        keep_cluster_inside_bounds(cluster, width, height)

def rotate_all_clusters(particles, grid, cell_size, omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cluster_id, cluster in clusters.items():
        if len(cluster) <= 1:
            for p in cluster:
                if p['shape'] == 'triangle':
                    p['orientation'] = (p['orientation'] + random.gauss(0.0, 0.1)) % (2 * math.pi)
            continue

        m = cluster_mass(cluster)
        com_x = sum(p['x'] * p['r']**2 for p in cluster) / m
        com_y = sum(p['y'] * p['r']**2 for p in cluster) / m

        rg = radius_of_gyration(cluster)
        rg_eff = max(rg, 5.0)
        I_eff = m * (rg_eff ** 2)

        omega_t = (omega_base / (1.0 + m / 1000.0)) + (omega_brownian / math.sqrt(max(1.0, I_eff))) * random.gauss(0.0, 1.0)
        dtheta = omega_t * dt
        cos_t = math.cos(dtheta)
        sin_t = math.sin(dtheta)

        for p in cluster:
            rx = p['x'] - com_x
            ry = p['y'] - com_y
            p['x'] = com_x + (rx * cos_t - ry * sin_t)
            p['y'] = com_y + (rx * sin_t + ry * cos_t)
            if p['shape'] == 'triangle':
                p['orientation'] = (p['orientation'] + dtheta) % (2 * math.pi)

def update_cluster_movement(particles, grid, cell_size):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cluster_id, cluster in clusters.items():
        if not cluster:
            continue
        tot_m = cluster_mass(cluster)
        com_x = sum(p['x'] * p['r']**2 for p in cluster) / tot_m
        com_y = sum(p['y'] * p['r']**2 for p in cluster) / tot_m

        avg_vx = sum(velocity / p['r'] * math.cos(p['angle']) for p in cluster) / len(cluster)
        avg_vy = sum(velocity / p['r'] * math.sin(p['angle']) for p in cluster) / len(cluster)
        vel_scale = 1.0 / (1.0 + math.log(1.0 + tot_m / 1000.0))
        avg_vx *= vel_scale
        avg_vy *= vel_scale

        br_angle = random.uniform(0, 2 * math.pi)
        mf = 1.0 / (tot_m ** 0.3)  # Balanced mobility factor for steady coalescence
        new_com_x = com_x + avg_vx + velocity * mf * math.cos(br_angle)
        new_com_y = com_y + avg_vy + velocity * mf * math.sin(br_angle)

        dx = new_com_x - com_x
        dy = new_com_y - com_y
        for p in cluster:
            p['x'] += dx
            p['y'] += dy
            p['angle'] = br_angle

        keep_cluster_inside_bounds(cluster, width, height)

def check_one_cluster(particles):
    first_id = particles[0]['cluster_id']
    for p in particles[1:]:
        if p['cluster_id'] != first_id:
            return False
    return True

# ---------------------------
# Core Simulation Driver
# ---------------------------
def simulate_mixed_particles(particles, N_total, max_time, store_every_k=1,
                            save_dir=save_dir, tag=None, radii_pair=(12.0, 12.0),
                            fraction_A_label=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_plots, exist_ok=True)

    cell_size = max(radii_pair) * 2.5
    grid_w = math.ceil(width / cell_size)
    grid_h = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_h)] for _ in range(grid_w)]
    update_spatial_grid(particles, grid, cell_size)

    num_saved_frames = (max_time // store_every_k) + 1
    trajectories = np.zeros((num_saved_frames, N_total, 2), dtype=np.float32)
    orientations = np.zeros((num_saved_frames, N_total), dtype=np.float32)

    global particles_ref
    particles_ref = particles

    # Initial cleanup
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
    enforce_eps_gap_grid(particles, grid, cell_size)

    t_save_idx = 0
    start_time = time.time()

    for time_step in range(max_time):
        update_spatial_grid(particles, grid, cell_size)

        # Rigid rotation for clusters
        rotate_all_clusters(particles, grid, cell_size)

        # Unattached single particles move individually
        clusters_map = defaultdict(list)
        for p in particles:
            clusters_map[p['cluster_id']].append(p)

        for p in particles:
            if len(clusters_map[p['cluster_id']]) == 1:
                p['angle'] = random.uniform(0, 2 * math.pi)
                mf = 1.0 / p['r']
                new_x = p['x'] + velocity * mf * math.cos(p['angle'])
                new_y = p['y'] + velocity * mf * math.sin(p['angle'])
                r_b = p['r_bound']
                if new_x < r_b or new_x > width - r_b:
                    p['angle'] = math.pi - p['angle']
                if new_y < r_b or new_y > height - r_b:
                    p['angle'] = -p['angle']
                p['x'] = max(r_b, min(width - r_b, new_x))
                p['y'] = max(r_b, min(height - r_b, new_y))

        # Check collisions across spatial grid
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
                        cluster_pair = tuple(sorted((p1['cluster_id'], p2['cluster_id'])))
                        if cluster_pair in checked_pairs:
                            continue
                        if mixed_collisions(p1, p2, grid, cell_size):
                            checked_pairs.add(cluster_pair)
                            collision_happened = True

        # Adaptive cleanup
        if collision_happened or time_step % 100 == 0:
            resolve_all_overlaps_rigid(particles, grid, cell_size, passes=1)
            enforce_eps_gap_grid(particles, grid, cell_size)

        # Move clusters
        update_cluster_movement(particles, grid, cell_size)
        enforce_world_bounds_all(particles, width, height)

        # Save trajectory snapshot
        if (time_step % store_every_k) == 0:
            for idx, p in enumerate(particles):
                trajectories[t_save_idx, idx, 0] = p['x']
                trajectories[t_save_idx, idx, 1] = p['y']
                orientations[t_save_idx, idx] = p.get('orientation', 0.0)
            t_save_idx += 1

        if check_one_cluster(particles):
            print(f"All particles merged into one cluster at timestep {time_step}!")
            break

    elapsed = time.time() - start_time
    trajectories = trajectories[:t_save_idx]
    orientations = orientations[:t_save_idx]

    # Build saved metadata & arrays
    radii_arr = np.array([p['r'] for p in particles], dtype=np.float32)
    shapes_arr = np.array([0 if p['shape'] == 'circle' else 1 for p in particles], dtype=np.int32)
    count_circles = sum(1 for p in particles if p['shape'] == 'circle')
    count_triangles = len(particles) - count_circles

    if tag is None:
        tag = f"mixed_tri_circ_N{N_total}_t{max_time}"

    traj_path = os.path.join(save_dir, f"traj_{tag}.npy")
    orient_path = os.path.join(save_dir, f"orient_{tag}.npy")
    radii_path = os.path.join(save_dir, f"radii_{tag}.npy")
    shapes_path = os.path.join(save_dir, f"shapes_{tag}.npy")
    meta_path = os.path.join(save_dir, f"meta_{tag}.json")

    np.save(traj_path, trajectories)
    np.save(orient_path, orientations)
    np.save(radii_path, radii_arr)
    np.save(shapes_path, shapes_arr)

    meta = {
        "simulation_type": "Triangle-Circle Mixed",
        "N_total": N_total,
        "count_A": count_circles,
        "count_B": count_triangles,
        "shape_A": "circle",
        "shape_B": "triangle",
        "fraction_A": float(count_circles / N_total),
        "radii_pair": list(radii_pair),
        "width": width,
        "height": height,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "saved_frames": int(trajectories.shape[0]),
        "final_clusters": len(set(p['cluster_id'] for p in particles))
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Plot final configuration
    plt.figure(figsize=(12, 8))
    for p in particles:
        if p['shape'] == 'circle':
            c = plt.Circle((p['x'], p['y']), p['r'], color='#1f77b4', fill=True, alpha=0.85, ec='black', lw=0.5)
            plt.gca().add_patch(c)
        else:
            corners = get_triangle_corners(p)
            poly = plt.Polygon(corners, closed=True, color='#d62728', fill=True, alpha=0.85, ec='black', lw=0.5)
            plt.gca().add_patch(poly)

    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Mixed Triangle-Circle Simulation (N={N_total}, Final Clusters: {meta['final_clusters']})")
    plot_file = os.path.join(save_plots, f"final_{tag}.png")
    plt.savefig(plot_file, dpi=180, bbox_inches='tight')
    plt.close()

    result = {
        "traj_path": traj_path,
        "orient_path": orient_path,
        "radii_path": radii_path,
        "shapes_path": shapes_path,
        "meta_path": meta_path,
        "elapsed_s": float(elapsed),
        "saved_frames": int(trajectories.shape[0]),
        "num_particles": len(particles),
        "final_clusters": meta["final_clusters"]
    }
    return result

def run_simulation(N_total=400, r_circle=12.0, r_triangle=12.0, fraction_circles=0.5,
                   max_time=40000, store_every_k=1, save_dir=save_dir, seed=None):
    """
    Spawns roughly half circles and half triangles and runs the mixed DLCA simulation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    count_circles = int(round(N_total * fraction_circles))
    count_triangles = N_total - count_circles

    radii_pair = (r_circle, r_triangle)
    cell_size = max(radii_pair) * 2.5
    grid_w = math.ceil(width / cell_size)
    grid_h = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_h)] for _ in range(grid_w)]

    particles = []
    cluster_id = 0

    # Local vertices for equilateral triangle
    tri_local = create_equilateral_triangle_vertices(r_triangle)

    # 1. Place Circles
    for _ in range(count_circles):
        attempts = 0
        while attempts < 10000:
            x = random.uniform(spawn_x0 + r_circle, spawn_x0 + spawn_width - r_circle)
            y = random.uniform(spawn_y0 + r_circle, spawn_y0 + spawn_height - r_circle)
            if check_spawn_overlap(particles, x, y, r_circle, grid, cell_size):
                p = {
                    'x': x, 'y': y, 'r': float(r_circle), 'r_bound': float(r_circle),
                    'shape': 'circle', 'cluster_id': cluster_id, 'cluster_members': [],
                    'angle': random.uniform(0, 2 * math.pi), 'id': len(particles)
                }
                particles.append(p)
                insert_into_grid(p, grid, cell_size)
                cluster_id += 1
                break
            attempts += 1

    # 2. Place Triangles
    for _ in range(count_triangles):
        attempts = 0
        while attempts < 10000:
            x = random.uniform(spawn_x0 + r_triangle, spawn_x0 + spawn_width - r_triangle)
            y = random.uniform(spawn_y0 + r_triangle, spawn_y0 + spawn_height - r_triangle)
            if check_spawn_overlap(particles, x, y, r_triangle, grid, cell_size):
                p = {
                    'x': x, 'y': y, 'r': float(r_triangle), 'r_bound': float(r_triangle),
                    'shape': 'triangle', 'local_vertices': tri_local,
                    'orientation': random.uniform(0, 2 * math.pi),
                    'cluster_id': cluster_id, 'cluster_members': [],
                    'angle': random.uniform(0, 2 * math.pi), 'id': len(particles)
                }
                particles.append(p)
                insert_into_grid(p, grid, cell_size)
                cluster_id += 1
                break
            attempts += 1

    for p in particles:
        p['cluster_members'] = [p]

    tag = f"comp{fraction_circles:.3f}_N{len(particles)}_t{max_time}"
    return simulate_mixed_particles(
        particles=particles,
        N_total=len(particles),
        max_time=max_time,
        store_every_k=store_every_k,
        save_dir=save_dir,
        tag=tag,
        radii_pair=radii_pair,
        fraction_A_label=fraction_circles,
        seed=None
    )

# ---------------------------
# GUI Launch Interface
# ---------------------------
def launch_ui():
    root = tk.Tk()
    root.title("Triangle-Circle Mixed DLCA Simulation")

    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    particles_var = tk.StringVar(value="400")
    r_circ_var = tk.StringVar(value="12.0")
    r_tri_var = tk.StringVar(value="12.0")
    frac_circ_var = tk.StringVar(value="0.5")
    status_var = tk.StringVar(value="Ready")

    ttk.Label(frm, text="Number of Particles (N):").grid(row=0, column=0, sticky="w", pady=3)
    ttk.Entry(frm, textvariable=particles_var, width=12).grid(row=0, column=1, sticky="w", pady=3)

    ttk.Label(frm, text="Circle Radius (r_A):").grid(row=1, column=0, sticky="w", pady=3)
    ttk.Entry(frm, textvariable=r_circ_var, width=12).grid(row=1, column=1, sticky="w", pady=3)

    ttk.Label(frm, text="Triangle Radius (r_B):").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Entry(frm, textvariable=r_tri_var, width=12).grid(row=2, column=1, sticky="w", pady=3)

    ttk.Label(frm, text="Circle Fraction (0.0 - 1.0):").grid(row=3, column=0, sticky="w", pady=3)
    ttk.Entry(frm, textvariable=frac_circ_var, width=12).grid(row=3, column=1, sticky="w", pady=3)

    status_lbl = ttk.Label(frm, textvariable=status_var, font=("Segoe UI", 9, "italic"))
    status_lbl.grid(row=4, column=0, columnspan=2, sticky="w", pady=6)

    def on_done(msg):
        status_var.set(msg)
        start_btn.config(state="normal")
        messagebox.showinfo("Simulation Complete", msg)

    def on_error(err):
        status_var.set(f"Error: {err}")
        start_btn.config(state="normal")
        messagebox.showerror("Simulation Error", str(err))

    def start_run():
        try:
            n_tot = int(particles_var.get())
            rc = float(r_circ_var.get())
            rt = float(r_tri_var.get())
            fc = float(frac_circ_var.get())
            if n_tot <= 0 or rc <= 0 or rt <= 0:
                raise ValueError("Parameters must be positive")
            if not (0.0 <= fc <= 1.0):
                raise ValueError("Fraction must be between 0.0 and 1.0")
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        start_btn.config(state="disabled")
        status_var.set("Running mixed triangle-circle simulation...")

        def worker():
            try:
                res = run_simulation(
                    N_total=n_tot,
                    r_circle=rc,
                    r_triangle=rt,
                    fraction_circles=fc,
                    max_time=max_time,
                    store_every_k=store_every_k,
                    save_dir=save_dir
                )
                msg = f"Done: {res['num_particles']} particles | Final clusters: {res['final_clusters']} | Time: {res['elapsed_s']:.2f}s"
                root.after(0, lambda: on_done(msg))
            except Exception as e:
                root.after(0, lambda: on_error(e))

        threading.Thread(target=worker, daemon=True).start()

    btns = ttk.Frame(root)
    btns.grid(row=5, column=0, columnspan=2, pady=8)
    start_btn = ttk.Button(btns, text="Start", command=start_run)
    quit_btn = ttk.Button(btns, text="Quit", command=root.destroy)
    start_btn.pack(side="left", padx=6)
    quit_btn.pack(side="left", padx=6)

    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triangle-Circle Mixed DLCA Simulation")
    parser.add_argument("--cli", action="store_true", help="Run simulation in CLI mode without GUI")
    parser.add_argument("-N", "--particles", type=int, default=400, help="Total number of particles")
    parser.add_argument("-t", "--max_time", type=int, default=40000, help="Maximum timesteps")
    args = parser.parse_args()

    if args.cli:
        print(f"Starting CLI Simulation: N={args.particles}, max_time={args.max_time}...")
        res = run_simulation(N_total=args.particles, max_time=args.max_time)
        print(f"Completed in {res['elapsed_s']:.2f}s | Final clusters: {res['final_clusters']}")
    else:
        launch_ui()
