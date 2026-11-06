"""
dlca_batch_keep_logic.py — rigid contacts, no overlaps (global cleanup + safety push)
Latest working version as of 19th Jan 2026
"""

import random
import math
import time
import json
import os
from collections import defaultdict
from collisions.collisions import dist, get_corners, project_polygon, get_mtv_square_square, quick_center_aabb_reject
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# ---------------------------
# Config
# ---------------------------
width = 1600
height = 1200

spawn_width = 800
spawn_height = 600

# width = 1200
# height = 1000

# spawn_width = 800
# spawn_height = 600


# spawn origin (top-left corner) — center the spawn box by default:
spawn_x0 = (width - spawn_width) / 2.0
spawn_y0 = (height - spawn_height) / 2.0

velocity = 5.0
max_time = 10000
min_mass = 10.0
min_Rg = 1e-8
min_pts_for_fit = 6
save_dir = "sim_npy_out_square"
save_plots = "plots_square"
store_every_k = 1
repeats = 1

# Visible contact gap and extra safety push (pixels)
EPS_CONTACT = 1.0
EPS_PUSH    = 0.2
ROT_OMEGA_BASE = 0.0     # baseline rad/step
ROT_BROWNIAN   = 0.12     # random jitter rad/step

# ---------------------------
# Grid helpers
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

def check(particles, x, y, r, grid, cell_size):
    cell_x, cell_y = int(x / cell_size), int(y / cell_size)
    for i in range(max(0, cell_x - 1), min(len(grid), cell_x + 2)):
        for j in range(max(0, cell_y - 1), min(len(grid[0]), cell_y + 2)):
            for p in grid[i][j]:
                if distance(x, y, p['x'], p['y']) <= (r + p['r'])**2:
                    return False
    return True

def keep_cluster_inside_bounds(cluster, width, height):
    # Calculate global min/max based on actual corners of the squares
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for p in cluster:
        # get_corners comes from your collisions.py code
        corners = get_corners(p) 
        for cx, cy in corners:
            if cx < min_x: min_x = cx
            if cx > max_x: max_x = cx
            if cy < min_y: min_y = cy
            if cy > max_y: max_y = cy

    dx = 0.0
    dy = 0.0

    if min_x < 0.0:      dx = -min_x
    elif max_x > width:  dx = width - max_x
    
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


# ---------------------------
# Cluster utilities
# ---------------------------
def get_cluster_properties(cluster_particles):
    total_mass = sum(p['r'] ** 2 for p in cluster_particles)
    if total_mass == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    com_x = sum(p['x'] * p['r'] ** 2 for p in cluster_particles) / total_mass
    com_y = sum(p['y'] * p['r'] ** 2 for p in cluster_particles) / total_mass
    avg_vx = sum(velocity / p['r'] * math.cos(p.get('angle', 0)) for p in cluster_particles) / len(cluster_particles)
    avg_vy = sum(velocity / p['r'] * math.sin(p.get('angle', 0)) for p in cluster_particles) / len(cluster_particles)
    velocity_scale = 1 / (1 + math.log(1 + total_mass / 1000))
    avg_vx *= velocity_scale
    avg_vy *= velocity_scale
    return com_x, com_y, avg_vx, avg_vy, total_mass

def radius_of_gyration(cluster):
    if not cluster:
        return 0.0
    masses = [p['r']**2 for p in cluster]
    M = sum(masses)
    if M == 0:
        return 0.0
    com_x = sum(p['x']*m for p, m in zip(cluster, masses)) / M
    com_y = sum(p['y']*m for p, m in zip(cluster, masses)) / M
    rg2 = sum(m*((p['x']-com_x)**2 + (p['y']-com_y)**2) for p, m in zip(cluster, masses)) / M
    return math.sqrt(rg2)

def cluster_mass(cluster):
    return sum(p['r']**2 for p in cluster)

def check_one_cluster(particles):
    return len(set(p['cluster_id'] for p in particles)) == 1

# ---------------------------
# Rigid contact separation
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

def _cluster_mass(cluster):
    return sum(pp['r']**2 for pp in cluster)

def resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    # 1. Cheap Reject (Bounding box check)
    # Using sqrt(2) * r as the bounding circle radius for a square
    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    safe_dist = (p1['r'] + p2['r']) * 1.415 + eps # 1.415 approx sqrt(2)
    if dist_sq > safe_dist**2:
        return False

    # 2. SAT Check (Accurate Square-Square)
    # get_mtv_square_square returns (mtv, -mtv) or None. 
    # mtv is the vector to push p1 OUT of p2.
    res = get_mtv_square_square(p1, p2, eps=1e-9)
    
    if res is None:
        return False

    mtv, _ = res
    
    # Calculate magnitude of the required push
    push_dist = math.hypot(mtv[0], mtv[1])
    if push_dist == 0:
        return False

    # Calculate unit vector for the push
    ux, uy = mtv[0] / push_dist, mtv[1] / push_dist

    # 'overlap' is how much we need to move to separate + the desired gap (eps)
    overlap = push_dist + eps

    c1 = p1['cluster_members']
    c2 = p2['cluster_members']
    m1 = max(_cluster_mass(c1), 1e-12)
    m2 = max(_cluster_mass(c2), 1e-12)
    tm = m1 + m2

    # Distribute the push based on cluster mass
    s1 = (m2 / tm) * overlap
    s2 = (m1 / tm) * overlap

    _move_cluster(c1, ux * s1, uy * s1)
    _move_cluster(c2, -ux * s2, -uy * s2)
    
    return True

def fully_separate_two_clusters(particles, grid, cell_size, cid1, cid2, eps=EPS_CONTACT, max_passes=10):
    # Note: Removed "rebuild_cluster_members" call if you are updating it elsewhere, 
    # but keeping it here is safer if lists get desynced.
    rebuild_cluster_members(particles) 
    
    c1 = [p for p in particles if p['cluster_id'] == cid1]
    c2 = [p for p in particles if p['cluster_id'] == cid2]
    
    if not c1 or not c2: return

    for _ in range(max_passes):
        moved = False
        # We only need to move if there is an overlap
        for a in c1:
            for b in c2:
                # Pre-check squared distance
                dx = a['x'] - b['x']
                dy = a['y'] - b['y']
                d2 = dx*dx + dy*dy
                limit = ((a['r'] + b['r']) * 1.42 + eps)**2
                
                if d2 < limit:
                    # Expensive SAT check
                    if resolve_overlap_pair_as_clusters(a, b, eps=eps):
                        moved = True
        
        # STOP! Do not update the spatial grid here. 
        # It is too expensive. Just continue iterating.
        if not moved:
            break

def rotate_cluster_about_com(cluster, angle_rad):
    """
    Rigidly rotate all particles in 'cluster' about its mass-weighted COM by angle_rad.
    Mass ~ r^2 (consistent with your COM definition).
    """
    if not cluster or angle_rad == 0.0:
        return

    # mass-weighted COM
    M = sum(p['r']**2 for p in cluster)
    if M <= 0.0:
        return
    cx = sum(p['x'] * (p['r']**2) for p in cluster) / M
    cy = sum(p['y'] * (p['r']**2) for p in cluster) / M

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    for p in cluster:
        ox = p['x'] - cx
        oy = p['y'] - cy
        rx = c * ox - s * oy
        ry = s * ox + c * oy
        p['x'] = cx + rx
        p['y'] = cy + ry
        
        # --- NEW: Rotate the square's own orientation ---
        p['orientation'] += angle_rad


# 1) Keep only ONE rotation, then snap inside bounds.
def rotate_all_clusters(particles, grid, cell_size,
                        omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cid, cluster in clusters.items():
        if not cluster:
            continue
        M = sum(p['r']**2 for p in cluster)
        if M <= 0.0:
            continue
        mass_scale = 1.0 / (1.0 + math.log(1.0 + M / 1000.0))
        omega = omega_base * mass_scale
        jitter = random.uniform(-omega_brownian, omega_brownian) * mass_scale
        angle = (omega + jitter) * dt

        rotate_cluster_about_com(cluster, angle)
        keep_cluster_inside_bounds(cluster, width, height)  # do not rotate again




def resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2, eps=EPS_CONTACT):
    for _ in range(passes):
        moved_any = False
        update_spatial_grid(particles, grid, cell_size)
        
        # Track which cluster pairs we have already tried to separate this pass
        processed_pairs = set()

        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x']/cell_size), int(p1['y']/cell_size)
            
            # 3x3 Grid Search
            for gx in range(max(0, cx-1), min(len(grid), cx+2)):
                for gy in range(max(0, cy-1), min(len(grid[0]), cy+2)):
                    for p2 in grid[gx][gy]:
                        # Standard checks
                        if p1 is p2 or p1['cluster_id'] == p2['cluster_id']:
                            continue
                        
                        # --- CRITICAL OPTIMIZATION ---
                        # Sort IDs to ensure (1, 2) is same as (2, 1)
                        c_pair = tuple(sorted((p1['cluster_id'], p2['cluster_id'])))
                        if c_pair in processed_pairs:
                            continue 
                        # -----------------------------

                        rs = p1['r'] + p2['r']
                        dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
                        
                        # Use squared distance check to avoid sqrt
                        # (rs * 1.42 + eps)^2
                        threshold = ((rs * 1.42) + eps)**2
                        
                        if dist_sq < threshold:
                            # We found a collision between these two clusters.
                            # Separate them ONCE.
                            fully_separate_two_clusters(
                                particles, grid, cell_size,
                                p1['cluster_id'], p2['cluster_id'],
                                eps=eps, max_passes=10 # Reduced passes (see point 2)
                            )
                            moved_any = True
                            # Mark this pair as processed so we don't do it again for other particles
                            processed_pairs.add(c_pair)

        if not moved_any:
            break

# ---------------------------
# Global, order-independent cleanup
# ---------------------------
def max_penetration_all(particles):
    """
    Maximum inter-cluster penetration depth (rs - d).
    Returns 0.0 if no inter-cluster overlaps exist.
    """
    n = len(particles)
    max_pen = 0.0
    for i in range(n):
        pi = particles[i]
        for j in range(i + 1, n):
            pj = particles[j]
            if pi['cluster_id'] == pj['cluster_id']:
                continue
            rs = pi['r'] + pj['r']
            d = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
            pen = rs - d
            if pen > max_pen:
                max_pen = pen
    return max_pen

def separate_all_overlaps_global(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=40):
    """
    Deterministic global cleanup:
    - Scan all inter-cluster pairs (O(N^2)).
    - If any pair overlaps, rigidly separate those two clusters with an extra push.
    - Repeat until no overlaps remain or max_rounds reached.
    """
    for _ in range(max_rounds):
        moved = False
        update_spatial_grid(particles, grid, cell_size)
        n = len(particles)
        for i in range(n):
            pi = particles[i]
            for j in range(i + 1, n):
                pj = particles[j]
                if pi['cluster_id'] == pj['cluster_id']:
                    continue
                rs = pi['r'] + pj['r']
                dx = pi['x'] - pj['x']; dy = pi['y'] - pj['y']
                # before: if dx*dx + dy*dy < rs*rs:
                dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                if dist < ((rs * 1.42) + eps):
                    fully_separate_two_clusters(
                        particles, grid, cell_size,
                        pi['cluster_id'], pj['cluster_id'],
                        eps=eps + EPS_PUSH, max_passes=10
                    )
                    moved = True


        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)
def enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=6):
    for _ in range(max_rounds):
        moved = False
        update_spatial_grid(particles, grid, cell_size)
        n = len(particles)
        for i in range(n):
            pi = particles[i]
            for j in range(i+1, n):
                pj = particles[j]
                if pi['cluster_id'] == pj['cluster_id']:
                    continue
                rs = pi['r'] + pj['r']
                dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                if dist < ((rs * 1.42) + eps):                    # “finish to EPS”
                    resolve_overlap_pair_as_clusters(pi, pj, eps=eps)
                    moved = True
        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)

def cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2):
    """
    Run the global cleanup a few macro passes until max penetration is zero.
    """
    for _ in range(outer_repeats):
        separate_all_overlaps_global(particles, grid, cell_size, eps=eps, max_rounds=20)
        if max_penetration_all(particles) <= 0.0:
            break
# ---------------------------
# Merge, collisions, movement
# ---------------------------
def merge_clusters(particles, cluster_id1, cluster_id2):
    cluster1 = [p for p in particles if p['cluster_id'] == cluster_id1]
    cluster2 = [p for p in particles if p['cluster_id'] == cluster_id2]
    new_cluster_id = min(cluster_id1, cluster_id2)
    other_cluster_id = max(cluster_id1, cluster_id2)
    for p in particles:
        if p['cluster_id'] == other_cluster_id:
            p['cluster_id'] = new_cluster_id
    merged_members = [p for p in particles if p['cluster_id'] == new_cluster_id]
    for p in merged_members:
        p['cluster_members'] = merged_members
    mass1 = sum(p['r'] ** 2 for p in cluster1)
    mass2 = sum(p['r'] ** 2 for p in cluster2)
    total_mass = mass1 + mass2
    _, _, vx1, vy1, _ = get_cluster_properties(cluster1)
    _, _, vx2, vy2, _ = get_cluster_properties(cluster2)
    denom = total_mass if total_mass > 0 else 1.0
    new_vx = (vx1 * mass1 + vx2 * mass2) / denom
    new_vy = (vy1 * mass1 + vy2 * mass2) / denom
    velocity_reduction = 1 / (1 + total_mass / 10000)
    new_vx *= velocity_reduction; new_vy *= velocity_reduction
    new_angle = math.atan2(new_vy, new_vx)
    for p in merged_members:
        p['angle'] = new_angle

def check_collision_and_merge(p1, p2, grid, cell_size):
    # 1. Same cluster check
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    # 2. Cheap bounding box reject
    if quick_center_aabb_reject(p1, p2, margin=0.0):
        return False

    # 3. Accurate SAT Check
    mtv_res = get_mtv_square_square(p1, p2)
    if mtv_res is None:
        return False # No collision

    # 4. Rigid Separation (Pre-merge)
    # Force separation using the square logic we defined in Step 5
    fully_separate_two_clusters(
        particles_ref, grid, cell_size, 
        p1['cluster_id'], p2['cluster_id'], 
        eps=0.001, 
        max_passes=10
    )
    
    update_spatial_grid(particles_ref, grid, cell_size)

    # 5. Merge
    merge_clusters(particles_ref, p1['cluster_id'], p2['cluster_id'])
    
    update_spatial_grid(particles_ref, grid, cell_size)
    return True

def update_cluster_movement(particles, grid, cell_size):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)
    for cluster_id, cluster in clusters.items():
        if not cluster:
            continue
        total_mass = sum(p['r'] ** 2 for p in cluster)
        com_x = sum(p['x'] * p['r'] ** 2 for p in cluster) / total_mass
        com_y = sum(p['y'] * p['r'] ** 2 for p in cluster) / total_mass
        avg_vx = sum(velocity / p['r'] * math.cos(p['angle']) for p in cluster) / len(cluster)
        avg_vy = sum(velocity / p['r'] * math.sin(p['angle']) for p in cluster) / len(cluster)
        vel_scale = 1 / (1 + math.log(1 + total_mass / 1000))
        avg_vx *= vel_scale; avg_vy *= vel_scale
        new_com_x = com_x + avg_vx
        new_com_y = com_y + avg_vy
        br_angle = random.uniform(0, 2 * math.pi)
        mf = 1 / math.sqrt(max(total_mass, 1e-12))
        new_com_x += velocity * mf * math.cos(br_angle)
        new_com_y += velocity * mf * math.sin(br_angle)
        final_angle = br_angle
        offsets = [(p['x'] - com_x, p['y'] - com_y, p['r']) for p in cluster]
        projected = [(new_com_x + ox, new_com_y + oy, r) for ox, oy, r in offsets]
        xs_left = [x - r for x, y, r in projected]
        xs_right = [x + r for x, y, r in projected]
        ys_top = [y - r for x, y, r in projected]
        ys_bottom = [y + r for x, y, r in projected]
        shift_x = shift_y = 0.0
        bounced_x = bounced_y = False
        if min(xs_left) < 0: shift_x, bounced_x = -min(xs_left), True
        elif max(xs_right) > width: shift_x, bounced_x = width - max(xs_right), True
        if min(ys_top) < 0: shift_y, bounced_y = -min(ys_top), True
        elif max(ys_bottom) > height: shift_y, bounced_y = height - max(ys_bottom), True
        new_com_x += shift_x; new_com_y += shift_y
        if bounced_x: final_angle = math.pi - final_angle
        if bounced_y: final_angle = -final_angle
        for idx, p in enumerate(cluster):
            ox, oy, r = offsets[idx]
            p['x'] = new_com_x + ox
            p['y'] = new_com_y + oy
            p['angle'] = final_angle

        keep_cluster_inside_bounds(cluster, width, height)
        
        
def compute_rdf(particles, width, height, dr=2.0, max_r=None):
    """
    Calculates the 2D Radial Distribution Function g(r).
    
    Args:
        particles: List of particle dicts.
        width, height: Simulation bounds.
        dr: Bin width (thickness of the rings).
        max_r: Maximum distance to calculate. Defaults to min(width, height)/2.
        
    Returns:
        r_vals (np.array): Midpoints of the bins.
        g_r (np.array): The calculated RDF values.
    """
    N = len(particles)
    if N < 2:
        return np.array([]), np.array([])

    # Convert particle positions to numpy array
    coords = np.array([[p['x'], p['y']] for p in particles])
    
    # Calculate all pair-wise distances (vectorized)
    # diff shape: (N, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    
    # dists shape: (N, N)
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Extract upper triangle to get unique pairs (excluding self-distance)
    # This flattens the array to shape (N*(N-1)/2, )
    pair_dists = dists[np.triu_indices(N, k=1)]
    
    if max_r is None:
        max_r = min(width, height) / 2.0
        
    # Define bins
    bins = np.arange(0, max_r + dr, dr)
    
    # Compute histogram
    hist, bin_edges = np.histogram(pair_dists, bins=bins)
    
    # Normalization
    # g(r) = (count in shell) / (area of shell * global density)
    # Note: We multiply hist by 2 because we only calculated pairs once (upper triangle),
    # but the density definition assumes centering on every particle.
    
    r_vals = bin_edges[:-1] + dr / 2
    
    # Area of annulus in 2D = pi * (r_outer^2 - r_inner^2) ~ 2 * pi * r * dr
    # Exact area:
    annulus_area = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    
    # Global number density
    rho = N / (width * height)
    
    # Calculate g(r)
    # We normalize by N because we are averaging over all particles as centers
    g_r = (hist * 2) / (annulus_area * rho * N)
    
    return r_vals, g_r
# ---------------------------
# Simulation per composition
# ---------------------------
def run_simulation_for_composition(N_total, radii_pair, fraction_A, max_time,
                                   store_every_k=1, save_dir=save_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_plots, exist_ok=True)

    

    count_A = int(round(N_total * fraction_A))
    count_B = N_total - count_A

    cell_size = max(radii_pair) * 2
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]

    particles = []
    cluster_id_counter = 0

    def place_particles_of_type(radius, count):
        nonlocal cluster_id_counter
        placed = 0
        max_attempts = 10000
        for _ in range(count):
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(spawn_x0 + radius, spawn_x0 + spawn_width - radius)
                y = random.uniform(spawn_y0 + radius, spawn_y0 + spawn_height - radius)
                if check(particles, x, y, radius, grid, cell_size):
                    # Inside the particle creation dict:
                    particle = {
                        'x': x, 'y': y, 'r': float(radius),
                        'shape': 'square',              # <--- NEW
                        'orientation': random.uniform(0,2*math.pi),            # <--- NEW (radians)
                        'cluster_id': cluster_id_counter,
                        'cluster_members': [], 
                        'angle': random.uniform(0, 2 * math.pi),
                        'id': len(particles)
                    }
                    particles.append(particle)
                    insert_into_grid(particles[-1], grid, cell_size)
                    cluster_id_counter += 1
                    placed += 1
                    break
                attempts += 1
            else:
                print(f"Warning: Could not place particle after {max_attempts} attempts")
        return placed

    place_particles_of_type(radii_pair[0], count_A)
    place_particles_of_type(radii_pair[1], count_B)

    for p in particles:
        p['cluster_members'] = [p]

    num_saved_frames = (max_time // store_every_k) + 1
    trajectories = np.zeros((num_saved_frames, N_total, 2), dtype=np.float32)
    orientations = np.zeros((num_saved_frames, N_total), dtype=np.float32)

    global_pairs = []
    time_evolution = []

    t_save_idx = 0
    start_time = time.time()

    # Initial cleanup
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
    cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=5)
    enforce_eps_gap_all(particles, grid, cell_size)

    for time_step in range(max_time):
        update_spatial_grid(particles, grid, cell_size)
        # Phase A: rigid rotation for each cluster
        rotate_all_clusters(particles, grid, cell_size,
                    omega_base=ROT_OMEGA_BASE,
                    omega_brownian=ROT_BROWNIAN,
                    dt=1.0)


        clusters_map = defaultdict(list)
        for p in particles:
            clusters_map[p['cluster_id']].append(p)

        for p in particles:
            if len(clusters_map[p['cluster_id']]) == 1:
                p['angle'] = random.uniform(0, 2 * math.pi)
                mf = 1.0 / p['r']
                new_x = p['x'] + velocity * mf * math.cos(p['angle'])
                new_y = p['y'] + velocity * mf * math.sin(p['angle'])
                if new_x < p['r'] or new_x > width - p['r']:
                    p['angle'] = math.pi - p['angle']
                if new_y < p['r'] or new_y > height - p['r']:
                    p['angle'] = -p['angle']
                p['x'] = max(p['r'], min(width - p['r'], new_x))
                p['y'] = max(p['r'], min(height - p['r'], new_y))

        checked_pairs = set()
        global particles_ref
        particles_ref = particles
        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x'] / cell_size), int(p1['y'] / cell_size)
            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for p2 in grid[gx][gy]:
                        if p1 is p2:
                            continue
                        if p1['cluster_id'] == p2['cluster_id']:
                            continue
                        cluster_pair = tuple(sorted((p1['cluster_id'], p2['cluster_id'])))
                        if cluster_pair in checked_pairs:
                            continue
                        if check_collision_and_merge(p1, p2, grid, cell_size):
                            update_spatial_grid(particles, grid, cell_size)
                            merge_clusters(particles, p1['cluster_id'], p2['cluster_id'])
                            update_spatial_grid(particles, grid, cell_size)
                            checked_pairs.add(cluster_pair)

        # Global cleanup after merges
        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=5)
        enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=6)

        # Move clusters and clean again
        update_cluster_movement(particles, grid, cell_size)
        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=5)
        enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=6)
        # Guarantee the rendered frame is inside the box
        enforce_world_bounds_all(particles, width, height)

        clusters = defaultdict(list)
        for p in particles:
            clusters[p['cluster_id']].append(p)

        cluster_stats = {}
        for cid, members in clusters.items():
            Rg = radius_of_gyration(members)
            M = cluster_mass(members)
            cluster_stats[cid] = {"N": len(members), "M": M, "Rg": Rg}
            if M >= min_mass and Rg > min_Rg:
                global_pairs.append((Rg, M))
        time_evolution.append({"t": time_step, "clusters": cluster_stats})

        if check_one_cluster(particles):
            print("All particles merged into one cluster at time step", time_step)
            break

        if (time_step % store_every_k) == 0:
            for idx, p in enumerate(particles):
                trajectories[t_save_idx, idx, 0] = p['x']
                trajectories[t_save_idx, idx, 1] = p['y']
                orientations[t_save_idx, idx] = p.get('orientation', 0.0)
            t_save_idx += 1
    elapsed = time.time() - start_time
    trajectories = trajectories[:t_save_idx]

    if len(global_pairs) >= min_pts_for_fit:
        unique = {}
        for rg, m in global_pairs:
            key = (round(rg, 8), round(m, 8))
            if key not in unique:
                unique[key] = (rg, m)
        Rg_vals, M_vals = zip(*unique.values())
        logRg = np.log(np.array(Rg_vals)); logM = np.log(np.array(M_vals))
        slope, intercept = np.polyfit(logRg, logM, 1)
        yhat = slope * logRg + intercept
        ss_res = np.sum((logM - yhat) ** 2)
        ss_tot = np.sum((logM - np.mean(logM)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        Df = float(slope)
    else:
        Df = float('nan'); intercept = float('nan'); r2 = float('nan')

    if len(time_evolution) > 0 and time_evolution[-1]["clusters"]:
        final_clusters = time_evolution[-1]["clusters"]
        largest_by_mass = max(final_clusters.values(), key=lambda c: c["M"])
        Rg_largest = largest_by_mass["Rg"]
        largest_by_n = max(final_clusters.values(), key=lambda c: c["N"])
        M_largest = largest_by_n["M"]
        Rg_largest_n = largest_by_n["Rg"]
    else:
        Rg_largest = 0.0; M_largest = 0.0; Rg_largest_n = 0.0

    radii_per_particle = np.array([p['r'] for p in particles], dtype=np.float32)
    tag = f"comp{fraction_A:.3f}_N{N_total}_t{max_time}"
    traj_path = os.path.join(save_dir, f"traj_{tag}.npy")
    radii_path = os.path.join(save_dir, f"radii_{tag}.npy")
    orient_path = os.path.join(save_dir, f"orient_{tag}.npy")
    meta_path = os.path.join(save_dir, f"meta_{tag}.json")
    np.save(traj_path, trajectories)
    np.save(radii_path, radii_per_particle)
    np.save(orient_path, orientations)
    meta = {
        "fraction_A": fraction_A,
        "count_A": int(round(N_total * fraction_A)),
        "count_B": N_total - int(round(N_total * fraction_A)),
        "radii_pair": list(radii_pair),
        "N_total": N_total,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "saved_frames": int(trajectories.shape[0]),
        "final_clusters": len(set(p['cluster_id'] for p in particles))
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    enforce_world_bounds_all(particles, width, height)
    # Simple Matplotlib visualization
    os.makedirs(save_plots, exist_ok=True)
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for p in particles:
        corners = get_corners(p) # Using helper from collisions.collisions.py
    
    # Create polygon patch
        poly = patches.Polygon(corners, closed=True, facecolor='red', alpha=1.0)
        ax.add_patch(poly)
        
        # Outline
        outline = patches.Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=0.8)
        ax.add_patch(outline)
    plt.xlim(0, width); plt.ylim(0, height); plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Final: {fraction_A:.2f} A, {len(set(p['cluster_id'] for p in particles))} clusters | N={len(particles)}")
    plt.savefig(os.path.join(save_plots, f"final_comp{fraction_A:.3f}.png"), dpi=180, bbox_inches='tight')
    plt.close()

    result = {
        "fraction": fraction_A,
        "Df": float(Df),
        "intercept": float(intercept) if 'intercept' in locals() else None,
        "r2": float(r2) if 'r2' in locals() else None,
        "Rg_largest": float(Rg_largest),
        "Rg_largest_n": float(Rg_largest_n),
        "Mass_largest": float(M_largest),
        "traj_path": traj_path,
        "radii_path": radii_path,
        "meta_path": meta_path,
        "elapsed_s": float(elapsed),
        "num_global_pairs": int(len(global_pairs)),
        "saved_frames": int(trajectories.shape[0]),
        "num_particles": len(particles),
        "num_clusters_final": len(set(p['cluster_id'] for p in particles))
    }
    # ---------------------------
    # RDF Calculation & Plotting
    # ---------------------------
    r_vals, g_r = compute_rdf(particles, width, height, dr=2.0)
    
    # Save RDF Data
    rdf_path = os.path.join(save_dir, f"rdf_{tag}.npy")
    np.save(rdf_path, np.column_stack((r_vals, g_r)))
    
    # Plot RDF
    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, g_r, label=f'RDF (t={max_time})')
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Random distribution')
    plt.xlabel("Distance r (pixels)")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function\n{fraction_A:.2f} A | N={N_total}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_plots, f"rdf_comp{fraction_A:.3f}.png"), dpi=150)
    plt.close()
    return result

# ---------------------------
# Batch runner across compositions
# ---------------------------
def sweep_compositions(N_total, radii_pair, fractions, repeats=1, max_time_local=max_time, store_every_k_local=1):
    all_results = []
    for f in fractions:
        df_list = []; rg_list = []
        for rep in range(repeats):
            seed = int(time.time()) % 100000 + rep
            res = run_simulation_for_composition(N_total=N_total,
                                                 radii_pair=radii_pair,
                                                 fraction_A=f,
                                                 max_time=max_time_local,
                                                 store_every_k=store_every_k_local,
                                                 save_dir=save_dir,
                                                 seed=seed)
            if res is None:
                continue
            Df_largest = math.log(res['Mass_largest']) / math.log(res['Rg_largest_n']) if res['Rg_largest_n'] > 0 else float('nan')
            print(f"Done comp={f:.3f} rep={rep} particles={res['num_particles']} Df={res['Df']:.4f} Df_largest={Df_largest:.4f} Rg_final={res['Rg_largest']:.3f}")
            all_results.append(res)
            df_list.append(Df_largest)
            rg_list.append(res['Rg_largest'])
        if df_list:
            print(f"comp {f:.3f}: Df mean {np.nanmean(df_list):.4f} std {np.nanstd(df_list):.4f}; Rg mean {np.mean(rg_list):.3f} std {np.std(rg_list):.3f}")

    # Optional summary plots
    if all_results:
        comps = sorted(set([r['fraction'] for r in all_results]))
        Df_mean = []; Df_std = []; Rg_mean = []; Rg_std = []
        for c in comps:
            vals = [r for r in all_results if r['fraction'] == c]
            Dfs = np.array([v['Df'] for v in vals], dtype=np.float64)
            Rgs = np.array([v['Rg_largest'] for v in vals], dtype=np.float64)
            Df_mean.append(np.nanmean(Dfs)); Df_std.append(np.nanstd(Dfs))
            Rg_mean.append(np.mean(Rgs)); Rg_std.append(np.std(Rgs))
        os.makedirs(save_plots, exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.errorbar(comps, Df_mean, yerr=Df_std, fmt='o-', capsize=4, label='Df (mean ± std)')
        plt.xlabel("Composition (fraction type A)"); plt.ylabel("Fractal dimension Df")
        plt.title("Df vs composition"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(save_plots, "Df_vs_composition.png")); plt.close()
        plt.figure(figsize=(7, 4))
        plt.errorbar(comps, Rg_mean, yerr=Rg_std, fmt='s-', capsize=4, label='Rg largest (mean ± std)')
        plt.xlabel("Composition (fraction type A)"); plt.ylabel("Rg (final)")
        plt.title("Rg (largest cluster at final time) vs composition"); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(save_plots, "Rg_vs_composition.png")); plt.close()
    return all_results

# ---------------------------
# Example usage
# ---------------------------
# --- Tkinter UI for DLCA simulation ---
import threading
import tkinter as tk
from tkinter import ttk, messagebox

def launch_ui():
    root = tk.Tk()
    root.title("DLCA Runner")

    # Vars
    particles_var = tk.StringVar(value="265")
    rA_var = tk.StringVar(value="2")
    rB_var = tk.StringVar(value="10")
    fracA_var = tk.StringVar(value="0.668")
    status_var = tk.StringVar(value="Idle")

    # Grid helpers
    def add_row(label_text, widget, row):
        ttk.Label(root, text=label_text).grid(row=row, column=0, padx=8, pady=6, sticky="e")
        widget.grid(row=row, column=1, padx=8, pady=6, sticky="we")

    # Inputs
    e_particles = ttk.Entry(root, textvariable=particles_var)
    e_rA = ttk.Entry(root, textvariable=rA_var)
    e_rB = ttk.Entry(root, textvariable=rB_var)
    e_fracA = ttk.Entry(root, textvariable=fracA_var)

    add_row("No. of particles", e_particles, 0)
    add_row("Radius A", e_rA, 1)
    add_row("Radius B", e_rB, 2)
    add_row("Fraction A (0-1)", e_fracA, 3)

    # Status
    ttk.Label(root, text="Status").grid(row=4, column=0, padx=8, pady=6, sticky="e")
    status_lbl = ttk.Label(root, textvariable=status_var)
    status_lbl.grid(row=4, column=1, padx=8, pady=6, sticky="we")

    # Button handlers
    def enable_ui(enable=True):
        state = "normal" if enable else "disabled"
        start_btn.config(state=state)

    def on_done(msg, results):
        status_var.set(msg)
        enable_ui(True)
        try:
            # Show a brief summary
            if results:
                last = results[-1]
                summary = (
                    f"Saved frames: {last.get('saved_frames')}\n"
                    f"Particles: {last.get('num_particles')}\n"
                    f"Final clusters: {last.get('num_clusters_final')}\n"
                    f"Df (fit): {last.get('Df')}\n"
                    f"Elapsed (s): {last.get('elapsed_s'):.2f}"
                )
            else:
                summary = "No results returned."
            messagebox.showinfo("Simulation complete", summary)
        except Exception as e:
            messagebox.showwarning("Completed with issues", f"{msg}\n\nNote: {e}")

    def on_error(err):
        status_var.set(f"Error: {err}")
        enable_ui(True)
        messagebox.showerror("Simulation failed", str(err))

    def start_run():
        # Validate inputs
        try:
            N_total = int(particles_var.get())
            rA = float(rA_var.get())
            rB = float(rB_var.get())
            fA = float(fracA_var.get())
            if N_total <= 0:
                raise ValueError("Number of particles must be positive")
            if rA <= 0 or rB <= 0:
                raise ValueError("Radii must be positive")
            if not (0.0 <= fA <= 1.0):
                raise ValueError("Fraction A must be between 0 and 1")
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        enable_ui(False)
        status_var.set("Running...")

        def worker():
            try:
                # Use existing globals max_time and store_every_k for consistency
                results = sweep_compositions(
                    N_total=N_total,
                    radii_pair=(rA, rB),
                    fractions=[fA],
                    repeats=1,
                    max_time_local=max_time,
                    store_every_k_local=store_every_k,
                )
                msg = f"Done: {len(results)} run(s) completed"
                root.after(0, lambda: on_done(msg, results))
            except Exception as e:
                root.after(0, lambda: on_error(e))

        threading.Thread(target=worker, daemon=True).start()

    # Buttons
    btns = ttk.Frame(root)
    btns.grid(row=5, column=0, columnspan=2, pady=8)
    start_btn = ttk.Button(btns, text="Start", command=start_run)
    quit_btn = ttk.Button(btns, text="Quit", command=root.destroy)
    start_btn.pack(side="left", padx=6)
    quit_btn.pack(side="left", padx=6)

    # Layout
    root.columnconfigure(1, weight=1)
    root.minsize(420, 220)

    root.mainloop()

# Replace the former __main__ block with UI
if __name__ == "__main__":
    print("Output:", save_dir, "| Plots:", save_plots)
    launch_ui()