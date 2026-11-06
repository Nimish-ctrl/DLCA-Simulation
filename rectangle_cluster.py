"""
dlca_batch_keep_logic.py — rigid contacts, no overlaps (global cleanup + safety push)
Updated to support Rectangles (Length & Width) for a Binary System
"""

import random
import math
import time
import json
import os
from collections import defaultdict
from collisions2 import dist, get_corners, project_polygon, get_mtv_square_square, quick_center_aabb_reject
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

spawn_x0 = (width - spawn_width) / 2.0
spawn_y0 = (height - spawn_height) / 2.0

velocity = 5.0
max_time = 10000
min_mass = 10.0
min_Rg = 1e-8
min_pts_for_fit = 6
save_dir = "sim_npy_out_rectangle"
save_plots = "plots_rectangle"
store_every_k = 1
repeats = 1

EPS_CONTACT = 1.0
EPS_PUSH    = 0.2
ROT_OMEGA_BASE = 0.0     
ROT_BROWNIAN   = 0.12   

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

def check_exact_sat(candidate, grid, cell_size, eps=0.0):
    cell_x, cell_y = int(candidate['x'] / cell_size), int(candidate['y'] / cell_size)
    
    for i in range(max(0, cell_x - 1), min(len(grid), cell_x + 2)):
        for j in range(max(0, cell_y - 1), min(len(grid[0]), cell_y + 2)):
            for p in grid[i][j]:
                # Broad Phase: Fast circle reject (Keeps the code running fast!)
                dist_sq = distance(candidate['x'], candidate['y'], p['x'], p['y'])
                limit_sq = (candidate['bounding_r'] + p['bounding_r'] + eps)**2
                
                if dist_sq > limit_sq:
                    continue # Bounding circles don't overlap, so rectangles definitely don't.
                
                # Narrow Phase: Precise SAT check
                # If get_mtv_square_square returns anything other than None, they overlap.
                if get_mtv_square_square(candidate, p, eps=eps) is not None:
                    return False 
                    
    return True

def keep_cluster_inside_bounds(cluster, width, height):
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for p in cluster:
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
    total_mass = sum(p['area'] for p in cluster_particles)
    if total_mass == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    com_x = sum(p['x'] * p['area'] for p in cluster_particles) / total_mass
    com_y = sum(p['y'] * p['area'] for p in cluster_particles) / total_mass
    avg_vx = sum(velocity / p['bounding_r'] * math.cos(p.get('angle', 0)) for p in cluster_particles) / len(cluster_particles)
    avg_vy = sum(velocity / p['bounding_r'] * math.sin(p.get('angle', 0)) for p in cluster_particles) / len(cluster_particles)
    velocity_scale = 1 / (1 + math.log(1 + total_mass / 1000))
    avg_vx *= velocity_scale
    avg_vy *= velocity_scale
    return com_x, com_y, avg_vx, avg_vy, total_mass

def radius_of_gyration(cluster):
    if not cluster:
        return 0.0
    masses = [p['area'] for p in cluster]
    M = sum(masses)
    if M == 0:
        return 0.0
    com_x = sum(p['x']*m for p, m in zip(cluster, masses)) / M
    com_y = sum(p['y']*m for p, m in zip(cluster, masses)) / M
    rg2 = sum(m*((p['x']-com_x)**2 + (p['y']-com_y)**2) for p, m in zip(cluster, masses)) / M
    return math.sqrt(rg2)

def cluster_mass(cluster):
    return sum(p['area'] for p in cluster)

def check_one_cluster(particles):
    return len(set(p['cluster_id'] for p in particles)) == 1

# ---------------------------
# Rigid contact separation
# ---------------------------
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
    return sum(pp['area'] for pp in cluster)

def resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    safe_dist = p1['bounding_r'] + p2['bounding_r'] + eps
    if dist_sq > safe_dist**2:
        return False

    res = get_mtv_square_square(p1, p2, eps=1e-9)
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
    m1 = max(_cluster_mass(c1), 1e-12)
    m2 = max(_cluster_mass(c2), 1e-12)
    tm = m1 + m2

    s1 = (m2 / tm) * overlap
    s2 = (m1 / tm) * overlap

    _move_cluster(c1, ux * s1, uy * s1)
    _move_cluster(c2, -ux * s2, -uy * s2)
    
    return True

def fully_separate_two_clusters(particles, grid, cell_size, cid1, cid2, eps=EPS_CONTACT, max_passes=10):
    rebuild_cluster_members(particles) 
    c1 = [p for p in particles if p['cluster_id'] == cid1]
    c2 = [p for p in particles if p['cluster_id'] == cid2]
    
    if not c1 or not c2: return False # <-- Return False if empty

    any_moved_at_all = False
    for _ in range(max_passes):
        moved = False
        for a in c1:
            for b in c2:
                dx = a['x'] - b['x']
                dy = a['y'] - b['y']
                d2 = dx*dx + dy*dy
                limit = (a['bounding_r'] + b['bounding_r'] + eps)**2
                
                if d2 < limit:
                    if resolve_overlap_pair_as_clusters(a, b, eps=eps):
                        moved = True
                        any_moved_at_all = True
        if not moved:
            break
            
    return any_moved_at_all # <-- Return whether a separation actually occurred

def rotate_cluster_about_com(cluster, angle_rad):
    if not cluster or angle_rad == 0.0:
        return

    M = sum(p['area'] for p in cluster)
    if M <= 0.0:
        return
    cx = sum(p['x'] * p['area'] for p in cluster) / M
    cy = sum(p['y'] * p['area'] for p in cluster) / M

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    for p in cluster:
        ox = p['x'] - cx
        oy = p['y'] - cy
        rx = c * ox - s * oy
        ry = s * ox + c * oy
        p['x'] = cx + rx
        p['y'] = cy + ry
        p['orientation'] += angle_rad

def rotate_all_clusters(particles, grid, cell_size, omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cid, cluster in clusters.items():
        if not cluster:
            continue
        M = sum(p['area'] for p in cluster)
        if M <= 0.0:
            continue
        mass_scale = 1.0 / (1.0 + math.log(1.0 + M))
        omega = omega_base * mass_scale
        jitter = random.uniform(-omega_brownian, omega_brownian) * mass_scale
        angle = (omega + jitter) * dt

        rotate_cluster_about_com(cluster, angle)
        keep_cluster_inside_bounds(cluster, width, height) 

def resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2, eps=EPS_CONTACT):
    for _ in range(passes):
        moved_any = False
        update_spatial_grid(particles, grid, cell_size)
        processed_pairs = set()

        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x']/cell_size), int(p1['y']/cell_size)
            
            for gx in range(max(0, cx-1), min(len(grid), cx+2)):
                for gy in range(max(0, cy-1), min(len(grid[0]), cy+2)):
                    for p2 in grid[gx][gy]:
                        if p1 is p2 or p1['cluster_id'] == p2['cluster_id']:
                            continue
                        
                        c_pair = tuple(sorted((p1['cluster_id'], p2['cluster_id'])))
                        if c_pair in processed_pairs:
                            continue 

                        rs = p1['bounding_r'] + p2['bounding_r']
                        dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
                        threshold = (rs + eps)**2
                        
                        if dist_sq < threshold:
                            # Only trip the "moved" flag if actual SAT separation happened
                            if fully_separate_two_clusters(
                                particles, grid, cell_size,
                                p1['cluster_id'], p2['cluster_id'],
                                eps=eps, max_passes=10 
                            ):
                                moved_any = True
                            processed_pairs.add(c_pair)

        if not moved_any:
            break

# ---------------------------
# Global, order-independent cleanup
# ---------------------------
# Notice: max_penetration_all() has been deleted entirely.

def separate_all_overlaps_global(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=40):
    any_global_move = False
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
                rs = pi['bounding_r'] + pj['bounding_r']
                dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                if dist < (rs + eps):
                    # Only flag true if actual separation occurred
                    if fully_separate_two_clusters(
                        particles, grid, cell_size,
                        pi['cluster_id'], pj['cluster_id'],
                        eps=eps + EPS_PUSH, max_passes=10
                    ):
                        moved = True
                        any_global_move = True
        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)
    return any_global_move

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
                rs = pi['bounding_r'] + pj['bounding_r']
                dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                if dist < (rs + eps):                    
                    if resolve_overlap_pair_as_clusters(pi, pj, eps=eps):
                        moved = True
        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)

def cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2):
    for _ in range(outer_repeats):
        # We now rely on the function returning True if overlaps existed
        any_overlaps_found = separate_all_overlaps_global(particles, grid, cell_size, eps=eps, max_rounds=20)
        if not any_overlaps_found:
            break # Exits cleanly!
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
    mass1 = sum(p['area'] for p in cluster1)
    mass2 = sum(p['area'] for p in cluster2)
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
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    if quick_center_aabb_reject(p1, p2, margin=0.0):
        return False

    mtv_res = get_mtv_square_square(p1, p2)
    if mtv_res is None:
        return False 

    fully_separate_two_clusters(
        particles_ref, grid, cell_size, 
        p1['cluster_id'], p2['cluster_id'], 
        eps=0.001, 
        max_passes=10
    )
    
    update_spatial_grid(particles_ref, grid, cell_size)
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
        total_mass = sum(p['area'] for p in cluster)
        com_x = sum(p['x'] * p['area'] for p in cluster) / total_mass
        com_y = sum(p['y'] * p['area'] for p in cluster) / total_mass
        avg_vx = sum(velocity / p['bounding_r'] * math.cos(p['angle']) for p in cluster) / len(cluster)
        avg_vy = sum(velocity / p['bounding_r'] * math.sin(p['angle']) for p in cluster) / len(cluster)
        vel_scale = 1 / (1 + math.log(1 + total_mass / 1000))
        avg_vx *= vel_scale; avg_vy *= vel_scale
        new_com_x = com_x + avg_vx
        new_com_y = com_y + avg_vy
        br_angle = random.uniform(0, 2 * math.pi)
        mf = 1 / math.sqrt(max(total_mass, 1e-12))
        new_com_x += velocity * mf * math.cos(br_angle)
        new_com_y += velocity * mf * math.sin(br_angle)
        final_angle = br_angle
        offsets = [(p['x'] - com_x, p['y'] - com_y, p['bounding_r']) for p in cluster]
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
    N = len(particles)
    if N < 2:
        return np.array([]), np.array([])

    coords = np.array([[p['x'], p['y']] for p in particles])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    pair_dists = dists[np.triu_indices(N, k=1)]
    
    if max_r is None:
        max_r = min(width, height) / 2.0
        
    bins = np.arange(0, max_r + dr, dr)
    hist, bin_edges = np.histogram(pair_dists, bins=bins)
    
    r_vals = bin_edges[:-1] + dr / 2
    annulus_area = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
    rho = N / (width * height)
    g_r = (hist * 2) / (annulus_area * rho * N)
    
    return r_vals, g_r

# ---------------------------
# Simulation per composition
# ---------------------------
def run_simulation_for_composition(N_total, dims_pair, fraction_A, max_time,
                                   store_every_k=1, save_dir=save_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_plots, exist_ok=True)

    count_A = int(round(N_total * fraction_A))
    count_B = N_total - count_A

    # Find maximum bounding dimension for grid sizing
    max_bounding_r = max(math.hypot(w, h)/2.0 for w, h in dims_pair)
    cell_size = max_bounding_r * 2.5 
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]

    particles = []
    cluster_id_counter = 0

    def place_particles_of_type(dims, count):
        nonlocal cluster_id_counter
        w, h = dims
        bounding_r = math.hypot(w, h) / 2.0
        area = w * h
        
        placed = 0
        max_attempts = 10000
        for _ in range(count):
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(spawn_x0 + bounding_r, spawn_x0 + spawn_width - bounding_r)
                y = random.uniform(spawn_y0 + bounding_r, spawn_y0 + spawn_height - bounding_r)
                spawn_orientation = random.uniform(0, 2*math.pi)
                
                # 1. Create a lightweight candidate for the SAT test
                candidate = {
                    'x': x, 'y': y, 
                    'w': float(w), 'h': float(h), 
                    'bounding_r': bounding_r,
                    'orientation': spawn_orientation  
                }
                
                # 2. Test exact rectangular overlap using SAT
                if check_exact_sat(candidate, grid, cell_size, eps=EPS_CONTACT):
                    
                    # 3. If it passes, build the full particle and append it
                    particle = {
                        'x': x, 'y': y, 
                        'w': float(w), 'h': float(h), 
                        'bounding_r': bounding_r,
                        'area': area,
                        'shape': 'rectangle', 
                        'orientation': spawn_orientation,
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

    place_particles_of_type(dims_pair[0], count_A)
    place_particles_of_type(dims_pair[1], count_B)

    for p in particles:
        p['cluster_members'] = [p]

    num_saved_frames = (max_time // store_every_k) + 1
    trajectories = np.zeros((num_saved_frames, N_total, 2), dtype=np.float32)
    orientations = np.zeros((num_saved_frames, N_total), dtype=np.float32)

    global_pairs = []
    time_evolution = []

    t_save_idx = 0
    start_time = time.time()

    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
    cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=5)
    enforce_eps_gap_all(particles, grid, cell_size)

    for time_step in range(max_time):
        update_spatial_grid(particles, grid, cell_size)
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
                mf = 1.0 / p['bounding_r']
                new_x = p['x'] + velocity * mf * math.cos(p['angle'])
                new_y = p['y'] + velocity * mf * math.sin(p['angle'])
                if new_x < p['bounding_r'] or new_x > width - p['bounding_r']:
                    p['angle'] = math.pi - p['angle']
                if new_y < p['bounding_r'] or new_y > height - p['bounding_r']:
                    p['angle'] = -p['angle']
                p['x'] = max(p['bounding_r'], min(width - p['bounding_r'], new_x))
                p['y'] = max(p['bounding_r'], min(height - p['bounding_r'], new_y))

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

        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=5)
        enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=6)

        update_cluster_movement(particles, grid, cell_size)
        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=5)
        enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=6)
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

    radii_per_particle = np.array([p['bounding_r'] for p in particles], dtype=np.float32)
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
        "dims_pair": list(dims_pair),
        "N_total": N_total,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "saved_frames": int(trajectories.shape[0]),
        "final_clusters": len(set(p['cluster_id'] for p in particles))
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    enforce_world_bounds_all(particles, width, height)
    
    os.makedirs(save_plots, exist_ok=True)
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for p in particles:
        corners = get_corners(p) 
        poly = patches.Polygon(corners, closed=True, facecolor='red', alpha=1.0)
        ax.add_patch(poly)
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
    
    r_vals, g_r = compute_rdf(particles, width, height, dr=2.0)
    rdf_path = os.path.join(save_dir, f"rdf_{tag}.npy")
    np.save(rdf_path, np.column_stack((r_vals, g_r)))
    
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
def sweep_compositions(N_total, dims_pair, fractions, repeats=1, max_time_local=max_time, store_every_k_local=1):
    all_results = []
    for f in fractions:
        df_list = []; rg_list = []
        for rep in range(repeats):
            seed = int(time.time()) % 100000 + rep
            res = run_simulation_for_composition(N_total=N_total,
                                                 dims_pair=dims_pair,
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
# Tkinter UI for DLCA simulation
# ---------------------------
import threading
import tkinter as tk
from tkinter import ttk, messagebox

def launch_ui():
    root = tk.Tk()
    root.title("DLCA Runner (Rectangles)")

    # Vars
    particles_var = tk.StringVar(value="265")
    lenA_var = tk.StringVar(value="10")
    widA_var = tk.StringVar(value="4")
    lenB_var = tk.StringVar(value="20")
    widB_var = tk.StringVar(value="6")
    fracA_var = tk.StringVar(value="0.668")
    status_var = tk.StringVar(value="Idle")

    def add_row(label_text, widget, row):
        ttk.Label(root, text=label_text).grid(row=row, column=0, padx=8, pady=6, sticky="e")
        widget.grid(row=row, column=1, padx=8, pady=6, sticky="we")

    # Inputs
    e_particles = ttk.Entry(root, textvariable=particles_var)
    e_lenA = ttk.Entry(root, textvariable=lenA_var)
    e_widA = ttk.Entry(root, textvariable=widA_var)
    e_lenB = ttk.Entry(root, textvariable=lenB_var)
    e_widB = ttk.Entry(root, textvariable=widB_var)
    e_fracA = ttk.Entry(root, textvariable=fracA_var)

    add_row("No. of particles", e_particles, 0)
    add_row("Length A", e_lenA, 1)
    add_row("Width A", e_widA, 2)
    add_row("Length B", e_lenB, 3)
    add_row("Width B", e_widB, 4)
    add_row("Fraction A (0-1)", e_fracA, 5)

    ttk.Label(root, text="Status").grid(row=6, column=0, padx=8, pady=6, sticky="e")
    status_lbl = ttk.Label(root, textvariable=status_var)
    status_lbl.grid(row=6, column=1, padx=8, pady=6, sticky="we")

    def enable_ui(enable=True):
        state = "normal" if enable else "disabled"
        start_btn.config(state=state)

    def on_done(msg, results):
        status_var.set(msg)
        enable_ui(True)
        try:
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
        try:
            N_total = int(particles_var.get())
            lA = float(lenA_var.get())
            wA = float(widA_var.get())
            lB = float(lenB_var.get())
            wB = float(widB_var.get())
            fA = float(fracA_var.get())
            if N_total <= 0:
                raise ValueError("Number of particles must be positive")
            if lA <= 0 or wA <= 0 or lB <= 0 or wB <= 0:
                raise ValueError("Dimensions must be positive")
            if not (0.0 <= fA <= 1.0):
                raise ValueError("Fraction A must be between 0 and 1")
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        enable_ui(False)
        status_var.set("Running...")

        def worker():
            try:
                results = sweep_compositions(
                    N_total=N_total,
                    dims_pair=((wA, lA), (wB, lB)),
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

    btns = ttk.Frame(root)
    btns.grid(row=7, column=0, columnspan=2, pady=8)
    start_btn = ttk.Button(btns, text="Start", command=start_run)
    quit_btn = ttk.Button(btns, text="Quit", command=root.destroy)
    start_btn.pack(side="left", padx=6)
    quit_btn.pack(side="left", padx=6)

    root.columnconfigure(1, weight=1)
    root.minsize(420, 260)

    root.mainloop()

if __name__ == "__main__":
    print("Output:", save_dir, "| Plots:", save_plots)
    launch_ui()