"""
dlca_mixed_system.py — Mixed Circles and Squares
Adapted from dlca_batch_keep_logic.py
"""

import random
import math
import time
import json
import os
from collections import defaultdict

# Ensure collisions.py is in the same folder and has these functions
from collisions import (
    dist, get_corners, project_polygon, 
    get_mtv_square_square, get_mtv_circle_circle, get_mtv_circle_square,
    quick_center_aabb_reject
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------
# Config
# ---------------------------
width = 2000
height = 1500
spawn_width = 1000
spawn_height = 1000
spawn_x0 = (width - spawn_width) / 2.0
spawn_y0 = (height - spawn_height) / 2.0

velocity = 5.0
max_time = 10000
min_mass = 10.0
min_Rg = 1e-8
min_pts_for_fit = 6
save_dir = "sim_npy_out_mixed"
save_plots = "plots_mixed"
store_every_k = 1
repeats = 1

EPS_CONTACT = 1.0
EPS_PUSH    = 0.2
ROT_OMEGA_BASE = 0.0     
ROT_BROWNIAN   = 0.12     

# ---------------------------
# Helpers for Mixed Shapes
# ---------------------------
def get_bounding_radius(p):
    """Returns the radius of the bounding circle for the shape."""
    if p['shape'] == 'square':
        return p['r'] * 1.415  # approx sqrt(2)
    return p['r']              # circle

def get_mtv_mixed(p1, p2, eps=1e-9):
    """
    Dispatches to the correct collision function based on shape pair.
    Returns (mtv_p1, mtv_p2) or None.
    """
    s1 = p1['shape']
    s2 = p2['shape']

    if s1 == 'square' and s2 == 'square':
        return get_mtv_square_square(p1, p2, eps)
    
    elif s1 == 'circle' and s2 == 'circle':
        return get_mtv_circle_circle(p1, p2, eps)
    
    elif s1 == 'circle' and s2 == 'square':
        return get_mtv_circle_square(p1, p2, eps)
    
    elif s1 == 'square' and s2 == 'circle':
        # collisions.py usually expects (circle, square). 
        # If we flip inputs, the output (mtv1, mtv2) refers to (p2, p1).
        # We must swap them back to match (p1, p2).
        res = get_mtv_circle_square(p2, p1, eps)
        if res:
            return (res[1], res[0]) # Swap: mtv_sq (p1), mtv_circ (p2)
        return None
        
    return None

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

def check_exact_placement_overlap(candidate, grid, cell_size, eps=0.0):
    """
    Broad Phase (Bounding Circle) + Narrow Phase (Exact MTV Check)
    """
    cell_x, cell_y = int(candidate['x'] / cell_size), int(candidate['y'] / cell_size)
    bounding_r1 = get_bounding_radius(candidate)
    
    for i in range(max(0, cell_x - 1), min(len(grid), cell_x + 2)):
        for j in range(max(0, cell_y - 1), min(len(grid[0]), cell_y + 2)):
            for p in grid[i][j]:
                # Broad Phase: Fast circle reject
                dist_sq = distance(candidate['x'], candidate['y'], p['x'], p['y'])
                bounding_r2 = get_bounding_radius(p)
                limit_sq = (bounding_r1 + bounding_r2 + eps)**2
                
                if dist_sq > limit_sq:
                    continue # Bounding circles don't overlap
                
                # Narrow Phase: Precise MTV check
                # If get_mtv_mixed returns anything other than None, they actually overlap
                if get_mtv_mixed(candidate, p, eps=eps) is not None:
                    return False 
                    
    return True

def keep_cluster_inside_bounds(cluster, width, height):
    # Calculate global min/max based on shape
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for p in cluster:
        if p['shape'] == 'square':
            corners = get_corners(p)
            for cx, cy in corners:
                min_x = min(min_x, cx); max_x = max(max_x, cx)
                min_y = min(min_y, cy); max_y = max(max_y, cy)
        else:
            # Circle bounds
            min_x = min(min_x, p['x'] - p['r']); max_x = max(max_x, p['x'] + p['r'])
            min_y = min(min_y, p['y'] - p['r']); max_y = max(max_y, p['y'] + p['r'])

    dx = 0.0; dy = 0.0
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
    if not cluster: return 0.0
    masses = [p['r']**2 for p in cluster]
    M = sum(masses)
    if M == 0: return 0.0
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

    # 1. Cheap Reject (Bounding Circle Check)
    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    
    # Calculate safe distance based on shapes
    r1_safe = get_bounding_radius(p1)
    r2_safe = get_bounding_radius(p2)
    safe_dist = r1_safe + r2_safe + eps

    if dist_sq > safe_dist**2:
        return False

    # 2. Accurate MTV Check (Dispatch based on shape)
    res = get_mtv_mixed(p1, p2, eps=1e-9)
    
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
    
    if not c1 or not c2: return

    for _ in range(max_passes):
        moved = False
        for a in c1:
            for b in c2:
                # Pre-check squared distance using dynamic bounding radii
                dx = a['x'] - b['x']
                dy = a['y'] - b['y']
                d2 = dx*dx + dy*dy
                
                ra = get_bounding_radius(a)
                rb = get_bounding_radius(b)
                limit = (ra + rb + eps)**2
                
                if d2 < limit:
                    if resolve_overlap_pair_as_clusters(a, b, eps=eps):
                        moved = True
        
        if not moved:
            break

def rotate_cluster_about_com(cluster, angle_rad):
    if not cluster or angle_rad == 0.0:
        return

    M = sum(p['r']**2 for p in cluster)
    if M <= 0.0: return
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
        
        # Only squares need orientation updates, but harmless for circles
        p['orientation'] += angle_rad

def rotate_all_clusters(particles, grid, cell_size,
                        omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cid, cluster in clusters.items():
        if not cluster: continue
        M = sum(p['r']**2 for p in cluster)
        if M <= 0.0: continue
        mass_scale = 1.0 / (1.0 + math.log(1.0 + M / 1000.0))
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
                        
                        # Dynamic safe distance check
                        ra = get_bounding_radius(p1)
                        rb = get_bounding_radius(p2)
                        threshold = (ra + rb + eps)**2
                        
                        dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
                        
                        if dist_sq < threshold:
                            fully_separate_two_clusters(
                                particles, grid, cell_size,
                                p1['cluster_id'], p2['cluster_id'],
                                eps=eps, max_passes=10
                            )
                            moved_any = True
                            processed_pairs.add(c_pair)

        if not moved_any:
            break

def max_penetration_all(particles):
    n = len(particles)
    max_pen = 0.0
    for i in range(n):
        pi = particles[i]
        for j in range(i + 1, n):
            pj = particles[j]
            if pi['cluster_id'] == pj['cluster_id']:
                continue
            
            # Simple approximation for max penetration check
            # (Strictly calculating max pen for mixed shapes is expensive)
            # We use bounding circles here for speed in this metric
            ra = get_bounding_radius(pi)
            rb = get_bounding_radius(pj)
            rs = ra + rb
            d = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
            pen = rs - d
            if pen > max_pen:
                max_pen = pen
    return max_pen

def separate_all_overlaps_global(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=40):
    for _ in range(max_rounds):
        moved = False
        update_spatial_grid(particles, grid, cell_size)
        processed_pairs = set() 

        for i in range(len(particles)):
            pi = particles[i]
            # Optimization: Use grid instead of O(N^2) here too if possible, 
            # but user logic was N^2. Keeping N^2 for "global cleanup" safety, 
            # but adding processed_pairs optimization.
            for j in range(i + 1, len(particles)):
                pj = particles[j]
                if pi['cluster_id'] == pj['cluster_id']:
                    continue
                
                c_pair = tuple(sorted((pi['cluster_id'], pj['cluster_id'])))
                if c_pair in processed_pairs:
                    continue

                ra = get_bounding_radius(pi)
                rb = get_bounding_radius(pj)
                dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                
                if dist < (ra + rb + eps):
                    fully_separate_two_clusters(
                        particles, grid, cell_size,
                        pi['cluster_id'], pj['cluster_id'],
                        eps=eps + EPS_PUSH, max_passes=10
                    )
                    moved = True
                    processed_pairs.add(c_pair)

        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)

def enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=6):
    for _ in range(max_rounds):
        moved = False
        update_spatial_grid(particles, grid, cell_size)
        processed_pairs = set()

        for i in range(len(particles)):
            pi = particles[i]
            cx, cy = int(pi['x']/cell_size), int(pi['y']/cell_size)
            # Grid search for gap enforcement
            for gx in range(max(0, cx-1), min(len(grid), cx+2)):
                for gy in range(max(0, cy-1), min(len(grid[0]), cy+2)):
                    for pj in grid[gx][gy]:
                        if pi is pj or pi['cluster_id'] == pj['cluster_id']:
                            continue
                        
                        c_pair = tuple(sorted((pi['cluster_id'], pj['cluster_id'])))
                        if c_pair in processed_pairs:
                            continue

                        ra = get_bounding_radius(pi)
                        rb = get_bounding_radius(pj)
                        dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                        
                        if dist < (ra + rb + eps):
                            resolve_overlap_pair_as_clusters(pi, pj, eps=eps)
                            moved = True
                            processed_pairs.add(c_pair)
        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)

def cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2):
    for _ in range(outer_repeats):
        separate_all_overlaps_global(particles, grid, cell_size, eps=eps, max_rounds=20)
        # Relaxed exit condition for mixed shapes (perfect 0.0 is hard with approx bounds)
        if max_penetration_all(particles) <= 0.1: 
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
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    if quick_center_aabb_reject(p1, p2, margin=0.0):
        return False

    # Dispatch to correct MTV function
    res = get_mtv_mixed(p1, p2)
    if res is None:
        return False # No collision

    # Rigid Separation (Pre-merge)
    fully_separate_two_clusters(
        particles_ref, grid, cell_size, 
        p1['cluster_id'], p2['cluster_id'], 
        eps=0.001, 
        max_passes=10
    )
    
    # We update grid sparsely, or not at all if logic is robust, 
    # but here we follow established pattern of safety.
    # removed update_spatial_grid call for speed, relying on outer loop updates

    merge_clusters(particles_ref, p1['cluster_id'], p2['cluster_id'])
    
    update_spatial_grid(particles_ref, grid, cell_size)
    return True

def update_cluster_movement(particles, grid, cell_size):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)
    for cluster_id, cluster in clusters.items():
        if not cluster: continue
        
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
        offsets = [(p['x'] - com_x, p['y'] - com_y) for p in cluster]
        
        # Simple projection of COM move (not checking rigid bounds exactly here for speed)
        # Bouncing Logic based on COM extents
        shift_x = shift_y = 0.0
        bounced_x = bounced_y = False
        
        # Calculate cluster extents relative to COM
        min_ox, max_ox = float('inf'), float('-inf')
        min_oy, max_oy = float('inf'), float('-inf')
        
        for p in cluster:
            # Conservative radius for bounds check during movement
            rb = get_bounding_radius(p)
            ox = p['x'] - com_x
            oy = p['y'] - com_y
            min_ox = min(min_ox, ox - rb)
            max_ox = max(max_ox, ox + rb)
            min_oy = min(min_oy, oy - rb)
            max_oy = max(max_oy, oy + rb)
            
        if new_com_x + min_ox < 0: 
            shift_x = -(new_com_x + min_ox); bounced_x = True
        elif new_com_x + max_ox > width: 
            shift_x = width - (new_com_x + max_ox); bounced_x = True
            
        if new_com_y + min_oy < 0: 
            shift_y = -(new_com_y + min_oy); bounced_y = True
        elif new_com_y + max_oy > height: 
            shift_y = height - (new_com_y + max_oy); bounced_y = True
            
        new_com_x += shift_x
        new_com_y += shift_y
        
        if bounced_x: final_angle = math.pi - final_angle
        if bounced_y: final_angle = -final_angle
        
        for idx, p in enumerate(cluster):
            ox, oy = offsets[idx]
            p['x'] = new_com_x + ox
            p['y'] = new_com_y + oy
            p['angle'] = final_angle

        keep_cluster_inside_bounds(cluster, width, height)

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

    cell_size = max(radii_pair) * 3 # Larger cell for safety with rotation
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]

    particles = []
    cluster_id_counter = 0

    def place_particles_of_type(radius, count, shape_type):
        nonlocal cluster_id_counter
        placed = 0
        max_attempts = 10000
        for _ in range(count):
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(spawn_x0 + radius, spawn_x0 + spawn_width - radius)
                y = random.uniform(spawn_y0 + radius, spawn_y0 + spawn_height - radius)
                spawn_orientation = random.uniform(0, 2 * math.pi)
                
                # 1. Create a lightweight candidate for the exact check
                candidate = {
                    'x': x, 'y': y, 'r': float(radius),
                    'shape': shape_type,
                    'orientation': spawn_orientation
                }
                
                # 2. Check using Mixed Logic (Broad + Narrow)
                if check_exact_placement_overlap(candidate, grid, cell_size, eps=EPS_CONTACT):
                    
                    # 3. If it passes, build the full particle and append it
                    particle = {
                        'x': x, 'y': y, 'r': float(radius),
                        'shape': shape_type,
                        'orientation': spawn_orientation, # Apply rotation here
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
                print(f"Warning: Could not place {shape_type} particle after {max_attempts} attempts")
        return placed

    # --- CONFIGURING MIXED SYSTEM ---
    # Type A = CIRCLES
    place_particles_of_type(radii_pair[0], count_A, shape_type='circle')
    # Type B = SQUARES
    place_particles_of_type(radii_pair[1], count_B, shape_type='square')

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
        
        rotate_all_clusters(particles, grid, cell_size,
                            omega_base=ROT_OMEGA_BASE,
                            omega_brownian=ROT_BROWNIAN,
                            dt=1.0)

        # Free movement for single particles
        clusters_map = defaultdict(list)
        for p in particles:
            clusters_map[p['cluster_id']].append(p)

        for p in particles:
            if len(clusters_map[p['cluster_id']]) == 1:
                p['angle'] = random.uniform(0, 2 * math.pi)
                mf = 1.0 / p['r']
                new_x = p['x'] + velocity * mf * math.cos(p['angle'])
                new_y = p['y'] + velocity * mf * math.sin(p['angle'])
                
                # Simple bounds bounce
                rb = get_bounding_radius(p)
                if new_x < rb or new_x > width - rb:
                    p['angle'] = math.pi - p['angle']
                if new_y < rb or new_y > height - rb:
                    p['angle'] = -p['angle']
                
                p['x'] = max(rb, min(width - rb, new_x))
                p['y'] = max(rb, min(height - rb, new_y))

        checked_pairs = set()
        global particles_ref
        particles_ref = particles
        
        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x'] / cell_size), int(p1['y'] / cell_size)
            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for p2 in grid[gx][gy]:
                        if p1 is p2: continue
                        if p1['cluster_id'] == p2['cluster_id']: continue
                        
                        cluster_pair = tuple(sorted((p1['cluster_id'], p2['cluster_id'])))
                        if cluster_pair in checked_pairs: continue
                        
                        if check_collision_and_merge(p1, p2, grid, cell_size):
                            # Grid updated inside collision function if needed
                            checked_pairs.add(cluster_pair)

        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2)
        
        update_cluster_movement(particles, grid, cell_size)
        
        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2)
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

    # --- Fractal Dimension Calculation ---
    if len(global_pairs) >= min_pts_for_fit:
        unique = {}
        for rg, m in global_pairs:
            key = (round(rg, 8), round(m, 8))
            if key not in unique: unique[key] = (rg, m)
        Rg_vals, M_vals = zip(*unique.values())
        logRg = np.log(np.array(Rg_vals)); logM = np.log(np.array(M_vals))
        slope, intercept = np.polyfit(logRg, logM, 1)
        ss_res = np.sum((logM - (slope * logRg + intercept)) ** 2)
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
        "count_A": count_A,
        "count_B": count_B,
        "radii_pair": list(radii_pair),
        "N_total": N_total,
        "max_time": max_time,
        "saved_frames": int(trajectories.shape[0]),
        "final_clusters": len(set(p['cluster_id'] for p in particles))
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- Mixed Visualization ---
    os.makedirs(save_plots, exist_ok=True)
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for p in particles:
        if p['shape'] == 'square':
            corners = get_corners(p)
            poly = patches.Polygon(corners, closed=True, facecolor='red', alpha=0.8, edgecolor='black')
            ax.add_patch(poly)
        else:
            circle = plt.Circle((p['x'], p['y']), p['r'], facecolor='blue', alpha=0.8, edgecolor='black')
            ax.add_patch(circle)

    plt.xlim(0, width); plt.ylim(0, height)
    ax.set_aspect('equal')
    plt.title(f"Mixed System: Circles (Blue) + Squares (Red)\nFinal Clusters: {len(set(p['cluster_id'] for p in particles))}")
    plt.savefig(os.path.join(save_plots, f"final_comp{fraction_A:.3f}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        "fraction": fraction_A, "Df": Df, "Rg_largest": Rg_largest, 
        "num_particles": len(particles), "elapsed_s": elapsed
    }

# ---------------------------
# Batch runner across compositions
# ---------------------------
def sweep_compositions(N_total, radii_pair, fractions, repeats=1, max_time_local=max_time, store_every_k_local=1):
    all_results = []
    for f in fractions:
        for rep in range(repeats):
            seed = int(time.time()) % 100000 + rep
            res = run_simulation_for_composition(
                N_total=N_total, radii_pair=radii_pair, fraction_A=f,
                max_time=max_time_local, store_every_k=store_every_k_local,
                save_dir=save_dir, seed=seed
            )
            if res:
                print(f"Done comp={f:.3f} rep={rep} Df={res['Df']:.4f}")
                all_results.append(res)
    return all_results

# ---------------------------
# UI Entry Point
# ---------------------------
import threading
import tkinter as tk
from tkinter import ttk, messagebox

def launch_ui():
    root = tk.Tk()
    root.title("Mixed DLCA Runner")

    # Vars
    particles_var = tk.StringVar(value="150")
    rA_var = tk.StringVar(value="4")
    rB_var = tk.StringVar(value="8")
    fracA_var = tk.StringVar(value="0.5")
    status_var = tk.StringVar(value="Idle")

    def add_row(label_text, widget, row):
        ttk.Label(root, text=label_text).grid(row=row, column=0, padx=8, pady=6, sticky="e")
        widget.grid(row=row, column=1, padx=8, pady=6, sticky="we")

    add_row("No. of particles", ttk.Entry(root, textvariable=particles_var), 0)
    add_row("Radius A (Circle)", ttk.Entry(root, textvariable=rA_var), 1)
    add_row("Radius B (Square)", ttk.Entry(root, textvariable=rB_var), 2)
    add_row("Fraction A (0-1)", ttk.Entry(root, textvariable=fracA_var), 3)

    ttk.Label(root, text="Status").grid(row=4, column=0, padx=8, pady=6, sticky="e")
    ttk.Label(root, textvariable=status_var).grid(row=4, column=1, padx=8, pady=6, sticky="we")

    def start_run():
        try:
            N = int(particles_var.get())
            rA = float(rA_var.get())
            rB = float(rB_var.get())
            fA = float(fracA_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid inputs")
            return

        start_btn.config(state="disabled")
        status_var.set("Running...")

        def worker():
            try:
                results = sweep_compositions(N, (rA, rB), [fA], 1, max_time_local=max_time)
                root.after(0, lambda: status_var.set(f"Done. Saved to {save_dir}"))
            except Exception as e:
                print(e)
                root.after(0, lambda: status_var.set("Error occurred"))
            finally:
                root.after(0, lambda: start_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    start_btn = ttk.Button(root, text="Start", command=start_run)
    start_btn.grid(row=5, column=0, columnspan=2, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    launch_ui()