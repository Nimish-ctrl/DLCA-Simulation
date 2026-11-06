"""
dlca_rectangle_circle_binary.py — DLCA with Binary System: Rectangles + Circles
Rigid contacts, no overlaps, with SAT collision detection
"""

import random
import math
import time
import json
import os
from collections import defaultdict
from collisions_rect_circle import (
    dist, get_corners, project_polygon, get_mtv_any_any, 
    quick_center_aabb_reject
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

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
save_dir = "sim_npy_out_rect_circle"
save_plots = "plots_rect_circle"
store_every_k = 1
repeats = 1

EPS_CONTACT = 1.0
EPS_PUSH    = 0.2
ROT_OMEGA_BASE = 0.0     
ROT_BROWNIAN   = 0.12   

# ---------------------------
# Particle Geometry Helpers
# ---------------------------
def get_particle_area(p):
    """Calculate particle area."""
    if p['shape'] == 'circle':
        return math.pi * p['radius']**2
    else:  # rectangle
        return p['w'] * p['h']

def get_bounding_radius(p):
    """Get bounding radius for broad phase collision detection."""
    return p['bounding_r']

def get_particle_bounds(p):
    """Get min/max x,y bounds for a particle."""
    if p['shape'] == 'circle':
        r = p['radius']
        return (p['x'] - r, p['x'] + r, p['y'] - r, p['y'] + r)
    else:  # rectangle
        corners = get_corners(p)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (min(xs), max(xs), min(ys), max(ys))

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

def check_exact_collision(candidate, grid, cell_size, eps=0.0):
    """
    Check if candidate overlaps with any particle in grid.
    Works for both circles and rectangles.
    """
    cell_x, cell_y = int(candidate['x'] / cell_size), int(candidate['y'] / cell_size)
    
    for i in range(max(0, cell_x - 1), min(len(grid), cell_x + 2)):
        for j in range(max(0, cell_y - 1), min(len(grid[0]), cell_y + 2)):
            for p in grid[i][j]:
                # Broad Phase: Fast circle reject
                dist_sq = distance(candidate['x'], candidate['y'], p['x'], p['y'])
                limit_sq = (candidate['bounding_r'] + p['bounding_r'] + eps)**2
                
                if dist_sq > limit_sq:
                    continue
                
                # Narrow Phase: Precise collision check
                if get_mtv_any_any(candidate, p, eps=eps) is not None:
                    return False 
                    
    return True

def keep_cluster_inside_bounds(cluster, width, height):
    """Keep cluster within simulation bounds."""
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for p in cluster:
        if p['shape'] == 'circle':
            r = p['radius']
            min_x = min(min_x, p['x'] - r)
            max_x = max(max_x, p['x'] + r)
            min_y = min(min_y, p['y'] - r)
            max_y = max(max_y, p['y'] + r)
        else:  # rectangle
            corners = get_corners(p)
            for cx, cy in corners:
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

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
def radius_of_gyration(cluster):
    if not cluster:
        return 0.0
    masses = [get_particle_area(p) for p in cluster]
    M = sum(masses)
    if M == 0:
        return 0.0
    com_x = sum(p['x']*m for p, m in zip(cluster, masses)) / M
    com_y = sum(p['y']*m for p, m in zip(cluster, masses)) / M
    rg2 = sum(m*((p['x']-com_x)**2 + (p['y']-com_y)**2) for p, m in zip(cluster, masses)) / M
    return math.sqrt(rg2)

def cluster_mass(cluster):
    return sum(get_particle_area(p) for p in cluster)

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
    return sum(get_particle_area(pp) for pp in cluster)

def resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT):
    """Resolve overlap between two particles using MTV."""
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    safe_dist = p1['bounding_r'] + p2['bounding_r'] + eps
    if dist_sq > safe_dist**2:
        return False

    res = get_mtv_any_any(p1, p2, eps=1e-9)
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
    
    if not c1 or not c2: 
        return False

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
            
    return any_moved_at_all

def rotate_cluster_about_com(cluster, angle_rad):
    """Rotate cluster about center of mass. Only affects rectangles."""
    if not cluster or angle_rad == 0.0:
        return

    M = sum(get_particle_area(p) for p in cluster)
    if M <= 0.0:
        return
    cx = sum(p['x'] * get_particle_area(p) for p in cluster) / M
    cy = sum(p['y'] * get_particle_area(p) for p in cluster) / M

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    for p in cluster:
        ox = p['x'] - cx
        oy = p['y'] - cy
        rx = c * ox - s * oy
        ry = s * ox + c * oy
        p['x'] = cx + rx
        p['y'] = cy + ry
        
        # Only update orientation for rectangles
        if p['shape'] == 'rectangle':
            p['orientation'] += angle_rad

def rotate_all_clusters(particles, grid, cell_size, omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cid, cluster in clusters.items():
        if not cluster:
            continue
        M = sum(get_particle_area(p) for p in cluster)
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
                            if fully_separate_two_clusters(
                                particles, grid, cell_size,
                                p1['cluster_id'], p2['cluster_id'],
                                eps=eps, max_passes=10 
                            ):
                                moved_any = True
                            processed_pairs.add(c_pair)

        if not moved_any:
            break

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
        any_overlaps_found = separate_all_overlaps_global(particles, grid, cell_size, eps=eps, max_rounds=20)
        if not any_overlaps_found:
            break

# ---------------------------
# Merge, collisions, movement
# ---------------------------
def merge_clusters(particles, cluster_id1, cluster_id2):
    new_cluster_id = min(cluster_id1, cluster_id2)
    other_cluster_id = max(cluster_id1, cluster_id2)
    for p in particles:
        if p['cluster_id'] == other_cluster_id:
            p['cluster_id'] = new_cluster_id
    merged_members = [p for p in particles if p['cluster_id'] == new_cluster_id]
    for p in merged_members:
        p['cluster_members'] = merged_members

def check_collision_and_merge(p1, p2, grid, cell_size):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    if quick_center_aabb_reject(p1, p2, margin=0.0):
        return False

    mtv_res = get_mtv_any_any(p1, p2)
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
        total_mass = sum(get_particle_area(p) for p in cluster)
        com_x = sum(p['x'] * get_particle_area(p) for p in cluster) / total_mass
        com_y = sum(p['y'] * get_particle_area(p) for p in cluster) / total_mass
        
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

# ---------------------------
# Main simulation
# ---------------------------
def run_simulation_binary(N_total, rect_dims, circle_r, fraction_rect, 
                         max_time, store_every_k=1, save_dir=save_dir, seed=None):
    """
    Run DLCA simulation with binary mixture of rectangles and circles.
    
    Args:
        N_total: Total number of particles
        rect_dims: (width, height) tuple for rectangles
        circle_r: Radius of circles
        fraction_rect: Fraction of particles that are rectangles (0-1)
        max_time: Maximum simulation time steps
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_plots, exist_ok=True)

    count_rect = int(round(N_total * fraction_rect))
    count_circle = N_total - count_rect

    rect_w, rect_h = rect_dims
    rect_bounding_r = math.hypot(rect_w, rect_h) / 2.0
    
    max_bounding_r = max(rect_bounding_r, circle_r)
    cell_size = max_bounding_r * 2.5 
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]

    particles = []
    cluster_id_counter = 0

    def place_rectangles(dims, count):
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
                
                candidate = {
                    'x': x, 'y': y, 
                    'w': float(w), 'h': float(h), 
                    'bounding_r': bounding_r,
                    'orientation': spawn_orientation,
                    'shape': 'rectangle'
                }
                
                if check_exact_collision(candidate, grid, cell_size, eps=EPS_CONTACT):
                    particle = {
                        'x': x, 'y': y, 
                        'w': float(w), 'h': float(h), 
                        'bounding_r': bounding_r,
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
                print(f"Warning: Could not place rectangle after {max_attempts} attempts")
        return placed

    def place_circles(r, count):
        nonlocal cluster_id_counter
        placed = 0
        
        max_attempts = 10000
        for _ in range(count):
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(spawn_x0 + r, spawn_x0 + spawn_width - r)
                y = random.uniform(spawn_y0 + r, spawn_y0 + spawn_height - r)
                
                candidate = {
                    'x': x, 'y': y,
                    'radius': float(r),
                    'bounding_r': float(r),
                    'shape': 'circle'
                }
                
                if check_exact_collision(candidate, grid, cell_size, eps=EPS_CONTACT):
                    particle = {
                        'x': x, 'y': y, 
                        'radius': float(r),
                        'bounding_r': float(r),
                        'orientation': 0.0,  # Circles don't need orientation
                        'angle': random.uniform(0, 2 * math.pi),
                        'cluster_id': cluster_id_counter,
                        'cluster_members': [],
                        'id': len(particles),
                        'shape': 'circle'
                    }
                    
                    particles.append(particle)
                    insert_into_grid(particles[-1], grid, cell_size)
                    cluster_id_counter += 1
                    placed += 1
                    break
                    
                attempts += 1
            else:
                print(f"Warning: Could not place circle after {max_attempts} attempts")
        return placed

    print(f"Placing {count_rect} rectangles and {count_circle} circles...")
    placed_rect = place_rectangles(rect_dims, count_rect)
    placed_circle = place_circles(circle_r, count_circle)
    print(f"Successfully placed {placed_rect} rectangles and {placed_circle} circles")

    for p in particles:
        p['cluster_members'] = [p]

    num_saved_frames = (max_time // store_every_k) + 1
    trajectories = np.zeros((num_saved_frames, N_total, 2), dtype=np.float32)
    orientations = np.zeros((num_saved_frames, N_total), dtype=np.float32)
    shapes = np.array([p['shape'] for p in particles])

    global_pairs = []
    t_save_idx = 0
    start_time = time.time()

    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
    cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=5)
    enforce_eps_gap_all(particles, grid, cell_size)

    print("Starting simulation...")
    for time_step in range(max_time):
        if time_step % 100 == 0:
            print(f"Time step: {time_step}/{max_time}")
            
        update_spatial_grid(particles, grid, cell_size)
        rotate_all_clusters(particles, grid, cell_size, dt=1.0)

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
                        if p1 is p2: continue
                        if p1['cluster_id'] == p2['cluster_id']: continue
                        
                        cluster_pair = tuple(sorted((p1['cluster_id'], p2['cluster_id'])))
                        if cluster_pair in checked_pairs: continue
                        
                        if check_collision_and_merge(p1, p2, grid, cell_size):
                            checked_pairs.add(cluster_pair)

        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2)
        
        update_cluster_movement(particles, grid, cell_size)
        
        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
        cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2)
        enforce_world_bounds_all(particles, width, height)

        if check_one_cluster(particles):
            print(f"All particles merged into one cluster at time step {time_step}")
            break

        if (time_step % store_every_k) == 0:
            for idx, p in enumerate(particles):
                trajectories[t_save_idx, idx, 0] = p['x']
                trajectories[t_save_idx, idx, 1] = p['y']
                orientations[t_save_idx, idx] = p.get('orientation', 0.0)
            t_save_idx += 1
            
    elapsed = time.time() - start_time
    trajectories = trajectories[:t_save_idx]
    orientations = orientations[:t_save_idx]
    
    clusters = defaultdict(list)
    for p in particles: 
        clusters[p['cluster_id']].append(p)
    
    for members in clusters.values():
        Rg = radius_of_gyration(members)
        M = cluster_mass(members)
        if M >= min_mass and Rg > min_Rg:
            global_pairs.append((Rg, M))

    Df = float('nan')
    if len(global_pairs) >= min_pts_for_fit:
        unique = {}
        for rg, m in global_pairs: 
            unique[(round(rg,6), round(m,6))] = (rg, m)
        Rg_vals, M_vals = zip(*unique.values())
        if len(Rg_vals) > 2:
            logRg = np.log(np.array(Rg_vals))
            logM = np.log(np.array(M_vals))
            slope, _ = np.polyfit(logRg, logM, 1)
            Df = float(slope)

    tag = f"frac{fraction_rect:.3f}_N{N_total}_t{max_time}"
    
    np.save(os.path.join(save_dir, f"traj_{tag}.npy"), trajectories)
    np.save(os.path.join(save_dir, f"shapes_{tag}.npy"), shapes)
    np.save(os.path.join(save_dir, f"orient_{tag}.npy"), orientations)
    
    meta = {
        "fraction_rect": fraction_rect,
        "rect_w": rect_w,
        "rect_h": rect_h,
        "circle_r": circle_r,
        "N_total": N_total,
        "max_time": max_time,
        "saved_frames": int(trajectories.shape[0]),
        "Df": Df,
        "count_rect": count_rect,
        "count_circle": count_circle
    }
    with open(os.path.join(save_dir, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Plot final configuration
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    for p in particles:
        if p['shape'] == 'circle':
            c = Circle((p['x'], p['y']), 
                      radius=p['radius'],
                      facecolor='blue', edgecolor='black', alpha=0.8, linewidth=0.5)
            ax.add_patch(c)
        else:  # rectangle
            corners = get_corners(p)
            poly = mpatches.Polygon(corners, closed=True, 
                                   facecolor='red', edgecolor='black', 
                                   alpha=0.8, linewidth=0.5)
            ax.add_patch(poly)
            
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Binary DLCA: Rectangles (red) + Circles (blue), Df={Df:.3f}")
    plt.savefig(os.path.join(save_plots, f"final_{tag}.png"), dpi=180)
    plt.close()

    print(f"Simulation complete! Time: {elapsed:.2f}s, Df: {Df:.3f}")
    return {"Df": Df, "elapsed": elapsed}

# ---------------------------
# UI
# ---------------------------
import threading
import tkinter as tk
from tkinter import ttk, messagebox

def launch_ui():
    root = tk.Tk()
    root.title("Binary DLCA: Rectangles + Circles")

    particles_var = tk.StringVar(value="200")
    rect_w_var = tk.StringVar(value="10")
    rect_h_var = tk.StringVar(value="20")
    circle_r_var = tk.StringVar(value="8")
    frac_rect_var = tk.StringVar(value="0.5")
    status_var = tk.StringVar(value="Idle")

    def add_row(label_text, widget, row):
        ttk.Label(root, text=label_text).grid(row=row, column=0, padx=8, pady=6, sticky="e")
        widget.grid(row=row, column=1, padx=8, pady=6, sticky="we")

    add_row("Total Particles", ttk.Entry(root, textvariable=particles_var), 0)
    add_row("Rectangle Width", ttk.Entry(root, textvariable=rect_w_var), 1)
    add_row("Rectangle Height", ttk.Entry(root, textvariable=rect_h_var), 2)
    add_row("Circle Radius", ttk.Entry(root, textvariable=circle_r_var), 3)
    add_row("Fraction Rectangles (0-1)", ttk.Entry(root, textvariable=frac_rect_var), 4)

    ttk.Label(root, text="Status").grid(row=5, column=0, padx=8, pady=6, sticky="e")
    ttk.Label(root, textvariable=status_var).grid(row=5, column=1, padx=8, pady=6, sticky="we")

    def start_run():
        try:
            N = int(particles_var.get())
            rect_w = float(rect_w_var.get())
            rect_h = float(rect_h_var.get())
            circle_r = float(circle_r_var.get())
            frac_r = float(frac_rect_var.get())
            
            if not (0 <= frac_r <= 1):
                messagebox.showerror("Error", "Fraction must be between 0 and 1")
                return
                
        except ValueError:
            messagebox.showerror("Error", "Invalid inputs")
            return

        start_btn.config(state="disabled")
        status_var.set("Running...")

        def worker():
            try:
                run_simulation_binary(N, (rect_w, rect_h), circle_r, frac_r, max_time) 
                root.after(0, lambda: status_var.set(f"Done! Saved to {save_dir}"))
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                root.after(0, lambda: status_var.set(f"Error: {str(e)}"))
            finally:
                root.after(0, lambda: start_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    start_btn = ttk.Button(root, text="Start Simulation", command=start_run)
    start_btn.grid(row=6, column=0, columnspan=2, pady=10)
    
    root.columnconfigure(1, weight=1)
    root.minsize(420, 280)
    
    root.mainloop()

if __name__ == "__main__":
    print(f"Output: {save_dir} | Plots: {save_plots}")
    launch_ui()
