"""
triangle_cluster.py — Generalized Triangle DLCA Simulation
Supports 4 triangle types: Equilateral, Right-Angled, Isosceles, and Scalene.
Uses stored local vertices, SAT/MTV collision physics, area-based mass proxy,
unbiased Brownian rotation, and spatial-grid acceleration.
"""

import sys
import argparse
import random
import math
import time
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collisions.collisions import (
    create_equilateral_triangle_vertices,
    create_right_triangle_vertices,
    create_isosceles_triangle_vertices,
    create_scalene_triangle_vertices,
    get_triangle_bounding_radius,
    get_triangle_area,
    get_triangle_self_inertia_ratio,
    get_triangle_corners,
    get_mtv_triangle_triangle
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
save_dir = os.path.join(SCRIPT_DIR, "sim_npy_out_triangle")
save_plots = os.path.join(SCRIPT_DIR, "plots_triangle")
store_every_k = 1

# Contact gap and safety push
EPS_CONTACT = 1.0
EPS_PUSH    = 0.2

# Unbiased Brownian rotation parameters
ROT_OMEGA_BASE = 0.0     # zero directional bias
ROT_BROWNIAN   = 0.03    # symmetric Brownian rotational jitter

# ---------------------------
# Grid & Geometry Helpers
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

def keep_cluster_inside_bounds(cluster, width, height):
    if not cluster:
        return
    min_x = min(p['x'] - p['r_bound'] for p in cluster)
    max_x = max(p['x'] + p['r_bound'] for p in cluster)
    min_y = min(p['y'] - p['r_bound'] for p in cluster)
    max_y = max(p['y'] + p['r_bound'] for p in cluster)

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
# Cluster Utilities (Area as Mass Proxy)
# ---------------------------
def get_cluster_properties(cluster_particles):
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

def radius_of_gyration(cluster):
    """
    Computes radius of gyration using exact polygon area and self-moment-of-inertia.
    R_g = sqrt( sum( m_i * ( ||x_i - x_com||^2 + I_self / m_i ) ) / M )
    """
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

def cluster_mass(cluster):
    return sum(p['area'] for p in cluster)

def check_one_cluster(particles):
    return len(set(p['cluster_id'] for p in particles)) == 1

# ---------------------------
# Rigid Contact Separation (SAT / MTV)
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

    # 1. Cheap Reject (Bounding circumradius check)
    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    safe_dist = (p1['r_bound'] + p2['r_bound']) + eps
    if dist_sq > safe_dist**2:
        return False

    # 2. Exact Polygon SAT Check
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
    m1 = max(_cluster_mass(c1), 1e-12)
    m2 = max(_cluster_mass(c2), 1e-12)
    tm = m1 + m2

    s1 = (m2 / tm) * overlap
    s2 = (m1 / tm) * overlap

    _move_cluster(c1, ux * s1, uy * s1)
    _move_cluster(c2, -ux * s2, -uy * s2)
    
    return True

def fully_separate_two_clusters(particles, grid, cell_size, cid1, cid2, eps=EPS_CONTACT, max_passes=5):
    rebuild_cluster_members(particles) 
    c1 = [p for p in particles if p['cluster_id'] == cid1]
    c2 = [p for p in particles if p['cluster_id'] == cid2]
    if not c1 or not c2: return

    for _ in range(max_passes):
        moved = False
        for a in c1:
            for b in c2:
                dx = a['x'] - b['x']
                dy = a['y'] - b['y']
                d2 = dx*dx + dy*dy
                limit = ((a['r_bound'] + b['r_bound']) + eps)**2
                
                if d2 < limit:
                    if resolve_overlap_pair_as_clusters(a, b, eps=eps):
                        moved = True
        if not moved:
            break

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

def rotate_all_clusters(particles, grid, cell_size,
                        omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cid, cluster in clusters.items():
        if not cluster:
            continue
        M = sum(p['area'] for p in cluster)
        if M <= 0.0:
            continue
        mass_scale = 1.0 / (1.0 + math.log(1.0 + M / 1000.0))
        # Unbiased symmetric Brownian rotational torque
        jitter = random.uniform(-omega_brownian, omega_brownian) * mass_scale
        angle = (omega_base + jitter) * dt

        rotate_cluster_about_com(cluster, angle)
        keep_cluster_inside_bounds(cluster, width, height)

def resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2, eps=EPS_CONTACT):
    rebuild_cluster_members(particles)
    for _ in range(passes):
        moved_any = False
        update_spatial_grid(particles, grid, cell_size)
        processed_pairs = set()

        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x'] / cell_size), int(p1['y'] / cell_size)

            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for p2 in grid[gx][gy]:
                        if p1 is p2 or p1['cluster_id'] == p2['cluster_id']:
                            continue

                        c_pair = tuple(sorted((p1['cluster_id'], p2['cluster_id'])))
                        if c_pair in processed_pairs:
                            continue

                        rs = p1['r_bound'] + p2['r_bound']
                        dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
                        threshold = (rs + eps)**2

                        if dist_sq < threshold:
                            if resolve_overlap_pair_as_clusters(p1, p2, eps=eps):
                                moved_any = True
                                processed_pairs.add(c_pair)

        if not moved_any:
            break

def cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2):
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=outer_repeats*2, eps=eps)

def enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=2):
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=max_rounds, eps=eps)
    enforce_world_bounds_all(particles, width, height)

# ---------------------------
# Merge, Collisions & Movement
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
    velocity_reduction = 1.0 / (1.0 + math.log(1.0 + total_mass / 1000.0))
    new_vx *= velocity_reduction; new_vy *= velocity_reduction
    new_angle = math.atan2(new_vy, new_vx)
    for p in merged_members:
        p['angle'] = new_angle

def check_collision_and_merge(p1, p2, grid, cell_size):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    if dist_sq > (p1['r_bound'] + p2['r_bound'] + EPS_CONTACT)**2:
        return False

    if resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT * 1.1):
        merge_clusters(particles_ref, p1['cluster_id'], p2['cluster_id'])
        return True

    return False

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
        avg_vx = sum(velocity / p['r_bound'] * math.cos(p.get('angle', 0)) for p in cluster) / len(cluster)
        avg_vy = sum(velocity / p['r_bound'] * math.sin(p.get('angle', 0)) for p in cluster) / len(cluster)
        vel_scale = 1.0 / (1.0 + math.log(1.0 + total_mass / 1000.0))
        avg_vx *= vel_scale; avg_vy *= vel_scale
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
        dx += shift_x; dy += shift_y
        if bounced_x: final_angle = math.pi - final_angle
        if bounced_y: final_angle = -final_angle
        for idx, p in enumerate(cluster):
            ox, oy, r = offsets[idx]
            p['x'] = com_x + dx + ox
            p['y'] = com_y + dy + oy
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
# Particle Creation Helper
# ---------------------------
def create_triangle_particle(shape_type, params, x, y, orientation, cluster_id, particle_id):
    """
    Creates a triangle particle with local vertices generated based on type:
    - 'equilateral': params is radius r
    - 'right': params is (base, height)
    - 'isosceles': params is (base, height)
    - 'scalene': params is (side_a, side_b, side_c)
    """
    st = shape_type.lower()
    if st in ['equilateral', 'eq']:
        r = float(params if isinstance(params, (int, float)) else params[0])
        verts = create_equilateral_triangle_vertices(r)
    elif st in ['right', 'right-angled', 'rt']:
        b, h = float(params[0]), float(params[1])
        verts = create_right_triangle_vertices(b, h)
    elif st in ['isosceles', 'iso']:
        b, h = float(params[0]), float(params[1])
        verts = create_isosceles_triangle_vertices(b, h)
    elif st in ['scalene', 'sc']:
        a, b, c = float(params[0]), float(params[1]), float(params[2])
        verts = create_scalene_triangle_vertices(a, b, c)
    else:
        raise ValueError(f"Unknown triangle type: {shape_type}")

    r_bound = get_triangle_bounding_radius(verts)
    area = get_triangle_area(verts)
    self_inertia = get_triangle_self_inertia_ratio(verts)

    return {
        'x': float(x),
        'y': float(y),
        'orientation': float(orientation),
        'local_vertices': verts,
        'r_bound': float(r_bound),
        'r': float(r_bound),  # backwards compatibility
        'area': float(area),
        'self_inertia_ratio': float(self_inertia),
        'shape': 'triangle',
        'triangle_type': st,
        'cluster_id': int(cluster_id),
        'cluster_members': [],
        'angle': random.uniform(0, 2 * math.pi),
        'id': int(particle_id)
    }

# ---------------------------
# Simulation Runner
# ---------------------------
def run_simulation(N_total, triangle_type, params_A, params_B=None, fraction_A=1.0,
                   max_time=10000, store_every_k=1, save_dir=save_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_plots, exist_ok=True)

    if params_B is None or fraction_A >= 1.0:
        params_B = params_A
        fraction_A = 1.0

    count_A = int(round(N_total * fraction_A))
    count_B = N_total - count_A

    # Sample particle to compute max cell size
    dummy_A = create_triangle_particle(triangle_type, params_A, 0, 0, 0, 0, 0)
    dummy_B = create_triangle_particle(triangle_type, params_B, 0, 0, 0, 0, 1)
    max_rbound = max(dummy_A['r_bound'], dummy_B['r_bound'])

    cell_size = max(20.0, max_rbound * 2.2)
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]

    particles = []
    cluster_id_counter = 0

    def place_particles(params, count):
        nonlocal cluster_id_counter
        placed = 0
        max_attempts = 15000
        for _ in range(count):
            attempts = 0
            # Test particle to get r_bound
            temp_p = create_triangle_particle(triangle_type, params, 0, 0, 0, 0, 0)
            rb = temp_p['r_bound']
            while attempts < max_attempts:
                x = random.uniform(spawn_x0 + rb, spawn_x0 + spawn_width - rb)
                y = random.uniform(spawn_y0 + rb, spawn_y0 + spawn_height - rb)
                if check_spawn_overlap(particles, x, y, rb, grid, cell_size):
                    orient = random.uniform(0, 2 * math.pi)
                    p = create_triangle_particle(triangle_type, params, x, y, orient, cluster_id_counter, len(particles))
                    particles.append(p)
                    insert_into_grid(particles[-1], grid, cell_size)
                    cluster_id_counter += 1
                    placed += 1
                    break
                attempts += 1
            else:
                print(f"Warning: Could not place particle after {max_attempts} attempts")
        return placed

    place_particles(params_A, count_A)
    if count_B > 0:
        place_particles(params_B, count_B)

    for p in particles:
        p['cluster_members'] = [p]

    num_saved_frames = (max_time // store_every_k) + 60
    trajectories = np.zeros((num_saved_frames, len(particles), 2), dtype=np.float32)
    orientations = np.zeros((num_saved_frames, len(particles)), dtype=np.float32)
    local_verts_arr = np.array([p['local_vertices'] for p in particles], dtype=np.float32)

    global_pairs = []
    time_evolution = []

    t_save_idx = 0
    start_time = time.time()

    # Initial overlap relaxation
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
    cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2)
    enforce_eps_gap_all(particles, grid, cell_size)

    for time_step in range(max_time):
        update_spatial_grid(particles, grid, cell_size)
        
        # Unbiased Brownian rotation for each cluster
        rotate_all_clusters(particles, grid, cell_size,
                            omega_base=ROT_OMEGA_BASE,
                            omega_brownian=ROT_BROWNIAN,
                            dt=1.0)

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
                            checked_pairs.add(cluster_pair)

        # Cleanup overlaps
        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)

        # Move clusters and clean again
        update_cluster_movement(particles, grid, cell_size)
        resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
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

        # Save trajectory frame
        if (time_step % store_every_k) == 0:
            for idx, p in enumerate(particles):
                trajectories[t_save_idx, idx, 0] = p['x']
                trajectories[t_save_idx, idx, 1] = p['y']
                orientations[t_save_idx, idx] = p.get('orientation', 0.0)
            t_save_idx += 1

        if check_one_cluster(particles):
            print(f"All particles merged into one cluster at time step {time_step}")
            # Hold the final cluster for 45 frames
            for _ in range(45):
                if t_save_idx < num_saved_frames:
                    for idx, p in enumerate(particles):
                        trajectories[t_save_idx, idx, 0] = p['x']
                        trajectories[t_save_idx, idx, 1] = p['y']
                        orientations[t_save_idx, idx] = p.get('orientation', 0.0)
                    t_save_idx += 1
            break

    elapsed = time.time() - start_time
    trajectories = trajectories[:t_save_idx]
    orientations = orientations[:t_save_idx]

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

    tag = f"{triangle_type}_N{len(particles)}_t{max_time}"
    traj_path = os.path.join(save_dir, f"traj_{tag}.npy")
    orient_path = os.path.join(save_dir, f"orient_{tag}.npy")
    verts_path = os.path.join(save_dir, f"vertices_{tag}.npy")
    radii_path = os.path.join(save_dir, f"radii_{tag}.npy")
    meta_path = os.path.join(save_dir, f"meta_{tag}.json")

    np.save(traj_path, trajectories)
    np.save(orient_path, orientations)
    np.save(verts_path, local_verts_arr)
    np.save(radii_path, np.array([p['r_bound'] for p in particles], dtype=np.float32))

    meta = {
        "triangle_type": triangle_type,
        "params_A": params_A if isinstance(params_A, (list, tuple)) else [params_A],
        "params_B": params_B if isinstance(params_B, (list, tuple)) else [params_B],
        "fraction_A": fraction_A,
        "count_A": count_A,
        "count_B": count_B,
        "N_total": len(particles),
        "width": width,
        "height": height,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "saved_frames": int(trajectories.shape[0]),
        "final_clusters": len(set(p['cluster_id'] for p in particles)),
        "Df": float(Df)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    enforce_world_bounds_all(particles, width, height)
    
    # Static plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for p in particles:
        corners = get_triangle_corners(p)
        color = 'cyan' if p['id'] < count_A else 'coral'
        poly = patches.Polygon(corners, closed=True, facecolor=color, alpha=0.85)
        ax.add_patch(poly)
        outline = patches.Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
        ax.add_patch(outline)
    plt.xlim(0, width); plt.ylim(0, height); plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Final: {triangle_type.capitalize()} Triangles | {len(set(p['cluster_id'] for p in particles))} clusters | N={len(particles)}")
    plt.savefig(os.path.join(save_plots, f"final_{tag}.png"), dpi=180, bbox_inches='tight')
    plt.close()

    result = {
        "triangle_type": triangle_type,
        "Df": float(Df),
        "intercept": float(intercept) if 'intercept' in locals() else None,
        "r2": float(r2) if 'r2' in locals() else None,
        "Rg_largest": float(Rg_largest),
        "Rg_largest_n": float(Rg_largest_n),
        "Mass_largest": float(M_largest),
        "traj_path": traj_path,
        "orient_path": orient_path,
        "verts_path": verts_path,
        "meta_path": meta_path,
        "elapsed_s": float(elapsed),
        "num_global_pairs": int(len(global_pairs)),
        "saved_frames": int(trajectories.shape[0]),
        "num_particles": len(particles),
        "num_clusters_final": len(set(p['cluster_id'] for p in particles)),
        "final_clusters": len(set(p['cluster_id'] for p in particles))
    }
    
    # RDF
    r_vals, g_r = compute_rdf(particles, width, height, dr=2.0)
    rdf_path = os.path.join(save_dir, f"rdf_{tag}.npy")
    np.save(rdf_path, np.column_stack((r_vals, g_r)))
    
    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, g_r, label=f'RDF (t={max_time})')
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Random distribution')
    plt.xlabel("Distance r (pixels)")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function\n{triangle_type.capitalize()} Triangles | N={len(particles)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_plots, f"rdf_{tag}.png"), dpi=150)
    plt.close()

    return result

# Backward compatible function name
def run_simulation_for_composition(N_total, radii_pair, fraction_A, max_time=10000,
                                   store_every_k=1, save_dir=save_dir, seed=None):
    return run_simulation(
        N_total=N_total,
        triangle_type='equilateral',
        params_A=radii_pair[0],
        params_B=radii_pair[1],
        fraction_A=fraction_A,
        max_time=max_time,
        store_every_k=store_every_k,
        save_dir=save_dir,
        seed=seed
    )

# ---------------------------
# Interactive Tkinter GUI
# ---------------------------
import threading
import tkinter as tk
from tkinter import ttk, messagebox

def launch_ui(initial_type="equilateral"):
    root = tk.Tk()
    root.title("Generalized Triangle DLCA Runner")

    # Variables
    type_var = tk.StringVar(value=initial_type.capitalize())
    particles_var = tk.StringVar(value="350")
    fracA_var = tk.StringVar(value="0.668")
    status_var = tk.StringVar(value="Idle")

    # Type A params
    pA_1 = tk.StringVar(value="8.0")
    pA_2 = tk.StringVar(value="16.0")
    pA_3 = tk.StringVar(value="18.0")

    # Type B params
    pB_1 = tk.StringVar(value="14.0")
    pB_2 = tk.StringVar(value="20.0")
    pB_3 = tk.StringVar(value="24.0")

    # Layout frames
    top_frame = ttk.LabelFrame(root, text="Triangle Type & Particle Settings")
    top_frame.pack(fill="x", padx=10, pady=6)

    params_frame = ttk.LabelFrame(root, text="Geometric Parameters")
    params_frame.pack(fill="x", padx=10, pady=6)

    # Type selector
    ttk.Label(top_frame, text="Triangle Type:").grid(row=0, column=0, padx=6, pady=4, sticky="e")
    type_combo = ttk.Combobox(
        top_frame, textvariable=type_var, state="readonly",
        values=["Equilateral", "Right-Angled", "Isosceles", "Scalene"]
    )
    type_combo.grid(row=0, column=1, padx=6, pady=4, sticky="w")

    ttk.Label(top_frame, text="No. of particles:").grid(row=1, column=0, padx=6, pady=4, sticky="e")
    ttk.Entry(top_frame, textvariable=particles_var, width=12).grid(row=1, column=1, padx=6, pady=4, sticky="w")

    ttk.Label(top_frame, text="Fraction Type A:").grid(row=2, column=0, padx=6, pady=4, sticky="e")
    ttk.Entry(top_frame, textvariable=fracA_var, width=12).grid(row=2, column=1, padx=6, pady=4, sticky="w")

    # Dynamic parameter labels and entries
    lbl_A1 = ttk.Label(params_frame, text="Radius A:")
    ent_A1 = ttk.Entry(params_frame, textvariable=pA_1, width=10)
    lbl_A2 = ttk.Label(params_frame, text="Height A:")
    ent_A2 = ttk.Entry(params_frame, textvariable=pA_2, width=10)
    lbl_A3 = ttk.Label(params_frame, text="Side C (A):")
    ent_A3 = ttk.Entry(params_frame, textvariable=pA_3, width=10)

    lbl_B1 = ttk.Label(params_frame, text="Radius B:")
    ent_B1 = ttk.Entry(params_frame, textvariable=pB_1, width=10)
    lbl_B2 = ttk.Label(params_frame, text="Height B:")
    ent_B2 = ttk.Entry(params_frame, textvariable=pB_2, width=10)
    lbl_B3 = ttk.Label(params_frame, text="Side C (B):")
    ent_B3 = ttk.Entry(params_frame, textvariable=pB_3, width=10)

    def update_param_fields(*args):
        st = type_var.get().lower()
        # Clear existing grid in params_frame
        for widget in params_frame.winfo_children():
            widget.grid_forget()

        if "equi" in st:
            lbl_A1.config(text="Radius A:")
            lbl_B1.config(text="Radius B:")
            pA_1.set("8.0")
            pB_1.set("16.0")

            lbl_A1.grid(row=0, column=0, padx=4, pady=3, sticky="e")
            ent_A1.grid(row=0, column=1, padx=4, pady=3)
            lbl_B1.grid(row=0, column=2, padx=4, pady=3, sticky="e")
            ent_B1.grid(row=0, column=3, padx=4, pady=3)

        elif "right" in st:
            lbl_A1.config(text="Base A:")
            lbl_A2.config(text="Height A:")
            lbl_B1.config(text="Base B:")
            lbl_B2.config(text="Height B:")
            pA_1.set("12.0"); pA_2.set("16.0")
            pB_1.set("18.0"); pB_2.set("24.0")

            lbl_A1.grid(row=0, column=0, padx=4, pady=3, sticky="e")
            ent_A1.grid(row=0, column=1, padx=4, pady=3)
            lbl_A2.grid(row=1, column=0, padx=4, pady=3, sticky="e")
            ent_A2.grid(row=1, column=1, padx=4, pady=3)

            lbl_B1.grid(row=0, column=2, padx=4, pady=3, sticky="e")
            ent_B1.grid(row=0, column=3, padx=4, pady=3)
            lbl_B2.grid(row=1, column=2, padx=4, pady=3, sticky="e")
            ent_B2.grid(row=1, column=3, padx=4, pady=3)

        elif "iso" in st:
            lbl_A1.config(text="Base A:")
            lbl_A2.config(text="Height A:")
            lbl_B1.config(text="Base B:")
            lbl_B2.config(text="Height B:")
            pA_1.set("14.0"); pA_2.set("18.0")
            pB_1.set("20.0"); pB_2.set("26.0")

            lbl_A1.grid(row=0, column=0, padx=4, pady=3, sticky="e")
            ent_A1.grid(row=0, column=1, padx=4, pady=3)
            lbl_A2.grid(row=1, column=0, padx=4, pady=3, sticky="e")
            ent_A2.grid(row=1, column=1, padx=4, pady=3)

            lbl_B1.grid(row=0, column=2, padx=4, pady=3, sticky="e")
            ent_B1.grid(row=0, column=3, padx=4, pady=3)
            lbl_B2.grid(row=1, column=2, padx=4, pady=3, sticky="e")
            ent_B2.grid(row=1, column=3, padx=4, pady=3)

        elif "scale" in st:
            lbl_A1.config(text="Side a (A):")
            lbl_A2.config(text="Side b (A):")
            lbl_A3.config(text="Side c (A):")
            lbl_B1.config(text="Side a (B):")
            lbl_B2.config(text="Side b (B):")
            lbl_B3.config(text="Side c (B):")
            pA_1.set("13.0"); pA_2.set("14.0"); pA_3.set("15.0")
            pB_1.set("18.0"); pB_2.set("20.0"); pB_3.set("22.0")

            lbl_A1.grid(row=0, column=0, padx=4, pady=3, sticky="e")
            ent_A1.grid(row=0, column=1, padx=4, pady=3)
            lbl_A2.grid(row=1, column=0, padx=4, pady=3, sticky="e")
            ent_A2.grid(row=1, column=1, padx=4, pady=3)
            lbl_A3.grid(row=2, column=0, padx=4, pady=3, sticky="e")
            ent_A3.grid(row=2, column=1, padx=4, pady=3)

            lbl_B1.grid(row=0, column=2, padx=4, pady=3, sticky="e")
            ent_B1.grid(row=0, column=3, padx=4, pady=3)
            lbl_B2.grid(row=1, column=2, padx=4, pady=3, sticky="e")
            ent_B2.grid(row=1, column=3, padx=4, pady=3)
            lbl_B3.grid(row=2, column=2, padx=4, pady=3, sticky="e")
            ent_B3.grid(row=2, column=3, padx=4, pady=3)

    type_combo.bind("<<ComboboxSelected>>", update_param_fields)
    update_param_fields()

    # Status
    status_frame = ttk.Frame(root)
    status_frame.pack(fill="x", padx=10, pady=4)
    ttk.Label(status_frame, text="Status:").pack(side="left", padx=4)
    ttk.Label(status_frame, textvariable=status_var, font=("Arial", 9, "italic")).pack(side="left", padx=4)

    # Button handlers
    def enable_ui(enable=True):
        state = "normal" if enable else "disabled"
        start_btn.config(state=state)
        type_combo.config(state="readonly" if enable else "disabled")

    def on_done(msg, res):
        status_var.set(msg)
        enable_ui(True)
        try:
            summary = (
                f"Triangle Type: {res.get('triangle_type').capitalize()}\n"
                f"Saved Frames: {res.get('saved_frames')}\n"
                f"Particles: {res.get('num_particles')}\n"
                f"Final Clusters: {res.get('num_clusters_final')}\n"
                f"Df: {res.get('Df'):.4f}\n"
                f"Elapsed: {res.get('elapsed_s'):.2f} s"
            )
            messagebox.showinfo("Simulation Complete", summary)
        except Exception as e:
            messagebox.showwarning("Completed with Notes", f"{msg}\n\nNote: {e}")

    def on_error(err):
        status_var.set(f"Error: {err}")
        enable_ui(True)
        messagebox.showerror("Simulation Error", str(err))

    def start_run():
        st = type_var.get().lower()
        try:
            N = int(particles_var.get())
            fA = float(fracA_var.get())
            if N <= 0: raise ValueError("Particle count must be positive")
            if not (0.0 <= fA <= 1.0): raise ValueError("Fraction A must be in [0, 1]")

            if "equi" in st:
                pA = float(pA_1.get())
                pB = float(pB_1.get())
                if pA <= 0 or pB <= 0: raise ValueError("Radii must be positive")
            elif "right" in st or "iso" in st:
                pA = (float(pA_1.get()), float(pA_2.get()))
                pB = (float(pB_1.get()), float(pB_2.get()))
                if min(pA) <= 0 or min(pB) <= 0: raise ValueError("Dimensions must be positive")
            elif "scale" in st:
                pA = (float(pA_1.get()), float(pA_2.get()), float(pA_3.get()))
                pB = (float(pB_1.get()), float(pB_2.get()), float(pB_3.get()))
                # Trigger scalene validation
                create_scalene_triangle_vertices(*pA)
                create_scalene_triangle_vertices(*pB)
            else:
                raise ValueError("Invalid triangle type")
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        enable_ui(False)
        status_var.set("Running simulation...")

        def worker():
            try:
                res = run_simulation(
                    N_total=N,
                    triangle_type=st,
                    params_A=pA,
                    params_B=pB,
                    fraction_A=fA,
                    max_time=max_time,
                    store_every_k=store_every_k
                )
                msg = f"Done: simulation completed ({res.get('num_clusters_final')} cluster(s))"
                root.after(0, lambda: on_done(msg, res))
            except Exception as e:
                root.after(0, lambda: on_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def generate_video_action():
        status_var.set("Generating video...")
        def vid_worker():
            try:
                import video_renderers.make_triangle_video as make_triangle_video
                make_triangle_video.create_video()
                status_var.set("Video generated successfully!")
                messagebox.showinfo("Video Ready", "Video has been generated and saved to the project folder!")
            except Exception as e:
                status_var.set(f"Video error: {e}")
                messagebox.showerror("Video Error", str(e))
        threading.Thread(target=vid_worker, daemon=True).start()

    # Buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=8)
    start_btn = ttk.Button(btn_frame, text="Start Simulation", command=start_run)
    start_btn.pack(side="left", padx=4)
    vid_btn = ttk.Button(btn_frame, text="Generate Video", command=generate_video_action)
    vid_btn.pack(side="left", padx=4)
    quit_btn = ttk.Button(btn_frame, text="Quit", command=root.destroy)
    quit_btn.pack(side="left", padx=4)

    root.minsize(440, 260)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized Triangle DLCA Simulation")
    parser.add_argument("--type", type=str, default="equilateral",
                        choices=["equilateral", "right", "right-angled", "isosceles", "scalene"],
                        help="Pre-selected triangle type for the GUI")
    args, _ = parser.parse_known_args()

    # Normalize type argument
    t_map = {
        "equilateral": "equilateral",
        "right": "right-angled",
        "right-angled": "right-angled",
        "isosceles": "isosceles",
        "scalene": "scalene"
    }
    sel_type = t_map.get(args.type.lower(), "equilateral")

    print("Output Data Dir:", save_dir)
    print("Output Plots Dir:", save_plots)
    launch_ui(initial_type=sel_type)
