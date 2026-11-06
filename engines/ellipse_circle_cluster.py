"""
dlca_ellipse_circle_binary.py — DLCA with Binary System: Ellipses + Circles
Fixed aspect ratio for ellipses, circles treated as special case.
"""

import random
import math
import time
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# ---------------------------
# Config
# ---------------------------
width = 2000
height = 0.75*width

spawn_width = 1000
spawn_height = 1000
spawn_x0 = (width - spawn_width) / 2.0
spawn_y0 = (height - spawn_height) / 2.0

velocity = 5.0
max_time = 10000
min_mass = 10.0
min_Rg = 1e-8
min_pts_for_fit = 6
save_dir = "sim_npy_out_binary"
save_plots = "plots_binary"
store_every_k = 1
repeats = 1

# Fixed aspect ratio for ellipses
ELLIPSE_ASPECT_RATIO = 2.0  # major = ASPECT_RATIO * minor

# Relaxed parameters
EPS_CONTACT = 0.5
EPS_PUSH    = 0.1
ROT_OMEGA_BASE = 0.0     
ROT_BROWNIAN   = 0.12     

# ---------------------------
# PARTICLE GEOMETRY HELPERS
# ---------------------------
def generate_ellipse_circles(x, y, a, b, angle):
    """
    Places circles along the major axis for ellipse approximation.
    """
    circles = []
    positions = [-0.9, -0.65, -0.35, 0, 0.35, 0.65, 0.9]
    
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)

    for t in positions:
        local_x = t * a               
        
        rot_x = local_x * cos_t
        rot_y = local_x * sin_t
        pos_x = x + rot_x
        pos_y = y + rot_y
        
        frac = t 
        radius = b * math.sqrt(max(0.0, 1.0 - frac * frac))
        
        radius = max(radius, 1e-6)
        circles.append({'x': pos_x, 'y': pos_y, 'r': radius})

    return circles

def generate_circle_representation(x, y, r):
    """
    For a perfect circle, we just need one circle at the center.
    """
    return [{'x': x, 'y': y, 'r': r}]

def update_particle_circles(p):
    """
    Update the circle representation based on particle type.
    """
    if p['shape'] == 'circle':
        p['circles'] = generate_circle_representation(p['x'], p['y'], p['radius'])
    else:  # ellipse
        p['circles'] = generate_ellipse_circles(
            p['x'], p['y'], 
            p['major'], p['minor'], 
            p['orientation']
        )

def get_bounding_radius(p):
    """
    Get the bounding radius for collision broad phase.
    """
    if p['shape'] == 'circle':
        return p['radius']
    else:  # ellipse
        return p['major']

def get_exact_aabb(p):
    """
    Calculates the exact axis-aligned bounding box half-extents.
    """
    if p['shape'] == 'circle':
        r = p['radius']
        return r, r
    else:  # ellipse
        a = p['major']
        b = p['minor']
        angle = p['orientation']
        c = math.cos(angle)
        s = math.sin(angle)
        
        ext_x = math.sqrt((a*c)**2 + (b*s)**2)
        ext_y = math.sqrt((a*s)**2 + (b*c)**2)
        
        return ext_x, ext_y

def get_particle_mass(p):
    """
    Calculate particle mass (area for 2D).
    """
    if p['shape'] == 'circle':
        return math.pi * p['radius']**2
    else:  # ellipse
        return math.pi * p['major'] * p['minor']

# ---------------------------
# Grid helpers
# ---------------------------
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

def check_exact_overlap(candidate, grid, cell_size, eps=0.0):
    """
    Check if candidate overlaps with any particle in grid.
    Works for both circles and ellipses.
    """
    cell_x, cell_y = int(candidate['x'] / cell_size), int(candidate['y'] / cell_size)
    
    for i in range(max(0, cell_x - 1), min(len(grid), cell_x + 2)):
        for j in range(max(0, cell_y - 1), min(len(grid[0]), cell_y + 2)):
            for p in grid[i][j]:
                # Broad Phase: Fast bounding circle reject
                dist_sq = (candidate['x'] - p['x'])**2 + (candidate['y'] - p['y'])**2
                min_dist = get_bounding_radius(candidate) + get_bounding_radius(p) + eps
                
                if dist_sq > min_dist**2:
                    continue
                
                # Narrow Phase: Exact multi-circle check
                if 'circles' not in p:
                    update_particle_circles(p)
                    
                for c1 in candidate['circles']:
                    for c2 in p['circles']:
                        d2 = (c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2
                        r_sum = c1['r'] + c2['r'] + eps
                        if d2 < r_sum**2:
                            return False
                            
    return True

# ---------------------------
# Boundary & Movement
# ---------------------------
def keep_cluster_inside_bounds(cluster, width, height):
    min_x_global = float('inf')
    max_x_global = float('-inf')
    min_y_global = float('inf')
    max_y_global = float('-inf')

    for p in cluster:
        ext_x, ext_y = get_exact_aabb(p)
        min_x_global = min(min_x_global, p['x'] - ext_x)
        max_x_global = max(max_x_global, p['x'] + ext_x)
        min_y_global = min(min_y_global, p['y'] - ext_y)
        max_y_global = max(max_y_global, p['y'] + ext_y)

    dx = 0.0; dy = 0.0
    
    if min_x_global < 0.0:      dx = -min_x_global
    elif max_x_global > width:  dx = width - max_x_global
    
    if min_y_global < 0.0:      dy = -min_y_global
    elif max_y_global > height: dy = height - max_y_global

    if dx != 0.0 or dy != 0.0:
        for p in cluster:
            p['x'] += dx
            p['y'] += dy
            update_particle_circles(p)

def enforce_world_bounds_all(particles, width, height):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)
    for cluster in clusters.values():
        keep_cluster_inside_bounds(cluster, width, height)

# ---------------------------
# Cluster Physics
# ---------------------------
def radius_of_gyration(cluster):
    if not cluster: return 0.0
    masses = [get_particle_mass(p) for p in cluster]
    M = sum(masses)
    if M == 0: return 0.0
    com_x = sum(p['x']*m for p, m in zip(cluster, masses)) / M
    com_y = sum(p['y']*m for p, m in zip(cluster, masses)) / M
    rg2 = sum(m*((p['x']-com_x)**2 + (p['y']-com_y)**2) for p, m in zip(cluster, masses)) / M
    return math.sqrt(rg2)

def cluster_mass(cluster):
    return sum(get_particle_mass(p) for p in cluster)

def check_one_cluster(particles):
    return len(set(p['cluster_id'] for p in particles)) == 1

# ---------------------------
# Rigid Movement Helpers
# ---------------------------
def _move_cluster(cluster, dx, dy):
    for q in cluster:
        q['x'] += dx
        q['y'] += dy
        update_particle_circles(q)

def rebuild_cluster_members(particles):
    by_id = defaultdict(list)
    for p in particles:
        by_id[p['cluster_id']].append(p)
    for p in particles:
        p['cluster_members'] = by_id[p['cluster_id']]

def _cluster_mass_val(cluster):
    return sum(get_particle_mass(pp) for pp in cluster)

# ---------------------------
# Collision Logic
# ---------------------------
def resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT):
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    safe_dist = get_bounding_radius(p1) + get_bounding_radius(p2) + eps
    if dist_sq > safe_dist**2:
        return False

    if 'circles' not in p1: update_particle_circles(p1)
    if 'circles' not in p2: update_particle_circles(p2)

    c1_list = p1['circles']
    c2_list = p2['circles']
    
    max_overlap = 0.0
    best_normal = (0, 0)
    collision_found = False

    for c1 in c1_list:
        for c2 in c2_list:
            dx = c1['x'] - c2['x']
            dy = c1['y'] - c2['y']
            d2 = dx*dx + dy*dy
            r_sum = c1['r'] + c2['r']
            
            if d2 < (r_sum + eps)**2:
                d = math.sqrt(d2)
                if d == 0: 
                    d = 0.001
                    nx, ny = 1.0, 0.0
                else:
                    nx, ny = dx/d, dy/d
                
                current_overlap = (r_sum - d) + eps
                
                if current_overlap > max_overlap:
                    max_overlap = current_overlap
                    best_normal = (nx, ny)
                    collision_found = True

    if not collision_found:
        return False

    ux, uy = best_normal
    overlap = max_overlap

    c1_members = p1['cluster_members']
    c2_members = p2['cluster_members']
    
    m1 = max(_cluster_mass_val(c1_members), 1e-12)
    m2 = max(_cluster_mass_val(c2_members), 1e-12)
    tm = m1 + m2

    s1 = (m2 / tm) * overlap
    s2 = (m1 / tm) * overlap

    _move_cluster(c1_members, ux * s1, uy * s1)
    _move_cluster(c2_members, -ux * s2, -uy * s2)
    
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
                dx = a['x'] - b['x']
                dy = a['y'] - b['y']
                d2 = dx*dx + dy*dy
                limit = (get_bounding_radius(a) + get_bounding_radius(b) + eps)**2
                
                if d2 < limit:
                    if resolve_overlap_pair_as_clusters(a, b, eps=eps):
                        moved = True
        if not moved:
            break

def rotate_cluster_about_com(cluster, angle_rad):
    """
    Rotate ellipses in cluster. Circles don't need orientation updates.
    """
    if not cluster or angle_rad == 0.0: return

    M = sum(get_particle_mass(p) for p in cluster)
    if M <= 0.0: return
    
    cx = sum(p['x'] * get_particle_mass(p) for p in cluster) / M
    cy = sum(p['y'] * get_particle_mass(p) for p in cluster) / M

    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    for p in cluster:
        ox = p['x'] - cx
        oy = p['y'] - cy
        rx = c * ox - s * oy
        ry = s * ox + c * oy
        p['x'] = cx + rx
        p['y'] = cy + ry
        
        # Only update orientation for ellipses
        if p['shape'] == 'ellipse':
            p['orientation'] += angle_rad
        
        update_particle_circles(p)

def rotate_all_clusters(particles, grid, cell_size,
                        omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)

    for cid, cluster in clusters.items():
        if not cluster: continue
        M = sum(get_particle_mass(p) for p in cluster)
        if M <= 0.0: continue
        mass_scale = 1.0 / (1.0 + math.log(1.0 + M / 100.0))
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
                        if c_pair in processed_pairs: continue 
                        
                        ra = get_bounding_radius(p1)
                        rb = get_bounding_radius(p2)
                        threshold = (ra + rb + eps)**2
                        
                        dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
                        
                        if dist_sq < threshold:
                            fully_separate_two_clusters(
                                particles, grid, cell_size,
                                p1['cluster_id'], p2['cluster_id'],
                                eps=eps, max_passes=5
                            )
                            moved_any = True
                            processed_pairs.add(c_pair)

        if not moved_any: break

def cleanup_until_clean(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=2):
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=outer_repeats*2, eps=eps)

def enforce_eps_gap_all(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=6):
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=max_rounds, eps=eps)
    enforce_world_bounds_all(particles, width, height)

# ---------------------------
# Merge Logic
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
    
    m1 = sum(get_particle_mass(p) for p in cluster1)
    m2 = sum(get_particle_mass(p) for p in cluster2)
    total_mass = m1 + m2
    
    vx1 = sum(math.cos(p['angle']) for p in cluster1) * m1
    vy1 = sum(math.sin(p['angle']) for p in cluster1) * m1
    vx2 = sum(math.cos(p['angle']) for p in cluster2) * m2
    vy2 = sum(math.sin(p['angle']) for p in cluster2) * m2
    
    new_vx = (vx1 + vx2) / total_mass
    new_vy = (vy1 + vy2) / total_mass
    new_angle = math.atan2(new_vy, new_vx)
    
    for p in merged_members:
        p['angle'] = new_angle

def check_collision_and_merge(p1, p2, grid, cell_size):
    if p1['cluster_id'] == p2['cluster_id']: return False

    merge_eps = EPS_CONTACT * 1.1 

    if resolve_overlap_pair_as_clusters(p1, p2, eps=merge_eps):
        merge_clusters(particles_ref, p1['cluster_id'], p2['cluster_id'])
        return True
        
    return False

def update_cluster_movement(particles, grid, cell_size):
    clusters = defaultdict(list)
    for p in particles:
        clusters[p['cluster_id']].append(p)
        
    for cluster_id, cluster in clusters.items():
        if not cluster: continue
        
        total_mass = sum(get_particle_mass(p) for p in cluster)
        com_x = sum(p['x'] * get_particle_mass(p) for p in cluster) / total_mass
        com_y = sum(p['y'] * get_particle_mass(p) for p in cluster) / total_mass
        
        # Velocity calculation (simplified for mixed shapes)
        avg_vx = sum(velocity * math.cos(p['angle']) for p in cluster) / len(cluster)
        avg_vy = sum(velocity * math.sin(p['angle']) for p in cluster) / len(cluster)
        
        vel_scale = 1 / (1 + math.log(1 + total_mass / 1000))
        avg_vx *= vel_scale; avg_vy *= vel_scale
        
        br_angle = random.uniform(0, 2 * math.pi)
        mf = 1 / math.sqrt(max(total_mass, 1e-12))
        dx = avg_vx + velocity * mf * math.cos(br_angle)
        dy = avg_vy + velocity * mf * math.sin(br_angle)
        
        final_angle = br_angle
        
        min_x_g, max_x_g = float('inf'), float('-inf')
        min_y_g, max_y_g = float('inf'), float('-inf')
        
        for p in cluster:
            ext_x, ext_y = get_exact_aabb(p)
            nx = p['x'] + dx
            ny = p['y'] + dy
            min_x_g = min(min_x_g, nx - ext_x)
            max_x_g = max(max_x_g, nx + ext_x)
            min_y_g = min(min_y_g, ny - ext_y)
            max_y_g = max(max_y_g, ny + ext_y)
            
        bounced_x = False
        bounced_y = False
        
        if min_x_g < 0: dx += -min_x_g; bounced_x = True
        elif max_x_g > width: dx += width - max_x_g; bounced_x = True
        
        if min_y_g < 0: dy += -min_y_g; bounced_y = True
        elif max_y_g > height: dy += height - max_y_g; bounced_y = True
        
        if bounced_x: final_angle = math.pi - final_angle
        if bounced_y: final_angle = -final_angle
        
        for p in cluster:
            p['x'] += dx
            p['y'] += dy
            p['angle'] = final_angle
            update_particle_circles(p)

        keep_cluster_inside_bounds(cluster, width, height)

# ---------------------------
# Main Runner
# ---------------------------
def run_simulation_binary(N_total, ellipse_minor_r, circle_r, fraction_ellipse, 
                         max_time, store_every_k=1, save_dir=save_dir, seed=None):
    """
    Run DLCA simulation with binary mixture of ellipses and circles.
    
    Args:
        N_total: Total number of particles
        ellipse_minor_r: Minor radius of ellipses
        circle_r: Radius of circles
        fraction_ellipse: Fraction of particles that are ellipses (0-1)
        max_time: Maximum simulation time steps
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_plots, exist_ok=True)

    count_ellipse = int(round(N_total * fraction_ellipse))
    count_circle = N_total - count_ellipse

    ellipse_major_r = ellipse_minor_r * ELLIPSE_ASPECT_RATIO
    
    # Cell size based on largest particle
    max_radius = max(ellipse_major_r, circle_r)
    cell_size = max_radius * 2.2 
    
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]

    particles = []
    cluster_id_counter = 0

    def place_ellipse_particles(minor_r, count):
        nonlocal cluster_id_counter
        placed = 0
        major_r = minor_r * ELLIPSE_ASPECT_RATIO
        
        max_attempts = 10000
        for _ in range(count):
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(spawn_x0 + major_r, spawn_x0 + spawn_width - major_r)
                y = random.uniform(spawn_y0 + major_r, spawn_y0 + spawn_height - major_r)
                orientation = random.uniform(0, 2 * math.pi)
                
                candidate = {
                    'x': x, 'y': y,
                    'minor': float(minor_r),
                    'major': float(major_r),
                    'orientation': orientation,
                    'shape': 'ellipse'
                }
                candidate['circles'] = generate_ellipse_circles(x, y, major_r, minor_r, orientation)
                
                if check_exact_overlap(candidate, grid, cell_size, eps=EPS_CONTACT):
                    particle = {
                        'x': x, 'y': y, 
                        'minor': float(minor_r),
                        'major': float(major_r),
                        'orientation': orientation,
                        'angle': random.uniform(0, 2 * math.pi),
                        'cluster_id': cluster_id_counter,
                        'cluster_members': [],
                        'id': len(particles),
                        'shape': 'ellipse',
                        'circles': candidate['circles']
                    }
                    
                    particles.append(particle)
                    insert_into_grid(particles[-1], grid, cell_size)
                    cluster_id_counter += 1
                    placed += 1
                    break
                    
                attempts += 1
            else:
                print(f"Warning: Could not place ellipse after {max_attempts} attempts")
        return placed

    def place_circle_particles(r, count):
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
                    'shape': 'circle'
                }
                candidate['circles'] = generate_circle_representation(x, y, r)
                
                if check_exact_overlap(candidate, grid, cell_size, eps=EPS_CONTACT):
                    particle = {
                        'x': x, 'y': y, 
                        'radius': float(r),
                        'orientation': 0.0,  # Circles don't need orientation
                        'angle': random.uniform(0, 2 * math.pi),
                        'cluster_id': cluster_id_counter,
                        'cluster_members': [],
                        'id': len(particles),
                        'shape': 'circle',
                        'circles': candidate['circles']
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
    
    # Place particles
    print(f"Placing {count_ellipse} ellipses and {count_circle} circles...")
    placed_ellipse = place_ellipse_particles(ellipse_minor_r, count_ellipse)
    placed_circle = place_circle_particles(circle_r, count_circle)
    print(f"Successfully placed {placed_ellipse} ellipses and {placed_circle} circles")

    for p in particles:
        p['cluster_members'] = [p]

    num_saved_frames = (max_time // store_every_k) + 1
    trajectories = np.zeros((num_saved_frames, N_total, 2), dtype=np.float32)
    orientations = np.zeros((num_saved_frames, N_total), dtype=np.float32)
    shapes = np.array([p['shape'] for p in particles])  # Store particle types

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
    for p in particles: clusters[p['cluster_id']].append(p)
    
    for members in clusters.values():
        Rg = radius_of_gyration(members)
        M = cluster_mass(members)
        if M >= min_mass and Rg > min_Rg:
            global_pairs.append((Rg, M))

    Df = float('nan')
    if len(global_pairs) >= min_pts_for_fit:
        unique = {}
        for rg, m in global_pairs: unique[(round(rg,6), round(m,6))] = (rg, m)
        Rg_vals, M_vals = zip(*unique.values())
        if len(Rg_vals) > 2:
            logRg = np.log(np.array(Rg_vals)); logM = np.log(np.array(M_vals))
            slope, _ = np.polyfit(logRg, logM, 1)
            Df = float(slope)

    tag = f"frac{fraction_ellipse:.3f}_N{N_total}_t{max_time}"
    
    np.save(os.path.join(save_dir, f"traj_{tag}.npy"), trajectories)
    np.save(os.path.join(save_dir, f"shapes_{tag}.npy"), shapes)
    np.save(os.path.join(save_dir, f"orient_{tag}.npy"), orientations)
    
    meta = {
        "fraction_ellipse": fraction_ellipse,
        "ellipse_minor_r": ellipse_minor_r,
        "ellipse_major_r": ellipse_major_r,
        "circle_r": circle_r,
        "N_total": N_total,
        "max_time": max_time,
        "saved_frames": int(trajectories.shape[0]),
        "Df": Df,
        "count_ellipse": count_ellipse,
        "count_circle": count_circle
    }
    with open(os.path.join(save_dir, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Plot final configuration
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    for p in particles:
        if p['shape'] == 'circle':
            c = Circle(xy=(p['x'], p['y']), 
                      radius=p['radius'],
                      facecolor='blue', edgecolor='black', alpha=0.8)
            ax.add_patch(c)
        else:  # ellipse
            e = Ellipse(xy=(p['x'], p['y']), 
                       width=p['major']*2, height=p['minor']*2, 
                       angle=math.degrees(p['orientation']),
                       facecolor='orange', edgecolor='black', alpha=0.8)
            ax.add_patch(e)
            
    plt.xlim(0, width); plt.ylim(0, height)
    plt.title(f"Binary DLCA: Ellipses (orange) + Circles (blue), Df={Df:.3f}")
    plt.savefig(os.path.join(save_plots, f"final_{tag}.png"))
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
    root.title("Binary DLCA: Ellipses + Circles")

    particles_var = tk.StringVar(value="100")
    ellipse_minor_var = tk.StringVar(value="4")
    circle_r_var = tk.StringVar(value="6")
    frac_ellipse_var = tk.StringVar(value="0.5")
    status_var = tk.StringVar(value="Idle")

    def add_row(label_text, widget, row):
        ttk.Label(root, text=label_text).grid(row=row, column=0, padx=8, pady=6, sticky="e")
        widget.grid(row=row, column=1, padx=8, pady=6, sticky="we")

    add_row("Total Particles", ttk.Entry(root, textvariable=particles_var), 0)
    add_row("Ellipse Minor Radius", ttk.Entry(root, textvariable=ellipse_minor_var), 1)
    add_row("Circle Radius", ttk.Entry(root, textvariable=circle_r_var), 2)
    add_row("Fraction Ellipses (0-1)", ttk.Entry(root, textvariable=frac_ellipse_var), 3)

    ttk.Label(root, text="Status").grid(row=4, column=0, padx=8, pady=6, sticky="e")
    ttk.Label(root, textvariable=status_var).grid(row=4, column=1, padx=8, pady=6, sticky="we")
    
    info_text = f"Note: Ellipse aspect ratio is fixed at {ELLIPSE_ASPECT_RATIO}:1 (major:minor)"
    ttk.Label(root, text=info_text, font=('Arial', 8, 'italic')).grid(
        row=5, column=0, columnspan=2, padx=8, pady=2)

    def start_run():
        try:
            N = int(particles_var.get())
            ellipse_minor = float(ellipse_minor_var.get())
            circle_r = float(circle_r_var.get())
            frac_e = float(frac_ellipse_var.get())
            
            if not (0 <= frac_e <= 1):
                messagebox.showerror("Error", "Fraction must be between 0 and 1")
                return
                
        except ValueError:
            messagebox.showerror("Error", "Invalid inputs")
            return

        start_btn.config(state="disabled")
        status_var.set("Running...")

        def worker():
            try:
                run_simulation_binary(N, ellipse_minor, circle_r, frac_e, max_time) 
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
    
    root.mainloop()

if __name__ == "__main__":
    launch_ui()