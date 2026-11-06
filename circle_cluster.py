"""
dlca_batch_keep_logic.py — TRULY OPTIMIZED: Reduced cleanup passes + adaptive cleanup
"""

import random
import math
import time
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
width = 3200
height = 2400

spawn_width = 1600
spawn_height = 1200

# spawn origin (top-left corner) — center the spawn box by default:
spawn_x0 = (width - spawn_width) / 2.0
spawn_y0 = (height - spawn_height) / 2.0

velocity = 5.0
max_time = 10000
min_mass = 10.0
min_Rg = 1e-8
min_pts_for_fit = 6
save_dir = "sim_npy_out"
save_plots = "plots"
store_every_k = 1
repeats = 1

# Visible contact gap and extra safety push (pixels)
EPS_CONTACT = 1.0
EPS_PUSH    = 0.2
ROT_OMEGA_BASE = 0.0      # Zero directional bias (unbiased random rotation)
ROT_BROWNIAN   = 0.03     # Symmetric Brownian rotational jitter rad/step

# Performance tuning - AGGRESSIVE OPTIMIZATION
ADAPTIVE_CLEANUP = True           # Only cleanup when needed
EARLY_TERMINATION_THRESHOLD = 0.1 # Stop cleanup if penetration < this
MAX_SEPARATION_PASSES = 20        # Reduced from 80
CLEANUP_OUTER_REPEATS = 1         # Reduced from 5
ENFORCE_GAP_ROUNDS = 2            # Reduced from 6

# ---------------------------
# Grid helpers
# ---------------------------
def distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

def update_spatial_grid(particles, grid, cell_size):
    """Full grid rebuild - O(N)"""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j].clear()
    for p in particles:
        cx = int(p['x'] / cell_size)
        cy = int(p['y'] / cell_size)
        if 0 <= cx < len(grid) and 0 <= cy < len(grid[0]):
            grid[cx][cy].append(p)

def update_particle_in_grid(p, grid, cell_size, old_x=None, old_y=None):
    """Incremental grid update - O(1) - only update changed cells"""
    new_cx = int(p['x'] / cell_size)
    new_cy = int(p['y'] / cell_size)
    
    if old_x is not None and old_y is not None:
        old_cx = int(old_x / cell_size)
        old_cy = int(old_y / cell_size)
        
        # Only update if cell changed
        if old_cx == new_cx and old_cy == new_cy:
            return
        
        # Remove from old cell
        if 0 <= old_cx < len(grid) and 0 <= old_cy < len(grid[0]):
            try:
                grid[old_cx][old_cy].remove(p)
            except ValueError:
                pass  # Already removed
    
    # Add to new cell
    if 0 <= new_cx < len(grid) and 0 <= new_cy < len(grid[0]):
        if p not in grid[new_cx][new_cy]:
            grid[new_cx][new_cy].append(p)

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
    # Compute how far the cluster extends beyond each boundary and translate it back in one rigid move.
    min_left = min(p['x'] - p['r'] for p in cluster)
    max_right = max(p['x'] + p['r'] for p in cluster)
    min_top = min(p['y'] - p['r'] for p in cluster)
    max_bot = max(p['y'] + p['r'] for p in cluster)

    dx = 0.0
    dy = 0.0
    if min_left < 0.0:
        dx += -min_left
    if max_right > width:
        dx += width - max_right
    if min_top < 0.0:
        dy += -min_top
    if max_bot > height:
        dy += height - max_bot

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
    return by_id

def _cluster_mass(cluster):
    return sum(pp['r']**2 for pp in cluster)

def resolve_overlap_pair_as_clusters(p1, p2, eps=EPS_CONTACT):
    # Rigidly separate the two clusters along line of centers to reach a gap of ~eps
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    dist = math.hypot(dx, dy)
    target = p1['r'] + p2['r']

    # Remove the early return; we want to also handle near-tangent (dist in [target, target+eps))
    if dist == 0.0:
        h = (p1.get('id',0)*73856093) ^ (p2.get('id',0)*19349663)
        ang = (h & 1023) * (2.0*math.pi/1024.0)
        ux, uy = math.cos(ang), math.sin(ang)
        dist = 1e-9
    else:
        ux, uy = dx / dist, dy / dist

    # Positive 'overlap' means "how much to push to achieve EPS gap"
    overlap = (target - dist) + eps
    if overlap <= 0.0:
        return False

    c1 = p1['cluster_members']; c2 = p2['cluster_members']
    m1 = max(_cluster_mass(c1), 1e-12)
    m2 = max(_cluster_mass(c2), 1e-12)
    tm = m1 + m2

    s1 = (m2 / tm) * overlap
    s2 = (m1 / tm) * overlap

    _move_cluster(c1, ux * s1, uy * s1)
    _move_cluster(c2, -ux * s2, -uy * s2)
    return True

def fully_separate_two_clusters(particles, grid, cell_size, cid1, cid2, eps=EPS_CONTACT, max_passes=None):
    """OPTIMIZED: Early termination + adaptive max_passes + grid-restricted candidate search"""
    if max_passes is None:
        max_passes = MAX_SEPARATION_PASSES
    
    by_id = rebuild_cluster_members(particles)
    c1 = by_id[cid1]
    c2 = by_id[cid2]
    
    for pass_num in range(max_passes):
        moved = False
        
        # Early termination: if clusters are small, don't need many passes
        if pass_num > 3 and len(c1) + len(c2) < 10:
            break
        
        for a in c1:
            cx, cy = int(a['x'] / cell_size), int(a['y'] / cell_size)
            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for b in grid[gx][gy]:
                        if b['cluster_id'] != cid2:
                            continue
                        rs = a['r'] + b['r']
                        dist = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
                        if dist < (rs + eps):
                            if resolve_overlap_pair_as_clusters(a, b, eps=eps):
                                moved = True
        if moved:
            update_spatial_grid(particles, grid, cell_size)
            continue
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


def rotate_all_clusters(particles, grid, cell_size,
                        omega_base=ROT_OMEGA_BASE, omega_brownian=ROT_BROWNIAN, dt=1.0):
    seen_clusters = set()
    for p in particles:
        cid = p['cluster_id']
        if cid in seen_clusters:
            continue
        cluster = p.get('cluster_members', [])
        if len(cluster) <= 1:
            continue
        seen_clusters.add(cid)
        
        # 1. Calculate properties
        M = sum(pp['r']**2 for pp in cluster)
        if M <= 0.0: continue
        
        # Calculate Center of Mass
        cx = sum(pp['x'] * (pp['r']**2) for pp in cluster) / M
        cy = sum(pp['y'] * (pp['r']**2) for pp in cluster) / M

        # 2. Dynamic Angle Limit (Anti-Tunneling)
        # Find the furthest particle from COM to determine max swing radius
        max_dist_sq = 0.0
        min_particle_r = 1e9
        for pp in cluster:
            d2 = (pp['x'] - cx)**2 + (pp['y'] - cy)**2
            if d2 > max_dist_sq:
                max_dist_sq = d2
            if pp['r'] < min_particle_r:
                min_particle_r = pp['r']
        
        max_dist = math.sqrt(max_dist_sq)
        
        # Standard angle calculation
        mass_scale = 1.0 / (1.0 + math.log(1.0 + M / 1000.0))
        omega = omega_base * mass_scale
        jitter = random.uniform(-omega_brownian, omega_brownian) * mass_scale
        raw_angle = (omega + jitter) * dt
        
        # LIMIT: Ensure the tip of the cluster doesn't move more than ~min_particle_r
        if max_dist > 0.001:
            max_angle = (0.5 * min_particle_r) / max_dist
            angle = max(-max_angle, min(max_angle, raw_angle))
        else:
            angle = raw_angle

        # 3. Rotate blindly
        rotate_cluster_about_com(cluster, angle)
        keep_cluster_inside_bounds(cluster, width, height)


def resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2, eps=EPS_CONTACT):
    """OPTIMIZED: Early termination when no overlaps found"""
    for _ in range(passes):
        moved_any = False
        update_spatial_grid(particles, grid, cell_size)
        
        processed_pairs = set()  # Avoid processing same cluster pair multiple times
        
        for i in range(len(particles)):
            p1 = particles[i]
            cx, cy = int(p1['x']/cell_size), int(p1['y']/cell_size)
            for gx in range(max(0, cx-1), min(len(grid), cx+2)):
                for gy in range(max(0, cy-1), min(len(grid[0]), cy+2)):
                    for p2 in grid[gx][gy]:
                        if p1 is p2 or p1['cluster_id']==p2['cluster_id']:
                            continue
                        
                        # Avoid processing same cluster pair multiple times
                        pair_key = tuple(sorted([p1['cluster_id'], p2['cluster_id']]))
                        if pair_key in processed_pairs:
                            continue
                        
                        rs = p1['r'] + p2['r']
                        dist = math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
                        if dist < (rs + eps):
                            fully_separate_two_clusters(
                                particles, grid, cell_size,
                                p1['cluster_id'], p2['cluster_id'],
                                eps=eps, max_passes=MAX_SEPARATION_PASSES
                            )
                            moved_any = True
                            processed_pairs.add(pair_key)
        if not moved_any:
            break

# ---------------------------
# OPTIMIZED: Grid-based cleanup with adaptive cleanup
# ---------------------------
def max_penetration_grid(particles, grid, cell_size):
    """
    OPTIMIZED: Grid-based maximum inter-cluster penetration depth.
    O(N) instead of O(N²) for sparse distributions.
    """
    max_pen = 0.0
    for i in range(len(particles)):
        pi = particles[i]
        cx, cy = int(pi['x'] / cell_size), int(pi['y'] / cell_size)
        for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
            for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                for pj in grid[gx][gy]:
                    if pi is pj or pi['cluster_id'] == pj['cluster_id']:
                        continue
                    rs = pi['r'] + pj['r']
                    d = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                    pen = rs - d
                    if pen > max_pen:
                        max_pen = pen
    return max_pen

def separate_all_overlaps_grid(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=40):
    """
    OPTIMIZED: Grid-based global cleanup with early termination.
    """
    for round_num in range(max_rounds):
        moved = False
        update_spatial_grid(particles, grid, cell_size)
        
        # EARLY TERMINATION: Check if penetration is small enough
        if round_num > 2:  # After a few rounds, check penetration
            max_pen = max_penetration_grid(particles, grid, cell_size)
            if max_pen < EARLY_TERMINATION_THRESHOLD:
                break
        
        # Track which cluster pairs we've already processed this round
        processed_pairs = set()
        
        for i in range(len(particles)):
            pi = particles[i]
            cx, cy = int(pi['x'] / cell_size), int(pi['y'] / cell_size)
            for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                    for pj in grid[gx][gy]:
                        if pi is pj or pi['cluster_id'] == pj['cluster_id']:
                            continue
                        
                        # Avoid processing same cluster pair multiple times
                        pair_key = tuple(sorted([pi['cluster_id'], pj['cluster_id']]))
                        if pair_key in processed_pairs:
                            continue
                        
                        rs = pi['r'] + pj['r']
                        dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                        if dist < (rs + eps):
                            fully_separate_two_clusters(
                                particles, grid, cell_size,
                                pi['cluster_id'], pj['cluster_id'],
                                eps=eps + EPS_PUSH, max_passes=MAX_SEPARATION_PASSES
                            )
                            moved = True
                            processed_pairs.add(pair_key)
        
        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)

def enforce_eps_gap_grid(particles, grid, cell_size, eps=EPS_CONTACT, max_rounds=None):
    """
    OPTIMIZED: Grid-based epsilon gap enforcement with early termination.
    """
    if max_rounds is None:
        max_rounds = ENFORCE_GAP_ROUNDS
    
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
                        rs = pi['r'] + pj['r']
                        dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                        if dist < (rs + eps):
                            resolve_overlap_pair_as_clusters(pi, pj, eps=eps)
                            moved = True
        if not moved:
            break
    enforce_world_bounds_all(particles, width, height)

def cleanup_until_clean_grid(particles, grid, cell_size, eps=EPS_CONTACT, outer_repeats=None):
    """
    OPTIMIZED: Grid-based cleanup with penetration check and early termination.
    """
    if outer_repeats is None:
        outer_repeats = CLEANUP_OUTER_REPEATS
    
    for _ in range(outer_repeats):
        separate_all_overlaps_grid(particles, grid, cell_size, eps=eps, max_rounds=40)
        update_spatial_grid(particles, grid, cell_size)
        max_pen = max_penetration_grid(particles, grid, cell_size)
        if max_pen <= EARLY_TERMINATION_THRESHOLD:
            break

# ---------------------------
# Merge, collisions, movement
# ---------------------------
def merge_clusters(particles, cluster_id1, cluster_id2):
    new_cluster_id = min(cluster_id1, cluster_id2)
    other_cluster_id = max(cluster_id1, cluster_id2)
    
    # Fast path: find representatives with cluster_members
    c1 = None
    c2 = None
    for p in particles:
        if p['cluster_id'] == new_cluster_id and c1 is None:
            c1 = p.get('cluster_members')
        elif p['cluster_id'] == other_cluster_id and c2 is None:
            c2 = p.get('cluster_members')
        if c1 is not None and c2 is not None:
            break
            
    if not c1:
        c1 = [p for p in particles if p['cluster_id'] == new_cluster_id]
    if not c2:
        c2 = [p for p in particles if p['cluster_id'] == other_cluster_id]

    merged_members = c1 + c2
    for p in c2:
        p['cluster_id'] = new_cluster_id
    for p in merged_members:
        p['cluster_members'] = merged_members

    mass1 = sum(p['r'] ** 2 for p in c1) if c1 else 1.0
    mass2 = sum(p['r'] ** 2 for p in c2) if c2 else 1.0
    total_mass = mass1 + mass2
    _, _, vx1, vy1, _ = get_cluster_properties(c1)
    _, _, vx2, vy2, _ = get_cluster_properties(c2)
    denom = total_mass if total_mass > 0 else 1.0
    new_vx = (vx1 * mass1 + vx2 * mass2) / denom
    new_vy = (vy1 * mass1 + vy2 * mass2) / denom
    velocity_reduction = 1 / (1 + total_mass / 10000)
    new_vx *= velocity_reduction; new_vy *= velocity_reduction
    new_angle = math.atan2(new_vy, new_vx)
    for p in merged_members:
        p['angle'] = new_angle

def circle_collisions(p1, p2, grid, cell_size):
    # 1. If already same cluster, ignore
    if p1['cluster_id'] == p2['cluster_id']:
        return False

    rs = p1['r'] + p2['r']
    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    
    # 2. Check collision
    if dist_sq >= rs**2:
        return False

    # 3. Rigid separation BEFORE merging
    fully_separate_two_clusters(
        particles_ref, grid, cell_size, 
        p1['cluster_id'], p2['cluster_id'], 
        eps=0.001,
        max_passes=MAX_SEPARATION_PASSES
    )
    
    # 4. Merge clusters
    merge_clusters(particles_ref, p1['cluster_id'], p2['cluster_id'])
    
    # 5. Update grid
    update_spatial_grid(particles_ref, grid, cell_size)
    return True

def update_cluster_movement(particles, grid, cell_size):
    seen_clusters = set()
    for p in particles:
        cid = p['cluster_id']
        if cid in seen_clusters:
            continue
        cluster = p.get('cluster_members', [])
        if len(cluster) <= 1:
            continue
        seen_clusters.add(cid)

        total_mass = sum(pp['r'] ** 2 for pp in cluster)
        com_x = sum(pp['x'] * pp['r'] ** 2 for pp in cluster) / total_mass
        com_y = sum(pp['y'] * pp['r'] ** 2 for pp in cluster) / total_mass
        avg_vx = sum(velocity / pp['r'] * math.cos(pp['angle']) for pp in cluster) / len(cluster)
        avg_vy = sum(velocity / pp['r'] * math.sin(pp['angle']) for pp in cluster) / len(cluster)
        vel_scale = 1.0 / (1.0 + math.log(1.0 + total_mass / 1000.0))
        avg_vx *= vel_scale; avg_vy *= vel_scale
        br_angle = random.uniform(0, 2 * math.pi)
        mf = 1.0 / max(total_mass, 1e-12) ** 0.3
        dx = avg_vx + velocity * mf * math.cos(br_angle)
        dy = avg_vy + velocity * mf * math.sin(br_angle)
        new_com_x = com_x + dx
        new_com_y = com_y + dy
        final_angle = br_angle
        offsets = [(pp['x'] - com_x, pp['y'] - com_y, pp['r']) for pp in cluster]
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
        for idx, pp in enumerate(cluster):
            ox, oy, r = offsets[idx]
            pp['x'] = new_com_x + ox
            pp['y'] = new_com_y + oy
            pp['angle'] = final_angle

        keep_cluster_inside_bounds(cluster, width, height)

# ---------------------------
# Simulation per composition
# ---------------------------
def simulate_particles(particles, N_total, max_time, store_every_k=1,
                       save_dir=save_dir, tag=None, radii_pair=(10.0, 10.0),
                       fraction_A_label=None, seed=None,
                       requested_N_total=None, placed_N_total=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    actual_N_total = len(particles)
    if requested_N_total is None:
        requested_N_total = N_total
    if placed_N_total is None:
        placed_N_total = actual_N_total

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_plots, exist_ok=True)

    cell_size = max(radii_pair) * 2
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [[[] for _ in range(grid_height)] for _ in range(grid_width)]
    update_spatial_grid(particles, grid, cell_size)

    num_saved_frames = (max_time // store_every_k) + 1
    trajectories = np.zeros((num_saved_frames, actual_N_total, 2), dtype=np.float32)

    global_pairs = []
    time_evolution = []

    # Initialize cluster count once before the timestep loop
    current_cluster_count = len(set(p['cluster_id'] for p in particles))
    cluster_count_evolution = []

    t_save_idx = 0
    start_time = time.time()

    # Initial cleanup - AGGRESSIVE OPTIMIZATION
    resolve_all_overlaps_rigid(particles, grid, cell_size, passes=2)
    cleanup_until_clean_grid(particles, grid, cell_size, eps=EPS_CONTACT)
    enforce_eps_gap_grid(particles, grid, cell_size)

    collision_happened = False

    for time_step in range(max_time):
        update_spatial_grid(particles, grid, cell_size)
        # Phase A: rigid rotation for each cluster
        rotate_all_clusters(particles, grid, cell_size,
                    omega_base=ROT_OMEGA_BASE,
                    omega_brownian=ROT_BROWNIAN,
                    dt=1.0)

        for p in particles:
            if len(p['cluster_members']) <= 1:
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
        collision_happened = False
        
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
                        if circle_collisions(p1, p2, grid, cell_size):
                            checked_pairs.add(cluster_pair)
                            collision_happened = True
                            current_cluster_count -= 1

        # Lightweight overlap cleanup
        if collision_happened:
            resolve_all_overlaps_rigid(particles, grid, cell_size, passes=1)

        # Move multi-particle clusters
        update_cluster_movement(particles, grid, cell_size)
        
        if collision_happened:
            resolve_all_overlaps_rigid(particles, grid, cell_size, passes=1)
        
        # Enforce boundary bounds
        enforce_world_bounds_all(particles, width, height)

        # Record fine-grained cluster count at every step
        cluster_count_evolution.append((time_step, current_cluster_count))

        is_save_step = (time_step % store_every_k) == 0

        # Early termination: all particles merged into a single cluster
        if current_cluster_count == 1:
            print(f"All particles merged into one cluster at time step {time_step}")
            if not is_save_step and t_save_idx < num_saved_frames:
                for idx, p in enumerate(particles):
                    trajectories[t_save_idx, idx, 0] = p['x']
                    trajectories[t_save_idx, idx, 1] = p['y']
                t_save_idx += 1
            break

        if is_save_step and t_save_idx < num_saved_frames:
            for idx, p in enumerate(particles):
                trajectories[t_save_idx, idx, 0] = p['x']
                trajectories[t_save_idx, idx, 1] = p['y']
            t_save_idx += 1

    elapsed = time.time() - start_time
    trajectories = trajectories[:t_save_idx]

    # Convert cluster count evolution to numpy array [time_step, cluster_count]
    clustercount_data = np.array(cluster_count_evolution, dtype=np.int32)

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
    clusterids_per_particle = np.array([p['cluster_id'] for p in particles], dtype=np.int32)
    if tag is None:
        if fraction_A_label is not None:
            tag = f"comp{fraction_A_label:.3f}_N{actual_N_total}_t{max_time}"
        else:
            tag = f"N{actual_N_total}_t{max_time}"
    traj_path = os.path.join(save_dir, f"traj_{tag}.npy")
    radii_path = os.path.join(save_dir, f"radii_{tag}.npy")
    clusterids_path = os.path.join(save_dir, f"clusterids_{tag}.npy")
    clustercount_path = os.path.join(save_dir, f"clustercount_{tag}.npy")
    meta_path = os.path.join(save_dir, f"meta_{tag}.json")
    np.save(traj_path, trajectories)
    np.save(radii_path, radii_per_particle)
    np.save(clusterids_path, clusterids_per_particle)
    np.save(clustercount_path, clustercount_data)
    meta = {
        "fraction_A": fraction_A_label,
        "count_A": int(round(placed_N_total * fraction_A_label)) if fraction_A_label is not None else None,
        "count_B": (placed_N_total - int(round(placed_N_total * fraction_A_label))) if fraction_A_label is not None else None,
        "radii_pair": list(radii_pair),
        "N_total": placed_N_total,
        "requested_N_total": requested_N_total,
        "placed_N_total": placed_N_total,
        "width": width,
        "height": height,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "saved_frames": int(trajectories.shape[0]),
        "final_clusters": int(current_cluster_count)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    enforce_world_bounds_all(particles, width, height)
    # Simple Matplotlib visualization
    os.makedirs(save_plots, exist_ok=True)
    plt.figure(figsize=(12, 8))
    for p in particles:
        color = 'red'
        circle = plt.Circle((p['x'], p['y']), p['r'], color=color, fill=True, alpha=1)
        plt.gca().add_patch(circle)
        outline = plt.Circle((p['x'], p['y']), p['r'], color='black', fill=False, linewidth=0.8)
        plt.gca().add_patch(outline)
    plt.xlim(0, width); plt.ylim(0, height); plt.gca().set_aspect('equal', adjustable='box')
    title_prefix = f"Final: {fraction_A_label:.2f} A, " if fraction_A_label is not None else f"Final ({tag}): "
    plt.title(f"{title_prefix}{current_cluster_count} clusters | N={len(particles)}")
    plot_filename = f"final_comp{fraction_A_label:.3f}.png" if fraction_A_label is not None else f"final_{tag}.png"
    plt.savefig(os.path.join(save_plots, plot_filename), dpi=180, bbox_inches='tight')
    plt.close()

    result = {
        "fraction": fraction_A_label,
        "Df": float(Df),
        "intercept": float(intercept) if 'intercept' in locals() else None,
        "r2": float(r2) if 'r2' in locals() else None,
        "Rg_largest": float(Rg_largest),
        "Rg_largest_n": float(Rg_largest_n),
        "Mass_largest": float(M_largest),
        "traj_path": traj_path,
        "radii_path": radii_path,
        "clusterids_path": clusterids_path,
        "clustercount_path": clustercount_path,
        "meta_path": meta_path,
        "elapsed_s": float(elapsed),
        "num_global_pairs": int(len(global_pairs)),
        "saved_frames": int(trajectories.shape[0]),
        "num_particles": len(particles),
        "num_clusters_final": int(current_cluster_count)
    }
    return result

def run_simulation_for_composition(N_total, radii_pair, fraction_A, max_time,
                                   store_every_k=1, save_dir=save_dir, seed=None,
                                   max_attempts=100000):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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
        for _ in range(count):
            attempts = 0
            while attempts < max_attempts:
                x = random.uniform(radius, width - radius)
                y = random.uniform(radius, height - radius)
                if check(particles, x, y, radius, grid, cell_size):
                    particle = {
                        'x': x, 'y': y, 'r': float(radius),
                        'cluster_id': cluster_id_counter,
                        'cluster_members': [], 'angle': random.uniform(0, 2 * math.pi),
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

    placed_A = place_particles_of_type(radii_pair[0], count_A)
    placed_B = place_particles_of_type(radii_pair[1], count_B)
    actual_N_total = placed_A + placed_B

    if actual_N_total != N_total:
        print(f"Warning: Requested {N_total} particles, but only successfully placed {actual_N_total} particles ({placed_A} of type A, {placed_B} of type B).")

    for p in particles:
        p['cluster_members'] = [p]

    tag = f"comp{fraction_A:.3f}_N{actual_N_total}_t{max_time}"
    return simulate_particles(
        particles=particles,
        N_total=actual_N_total,
        max_time=max_time,
        store_every_k=store_every_k,
        save_dir=save_dir,
        tag=tag,
        radii_pair=radii_pair,
        fraction_A_label=fraction_A,
        seed=None,
        requested_N_total=N_total,
        placed_N_total=actual_N_total
    )

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
            print(f"Done comp={f:.3f} rep={rep} particles={res['num_particles']} Df={res['Df']:.4f} Df_largest={Df_largest:.4f} Rg_final={res['Rg_largest']:.3f} time={res['elapsed_s']:.2f}s")
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
# Tkinter UI
# ---------------------------
import threading
import tkinter as tk
from tkinter import ttk, messagebox

def launch_ui():
    root = tk.Tk()
    root.title("DLCA Runner - TRULY OPTIMIZED")

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
        status_var.set("Running (TRULY OPTIMIZED)...")

        def worker():
            try:
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Circle DLCA Simulation")
    parser.add_argument("--cli", action="store_true", help="Run simulation in CLI mode")
    parser.add_argument("-N", "--particles", type=int, default=5000, help="Number of particles")
    parser.add_argument("-t", "--max_time", type=int, default=5000, help="Maximum timesteps")
    parser.add_argument("--rA", type=float, default=6.0, help="Radius A")
    parser.add_argument("--rB", type=float, default=10.0, help="Radius B")
    parser.add_argument("--fracA", type=float, default=0.668, help="Fraction A")
    parser.add_argument("--width", type=int, default=2400, help="Arena width")
    parser.add_argument("--height", type=int, default=1800, help="Arena height")
    args = parser.parse_args()

    if args.cli:
        print("=" * 70)
        print(f"STARTING DIRECT CLI SIMULATION: N={args.particles}, max_time={args.max_time}...")
        print("=" * 70)
        width = args.width
        height = args.height
        spawn_width = int(args.width * 0.85)
        spawn_height = int(args.height * 0.85)
        spawn_x0 = (width - spawn_width) / 2.0
        spawn_y0 = (height - spawn_height) / 2.0

        res = run_simulation_for_composition(
            N_total=args.particles,
            radii_pair=(args.rA, args.rB),
            fraction_A=args.fracA,
            max_time=args.max_time,
            store_every_k=1,
            seed=42
        )
        print("=" * 70)
        print(f"SIMULATION COMPLETE:")
        print(f"  Particles: {res['num_particles']}")
        print(f"  Final clusters: {res['num_clusters_final']}")
        print(f"  Saved frames: {res['saved_frames']}")
        print(f"  Elapsed time: {res['elapsed_s']:.2f}s")
        print(f"  Trajectory path: {res['traj_path']}")
        print("=" * 70)
    else:
        print("=" * 70)
        print("TRULY OPTIMIZED VERSION")
        print("=" * 70)
        print(f"✓ Grid-based cleanup (O(N) instead of O(N²))")
        print(f"✓ Adaptive cleanup (only when collisions occur)")
        print(f"✓ Early termination (penetration < {EARLY_TERMINATION_THRESHOLD})")
        print(f"✓ Reduced passes: separation={MAX_SEPARATION_PASSES}, cleanup={CLEANUP_OUTER_REPEATS}, gap={ENFORCE_GAP_ROUNDS}")
        print("=" * 70)
        print(f"Output: {save_dir} | Plots: {save_plots}")
        launch_ui()