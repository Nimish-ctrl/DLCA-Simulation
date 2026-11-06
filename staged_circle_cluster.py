"""
staged_circle_cluster.py — Two-stage staged DLCA simulation for circle particles.

Stage 1: Runs 8 independent 500-particle circle simulations concurrently via multiprocessing.
Stage 2: Combines the 8 final cluster states into a single 4000-particle tiled system
         and runs the continuation simulation.
"""

import os
import sys
import glob
import math
import time
import json
import random
import multiprocessing as mp
import numpy as np

import circle_cluster

def _worker_stage1(batch_idx, n_total, radii_pair, fraction_a, max_time, store_every_k, batch_save_dir, seed):
    """Worker function executed in a separate OS process for Stage 1."""
    print(f"[Stage 1] Starting Batch {batch_idx}: N={n_total}, seed={seed}, out={batch_save_dir}")
    t0 = time.time()
    res = circle_cluster.run_simulation_for_composition(
        N_total=n_total,
        radii_pair=radii_pair,
        fraction_A=fraction_a,
        max_time=max_time,
        store_every_k=store_every_k,
        save_dir=batch_save_dir,
        seed=seed
    )
    elapsed = time.time() - t0
    final_clusters = res.get('num_clusters_final', 'N/A')
    print(f"[Stage 1] Finished Batch {batch_idx} in {elapsed:.2f}s | Final clusters: {final_clusters}")
    return res

def run_stage1_parallel(num_batches=8, N_total=500, radii_pair=(8.0, 16.0), fraction_A=0.668,
                        max_time=10000, store_every_k=1, base_save_dir="sim_npy_out", base_seed=1000):
    """
    Launches `num_batches` Stage 1 simulations concurrently as separate OS processes.
    Each batch gets its own distinct save directory to prevent output collisions.
    """
    print("=" * 70)
    print(f"STAGE 1: Launching {num_batches} parallel simulations (N={N_total} each)...")
    print("=" * 70)
    
    processes = []
    batch_dirs = []
    
    for i in range(num_batches):
        batch_dir = os.path.join(base_save_dir, f"batch_{i}")
        batch_dirs.append(batch_dir)
        seed = base_seed + i
        p = mp.Process(
            target=_worker_stage1,
            args=(i, N_total, radii_pair, fraction_A, max_time, store_every_k, batch_dir, seed)
        )
        processes.append(p)
        p.start()
        
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Batch {i} process exited with code {p.exitcode}")
            
    print("=" * 70)
    print(f"STAGE 1 COMPLETE: All {num_batches} batches finished.")
    print("=" * 70)
    return batch_dirs

def compute_stage2_diffusion_parameters(radii_pair=(8.0, 16.0), fraction_A=0.668, N_batch=500, margin=4.0, safety_factor=2.5):
    """
    Computes expected step size, diffusion distance, and required Stage 2 simulation time.
    """
    velocity = circle_cluster.velocity
    r_dom = radii_pair[0] if fraction_A >= 0.5 else radii_pair[1]
    cluster_mass = N_batch * (r_dom ** 2)
    step_size = velocity * (1.0 / math.sqrt(cluster_mass))
    expected_distance = safety_factor * margin
    max_time_stage2 = int(math.ceil((expected_distance / step_size) ** 2))
    return step_size, expected_distance, margin, max_time_stage2, cluster_mass

def run_stage2_continuation(num_batches=8, base_save_dir="sim_npy_out",
                            stage2_save_dir="sim_npy_out/stage2_combined",
                            tag=None,
                            max_time=None, store_every_k=1,
                            radii_pair=(8.0, 16.0), fraction_A=0.668,
                            margin=4.0, safety_factor=2.5,
                            border=20.0, seed=42, cols=4, rows=2):
    """
    Loads final states of all Stage 1 batches, tiles them onto a combined canvas
    with tightly calibrated tile margins based on random walk diffusion physics,
    renumbers cluster IDs, and runs the Stage 2 continuation simulation.
    """
    # 1. Diffusion physics calculations
    step_size, expected_distance, margin, calc_max_time, cluster_mass = compute_stage2_diffusion_parameters(
        radii_pair=radii_pair,
        fraction_A=fraction_A,
        N_batch=500,
        margin=margin,
        safety_factor=safety_factor
    )

    if max_time is None:
        max_time = calc_max_time

    print("=" * 70)
    print("STAGE 2: DIFFUSION-BASED TILE SIZING & CONTINUATION PLANNING")
    print("=" * 70)
    print(f"  Velocity constant:           {circle_cluster.velocity:.2f} px/step")
    print(f"  Dominant particle radius:    {radii_pair[0] if fraction_A >= 0.5 else radii_pair[1]:.2f} px")
    print(f"  Estimated 500-cluster mass:  {cluster_mass:.1f}")
    print(f"  Step size per timestep:      {step_size:.5f} px/step")
    print(f"  Chosen margin (gap):         {margin:.2f} px")
    print(f"  Expected diffusion distance: {expected_distance:.2f} px (Safety factor: {safety_factor:.1f}x)")
    print(f"  Chosen max_time_stage2:      {max_time} steps")
    print("=" * 70)

    # 2. Load all batch outputs and calculate cluster bounding boxes
    batches_data = []
    for b in range(num_batches):
        batch_dir = os.path.join(base_save_dir, f"batch_{b}")
        traj_files = glob.glob(os.path.join(batch_dir, "traj_*.npy"))
        radii_files = glob.glob(os.path.join(batch_dir, "radii_*.npy"))
        cids_files = glob.glob(os.path.join(batch_dir, "clusterids_*.npy"))

        if not traj_files or not radii_files or not cids_files:
            raise FileNotFoundError(f"Missing Stage 1 output files in {batch_dir}")

        traj = np.load(traj_files[0])
        radii = np.load(radii_files[0])
        cids = np.load(cids_files[0])
        final_pos = traj[-1]  # shape: (N, 2)

        xmin = float(np.min(final_pos[:, 0] - radii))
        xmax = float(np.max(final_pos[:, 0] + radii))
        ymin = float(np.min(final_pos[:, 1] - radii))
        ymax = float(np.max(final_pos[:, 1] + radii))
        w = xmax - xmin
        h = ymax - ymin

        batches_data.append({
            'b': b,
            'pos': final_pos,
            'radii': radii,
            'cids': cids,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'w': w,
            'h': h
        })

    # 3. Calculate grid column widths and row heights for tight tile alignment
    col_widths = [max(batches_data[r * cols + c]['w'] for r in range(rows)) for c in range(cols)]
    row_heights = [max(batches_data[r * cols + c]['h'] for c in range(cols)) for r in range(rows)]

    col_x = [border]
    for c in range(cols - 1):
        col_x.append(col_x[-1] + col_widths[c] + margin)

    row_y = [border]
    for r in range(rows - 1):
        row_y.append(row_y[-1] + row_heights[r] + margin)

    total_width = col_x[-1] + col_widths[-1] + border
    total_height = row_y[-1] + row_heights[-1] + border

    # 4. Place particles into combined list with renumbered cluster IDs
    combined_particles = []
    running_id = 0

    for b_idx, d in enumerate(batches_data):
        col = b_idx % cols
        row = b_idx // cols

        pad_x = (col_widths[col] - d['w']) / 2.0
        pad_y = (row_heights[row] - d['h']) / 2.0

        off_x = col_x[col] + pad_x
        off_y = row_y[row] + pad_y

        cid_offset = b_idx * 1_000_000
        pos = d['pos']
        radii = d['radii']
        cids = d['cids']

        print(f"  Positioning Batch {b_idx}: {len(radii)} particles -> Tile ({col}, {row}) at offset ({off_x:.1f}, {off_y:.1f})")

        for idx in range(len(radii)):
            px = float((pos[idx, 0] - d['xmin']) + off_x)
            py = float((pos[idx, 1] - d['ymin']) + off_y)
            pr = float(radii[idx])
            pcid = int(cids[idx] + cid_offset)

            particle = {
                'x': px,
                'y': py,
                'r': pr,
                'cluster_id': pcid,
                'cluster_members': [],
                'angle': random.uniform(0, 2 * math.pi),
                'id': running_id
            }
            combined_particles.append(particle)
            running_id += 1

    # Rebuild cluster members for the combined particle list
    circle_cluster.rebuild_cluster_members(combined_particles)

    # Reassign global width and height to the total canvas size
    circle_cluster.width = total_width
    circle_cluster.height = total_height

    print(f"Combined system: N_total={len(combined_particles)}, canvas={total_width:.1f}x{total_height:.1f}")
    print(f"Initial unique clusters in Stage 2: {len(set(p['cluster_id'] for p in combined_particles))}")

    if tag is None:
        tag = f"stage2_N{len(combined_particles)}"

    # Run Stage 2 simulation
    res = circle_cluster.simulate_particles(
        particles=combined_particles,
        N_total=len(combined_particles),
        max_time=max_time,
        store_every_k=store_every_k,
        save_dir=stage2_save_dir,
        tag=tag,
        radii_pair=radii_pair,
        fraction_A_label=None,
        seed=seed
    )

    print("=" * 70)
    print(f"STAGE 2 COMPLETE:")
    print(f"  Final clusters: {res.get('num_clusters_final')}")
    print(f"  Elapsed time: {res.get('elapsed_s'):.2f}s")
    print(f"  Traj path: {res.get('traj_path')}")
    print("=" * 70)
    return res

if __name__ == "__main__":
    # Standard settings for 5000 particles (10 batches of 500)
    NUM_BATCHES = 10
    COLS = 5
    ROWS = 2
    N_BATCH = 500
    RADII = (8.0, 16.0)
    FRAC_A = 0.668
    
    # Stage 1 settings
    MAX_TIME_STAGE1 = 10000
    K_STORE_STAGE1 = 1

    # Stage 2 settings (separate and distinct)
    MARGIN_STAGE2 = 4.0      # gap between adjacent clusters in pixels
    SAFETY_FACTOR_STAGE2 = 2.5 # expected diffusion distance = 2.5 * margin
    K_STORE_STAGE2 = 1       # dense storage for smooth Stage 2 video rendering

    # Stage 1
    run_stage1_parallel(
        num_batches=NUM_BATCHES,
        N_total=N_BATCH,
        radii_pair=RADII,
        fraction_A=FRAC_A,
        max_time=MAX_TIME_STAGE1,
        store_every_k=K_STORE_STAGE1,
        base_save_dir="sim_npy_out",
        base_seed=1000
    )

    # Stage 2
    run_stage2_continuation(
        num_batches=NUM_BATCHES,
        base_save_dir="sim_npy_out",
        stage2_save_dir="sim_npy_out/stage2_combined",
        max_time=None,  # solved automatically from MARGIN_STAGE2 & diffusion physics
        store_every_k=K_STORE_STAGE2,
        radii_pair=RADII,
        fraction_A=FRAC_A,
        margin=MARGIN_STAGE2,
        safety_factor=SAFETY_FACTOR_STAGE2,
        seed=42,
        cols=COLS,
        rows=ROWS
    )
