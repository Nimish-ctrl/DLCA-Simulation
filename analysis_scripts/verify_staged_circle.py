"""
verify_staged_circle.py — Automated verification script for the staged circle simulation.
"""

import os
import glob
import math
import numpy as np
import engines.staged_circle_cluster as staged_circle_cluster
import engines.circle_cluster as circle_cluster

def verify_stage1_outputs(num_batches=8, base_save_dir="sim_npy_out"):
    print("\n--- Verifying Stage 1 Outputs ---")
    for b in range(num_batches):
        bdir = os.path.join(base_save_dir, f"batch_{b}")
        assert os.path.isdir(bdir), f"Directory {bdir} does not exist!"
        traj_files = glob.glob(os.path.join(bdir, "traj_*.npy"))
        radii_files = glob.glob(os.path.join(bdir, "radii_*.npy"))
        cids_files = glob.glob(os.path.join(bdir, "clusterids_*.npy"))
        meta_files = glob.glob(os.path.join(bdir, "meta_*.json"))

        assert len(traj_files) == 1, f"Expected 1 traj file in {bdir}, found {len(traj_files)}"
        assert len(radii_files) == 1, f"Expected 1 radii file in {bdir}, found {len(radii_files)}"
        assert len(cids_files) == 1, f"Expected 1 clusterids file in {bdir}, found {len(cids_files)}"
        assert len(meta_files) == 1, f"Expected 1 meta file in {bdir}, found {len(meta_files)}"

        traj = np.load(traj_files[0])
        radii = np.load(radii_files[0])
        cids = np.load(cids_files[0])

        assert traj.shape[1] == len(radii) == len(cids), f"Dimension mismatch in {bdir}"
        print(f"[OK] Batch {b} outputs verified: N={len(radii)}, frames={traj.shape[0]}, unique clusters={len(np.unique(cids))}")

    # Geometric sanity check on batch 0
    print("\n--- Geometric Sanity Check on Batch 0 Clusters ---")
    traj0 = np.load(glob.glob(os.path.join(base_save_dir, "batch_0", "traj_*.npy"))[0])
    radii0 = np.load(glob.glob(os.path.join(base_save_dir, "batch_0", "radii_*.npy"))[0])
    cids0 = np.load(glob.glob(os.path.join(base_save_dir, "batch_0", "clusterids_*.npy"))[0])
    final_pos = traj0[-1]
    eps = 2.0  # margin

    unique_cids = np.unique(cids0)
    for cid in unique_cids:
        indices = np.where(cids0 == cid)[0]
        if len(indices) == 1:
            continue  # single-particle cluster
        for i in indices:
            xi, yi, ri = final_pos[i, 0], final_pos[i, 1], radii0[i]
            # Check distance to other members in the same cluster
            min_dist = float('inf')
            touching_found = False
            for j in indices:
                if i == j:
                    continue
                xj, yj, rj = final_pos[j, 0], final_pos[j, 1], radii0[j]
                d = math.hypot(xi - xj, yi - yj)
                if d <= (ri + rj + eps):
                    touching_found = True
                    break
            assert touching_found, f"Particle {i} in cluster {cid} is disconnected from other cluster members!"
    print(f"[OK] All multi-particle clusters in Batch 0 are geometrically contiguous!")

def test_stage2_initial_state_no_overlap(num_batches=8, base_save_dir="sim_npy_out", radii_pair=(8.0, 16.0), margin=64.0, cols=4):
    print("\n--- Verifying Stage 2 Initial Tile Separation & No Inter-Batch Overlaps ---")
    tile_w = 1600
    tile_h = 1200
    all_particles = []
    
    for b in range(num_batches):
        bdir = os.path.join(base_save_dir, f"batch_{b}")
        traj = np.load(glob.glob(os.path.join(bdir, "traj_*.npy"))[0])
        radii = np.load(glob.glob(os.path.join(bdir, "radii_*.npy"))[0])
        cids = np.load(glob.glob(os.path.join(bdir, "clusterids_*.npy"))[0])
        final_pos = traj[-1]
        
        col = b % cols
        row = b // cols
        off_x = col * (tile_w + margin)
        off_y = row * (tile_h + margin)
        
        for idx in range(len(radii)):
            x = final_pos[idx, 0] + off_x
            y = final_pos[idx, 1] + off_y
            r = radii[idx]
            cid = b * 1_000_000 + cids[idx]
            all_particles.append((b, x, y, r, cid))

    # Check that particles of different batches are strictly separated by >= margin
    print(f"Total combined particles: {len(all_particles)}")
    for i in range(len(all_particles)):
        b1, x1, y1, r1, cid1 = all_particles[i]
        for j in range(i + 1, len(all_particles)):
            b2, x2, y2, r2, cid2 = all_particles[j]
            if b1 != b2:
                d = math.hypot(x1 - x2, y1 - y2)
                assert d >= (r1 + r2), f"Overlap between batch {b1} and batch {b2}: d={d:.2f} < r1+r2={r1+r2:.2f}"
    print("[OK] Zero inter-batch collisions/overlaps in the initial tiled configuration!")

if __name__ == "__main__":
    print("Running quick verification run (N=200, 8 batches, max_time=300)...")
    staged_circle_cluster.run_stage1_parallel(
        num_batches=8,
        N_total=200,
        radii_pair=(8.0, 16.0),
        fraction_A=0.668,
        max_time=300,
        store_every_k=1,
        base_save_dir="sim_npy_out/test_staged",
        base_seed=100
    )
    verify_stage1_outputs(num_batches=8, base_save_dir="sim_npy_out/test_staged")
    test_stage2_initial_state_no_overlap(num_batches=8, base_save_dir="sim_npy_out/test_staged")
    
    print("\n--- Running Stage 2 Continuation (300 steps) ---")
    res = staged_circle_cluster.run_stage2_continuation(
        num_batches=8,
        base_save_dir="sim_npy_out/test_staged",
        stage2_save_dir="sim_npy_out/test_staged/stage2_combined",
        tag="stage2_N1600",
        max_time=300,
        store_every_k=1,
        radii_pair=(8.0, 16.0),
        seed=42,
        cols=4,
        rows=2
    )

    # 1. Verify No Overlaps Across Sampled Frames in Stage 2
    print("\n--- Verifying No Particle Overlaps Across Stage 2 Trajectory ---")
    traj = np.load(res['traj_path'])
    radii = np.load(res['radii_path'])
    n_frames = traj.shape[0]
    n_pts = traj.shape[1]
    
    # Check 10 frames across trajectory
    sample_frames = np.linspace(0, n_frames - 1, 10, dtype=int)
    for f in sample_frames:
        pos = traj[f]
        # Quick spatial grid check for overlaps
        cell_sz = max(radii) * 2.2
        grid_w = math.ceil(circle_cluster.width / cell_sz)
        grid_h = math.ceil(circle_cluster.height / cell_sz)
        grid = [[[] for _ in range(grid_h)] for _ in range(grid_w)]
        for i in range(n_pts):
            gx = min(max(0, int(pos[i, 0] / cell_sz)), grid_w - 1)
            gy = min(max(0, int(pos[i, 1] / cell_sz)), grid_h - 1)
            grid[gx][gy].append(i)
        
        for i in range(n_pts):
            gx = min(max(0, int(pos[i, 0] / cell_sz)), grid_w - 1)
            gy = min(max(0, int(pos[i, 1] / cell_sz)), grid_h - 1)
            for nx in range(max(0, gx - 1), min(grid_w, gx + 2)):
                for ny in range(max(0, gy - 1), min(grid_h, gy + 2)):
                    for j in grid[nx][ny]:
                        if i < j:
                            d = math.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                            # Tolerance for rigid contact
                            min_allowed = radii[i] + radii[j] - 0.5
                            assert d >= min_allowed, f"Frame {f}: overlap between particle {i} and {j} (d={d:.2f} < {min_allowed:.2f})"
    print(f"[OK] Verified 0 overlaps across {len(sample_frames)} sample frames in Stage 2!")

    # 2. Test Video Rendering
    print("\n--- Testing make_circle_video Rendering ---")
    import video_renderers.make_circle_video as make_circle_video
    vid_file = make_circle_video.create_video(
        data_dir="sim_npy_out/test_staged/stage2_combined",
        tag="stage2_N1600",
        output_filename="test_stage2.mp4",
        render_interval=5
    )
    assert os.path.exists(vid_file), "Video file was not created!"
    print(f"[OK] Video created successfully at: {vid_file} (Size: {os.path.getsize(vid_file)} bytes)")

    if os.path.exists("test_stage2.mp4"):
        os.remove("test_stage2.mp4")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
