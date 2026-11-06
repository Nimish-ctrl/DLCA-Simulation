"""
sweep_circle_area_fraction.py — Area-Fraction Sweep Orchestrator for Circle DLCA
===============================================================================
Runs a non-interactive, multi-process parameter sweep over particle counts / area fractions,
logs fine-grained cluster counts, extracts step-count checkpoints, validates placed area fraction,
and writes a self-contained sweep manifest.
"""

import os
import sys
import time
import json
import csv
import math
import argparse
from datetime import datetime
import multiprocessing as mp
import numpy as np

import engines.circle_cluster as circle_cluster

# ==============================================================================
# Default Sweep Configuration
# ==============================================================================
DEFAULT_COUNTS = [100, 150, 200, 250, 300, 500, 1000, 2000, 3000, 4000, 5000]
DEFAULT_CHECKPOINTS = [500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]
DEFAULT_MAX_TIME = 200000
DEFAULT_STORE_EVERY_K = 100
DEFAULT_MAX_ATTEMPTS = 100000
DEFAULT_WORKERS = min(4, max(1, mp.cpu_count() - 1))
DEFAULT_WIDTH = 3200
DEFAULT_HEIGHT = 2400
DEFAULT_RADII = (8.0, 16.0)
DEFAULT_FRAC_A = 0.668
DEFAULT_OUTPUT_BASE = "sweeps"


def _worker_run_simulation(task_args):
    """
    Worker task executed in a separate process pool worker.
    Runs a single simulation for a given particle count, extracts checkpoint counts,
    and returns full structured results.
    """
    (
        n_req,
        run_save_dir,
        max_time,
        store_every_k,
        radii_pair,
        fraction_A,
        domain_width,
        domain_height,
        checkpoints,
        seed,
        max_attempts
    ) = task_args

    os.makedirs(run_save_dir, exist_ok=True)

    # Set domain geometry in worker process
    circle_cluster.width = domain_width
    circle_cluster.height = domain_height

    # Suppress worker stdout to keep sweep silent
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    t0 = time.time()
    try:
        res = circle_cluster.run_simulation_for_composition(
            N_total=n_req,
            radii_pair=radii_pair,
            fraction_A=fraction_A,
            max_time=max_time,
            store_every_k=store_every_k,
            save_dir=run_save_dir,
            seed=seed,
            max_attempts=max_attempts
        )
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

    elapsed = time.time() - t0

    # Load output data
    radii = np.load(res['radii_path'])
    actual_placed = len(radii)
    
    # Accurate Area Fraction Calculation: Sum(pi * r^2) / (width * height)
    total_particle_area = float(np.sum(np.pi * (radii.astype(np.float64) ** 2)))
    domain_area = float(domain_width * domain_height)
    area_fraction = float(total_particle_area / domain_area)

    # Load and process cluster count evolution
    clustercount_data = np.load(res['clustercount_path'])  # shape: (T, 2)
    final_clusters = int(res['num_clusters_final'])
    final_step = int(clustercount_data[-1, 0]) if len(clustercount_data) > 0 else 0
    fully_coalesced = bool(final_clusters == 1)

    # Checkpoint Extraction: lookup cluster count at closest step <= checkpoint
    # If run stopped early (e.g. coalesced to 1 cluster), value at/beyond final step is final_clusters (1)
    checkpoint_results = {}
    for cp in checkpoints:
        if len(clustercount_data) == 0:
            checkpoint_results[cp] = actual_placed
        else:
            valid_rows = clustercount_data[clustercount_data[:, 0] <= cp]
            if len(valid_rows) > 0:
                checkpoint_results[cp] = int(valid_rows[-1, 1])
            else:
                checkpoint_results[cp] = int(clustercount_data[0, 1])

    return {
        "requested_particles": int(n_req),
        "placed_particles": int(actual_placed),
        "area_fraction": area_fraction,
        "total_particle_area": total_particle_area,
        "domain_area": domain_area,
        "final_clusters": final_clusters,
        "fully_coalesced": fully_coalesced,
        "final_step": final_step,
        "elapsed_seconds": elapsed,
        "subfolder": os.path.abspath(run_save_dir),
        "traj_path": os.path.abspath(res['traj_path']),
        "radii_path": os.path.abspath(res['radii_path']),
        "clusterids_path": os.path.abspath(res['clusterids_path']),
        "clustercount_path": os.path.abspath(res['clustercount_path']),
        "meta_path": os.path.abspath(res['meta_path']),
        "checkpoints": checkpoint_results
    }


def _load_run_from_disk(subfolder, domain_width, domain_height, checkpoints):
    """Attempts to load a completed run from a subfolder on disk."""
    import glob
    meta_files = glob.glob(os.path.join(subfolder, "meta_*.json"))
    rad_files = glob.glob(os.path.join(subfolder, "radii_*.npy"))
    cc_files = glob.glob(os.path.join(subfolder, "clustercount_*.npy"))
    traj_files = glob.glob(os.path.join(subfolder, "traj_*.npy"))
    cid_files = glob.glob(os.path.join(subfolder, "clusterids_*.npy"))
    if not (meta_files and rad_files and cc_files and traj_files):
        return None

    try:
        with open(meta_files[0], "r") as f:
            meta = json.load(f)
        radii = np.load(rad_files[0])
        cc = np.load(cc_files[0])
    except Exception:
        return None

    if len(radii) == 0 or len(cc) == 0:
        return None

    actual_placed = len(radii)
    total_particle_area = float(np.sum(np.pi * (radii.astype(np.float64) ** 2)))
    domain_area = float(domain_width * domain_height)
    area_fraction = float(total_particle_area / domain_area)

    final_clusters = int(cc[-1, 1])
    final_step = int(cc[-1, 0])
    fully_coalesced = bool(final_clusters == 1)

    checkpoint_results = {}
    for cp in checkpoints:
        valid_rows = cc[cc[:, 0] <= cp]
        if len(valid_rows) > 0:
            checkpoint_results[cp] = int(valid_rows[-1, 1])
        else:
            checkpoint_results[cp] = int(cc[0, 1])

    return {
        "requested_particles": int(meta.get("requested_N_total", actual_placed)),
        "placed_particles": int(actual_placed),
        "area_fraction": area_fraction,
        "total_particle_area": total_particle_area,
        "domain_area": domain_area,
        "final_clusters": final_clusters,
        "fully_coalesced": fully_coalesced,
        "final_step": final_step,
        "elapsed_seconds": 0.0,
        "subfolder": os.path.abspath(subfolder),
        "traj_path": os.path.abspath(traj_files[0]),
        "radii_path": os.path.abspath(rad_files[0]),
        "clusterids_path": os.path.abspath(cid_files[0]) if cid_files else "",
        "clustercount_path": os.path.abspath(cc_files[0]),
        "meta_path": os.path.abspath(meta_files[0]),
        "checkpoints": checkpoint_results
    }


def run_area_fraction_sweep(
    particle_counts=None,
    checkpoints=None,
    max_time=DEFAULT_MAX_TIME,
    store_every_k=DEFAULT_STORE_EVERY_K,
    num_workers=DEFAULT_WORKERS,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    radii_pair=DEFAULT_RADII,
    fraction_A=DEFAULT_FRAC_A,
    output_base=DEFAULT_OUTPUT_BASE,
    base_seed=1000,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    merge_with=None,
    resume_dir=None
):
    """
    Executes the area fraction sweep concurrently using a multiprocessing worker pool,
    with optional merging of previous run rows from an existing manifest or resuming an incomplete sweep.
    """
    if particle_counts is None:
        particle_counts = list(DEFAULT_COUNTS)
    if checkpoints is None:
        checkpoints = list(DEFAULT_CHECKPOINTS)

    # Sort counts for clean progression
    particle_counts = sorted(list(particle_counts))
    checkpoints = sorted(list(checkpoints))

    # 1. Determine Sweep Directory (new or resume)
    if resume_dir and os.path.isdir(resume_dir):
        sweep_dir = os.path.abspath(resume_dir)
        timestamp_str = os.path.basename(sweep_dir).replace("sweep_circles_", "")
        print(f"[Sweep] Resuming sweep in existing directory: {sweep_dir}")
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir_name = f"sweep_circles_{timestamp_str}"
        sweep_dir = os.path.abspath(os.path.join(output_base, sweep_dir_name))
        os.makedirs(sweep_dir, exist_ok=True)

    # Check for already completed runs if resuming
    completed_results = []
    pending_counts = []
    for n_req in particle_counts:
        subfolder = os.path.join(sweep_dir, f"N_{n_req}")
        if os.path.isdir(subfolder):
            loaded = _load_run_from_disk(subfolder, width, height, checkpoints)
            if loaded is not None:
                print(f"[Sweep] Found completed run for N={n_req} on disk. Skipping re-computation.")
                completed_results.append(loaded)
                continue
        pending_counts.append(n_req)

    # 2. Build Task Argument List for Worker Pool
    tasks = []
    for idx, n_req in enumerate(pending_counts):
        run_subfolder = os.path.join(sweep_dir, f"N_{n_req}")
        seed = base_seed + idx
        task_args = (
            n_req,
            run_subfolder,
            max_time,
            store_every_k,
            radii_pair,
            fraction_A,
            width,
            height,
            checkpoints,
            seed,
            max_attempts
        )
        tasks.append(task_args)

    sweep_start_time = time.time()
    results_list = list(completed_results)

    # 3. Process Pool Execution (Dynamic Work Queue)
    if tasks:
        print(f"[Sweep] Launching {len(tasks)} pending run(s): {pending_counts} with {min(num_workers, len(tasks))} worker(s)...")
        effective_workers = min(num_workers, len(tasks))
        with mp.Pool(processes=effective_workers) as pool:
            # imap_unordered pulls tasks dynamically as workers become free
            newly_completed = list(pool.imap_unordered(_worker_run_simulation, tasks))
            results_list.extend(newly_completed)

    sweep_total_time = time.time() - sweep_start_time

    # 4. Handle Merging with Base Manifest if Requested
    if merge_with:
        if os.path.isdir(merge_with):
            base_json_path = os.path.join(merge_with, "manifest.json")
        else:
            base_json_path = merge_with
        
        if os.path.exists(base_json_path):
            with open(base_json_path, "r") as f_base:
                base_manifest = json.load(f_base)
            base_runs = base_manifest.get("runs", [])
            # Filter out particle counts that were re-run
            re_run_counts = set(r["requested_particles"] for r in results_list)
            kept_base_runs = [r for r in base_runs if r["requested_particles"] not in re_run_counts]
            # Combine kept base runs + newly run results
            results_list = kept_base_runs + results_list
        else:
            print(f"Warning: merge_with path '{base_json_path}' not found. Writing only current results.")

    # Sort results by placed particle count / area fraction
    results_list.sort(key=lambda x: x["placed_particles"])

    # 5. Generate Sweep Manifest (CSV & JSON)
    manifest_csv_path = os.path.join(sweep_dir, "manifest.csv")
    manifest_json_path = os.path.join(sweep_dir, "manifest.json")

    # CSV Header
    fieldnames = [
        "requested_particles",
        "placed_particles",
        "area_fraction",
        "final_clusters",
        "fully_coalesced",
        "final_step",
        "elapsed_seconds",
        "subfolder"
    ]
    for cp in checkpoints:
        fieldnames.append(f"clusters_at_step_{cp}")

    # Write CSV
    with open(manifest_csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_list:
            row = {
                "requested_particles": r["requested_particles"],
                "placed_particles": r["placed_particles"],
                "area_fraction": f"{r['area_fraction']:.6f}",
                "final_clusters": r["final_clusters"],
                "fully_coalesced": r["fully_coalesced"],
                "final_step": r["final_step"],
                "elapsed_seconds": f"{r['elapsed_seconds']:.2f}",
                "subfolder": r["subfolder"]
            }
            for cp in checkpoints:
                row[f"clusters_at_step_{cp}"] = r["checkpoints"].get(str(cp), r["checkpoints"].get(cp, ""))
            writer.writerow(row)

    # Write JSON Manifest
    manifest_data = {
        "timestamp": timestamp_str,
        "total_runs": len(results_list),
        "total_wall_clock_seconds": sweep_total_time,
        "domain_width": width,
        "domain_height": height,
        "radii_pair": list(radii_pair),
        "fraction_A": fraction_A,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "checkpoints": checkpoints,
        "runs": results_list
    }
    with open(manifest_json_path, "w") as f_json:
        json.dump(manifest_data, f_json, indent=2)

    # 6. Print Single Final Sweep Summary Table
    _print_final_summary(results_list, sweep_dir, manifest_csv_path, sweep_total_time, checkpoints, max_time)

    return {
        "sweep_dir": sweep_dir,
        "manifest_csv": manifest_csv_path,
        "manifest_json": manifest_json_path,
        "total_runs": len(results_list),
        "total_time_s": sweep_total_time,
        "results": results_list
    }


def _print_final_summary(results, sweep_dir, manifest_path, total_time, checkpoints, max_time):
    """Prints a single clean summary table after all runs finish."""
    print("\n" + "=" * 95)
    print("                         CIRCLE DLCA AREA-FRACTION SWEEP SUMMARY")
    print("=" * 95)
    print(f"  Total Runs Completed:     {len(results)}")
    print(f"  Total Wall-Clock Time:    {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"  Safety Step Cap:          {max_time}")
    print(f"  Output Directory:         {sweep_dir}")
    print(f"  Manifest File:            {manifest_path}")
    print("=" * 95)

    # Header
    hdr = f"{'Req N':>6} | {'Placed N':>8} | {'Area Frac':>10} | {'Final Clust':>11} | {'Coalesced':>9} | {'Final Step':>10} | {'Time (s)':>8}"
    # Add first 3 checkpoints to table if present
    sample_cps = checkpoints[:3]
    for cp in sample_cps:
        hdr += f" | {f'N@{cp}':>6}"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        coalesced_str = "YES" if r.get("fully_coalesced") else "NO"
        line = f"{r['requested_particles']:>6} | {r['placed_particles']:>8} | {r['area_fraction']:>10.5f} | {r['final_clusters']:>11} | {coalesced_str:>9} | {r['final_step']:>10} | {r['elapsed_seconds']:>8.2f}"
        for cp in sample_cps:
            val = r["checkpoints"].get(str(cp), r["checkpoints"].get(cp, "-"))
            line += f" | {val:>6}"
        print(line)

    print("=" * len(hdr))

    # Coalescence Analysis / Flags
    uncoalesced = [r for r in results if not r.get("fully_coalesced", False)]
    if uncoalesced:
        print(f"\n[!] Note: {len(uncoalesced)} of {len(results)} run(s) did NOT reach full coalescence within the safety cap ({max_time} steps):")
        for u in uncoalesced:
            print(f"    - N={u['requested_particles']:<5} (Area Frac = {u['area_fraction']:.5f}): reached step {u['final_step']} with {u['final_clusters']} clusters remaining.")
    else:
        print(f"\n[✓] All {len(results)} runs reached full coalescence (1 cluster) before hitting the safety cap.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circle DLCA Area-Fraction Parameter Sweep")
    parser.add_argument(
        "--counts", nargs="+", type=int, default=DEFAULT_COUNTS,
        help="List of particle counts representing area fractions"
    )
    parser.add_argument(
        "--checkpoints", nargs="+", type=int, default=DEFAULT_CHECKPOINTS,
        help="Timestep checkpoints to extract cluster counts at"
    )
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME, help="Maximum timesteps per run")
    parser.add_argument("--store-every-k", type=int, default=DEFAULT_STORE_EVERY_K, help="Trajectory save cadence")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="Max placement retry attempts")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of concurrent worker processes")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Simulation domain width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Simulation domain height")
    parser.add_argument("--rA", type=float, default=DEFAULT_RADII[0], help="Radius A")
    parser.add_argument("--rB", type=float, default=DEFAULT_RADII[1], help="Radius B")
    parser.add_argument("--fracA", type=float, default=DEFAULT_FRAC_A, help="Fraction A")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_BASE, help="Base directory for sweep outputs")
    parser.add_argument("--merge-with", type=str, default=None, help="Path to base manifest to merge with")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing sweep directory to resume")

    args = parser.parse_args()

    run_area_fraction_sweep(
        particle_counts=args.counts,
        checkpoints=args.checkpoints,
        max_time=args.max_time,
        store_every_k=args.store_every_k,
        num_workers=args.workers,
        width=args.width,
        height=args.height,
        radii_pair=(args.rA, args.rB),
        fraction_A=args.fracA,
        output_base=args.output_dir,
        max_attempts=args.max_attempts,
        merge_with=args.merge_with,
        resume_dir=args.resume
    )
