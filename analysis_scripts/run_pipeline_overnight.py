"""
run_pipeline_overnight.py
Automated pipeline that monitors N=5000 completion in sweep_circles_20261010_230233,
writes the base manifest, then runs N=7000, 9000, 11000, and produces the complete final dataset.
"""

import os
import sys
import time
import json
import csv
import numpy as np

import engines.circle_cluster as circle_cluster
import analysis_scripts.sweep_circle_area_fraction as sweep_circle_area_fraction

BASE_SWEEP_DIR = r"sweeps/sweep_circles_20261010_230233"
EXTENDED_COUNTS = [7000, 9000, 11000]
CHECKPOINTS = [500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 150000, 200000]
DOMAIN_WIDTH = 3200
DOMAIN_HEIGHT = 2400
MAX_TIME = 200000
STORE_EVERY_K = 100

def wait_for_n5000():
    """Waits until N_5000 subfolder has all output files."""
    n5000_dir = os.path.join(BASE_SWEEP_DIR, "N_5000")
    print("[Pipeline] Monitoring N_5000 for completion...")
    while True:
        if os.path.exists(n5000_dir):
            files = os.listdir(n5000_dir)
            has_cc = any(f.startswith("clustercount_") for f in files)
            has_meta = any(f.startswith("meta_") for f in files)
            has_radii = any(f.startswith("radii_") for f in files)
            has_traj = any(f.startswith("traj_") for f in files)
            if has_cc and has_meta and has_radii and has_traj:
                print("[Pipeline] N=5000 is COMPLETE!")
                time.sleep(5)  # give file system a moment to settle
                break
        time.sleep(15)

def ensure_base_manifest():
    """Ensures base manifest.json and manifest.csv exist in BASE_SWEEP_DIR."""
    base_json = os.path.join(BASE_SWEEP_DIR, "manifest.json")
    base_csv = os.path.join(BASE_SWEEP_DIR, "manifest.csv")
    
    # Load all 11 runs from disk
    runs = []
    base_counts = [100, 150, 200, 250, 300, 500, 1000, 2000, 3000, 4000, 5000]
    for n in base_counts:
        sub = os.path.join(BASE_SWEEP_DIR, f"N_{n}")
        run_data = sweep_circle_area_fraction._load_run_from_disk(sub, DOMAIN_WIDTH, DOMAIN_HEIGHT, CHECKPOINTS)
        if run_data is not None:
            runs.append(run_data)
        else:
            print(f"[Pipeline] Warning: N={n} data could not be loaded from {sub}")
            
    runs.sort(key=lambda x: x["placed_particles"])
    
    # Write CSV
    fieldnames = [
        "requested_particles", "placed_particles", "area_fraction",
        "final_clusters", "fully_coalesced", "final_step", "elapsed_seconds", "subfolder"
    ]
    for cp in CHECKPOINTS:
        fieldnames.append(f"clusters_at_step_{cp}")
        
    with open(base_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
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
            for cp in CHECKPOINTS:
                row[f"clusters_at_step_{cp}"] = r["checkpoints"].get(str(cp), r["checkpoints"].get(cp, ""))
            writer.writerow(row)
            
    manifest_data = {
        "timestamp": "20261010_230233",
        "total_runs": len(runs),
        "total_wall_clock_seconds": 0.0,
        "domain_width": DOMAIN_WIDTH,
        "domain_height": DOMAIN_HEIGHT,
        "radii_pair": [8.0, 16.0],
        "fraction_A": 0.668,
        "max_time": MAX_TIME,
        "store_every_k": STORE_EVERY_K,
        "checkpoints": CHECKPOINTS,
        "runs": runs
    }
    with open(base_json, "w") as f_json:
        json.dump(manifest_data, f_json, indent=2)
        
    print(f"[Pipeline] Base manifest verified with {len(runs)} completed runs.")
    return base_json

def run_extended_sweep(base_json_path):
    """Runs N=7000, 9000, 11000 and merges with base manifest."""
    print(f"[Pipeline] Starting extended sweep for N={EXTENDED_COUNTS} merging with {base_json_path}...")
    res = sweep_circle_area_fraction.run_area_fraction_sweep(
        particle_counts=EXTENDED_COUNTS,
        checkpoints=CHECKPOINTS,
        max_time=MAX_TIME,
        store_every_k=STORE_EVERY_K,
        num_workers=4,
        width=DOMAIN_WIDTH,
        height=DOMAIN_HEIGHT,
        radii_pair=(8.0, 16.0),
        fraction_A=0.668,
        output_base="sweeps",
        max_attempts=100000,
        merge_with=base_json_path
    )
    print("\n[Pipeline] ALL RUNS COMPLETED SUCCESSFULLY!")
    print(f"[Pipeline] Output directory: {res['sweep_dir']}")
    print(f"[Pipeline] Manifest CSV: {res['manifest_csv']}")
    print(f"[Pipeline] Manifest JSON: {res['manifest_json']}")

if __name__ == "__main__":
    wait_for_n5000()
    base_json = ensure_base_manifest()
    run_extended_sweep(base_json)
