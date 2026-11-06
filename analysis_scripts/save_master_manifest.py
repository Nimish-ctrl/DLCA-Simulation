import os, json, csv
import analysis_scripts.sweep_circle_area_fraction as sweep_circle_area_fraction

base_json = r"sweeps/sweep_circles_20261010_230233/manifest.json"
with open(base_json) as f:
    base_data = json.load(f)

all_runs = list(base_data['runs'])
ext_dir = r"sweeps/sweep_circles_20261011_021023"
checkpoints = base_data['checkpoints']

for d in ['N_7000', 'N_9000']:
    sub = os.path.join(ext_dir, d)
    r_data = sweep_circle_area_fraction._load_run_from_disk(sub, 3200, 2400, checkpoints)
    if r_data:
        all_runs.append(r_data)

all_runs.sort(key=lambda x: x['placed_particles'])

fieldnames = [
    'requested_particles', 'placed_particles', 'area_fraction',
    'final_clusters', 'fully_coalesced', 'final_step', 'elapsed_seconds', 'subfolder'
]
for cp in checkpoints:
    fieldnames.append(f'clusters_at_step_{cp}')

out_csv = os.path.join(ext_dir, 'manifest.csv')
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in all_runs:
        row = {
            'requested_particles': r['requested_particles'],
            'placed_particles': r['placed_particles'],
            'area_fraction': f"{r['area_fraction']:.6f}",
            'final_clusters': r['final_clusters'],
            'fully_coalesced': r['fully_coalesced'],
            'final_step': r['final_step'],
            'elapsed_seconds': f"{r['elapsed_seconds']:.2f}",
            'subfolder': r['subfolder']
        }
        for cp in checkpoints:
            row[f'clusters_at_step_{cp}'] = r['checkpoints'].get(str(cp), r['checkpoints'].get(cp, ''))
        writer.writerow(row)

out_json = os.path.join(ext_dir, 'manifest.json')
master_manifest = {
    'timestamp': '20261011_021023',
    'total_runs': len(all_runs),
    'domain_width': 3200,
    'domain_height': 2400,
    'radii_pair': [8.0, 16.0],
    'fraction_A': 0.668,
    'max_time': 200000,
    'store_every_k': 100,
    'checkpoints': checkpoints,
    'runs': all_runs
}
with open(out_json, 'w') as f:
    json.dump(master_manifest, f, indent=2)

print("Saved master manifest to", ext_dir)
