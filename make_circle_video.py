"""
make_circle_video.py — Video Renderer for Circle DLCA Simulation
================================================================
Loads trajectory, radii, and metadata from `sim_npy_out/` and renders
a high-quality animation showing random Brownian walk, mini-cluster nucleation,
and large-scale cluster aggregation up to the final coalesced state.
"""

import os
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle

# --- FFMPEG AUTO-FIX ---
try:
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Found FFmpeg at: {plt.rcParams['animation.ffmpeg_path']}")
except ImportError:
    print("WARNING: 'imageio-ffmpeg' not found. If saving fails, run: pip install imageio-ffmpeg")
# -----------------------

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "sim_npy_out")
DEFAULT_WIDTH = 2400
DEFAULT_HEIGHT = 1800
FPS = 30
HOLD_FRAMES = 45

# Colors
COLOR_A = '#1f77b4'  # Royal Blue (Type A / Main)
COLOR_B = '#ff7f0e'  # Amber / Orange (Type B)
COLOR_ALT = '#2ca02c' # Green

# Multi-batch Palette (for staged data if present)
BATCH_COLORS = [
    '#E6194B', '#3CB44B', '#4363D8', '#F58231',
    '#911EB4', '#42D4F4', '#FFE119', '#F032E6',
    '#A9A9A9', '#800000', '#008080', '#000075'
]
# ==========================================

def load_simulation_data(data_dir=DATA_DIR, tag=None):
    """
    Loads latest trajectory, radii, clusterids, and metadata.
    """
    if tag is not None:
        traj_path = os.path.join(data_dir, f"traj_{tag}.npy")
        radii_path = os.path.join(data_dir, f"radii_{tag}.npy")
        cids_path = os.path.join(data_dir, f"clusterids_{tag}.npy")
        meta_path = os.path.join(data_dir, f"meta_{tag}.json")
    else:
        traj_path = ""

    if not os.path.exists(traj_path):
        # Look directly in data_dir, and if empty look in stage2_combined
        candidates = glob.glob(os.path.join(data_dir, "traj_*.npy"))
        if not candidates:
            sub = os.path.join(data_dir, "stage2_combined")
            if os.path.exists(sub):
                candidates = glob.glob(os.path.join(sub, "traj_*.npy"))
        if not candidates:
            raise FileNotFoundError(f"No trajectory files found in {data_dir}")

        traj_path = max(candidates, key=os.path.getmtime)
        dir_found = os.path.dirname(traj_path)
        base = os.path.basename(traj_path).replace("traj_", "").replace(".npy", "")
        radii_path = os.path.join(dir_found, f"radii_{base}.npy")
        cids_path = os.path.join(dir_found, f"clusterids_{base}.npy")
        meta_path = os.path.join(dir_found, f"meta_{base}.json")

    print(f"Loading trajectory: {traj_path}")
    traj_data = np.load(traj_path)
    radii_data = np.load(radii_path)
    
    if os.path.exists(cids_path):
        cids_data = np.load(cids_path)
    else:
        cids_data = np.zeros(traj_data.shape[1], dtype=np.int32)

    sim_width = DEFAULT_WIDTH
    sim_height = DEFAULT_HEIGHT
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                sim_width = meta.get("width", DEFAULT_WIDTH)
                sim_height = meta.get("height", DEFAULT_HEIGHT)
        except Exception as e:
            print(f"Note: Could not parse metadata: {e}")

    return traj_data, radii_data, cids_data, sim_width, sim_height

def create_video(data_dir=DATA_DIR, tag=None, output_filename=None, render_interval=None):
    # 1. Load Data
    try:
        full_traj, radii, cids, sim_width, sim_height = load_simulation_data(data_dir, tag)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    num_sim_frames = full_traj.shape[0]
    num_particles = full_traj.shape[1]

    # 2. Slow-motion frame sampling:
    # Maintain the consistent slow-mo pace of the initial phase (step = 3) throughout
    # the entire simulation for maximum clarity during cluster mergers
    if render_interval is not None and render_interval > 0:
        step = render_interval
    else:
        step = 3  # Matches the ~2.7-3.0 sim steps/frame of the initial slow-mo phase
        
    video_traj = full_traj[::step]

    # Append 45 static hold frames of final state
    hold_frames = np.repeat(video_traj[-1:], HOLD_FRAMES, axis=0)
    video_traj = np.concatenate([video_traj, hold_frames], axis=0)
    num_frames = video_traj.shape[0]

    print(f"Simulation frames: {num_sim_frames} -> Video frames: {num_frames} ({num_frames/FPS:.1f}s at {FPS} FPS)")
    print(f"Canvas size: {sim_width:.1f} x {sim_height:.1f} | Total particles: {num_particles}")

    # 3. Setup Plot
    aspect_ratio = sim_width / max(sim_height, 1)
    fig_w = 12
    fig_h = max(5, fig_w / aspect_ratio)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    ax.set_xlim(0, sim_width)
    ax.set_ylim(0, sim_height)
    ax.set_aspect('equal')
    ax.set_title(f"DLCA Circle Cluster Simulation (N = {num_particles})", fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.25)

    # 4. Color Assignment
    # Check if radii are bidisperse or monodisperse
    unique_radii = np.unique(np.round(radii, 2))
    is_bidisperse = len(unique_radii) > 1

    # Check if staged batch ids (cids >= 1_000_000)
    has_batch_ids = np.max(cids) >= 1_000_000

    patches = []
    for i in range(num_particles):
        r = float(radii[i])
        if has_batch_ids:
            b_idx = int(cids[i] // 1_000_000)
            color = BATCH_COLORS[b_idx % len(BATCH_COLORS)]
        elif is_bidisperse:
            if abs(r - unique_radii[0]) < 1e-3:
                color = COLOR_A
            else:
                color = COLOR_B
        else:
            color = COLOR_A

        circle = Circle((0, 0), r, facecolor=color, edgecolor='black', linewidth=0.35, alpha=0.88)
        patches.append(circle)
        ax.add_patch(circle)

    # Legend
    if has_batch_ids:
        batches_present = np.unique((cids // 1_000_000).astype(int))
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=BATCH_COLORS[b % len(BATCH_COLORS)],
                       markeredgecolor='black', markersize=8, label=f"Batch {b}")
            for b in batches_present
        ]
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.12, 1.0),
                  fontsize=8, framealpha=0.85, title="Origin Batch")
    elif is_bidisperse:
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_A,
                       markeredgecolor='black', markersize=8, label=f"Radius {unique_radii[0]:.1f} px"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_B,
                       markeredgecolor='black', markersize=8, label=f"Radius {unique_radii[1]:.1f} px")
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.85)
    else:
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_A,
                       markeredgecolor='black', markersize=8, label=f"Circles (r = {unique_radii[0]:.1f} px)")
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.85)

    fig.tight_layout()

    # 5. Animation Update Function
    def update(frame_idx):
        if frame_idx % 50 == 0:
            print(f"Rendering frame {frame_idx}/{num_frames}...", end='\r')
        current_positions = video_traj[frame_idx]
        for i, circle in enumerate(patches):
            circle.center = (current_positions[i, 0], current_positions[i, 1])
        return patches

    # 6. Generate and Save Video
    anim = FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000/FPS, blit=True
    )

    if output_filename is None:
        output_filename = os.path.join(SCRIPT_DIR, f"dlca_circle_{num_particles}.mp4")

    print(f"Saving video to {output_filename}...")
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='DLCA Circle Cluster'), bitrate=2500)

    try:
        anim.save(output_filename, writer=writer)
        print(f"\nSUCCESS: Video saved to {os.path.abspath(output_filename)}")
        # Also copy to standard alias
        alt_name = os.path.join(SCRIPT_DIR, "dlca_circle.mp4")
        if os.path.abspath(output_filename) != os.path.abspath(alt_name):
            import shutil
            shutil.copyfile(output_filename, alt_name)
            print(f"SUCCESS: Video also copied to {os.path.abspath(alt_name)}")
    except FileNotFoundError:
        print("\nERROR: FFmpeg not found.")
        print("Please run: 'pip install imageio-ffmpeg' to fix this automatically.")

    plt.close()
    return output_filename

if __name__ == "__main__":
    create_video()
