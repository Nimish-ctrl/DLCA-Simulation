"""
make_triangle_circle_video.py — Video Renderer for Mixed Triangle-Circle DLCA Simulation
========================================================================================
Loads mixed simulation trajectories, orientations, and shapes from `sim_npy_out_triangle_circle/`
and renders an animation distinguishing Circles (Blue) and Triangles (Red).
"""

import os
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Polygon

from collisions.collisions import (
    create_equilateral_triangle_vertices,
    get_triangle_corners
)

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
DATA_DIR = os.path.join(SCRIPT_DIR, "sim_npy_out_triangle_circle")
WIDTH = 1600
HEIGHT = 1200
FPS = 30
HOLD_FRAMES = 45

# Colors
COLOR_CIRCLE = '#1f77b4'    # Blue
COLOR_TRIANGLE = '#d62728'  # Red
# ==========================================

def get_triangle_world_corners(cx, cy, r, orientation):
    """Computes the 3 world-space corners of an equilateral triangle."""
    base_angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
    return [[cx + r * math.cos(a + orientation), cy + r * math.sin(a + orientation)] for a in base_angles]

def load_latest_simulation():
    search_path = os.path.join(DATA_DIR, "traj_*.npy")
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No trajectory files found in {DATA_DIR}")

    latest_traj = max(files, key=os.path.getmtime)
    base_name = os.path.basename(latest_traj).replace("traj_", "").replace(".npy", "")

    radii_path = os.path.join(DATA_DIR, f"radii_{base_name}.npy")
    orient_path = os.path.join(DATA_DIR, f"orient_{base_name}.npy")
    shapes_path = os.path.join(DATA_DIR, f"shapes_{base_name}.npy")
    meta_path = os.path.join(DATA_DIR, f"meta_{base_name}.json")

    print(f"Loading trajectory: {latest_traj}")
    traj_data = np.load(latest_traj)
    radii_data = np.load(radii_path)

    if os.path.exists(orient_path):
        orient_data = np.load(orient_path)
    else:
        print("WARNING: orient file not found. Triangles will have fixed orientation.")
        orient_data = np.zeros((traj_data.shape[0], traj_data.shape[1]), dtype=np.float32)

    count_A = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            count_A = meta.get("count_A")

    if os.path.exists(shapes_path):
        shapes_data = np.load(shapes_path)
    elif count_A is not None:
        shapes_data = np.array([0 if i < count_A else 1 for i in range(len(radii_data))], dtype=np.int32)
    else:
        shapes_data = np.array([0 if i < len(radii_data)//2 else 1 for i in range(len(radii_data))], dtype=np.int32)

    return traj_data, radii_data, orient_data, shapes_data, base_name

def create_video(output_filename=None):
    try:
        full_traj, radii, full_orient, shapes, base_name = load_latest_simulation()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    num_sim_frames = full_traj.shape[0]
    num_particles = full_traj.shape[1]

    # Non-uniform sampling: First 5 seconds (150 video frames at 30 FPS) dedicated to
    # the initial Brownian motion and mini-cluster formation
    EARLY_VIDEO_FRAMES = 150   # 5.0 seconds at 30 FPS
    LATE_VIDEO_FRAMES = 200    # Remaining aggregation phase
    EARLY_SIM_FRACTION = 0.06  # First 6% of simulation time

    if num_sim_frames <= (EARLY_VIDEO_FRAMES + LATE_VIDEO_FRAMES):
        sampled_indices = np.arange(num_sim_frames)
    else:
        split_idx = max(1, int(num_sim_frames * EARLY_SIM_FRACTION))
        early_indices = np.linspace(0, split_idx - 1, EARLY_VIDEO_FRAMES, dtype=int)
        late_indices = np.linspace(split_idx, num_sim_frames - 1, LATE_VIDEO_FRAMES, dtype=int)
        sampled_indices = np.unique(np.concatenate([early_indices, late_indices]))

    video_traj = full_traj[sampled_indices]
    video_orient = full_orient[sampled_indices]

    # Append static hold frames
    hold_traj = np.repeat(video_traj[-1:], HOLD_FRAMES, axis=0)
    hold_orient = np.repeat(video_orient[-1:], HOLD_FRAMES, axis=0)
    video_traj = np.concatenate([video_traj, hold_traj], axis=0)
    video_orient = np.concatenate([video_orient, hold_orient], axis=0)

    num_frames = video_traj.shape[0]
    print(f"Simulation frames: {num_sim_frames} -> Video frames: {num_frames} ({num_frames/FPS:.1f}s at {FPS} FPS)")
    print(f"Total particles: {num_particles} (Circles: {sum(shapes==0)}, Triangles: {sum(shapes==1)})")

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    ax.set_title(f"Mixed DLCA Simulation — Circles & Triangles (N={num_particles})", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.25)

    patches_list = []
    for i in range(num_particles):
        r = float(radii[i])
        if shapes[i] == 0:  # Circle
            circle = Circle((0, 0), r, facecolor=COLOR_CIRCLE, edgecolor='black', linewidth=0.5, alpha=0.85)
            patches_list.append(circle)
            ax.add_patch(circle)
        else:  # Triangle
            poly = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, facecolor=COLOR_TRIANGLE, edgecolor='black', linewidth=0.5, alpha=0.85)
            patches_list.append(poly)
            ax.add_patch(poly)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_CIRCLE, markeredgecolor='black', markersize=9, label='Circles'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLOR_TRIANGLE, markeredgecolor='black', markersize=9, label='Triangles')
    ]
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.85, fontsize=9)
    fig.tight_layout()

    def update(frame_idx):
        if frame_idx % 50 == 0:
            print(f"Rendering frame {frame_idx}/{num_frames}...", end='\r')
        pos = video_traj[frame_idx]
        orient = video_orient[frame_idx]

        for i, patch in enumerate(patches_list):
            cx, cy = pos[i, 0], pos[i, 1]
            r = radii[i]
            if shapes[i] == 0:
                patch.center = (cx, cy)
            else:
                corners = get_triangle_world_corners(cx, cy, r, orient[i])
                patch.set_xy(corners)
        return patches_list

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/FPS, blit=True)

    if output_filename is None:
        output_filename = os.path.join(SCRIPT_DIR, f"dlca_triangle_circle_{num_particles}.mp4")

    print(f"Saving video to {output_filename}...")
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='DLCA Mixed Triangle-Circle'), bitrate=2500)

    try:
        anim.save(output_filename, writer=writer)
        print(f"\nSUCCESS: Video saved to {os.path.abspath(output_filename)}")
        # Copy to general name as well
        gen_name = os.path.join(SCRIPT_DIR, "dlca_triangle_circle.mp4")
        if os.path.abspath(output_filename) != os.path.abspath(gen_name):
            import shutil
            shutil.copyfile(output_filename, gen_name)
    except FileNotFoundError:
        print("\nERROR: FFmpeg not found.")
        print("Please run: 'pip install imageio-ffmpeg' to fix this.")

    plt.close()
    return output_filename

if __name__ == "__main__":
    create_video()
