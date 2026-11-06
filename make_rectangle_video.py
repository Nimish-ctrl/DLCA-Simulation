import os
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# --- FFMPEG AUTO-FIX ---
try:
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Found FFmpeg at: {plt.rcParams['animation.ffmpeg_path']}")
except ImportError:
    print("WARNING: 'imageio-ffmpeg' not found. If saving fails, run: pip install imageio-ffmpeg")
# -----------------------

# ---------------------------
# Configuration
# ---------------------------
save_dir = "sim_npy_out_rectangle" # Make sure this matches your actual data folder
video_dir = "videos"
width = 1600
height = 1200
fps = 30
render_interval = 5 # Speed up rendering like in your square script

def get_rect_corners(x, y, w, h, angle):
    """Calculates the 4 corners of a rotated rectangle."""
    c = math.cos(angle)
    s = math.sin(angle)
    
    dx1, dy1 = w/2 * c, w/2 * s
    dx2, dy2 = -h/2 * s, h/2 * c
    
    return [
        (x + dx1 + dx2, y + dy1 + dy2), # Top Right
        (x - dx1 + dx2, y - dy1 + dy2), # Top Left
        (x - dx1 - dx2, y - dy1 - dy2), # Bottom Left
        (x + dx1 - dx2, y + dy1 - dy2)  # Bottom Right
    ]

def main():
    os.makedirs(video_dir, exist_ok=True)

    # 1. Find the most recent meta.json file automatically
    meta_files = glob.glob(os.path.join(save_dir, "meta_comp1.000_N768_t10000.json"))
    if not meta_files:
        print(f"No meta.json files found in '{save_dir}'. Run a simulation first.")
        return

    latest_meta = max(meta_files, key=os.path.getctime)
    tag = os.path.basename(latest_meta).replace("meta_", "").replace(".json", "")
    
    print(f"Loading data for tag: {tag}")

    # 2. Load Metadata
    with open(latest_meta, "r") as f:
        meta = json.load(f)

    count_A = meta["count_A"]
    count_B = meta["count_B"]
    N_total = meta["N_total"]
    
    # Reconstruct exact dimensions for each particle index
    if "dims_pair" in meta:
        (wA, hA), (wB, hB) = meta["dims_pair"]
    elif "radii_pair" in meta:
        rA, rB = meta["radii_pair"]
        (wA, hA) = (rA * 2, rA * 2)
        (wB, hB) = (rB * 2, rB * 2)
        print("Note: Loaded an older square simulation. Rendering as squares.")
    else:
        print("Error: Could not find dimensions in meta.json!")
        return

    particle_dims = []
    for _ in range(count_A):
        particle_dims.append((wA, hA))
    for _ in range(count_B):
        particle_dims.append((wB, hB))

    # 3. Load Trajectories and Orientations
    traj_path = os.path.join(save_dir, f"traj_{tag}.npy")
    orient_path = os.path.join(save_dir, f"orient_{tag}.npy")
    
    full_traj = np.load(traj_path) # Shape: (frames, N_total, 2)
    full_orient = np.load(orient_path) # Shape: (frames, N_total)
    
    # Apply render interval
    trajectories = full_traj[::render_interval]
    orientations = full_orient[::render_interval]
    num_frames = trajectories.shape[0]

    print(f"Original frames: {full_traj.shape[0]}")
    print(f"Rendering every {render_interval}th frame.")
    print(f"Video frames to render: {num_frames}")

    # 4. Setup Matplotlib Figure
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"DLCA Simulation (Speedup: {render_interval}x)\n{count_A} Type A (Blue), {count_B} Type B (Orange)")
    
    # Create empty polygon patches for all particles
    polys = []
    for i in range(N_total):
        # Color Type A blue, Type B orange
        color = '#1f77b4' if i < count_A else '#ff7f0e'
        poly = patches.Polygon(
            np.zeros((4, 2)), 
            closed=True, 
            facecolor=color, 
            edgecolor='black', 
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(poly)
        polys.append(poly)

    # 5. Animation Update Function
    def init():
        return polys

    def update(frame):
        # Print progress every 50 frames
        if frame % 50 == 0:
            print(f"Rendering frame {frame}/{num_frames}...", end='\r')
            
        for i, poly in enumerate(polys):
            x, y = trajectories[frame, i]
            angle = orientations[frame, i]
            w, h = particle_dims[i]
            
            corners = get_rect_corners(x, y, w, h, angle)
            poly.set_xy(corners)
            
        return polys

    # 6. Generate and Save Animation
    print(f"Generating animation at {fps} FPS...")
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, 
        init_func=init, blit=True, interval=1000/fps
    )

    output_file = os.path.join(video_dir, f"video_{tag}.mp4")
    
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='DLCA_Sim'), bitrate=2000)
        ani.save(output_file, writer=writer)
        print(f"\nSUCCESS: Video saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"\nFailed to save MP4. Error: {e}")
        
        # Fallback to GIF
        gif_file = output_file.replace('.mp4', '.gif')
        print(f"Attempting to save as GIF instead to: {gif_file} ...")
        ani.save(gif_file, writer='pillow', fps=fps)
        print("GIF saved successfully.")

    plt.close()

if __name__ == "__main__":
    main()