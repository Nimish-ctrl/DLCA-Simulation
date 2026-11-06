"""
Animation script for binary DLCA system (ellipses + circles)
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse, Circle

# --- FFMPEG AUTO-FIX ---
try:
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Found FFmpeg at: {plt.rcParams['animation.ffmpeg_path']}")
except ImportError:
    print("WARNING: 'imageio-ffmpeg' not found. pip install imageio-ffmpeg")
# -----------------------

DATA_DIR = "sim_npy_out_binary"  # Updated for binary system
WIDTH = 2000           
HEIGHT = 0.75*2000         
FPS = 30              
RENDER_INTERVAL = 5  


# Colors for different particle types
ELLIPSE_COLOR = 'orange'
CIRCLE_COLOR = 'blue'

def load_latest_simulation():
    search_path = os.path.join(DATA_DIR, "traj_*.npy")
    files = glob.glob(search_path)
    if not files: 
        raise FileNotFoundError(f"No trajectory files found in {DATA_DIR}")
    
    latest_traj = max(files, key=os.path.getmtime)
    base_name = os.path.basename(latest_traj).replace("traj_", "").replace(".npy", "")
    
    traj_data = np.load(latest_traj)
    shapes_data = np.load(os.path.join(DATA_DIR, f"shapes_{base_name}.npy"))  # New: particle types
    orient_data = np.load(os.path.join(DATA_DIR, f"orient_{base_name}.npy"))
    
    print(f"Loaded simulation: {base_name}")
    print(f"Total frames: {traj_data.shape[0]}")
    print(f"Total particles: {traj_data.shape[1]}")
    print(f"Ellipses: {np.sum(shapes_data == 'ellipse')}")
    print(f"Circles: {np.sum(shapes_data == 'circle')}")
    
    return traj_data, shapes_data, orient_data

def create_video():
    try:
        full_traj, shapes, full_orient = load_latest_simulation()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    video_traj = full_traj[::RENDER_INTERVAL]
    video_orient = full_orient[::RENDER_INTERVAL]
    num_frames, num_particles = video_traj.shape[:2]
    
    print(f"Creating video with {num_frames} frames...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    ax.set_title("Binary DLCA: Ellipses (Orange) + Circles (Blue)")
    ax.set_facecolor('#f0f0f0')
    
    patches = []
    patch_types = []  # Track which patches are ellipses vs circles
    
    # Load metadata to get particle dimensions
    meta_files = glob.glob(os.path.join(DATA_DIR, "meta_*.json"))
    if meta_files:
        import json
        with open(meta_files[-1], 'r') as f:
            meta = json.load(f)
            ellipse_minor = meta.get('ellipse_minor_r', 4.0)
            ellipse_major = meta.get('ellipse_major_r', 8.0)
            circle_radius = meta.get('circle_r', 6.0)
            print(f"Using dimensions from metadata:")
            print(f"  Ellipse: {ellipse_minor} x {ellipse_major}")
            print(f"  Circle: {circle_radius}")
    else:
        # Default values if metadata not found
        ellipse_minor = 4.0
        ellipse_major = 8.0
        circle_radius = 6.0
        print("Warning: No metadata found, using default dimensions")
    
    # Create patches based on particle type
    for i in range(num_particles):
        if shapes[i] == 'circle':
            # Create circle patch
            c = Circle((0, 0), radius=circle_radius, 
                      color=CIRCLE_COLOR, alpha=0.8, 
                      ec='black', lw=0.5)
            patches.append(c)
            patch_types.append('circle')
        else:  # ellipse
            # Create ellipse patch
            e = Ellipse((0, 0), 
                       width=ellipse_major*2, 
                       height=ellipse_minor*2, 
                       angle=0,
                       color=ELLIPSE_COLOR, alpha=0.8, 
                       ec='black', lw=0.5)
            patches.append(e)
            patch_types.append('ellipse')
        
        ax.add_patch(patches[-1])

    def update(frame):
        if frame % 50 == 0: 
            print(f"Rendering frame {frame}/{num_frames}...", end='\r')
        
        for i, (patch, ptype) in enumerate(zip(patches, patch_types)):
            # Update position for all particles
            patch.center = tuple(video_traj[frame, i])
            
            # Only update angle for ellipses (circles don't need rotation)
            if ptype == 'ellipse':
                patch.angle = np.degrees(video_orient[frame, i])
        
        return patches

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/FPS, blit=True)
    OUTPUT_FILENAME = f"dlca_binary_ellipse_circle_{num_particles}.mp4" 
    
    print(f"\nSaving animation to {OUTPUT_FILENAME}...")
    writer = FFMpegWriter(fps=FPS, bitrate=3000)
    anim.save(OUTPUT_FILENAME, writer=writer)
    print(f"✓ Successfully saved to {OUTPUT_FILENAME}")
    plt.close()

if __name__ == "__main__":
    create_video()