"""
Animation script for binary DLCA system (rectangles + circles)
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle
import math

# --- FFMPEG AUTO-FIX ---
try:
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Found FFmpeg at: {plt.rcParams['animation.ffmpeg_path']}")
except ImportError:
    print("WARNING: 'imageio-ffmpeg' not found. pip install imageio-ffmpeg")
# -----------------------

DATA_DIR = "sim_npy_out_rect_circle"
WIDTH = 1600           
HEIGHT = 1200         
FPS = 30              
RENDER_INTERVAL = 5  
OUTPUT_FILENAME = "dlca_binary_rect_circle.mp4" 

# Colors for different particle types
RECT_COLOR = 'red'
CIRCLE_COLOR = 'blue'

def get_rect_corners_rotated(x, y, w, h, angle_rad):
    """Calculate corners of a rotated rectangle."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    
    dx1, dy1 = w/2 * c, w/2 * s
    dx2, dy2 = -h/2 * s, h/2 * c
    
    return [
        (x + dx1 + dx2, y + dy1 + dy2),  # Top Right
        (x - dx1 + dx2, y - dy1 + dy2),  # Top Left
        (x - dx1 - dx2, y - dy1 - dy2),  # Bottom Left
        (x + dx1 - dx2, y + dy1 - dy2)   # Bottom Right
    ]

def load_latest_simulation():
    search_path = os.path.join(DATA_DIR, "traj_*.npy")
    files = glob.glob(search_path)
    if not files: 
        raise FileNotFoundError(f"No trajectory files found in {DATA_DIR}")
    
    latest_traj = max(files, key=os.path.getmtime)
    base_name = os.path.basename(latest_traj).replace("traj_", "").replace(".npy", "")
    
    traj_data = np.load(latest_traj)
    shapes_data = np.load(os.path.join(DATA_DIR, f"shapes_{base_name}.npy"))
    orient_data = np.load(os.path.join(DATA_DIR, f"orient_{base_name}.npy"))
    
    print(f"Loaded simulation: {base_name}")
    print(f"Total frames: {traj_data.shape[0]}")
    print(f"Total particles: {traj_data.shape[1]}")
    print(f"Rectangles: {np.sum(shapes_data == 'rectangle')}")
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
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    ax.set_title("Binary DLCA: Rectangles (Red) + Circles (Blue)")
    ax.set_facecolor('#f0f0f0')
    
    patches = []
    patch_types = []
    
    # Load metadata to get particle dimensions
    meta_files = glob.glob(os.path.join(DATA_DIR, "meta_*.json"))
    if meta_files:
        import json
        with open(meta_files[-1], 'r') as f:
            meta = json.load(f)
            rect_w = meta.get('rect_w', 10.0)
            rect_h = meta.get('rect_h', 20.0)
            circle_radius = meta.get('circle_r', 8.0)
            print(f"Using dimensions from metadata:")
            print(f"  Rectangle: {rect_w} x {rect_h}")
            print(f"  Circle: {circle_radius}")
    else:
        rect_w = 10.0
        rect_h = 20.0
        circle_radius = 8.0
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
        else:  # rectangle
            # For rectangles, we'll use Polygon since Rectangle doesn't support rotation well
            from matplotlib.patches import Polygon
            poly = Polygon([(0,0), (0,0), (0,0), (0,0)], 
                         closed=True,
                         color=RECT_COLOR, alpha=0.8, 
                         ec='black', lw=0.5)
            patches.append(poly)
            patch_types.append(('rectangle', rect_w, rect_h))
        
        ax.add_patch(patches[-1])

    def update(frame):
        if frame % 50 == 0: 
            print(f"Rendering frame {frame}/{num_frames}...", end='\r')
        
        for i, (patch, ptype) in enumerate(zip(patches, patch_types)):
            if ptype == 'circle':
                # Update circle position
                patch.center = tuple(video_traj[frame, i])
            else:  # rectangle
                # Update rectangle position and rotation
                _, w, h = ptype
                x, y = video_traj[frame, i]
                angle = video_orient[frame, i]
                
                # Calculate rotated corners
                corners = get_rect_corners_rotated(x, y, w, h, angle)
                patch.set_xy(corners)
        
        return patches

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/FPS, blit=True)
    
    print(f"\nSaving animation to {OUTPUT_FILENAME}...")
    writer = FFMpegWriter(fps=FPS, bitrate=3000)
    anim.save(OUTPUT_FILENAME, writer=writer)
    print(f"✓ Successfully saved to {OUTPUT_FILENAME}")
    plt.close()

if __name__ == "__main__":
    create_video()
