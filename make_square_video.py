import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon  # <--- CHANGED from Circle to Polygon

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
DATA_DIR = "sim_npy_out_square" 
WIDTH = 1600           
HEIGHT = 1200          
FPS = 30              

# Render Interval: 1 = all frames, 5 = every 5th frame
RENDER_INTERVAL = 5

 
# ==========================================

def get_corners(cx, cy, r, angle):
    """
    Calculates the 4 corners of a square centered at (cx, cy) 
    with half-width 'r' and rotation 'angle' (radians).
    """
    c = np.cos(angle)
    s = np.sin(angle)
    # Local offsets for corners: (r,r), (-r,r), (-r,-r), (r,-r)
    corners = []
    for dx, dy in [(r, r), (-r, r), (-r, -r), (r, -r)]:
        # Rotate
        rx = dx * c - dy * s
        ry = dx * s + dy * c
        # Translate
        corners.append([cx + rx, cy + ry])
    return np.array(corners)

def load_latest_simulation():
    # Find latest trajectory file
    search_path = os.path.join(DATA_DIR, "traj_*.npy")
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No trajectory files found in {DATA_DIR}")
    
    latest_traj = max(files, key=os.path.getmtime)
    base_name = os.path.basename(latest_traj).replace("traj_", "")
    
    # Construct paths for radii and orientation
    radii_path = os.path.join(DATA_DIR, f"radii_{base_name}")
    orient_path = os.path.join(DATA_DIR, f"orient_{base_name}") # <--- NEW
    
    print(f"Loading data from: {latest_traj}")
    traj_data = np.load(latest_traj)
    radii_data = np.load(radii_path)
    
    # Try to load orientation, default to zeros if missing
    if os.path.exists(orient_path):
        print(f"Loading orientation from: {orient_path}")
        orient_data = np.load(orient_path)
    else:
        print("WARNING: Orientation file not found. Squares will not rotate.")
        orient_data = np.zeros((traj_data.shape[0], traj_data.shape[1]))
    
    return traj_data, radii_data, orient_data

def create_video():
    # 1. Load Data
    try:
        full_traj, radii, full_orient = load_latest_simulation()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Slice Frames (apply render interval)
    video_traj = full_traj[::RENDER_INTERVAL]
    video_orient = full_orient[::RENDER_INTERVAL]
    
    num_frames = video_traj.shape[0]
    num_particles = video_traj.shape[1]
    
    print(f"Original frames: {full_traj.shape[0]}")
    print(f"Rendering every {RENDER_INTERVAL}th frame.")
    print(f"Video frames to render: {num_frames}")

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal')
    ax.set_title(f"DLCA Square Simulation (Speedup: {RENDER_INTERVAL}x)")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Initialize Particles (Polygons)
    unique_radii = np.unique(radii)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_radii)))
    radius_to_color = {r: colors[i] for i, r in enumerate(unique_radii)}
    
    patches = []
    for i in range(num_particles):
        r = radii[i]
        c = radius_to_color.get(r, 'blue')
        # Create a placeholder polygon. We will update vertices in the loop.
        # Initial corners don't matter much, just needs to be valid.
        poly = Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, color=c, alpha=0.8, ec='black', lw=0.5)
        patches.append(poly)
        ax.add_patch(poly)

    # 5. Animation Update
    def update(frame_idx):
        if frame_idx % 50 == 0:
            print(f"Rendering frame {frame_idx}/{num_frames}...", end='\r')
        
        current_positions = video_traj[frame_idx]
        current_angles = video_orient[frame_idx]
        
        for i, poly in enumerate(patches):
            cx, cy = current_positions[i]
            angle = current_angles[i]
            r = radii[i]
            
            # Calculate new corners based on pos, radius, and angle
            new_corners = get_corners(cx, cy, r, angle)
            poly.set_xy(new_corners)
            
        return patches

    # 6. Generate and Save
    anim = FuncAnimation(
        fig, update, frames=num_frames, 
        interval=1000/FPS, blit=True
    )
    OUTPUT_FILENAME = f"dlca_squares_{num_particles}.mp4"

    print(f"Saving video to {OUTPUT_FILENAME} (this may take a moment)...")
    
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='SimUser'), bitrate=2000)
    
    try:
        anim.save(OUTPUT_FILENAME, writer=writer)
        print(f"\nSUCCESS: Video saved to {os.path.abspath(OUTPUT_FILENAME)}")
    except FileNotFoundError:
        print("\nERROR: FFmpeg not found.")
        print("Please run: 'pip install imageio-ffmpeg' to fix this automatically.")
    
    plt.close()

if __name__ == "__main__":
    create_video()