import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Polygon

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
DATA_DIR = "sim_npy_out_mixed"  # Matches the save_dir from your simulation
WIDTH = 2000          
HEIGHT = 1500       
FPS = 30              

# Render Interval: 1 = all frames, 2 = every 2nd frame (2x speed), etc.
RENDER_INTERVAL = 5  

OUTPUT_FILENAME = "dlca_mixed_system.mp4" 
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
    # 1. Find latest trajectory file
    search_path = os.path.join(DATA_DIR, "traj_*.npy")
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No trajectory files found in {DATA_DIR}")
    
    latest_traj = max(files, key=os.path.getmtime)
    base_name = os.path.basename(latest_traj).replace("traj_", "").replace(".npy", "")
    
    # 2. Construct paths for other files
    radii_path = os.path.join(DATA_DIR, f"radii_{base_name}.npy")
    orient_path = os.path.join(DATA_DIR, f"orient_{base_name}.npy")
    meta_path = os.path.join(DATA_DIR, f"meta_{base_name}.json")
    
    print(f"Loading data from: {latest_traj}")
    traj_data = np.load(latest_traj)
    radii_data = np.load(radii_path)
    
    # 3. Load Orientation (default to 0 if missing)
    if os.path.exists(orient_path):
        orient_data = np.load(orient_path)
    else:
        print("WARNING: Orientation file not found. Squares will not rotate.")
        orient_data = np.zeros((traj_data.shape[0], traj_data.shape[1]))
        
    # 4. Load Metadata to distinguish Shapes
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            count_A = meta.get("count_A", 0)
            # fraction_A = meta.get("fraction_A", 0.0)
            print(f"Metadata loaded. Type A (Circles): {count_A}, Type B (Squares): {len(radii_data) - count_A}")
    else:
        # Fallback: Assume all circles if no meta found
        print("WARNING: Meta file not found. Assuming all particles are circles.")
        count_A = len(radii_data) 

    return traj_data, radii_data, orient_data, count_A

def create_video():
    # 1. Load Data
    try:
        full_traj, radii, full_orient, count_A = load_latest_simulation()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Slice Frames
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
    ax.set_title(f"Mixed System DLCA (Speedup: {RENDER_INTERVAL}x)")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Initialize Particles
    # We use a list to store the patch objects so we can update them in the loop
    patches = []
    
    # Color settings
    color_circle = 'blue'
    color_square = 'red'

    for i in range(num_particles):
        r = radii[i]
        
        # Determine Shape based on index and count_A
        if i < count_A:
            # --- CIRCLE (Type A) ---
            # Center at (0,0) initially, will be moved by update()
            circle = Circle((0, 0), r, color=color_circle, alpha=0.8, ec='black', lw=0.5)
            patches.append(circle)
            ax.add_patch(circle)
        else:
            # --- SQUARE (Type B) ---
            # Placeholder polygon
            poly = Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, color=color_square, alpha=0.8, ec='black', lw=0.5)
            patches.append(poly)
            ax.add_patch(poly)

    # 5. Animation Update
    def update(frame_idx):
        if frame_idx % 50 == 0:
            print(f"Rendering frame {frame_idx}/{num_frames}...", end='\r')
        
        current_positions = video_traj[frame_idx]
        current_angles = video_orient[frame_idx]
        
        for i, patch in enumerate(patches):
            x, y = current_positions[i]
            r = radii[i]
            
            if isinstance(patch, Circle):
                # Update Circle
                patch.set_center((x, y))
            else:
                # Update Square (Polygon)
                angle = current_angles[i]
                new_corners = get_corners(x, y, r, angle)
                patch.set_xy(new_corners)
            
        return patches

    # 6. Generate and Save
    anim = FuncAnimation(
        fig, update, frames=num_frames, 
        interval=1000/FPS, blit=True
    )

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