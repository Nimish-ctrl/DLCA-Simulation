import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon

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
DATA_DIR = os.path.join(SCRIPT_DIR, "sim_npy_out_triangle_single") 
WIDTH = 1600           
HEIGHT = 1200          
FPS = 30              

# Render Interval: dynamically computed for smooth ~12s video
RENDER_INTERVAL = 5

# ==========================================

def get_triangle_corners(cx, cy, r, angle):
    """
    Calculates the 3 corners of an equilateral triangle centered at (cx, cy)
    with circumradius 'r' and rotation 'angle' (radians).
    """
    base_angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
    corners = []
    for base_angle in base_angles:
        a = base_angle + angle
        corners.append([cx + r * math.cos(a), cy + r * math.sin(a)])
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
    orient_path = os.path.join(DATA_DIR, f"orient_{base_name}")
    meta_path = os.path.join(DATA_DIR, f"meta_{base_name.replace('.npy', '.json')}")
    
    print(f"Loading data from: {latest_traj}")
    traj_data = np.load(latest_traj)
    radii_data = np.load(radii_path)
    
    # Try to load orientation, default to zeros if missing
    if os.path.exists(orient_path):
        print(f"Loading orientation from: {orient_path}")
        orient_data = np.load(orient_path)
    else:
        print("WARNING: Orientation file not found. Triangles will not rotate.")
        orient_data = np.zeros((traj_data.shape[0], traj_data.shape[1]))
    
    sim_width = WIDTH
    sim_height = HEIGHT
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
                sim_width = meta.get("width", WIDTH)
                sim_height = meta.get("height", HEIGHT)
        except Exception:
            pass

    return traj_data, radii_data, orient_data, sim_width, sim_height

def create_video():
    # 1. Load Data
    try:
        full_traj, radii, full_orient, sim_width, sim_height = load_latest_simulation()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Non-uniform frame sampling: dense at start, sparse later
    #    First ~4s of video (120 frames) covers first 10% of simulation (early aggregation detail)
    #    Remaining ~8s (230 frames) covers the other 90% (bulk cluster merging)
    total_raw_frames = full_traj.shape[0]
    target_frames = 350
    early_video_frames = 120          # ~4 seconds at 30 FPS for early detail
    late_video_frames = target_frames - early_video_frames  # ~230 frames
    early_sim_cutoff = int(total_raw_frames * 0.10)  # first 10% of simulation
    
    if total_raw_frames <= target_frames:
        frame_indices = np.arange(total_raw_frames)
    else:
        early_indices = np.linspace(0, early_sim_cutoff - 1, early_video_frames, dtype=int)
        late_indices = np.linspace(early_sim_cutoff, total_raw_frames - 1, late_video_frames, dtype=int)
        frame_indices = np.concatenate([early_indices, late_indices])
        frame_indices = np.unique(frame_indices)  # remove any duplicates, keep sorted
    
    video_traj = full_traj[frame_indices]
    video_orient = full_orient[frame_indices]
    
    num_frames = video_traj.shape[0]
    num_particles = video_traj.shape[1]
    
    print(f"Original frames: {total_raw_frames}")
    print(f"Non-uniform sampling: {early_video_frames} frames for first 10% + {late_video_frames} frames for rest")
    print(f"Video frames to render: {num_frames} (approx {num_frames/FPS:.1f}s video at {FPS} FPS)")

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, sim_width)
    ax.set_ylim(0, sim_height)
    ax.set_aspect('equal')
    ax.set_title(f"DLCA Equilateral Triangle Simulation ({num_particles} particles)")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Initialize Particles (Polygons)
    patches = []
    for i in range(num_particles):
        r = radii[i]
        poly = Polygon([[0,0],[0,0],[0,0]], closed=True, color='cyan', alpha=0.85, ec='black', lw=0.6)
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
            new_corners = get_triangle_corners(cx, cy, r, angle)
            poly.set_xy(new_corners)
            
        return patches

    # 6. Generate and Save
    anim = FuncAnimation(
        fig, update, frames=num_frames, 
        interval=1000/FPS, blit=True
    )
    OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, f"dlca_triangles_same_size_{num_particles}.mp4")
    OUTPUT_FILENAME_ALT = os.path.join(SCRIPT_DIR, f"dlca_triangles_{num_particles}.mp4")

    print(f"Saving video to {OUTPUT_FILENAME} (this may take a moment)...")
    
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='SimUser'), bitrate=2000)
    
    try:
        anim.save(OUTPUT_FILENAME, writer=writer)
        print(f"\nSUCCESS: Video saved to {os.path.abspath(OUTPUT_FILENAME)}")
        import shutil
        shutil.copyfile(OUTPUT_FILENAME, OUTPUT_FILENAME_ALT)
        print(f"SUCCESS: Video also updated at {os.path.abspath(OUTPUT_FILENAME_ALT)}")
    except FileNotFoundError:
        print("\nERROR: FFmpeg not found.")
        print("Please run: 'pip install imageio-ffmpeg' to fix this automatically.")
    
    plt.close()

if __name__ == "__main__":
    create_video()
