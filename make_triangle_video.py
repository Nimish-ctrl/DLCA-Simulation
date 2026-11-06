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
DATA_DIR = os.path.join(SCRIPT_DIR, "sim_npy_out_triangle") 
WIDTH = 1600           
HEIGHT = 1200          
FPS = 30              

# ==========================================

def get_triangle_world_corners(cx, cy, angle, local_verts):
    """
    Transforms stored local vertices of a triangle by rotation angle
    and translation (cx, cy).
    """
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)
    corners = []
    for lx, ly in local_verts:
        wx = cx + cos_t * lx - sin_t * ly
        wy = cy + sin_t * lx + cos_t * ly
        corners.append([wx, wy])
    return np.array(corners)

def load_latest_simulation():
    # Find latest trajectory file
    search_path = os.path.join(DATA_DIR, "traj_*.npy")
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No trajectory files found in {DATA_DIR}")
    
    latest_traj = max(files, key=os.path.getmtime)
    base_name = os.path.basename(latest_traj).replace("traj_", "")
    
    # Construct paths
    orient_path = os.path.join(DATA_DIR, f"orient_{base_name}")
    verts_path = os.path.join(DATA_DIR, f"vertices_{base_name}")
    radii_path = os.path.join(DATA_DIR, f"radii_{base_name}")
    meta_path = os.path.join(DATA_DIR, f"meta_{base_name.replace('.npy', '.json')}")
    
    print(f"Loading data from: {latest_traj}")
    traj_data = np.load(latest_traj)
    num_particles = traj_data.shape[1]
    
    # Load orientation
    if os.path.exists(orient_path):
        orient_data = np.load(orient_path)
    else:
        print("WARNING: Orientation file not found. Defaulting to zeros.")
        orient_data = np.zeros((traj_data.shape[0], num_particles))
        
    # Load stored local vertices (shape: N, 3, 2)
    if os.path.exists(verts_path):
        local_verts = np.load(verts_path)
    else:
        print("WARNING: Vertices file not found. Deriving equilateral vertices from radii.")
        if os.path.exists(radii_path):
            radii_data = np.load(radii_path)
        else:
            radii_data = np.full(num_particles, 10.0)
        base_angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
        local_verts = np.zeros((num_particles, 3, 2), dtype=np.float32)
        for i, r in enumerate(radii_data):
            for k, a in enumerate(base_angles):
                local_verts[i, k, 0] = r * math.cos(a)
                local_verts[i, k, 1] = r * math.sin(a)
                
    # Read triangle type and domain dimensions from metadata if available
    triangle_type = "Triangle"
    sim_width = WIDTH
    sim_height = HEIGHT
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
                triangle_type = meta.get("triangle_type", "Triangle").capitalize()
                sim_width = meta.get("width", WIDTH)
                sim_height = meta.get("height", HEIGHT)
        except Exception:
            pass

    return traj_data, orient_data, local_verts, triangle_type, sim_width, sim_height

def create_video():
    # 1. Load Data
    try:
        full_traj, full_orient, local_verts, triangle_type, sim_width, sim_height = load_latest_simulation()
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
    ax.set_title(f"DLCA {triangle_type} Simulation ({num_particles} particles)")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Color Assignment
    # Determine distinct particle shapes by perimeter/size
    perimeters = np.array([
        math.hypot(v[1,0]-v[0,0], v[1,1]-v[0,1]) +
        math.hypot(v[2,0]-v[1,0], v[2,1]-v[1,1]) +
        math.hypot(v[0,0]-v[2,0], v[0,1]-v[2,1])
        for v in local_verts
    ])
    unique_perims = np.unique(np.round(perimeters, 2))
    colors = ['cyan', 'coral', 'mediumpurple', 'gold']
    perim_to_color = {p: colors[i % len(colors)] for i, p in enumerate(unique_perims)}
    
    patches_list = []
    for i in range(num_particles):
        p_key = round(perimeters[i], 2)
        c = perim_to_color.get(p_key, 'cyan')
        poly = Polygon([[0,0],[0,0],[0,0]], closed=True, facecolor=c, alpha=0.85, edgecolor='black', lw=0.6)
        patches_list.append(poly)
        ax.add_patch(poly)

    # 5. Animation Update
    def update(frame_idx):
        if frame_idx % 50 == 0:
            print(f"Rendering frame {frame_idx}/{num_frames}...", end='\r')
        
        current_positions = video_traj[frame_idx]
        current_angles = video_orient[frame_idx]
        
        for i, poly in enumerate(patches_list):
            cx, cy = current_positions[i]
            angle = current_angles[i]
            verts = local_verts[i]
            
            # Reconstruct world vertices by rotating and translating stored local vertices
            new_corners = get_triangle_world_corners(cx, cy, angle, verts)
            poly.set_xy(new_corners)
            
        return patches_list

    # 6. Generate and Save Video
    anim = FuncAnimation(
        fig, update, frames=num_frames, 
        interval=1000/FPS, blit=True
    )
    OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, f"dlca_{triangle_type.lower()}_{num_particles}.mp4")

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
