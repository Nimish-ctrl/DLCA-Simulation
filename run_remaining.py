import make_triangle_same_size_video
import make_triangle_video
import glob, os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_npy_out_triangle")

# 1. Re-render monodisperse
print("=== Re-rendering monodisperse video ===")
make_triangle_same_size_video.create_video()
print("Monodisperse DONE\n")

# 2. Re-render each generalized type by touching the latest file for each type
types = ['equilateral', 'right-angled', 'isosceles', 'scalene']
for t in types:
    pattern = os.path.join(DATA_DIR, f"traj_{t}_*.npy")
    files = glob.glob(pattern)
    if files:
        latest = max(files, key=os.path.getmtime)
        # Touch to make it the latest
        os.utime(latest, None)
        print(f"=== Re-rendering {t} video ===")
        make_triangle_video.create_video()
        print(f"{t} DONE\n")
    else:
        print(f"No data found for {t}, skipping\n")

print("ALL VIDEOS RE-RENDERED")
