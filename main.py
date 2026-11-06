"""
DLCA Cluster Simulation Manager

How to use this program:
1. Run this script from your terminal or command prompt: `python main.py`
2. You will see a numbered menu listing all available simulation types.
3. Type the number corresponding to the simulation you want to run and press Enter.
4. The chosen simulation will open in a new GUI window where you can adjust parameters and click "Start".
5. Wait for the simulation to finish. Once it says "Done" in the status bar, close the simulation window.
6. The script will then ask if you want to generate a video from the generated data.
7. Type 'y' for yes or 'n' for no. If you choose 'y', the video generation script will run automatically.
8. You will be returned to the main menu to run another simulation, or you can type '0' to exit.
"""

import os
import sys
import subprocess

def main():
    simulations = {
        "1": ("Circle Cluster", "circle_cluster.py", "make_circle_video.py"),
        "2": ("Square Cluster", "square_cluster.py", "make_square_video.py"),
        "3": ("Rectangle Cluster", "rectangle_cluster.py", "make_rectangle_video.py"),
        "4": ("Ellipse Cluster", "ellipse_cluster.py", None),
        "5": ("Square-Circle Cluster", "square_circle_cluster.py", "make_square_circle_video.py"),
        "6": ("Rectangle-Circle Cluster", "rectangle_circle_cluster.py", "make_rectangle_circle_video.py"),
        "7": ("Ellipse-Circle Cluster", "ellipse_circle_cluster.py", "make_ellipse_circle_video.py"),
        "8": ("Triangle Cluster (Different Sizes / Bidisperse)", "triangle_cluster.py", "make_triangle_video.py"),
        "9": ("Triangle Cluster (Same Size / Monodisperse)", "triangle_same_size_cluster.py", "make_triangle_same_size_video.py"),
        "10": ("Superellipse Cluster", "superellipse_cluster.py", "make_superellipse_video.py"),
        "11": ("Triangle-Circle Cluster", "triangle_circle_cluster.py", "make_triangle_circle_video.py"),
    }

    # Ensure we are in the correct directory (where main.py is located)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    while True:
        print("\n" + "="*40)
        print("    DLCA Cluster Simulation Menu")
        print("="*40)
        for key, (name, sim_file, vid_file) in simulations.items():
            print(f" {key}. {name}")
        print(" 0. Exit")
        print("="*40)
        
        choice = input("\nEnter your choice (0-11): ").strip()
        
        if choice == "0":
            print("Exiting program.")
            break
            
        if choice in simulations:
            name, sim_file, vid_file = simulations[choice]
            
            if not os.path.exists(sim_file):
                print(f"\n[!] Error: {sim_file} not found in the current directory.")
                continue

            cmd = [sys.executable, sim_file]

            # Sub-menu for Generalized Triangle Cluster
            if choice == "8":
                print("\n  --- Select Triangle Geometry ---")
                print("  1. Equilateral Triangles")
                print("  2. Right-Angled Triangles")
                print("  3. Isosceles Triangles")
                print("  4. Scalene Triangles")
                print("  0. Back to Main Menu")
                t_choice = input("  Enter choice (1-4, or 0): ").strip()
                t_type_map = {
                    "1": "equilateral",
                    "2": "right-angled",
                    "3": "isosceles",
                    "4": "scalene"
                }
                if t_choice == "0":
                    continue
                elif t_choice in t_type_map:
                    cmd.extend(["--type", t_type_map[t_choice]])
                else:
                    print("  [i] Invalid selection; defaulting to Equilateral.")
                    cmd.extend(["--type", "equilateral"])

            print(f"\n[*] Running {name} simulation ({sim_file})...")
            print("    (Please interact with the simulation window. Close it when done.)")
            
            try:
                # Run the simulation file
                subprocess.run(cmd)
            except Exception as e:
                print(f"\n[!] Error running simulation: {e}")
                continue
                
            print(f"\n[*] Simulation window closed.")
            
            if vid_file:
                if os.path.exists(vid_file):
                    while True:
                        ans = input(f"\n[?] Do you want to generate a video for the generated data? (y/n): ").strip().lower()
                        if ans in ['y', 'yes', 'n', 'no']:
                            break
                        print("    Please answer 'y' or 'n'.")
                        
                    if ans in ['y', 'yes']:
                        print(f"\n[*] Running video generation script ({vid_file})...")
                        try:
                            subprocess.run([sys.executable, vid_file])
                            print(f"[*] Video generation completed.")
                        except Exception as e:
                            print(f"\n[!] Error running video script: {e}")
                    else:
                        print("    Skipping video generation.")
                else:
                    print(f"\n[i] Video script '{vid_file}' is registered but not found in the directory.")
            else:
                print("\n[i] No video generation script is available for this cluster type.")
                
        else:
            print("\n[!] Invalid choice. Please try again.")

if __name__ == "__main__":
    # Ensure standard output is not buffered improperly
    sys.stdout.reconfigure(line_buffering=True)
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting.")
        sys.exit(0)
