# DLCA Simulation Suite — Final Project

A unified scientific laboratory and graphical suite for simulating **Diffusion-Limited Cluster Aggregation (DLCA)** across multiple 2D particle geometries, tracking live kinetics in real time, automatically generating publication-grade plots, and rendering high-resolution MP4 aggregation animations.

---

## 📁 Repository Structure

The project is cleanly divided into specialized directories for easy navigation:

```text
dlca_final_project/
├── app.py                     # Streamlit GUI frontend (Interactive Dashboard)
├── dlca_gui_backend.py        # Unified backend controller & plotting engine
├── run_gui.bat                # One-click Windows desktop launcher
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Standard Git ignore file
├── README.md                  # Comprehensive documentation
│
├── engines/                   # Core Brownian aggregation simulation modules
│   ├── circle_cluster.py              # Circle aggregation engine
│   ├── triangle_cluster.py            # Triangle aggregation engine (Equilateral, Right, Isosceles, Scalene)
│   ├── triangle_same_size_cluster.py  # Equal-sized triangle aggregation
│   ├── square_cluster.py              # Square aggregation engine
│   ├── rectangle_cluster.py           # Rectangle aggregation engine
│   ├── ellipse_cluster.py             # Ellipse aggregation engine
│   ├── superellipse_cluster.py        # Superellipse aggregation engine
│   ├── square_circle_cluster.py       # Binary mixture: Square-Circle aggregation
│   ├── triangle_circle_cluster.py     # Binary mixture: Triangle-Circle aggregation
│   ├── ellipse_circle_cluster.py      # Binary mixture: Ellipse-Circle aggregation
│   ├── rectangle_circle_cluster.py    # Binary mixture: Rectangle-Circle aggregation
│   └── staged_circle_cluster.py       # Staged circle aggregation engine
│
├── collisions/                # 2D geometry collision detection & Separating Axis Theorem (SAT)
│   ├── collisions.py                  # Primary SAT polygon & circle collision algorithms
│   ├── collisions2.py                 # Specialized collision routines for rectangular systems
│   ├── collisionsellipse.py           # Elliptical collision utilities
│   └── collisions_rect_circle.py      # Rectangle-Circle contact & penetration detection
│
├── video_renderers/           # Dedicated true-geometry MP4 video generators
│   ├── make_circle_video.py
│   ├── make_triangle_video.py
│   ├── make_triangle_same_size_video.py
│   ├── make_triangle_circle_video.py
│   ├── make_square_video.py
│   ├── make_square_circle_video.py
│   ├── make_rectangle_video.py
│   ├── make_rectangle_circle_video.py
│   ├── make_ellipse_circle_video.py
│   └── make_superellipse_video.py
│
└── analysis_scripts/          # Batch runners, parameter sweeps & standalone plotting scripts
    ├── generate_analysis_plots.py     # Statistical plot generator
    ├── main.py                        # CLI test runner
    ├── render_circle_final_pngs.py    # Batch snapshot renderer
    ├── run_pipeline_overnight.py      # Long-run parameter pipeline
    ├── run_remaining.py               # Auxiliary batch processor
    ├── save_master_manifest.py        # Manifest database builder
    ├── sweep_circle_area_fraction.py  # Area fraction (phi) parameter sweep for circles
    ├── sweep_triangle_area_fraction.py# Area fraction parameter sweep for triangles
    └── verify_staged_circle.py        # Verification harness
```

---

## 🌟 Key Capabilities

- **12 Particle Geometries & Binary Mixtures**: Monodisperse and bidistributed circles, triangles, squares, rectangles, ellipses, and superellipses.
- **Interactive Web GUI (Streamlit)**:
  - Real-time aggregation monitoring (live particle count, active cluster count, steps/second, dynamic ETA).
  - Live Altair kinetic decay curves tracking cluster dissolution over time.
  - Interactive **⏹ Stop Simulation** button allowing instant halting with full intermediate data and plot recovery.
  - On-demand or automatic true-geometry MP4 animation rendering via `imageio-ffmpeg`.
  - Comprehensive run history browser with instant download buttons for high-resolution PNGs and MP4s.
- **Scientific Analysis & Data Export**:
  - Trajectories saved in compressed NumPy format (`traj_*.npy`, `orient_*.npy`, `clusterids_*.npy`, `radii_*.npy`).
  - Automated generation of publication-ready spatial structure snapshots and kinetic cluster decay curves.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.9+** installed.

Clone this repository, navigate into the directory, and install the dependencies:
```bash
git clone https://github.com/Nimish-ctrl/DLCA-Simulation.git
cd DLCA-Simulation
pip install -r requirements.txt
```

### 2. Launch the Application
- **On Windows**: Simply double-click **`run_gui.bat`**.
- **Via Terminal**:
  ```bash
  python -m streamlit run app.py
  ```

Your browser will automatically open to:
```text
http://localhost:8501
```

---

## ⚙️ How to Use the GUI

1. **Select Geometry & Parameters**: Use the left sidebar to choose particle shapes, dispersion mode, arena domain dimensions, and initial packing density (by area fraction $\phi$ or particle count $N$).
2. **Launch Simulation**: Click **`▶ Launch DLCA Simulation`**.
3. **Monitor Live Progress**: Watch the live step-by-step counter, cluster merge rate, speed, ETA, and real-time kinetic decay curve.
4. **Instant Halting (Stop/Cancel)**: Click **`⏹ Stop Simulation`** at any point. All intermediate particle coordinates, trajectories, and plots are safely preserved on disk.
5. **View & Export Results**:
   - Inspect the high-resolution final structure snapshot.
   - Download `.png` plots for papers or presentations.
   - Play and download MP4 video animations directly in the browser.
   - Access previous simulations under the **Run History** tab.

---

## 📦 Output Artifacts

All simulation runs are automatically cataloged inside the `gui_runs/` folder:
- `summary.json`: Detailed simulation metadata (runtime, particle counts, final clusters, parameters).
- `final_structure.png`: Publication-ready plot of final spatial agglomerate.
- `cluster_decay.png`: Kinetic decay curve showing active cluster count vs. timestep.
- `simulation_video.mp4`: True-geometry video showing particle Brownian motion and aggregation.
- `traj_*.npy`, `orient_*.npy`, `clusterids_*.npy`: Raw trajectory and orientation arrays for further post-processing.

---

## 📄 License
This project is open-source and available under the MIT License.
