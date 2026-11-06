# DLCA Simulation Suite — Final Project

A unified scientific laboratory and graphical suite for simulating **Diffusion-Limited Cluster Aggregation (DLCA)** across multiple 2D particle geometries, tracking live kinetics in real time, automatically generating publication-grade plots, and rendering high-resolution MP4 aggregation animations.

---

## 🌟 Features

- **Multi-Geometry Physics Engine**:
  - **Single Species**: Circles, Equilateral / Right / Isosceles / Scalene Triangles, Squares, Rectangles, Ellipses, Superellipses.
  - **Binary Mixtures**: Circle-Square, Circle-Triangle, Circle-Ellipse, Circle-Rectangle.
  - **Dispersions**: Monodisperse and Bidistributed sizes with customizable mixing fractions.
- **Interactive Web GUI (Streamlit)**:
  - Real-time aggregation monitoring (live particle count, cluster count, SPS speed, dynamic ETA).
  - Live Altair kinetic decay curves tracking cluster dissolution over time.
  - Interactive **⏹ Stop Simulation** button allowing instant halting with full intermediate data and plot recovery.
  - On-demand or automatic true-geometry MP4 animation rendering via `imageio-ffmpeg`.
  - Comprehensive run history browser with instant download buttons for high-resolution PNGs and MP4s.
- **Scientific Analysis & Data Export**:
  - Trajectories saved in compressed NumPy format (`traj_*.npy`, `orient_*.npy`, `clusterids_*.npy`, `radii_*.npy`).
  - Automated generation of publication-ready spatial structure snapshots and kinetic cluster decay curves.

---

## 📁 Repository Structure

```text
dlca_final_project/
├── app.py                            # Streamlit GUI frontend
├── dlca_gui_backend.py               # Unified backend engine & simulation coordinator
├── run_gui.bat                       # One-click Windows launcher (starts the GUI)
│
├── circle_cluster.py                 # Core engine: Circle Brownian aggregation
├── triangle_cluster.py               # Core engine: Triangle aggregation (various types)
├── triangle_same_size_cluster.py     # Core engine: Equal-sized triangle aggregation
├── square_cluster.py                 # Core engine: Square particle aggregation
├── rectangle_cluster.py              # Core engine: Rectangle particle aggregation
├── ellipse_cluster.py                # Core engine: Ellipse aggregation
├── superellipse_cluster.py           # Core engine: Superellipse aggregation
│
├── square_circle_cluster.py          # Mixture engine: Square-Circle aggregation
├── triangle_circle_cluster.py        # Mixture engine: Triangle-Circle aggregation
├── ellipse_circle_cluster.py         # Mixture engine: Ellipse-Circle aggregation
├── rectangle_circle_cluster.py       # Mixture engine: Rectangle-Circle aggregation
│
├── collisions.py                     # High-performance 2D collision detection & SAT algorithms
├── collisions2.py                    # Collision helper routines for rectangles
├── collisionsellipse.py              # Elliptical collision detection utilities
├── collisions_rect_circle.py         # Rectangle-Circle collision utilities
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Standard Git ignore file
└── README.md                         # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.9+** installed.

Clone or download this repository, open a terminal in this directory, and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the GUI
- **On Windows**: Simply double-click **`run_gui.bat`**.
- **Via Terminal**:
  ```bash
  python -m streamlit run app.py
  ```

Once launched, your default web browser will automatically open:
```text
http://localhost:8501
```

---

## ⚙️ Using the GUI

1. **Select Geometry & Parameters**: Use the left sidebar to choose particle shapes, dispersion mode, arena domain size, and initial packing density (by area fraction $\phi$ or particle count $N$).
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
