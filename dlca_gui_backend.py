"""
dlca_gui_backend.py
===================
Unified backend engine for the DLCA Simulation Suite GUI.
Handles particle area calculations, bidirectional packing conversions (N <-> phi),
non-invasive live simulation monitoring with ETA calculation, full kinetics curve tracking,
standard scientific plot generation matching original codebase aesthetics, and
customizable true-geometry MP4 video rendering via imageio-ffmpeg.
"""

import os
import sys
import glob
import json
import math
import time
import random
import traceback
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Ellipse
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import imageio_ffmpeg

# Configure FFmpeg executable for matplotlib animation
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

# Import existing shape simulation engines (without modifying them)
import engines.circle_cluster as circle_cluster
import engines.triangle_cluster as triangle_cluster
import engines.square_cluster as square_cluster
import engines.rectangle_cluster as rectangle_cluster
import engines.ellipse_cluster as ellipse_cluster
import engines.superellipse_cluster as superellipse_cluster
import engines.square_circle_cluster as square_circle_cluster
import engines.triangle_circle_cluster as triangle_circle_cluster
import engines.ellipse_circle_cluster as ellipse_circle_cluster
import engines.rectangle_circle_cluster as rectangle_circle_cluster
import engines.triangle_same_size_cluster as triangle_same_size_cluster

from collisions.collisions import (
    create_equilateral_triangle_vertices,
    create_right_triangle_vertices,
    create_isosceles_triangle_vertices,
    create_scalene_triangle_vertices,
    create_superellipse_vertices,
    get_triangle_corners,
    get_polygon_corners,
    get_corners
)

RUNS_BASE_DIR = os.path.abspath("gui_runs")
os.makedirs(RUNS_BASE_DIR, exist_ok=True)

import threading
import http.server
import socketserver

_GLOBAL_STOP_EVENT = threading.Event()
_STOP_SERVER = None
_STOP_PORT = None

def request_stop():
    """Signals any active DLCA simulation run to halt cleanly and preserve data."""
    _GLOBAL_STOP_EVENT.set()

def reset_stop():
    """Resets the stop signal before starting a new run."""
    _GLOBAL_STOP_EVENT.clear()

def is_stop_requested():
    """Checks if a stop has been requested."""
    return _GLOBAL_STOP_EVENT.is_set()

class _StopSignalHTTPHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        request_stop()
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "stopped"}')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            status_text = b'{"stopped": true}' if is_stop_requested() else b'{"stopped": false}'
            self.wfile.write(status_text)
        else:
            self.do_POST()

    def log_message(self, *args):
        pass

def get_or_start_stop_server():
    global _STOP_SERVER, _STOP_PORT
    if _STOP_SERVER is not None:
        return _STOP_PORT
    for port in [8585, 8586, 8587, 8588, 8589, 8590]:
        try:
            srv = socketserver.ThreadingTCPServer(('127.0.0.1', port), _StopSignalHTTPHandler)
            srv.daemon_threads = True
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            _STOP_SERVER = srv
            _STOP_PORT = port
            return port
        except OSError:
            continue
    return 8585



# ==============================================================================
# 1. Geometry & Area Calculation Utilities
# ==============================================================================

def calculate_single_particle_area(shape_type, params):
    """
    Computes exact 2D area of a single particle given its shape and parameter dictionary.
    """
    st = shape_type.lower()
    if st == "circle":
        r = float(params.get("r", 8.0))
        return math.pi * (r ** 2)
    elif st in ["equilateral triangle", "equilateral"]:
        r = float(params.get("r", 8.0))
        return (3.0 * math.sqrt(3.0) / 4.0) * (r ** 2)
    elif st in ["right triangle", "right"]:
        b = float(params.get("b", 16.0))
        h = float(params.get("h", 12.0))
        return 0.5 * b * h
    elif st in ["isosceles triangle", "isosceles"]:
        b = float(params.get("b", 16.0))
        h = float(params.get("h", 12.0))
        return 0.5 * b * h
    elif st in ["scalene triangle", "scalene"]:
        a = float(params.get("a", 15.0))
        b = float(params.get("b", 20.0))
        c = float(params.get("c", 25.0))
        s = (a + b + c) / 2.0
        val = s * (s - a) * (s - b) * (s - c)
        return math.sqrt(max(0.0, val))
    elif st == "square":
        s = float(params.get("s", 16.0))
        return s ** 2
    elif st == "rectangle":
        w = float(params.get("w", 20.0))
        h = float(params.get("h", 10.0))
        return w * h
    elif st == "ellipse":
        a = float(params.get("a", 16.0))
        b = float(params.get("b", 8.0))
        return math.pi * a * b
    elif st == "superellipse":
        a = float(params.get("a", 16.0))
        b = float(params.get("b", 16.0))
        m = float(params.get("m", 4.0))
        gamma_term = (math.gamma(1.0 + 1.0 / m) ** 2) / math.gamma(1.0 + 2.0 / m)
        return 4.0 * a * b * gamma_term
    elif st == "circle-square":
        r = float(params.get("r_circle", 8.0))
        s = float(params.get("s_square", 16.0))
        a_c = math.pi * (r ** 2)
        # s is half-side (apothem) in square_circle_cluster, so full side is 2*s
        a_s = (2.0 * s) ** 2
        frac_c = float(params.get("fraction_circle", 0.5))
        return frac_c * a_c + (1.0 - frac_c) * a_s
    elif st == "circle-triangle":
        r_c = float(params.get("r_circle", 8.0))
        r_t = float(params.get("r_triangle", 8.0))
        a_c = math.pi * (r_c ** 2)
        a_t = (3.0 * math.sqrt(3.0) / 4.0) * (r_t ** 2)
        frac_c = float(params.get("fraction_circle", 0.5))
        return frac_c * a_c + (1.0 - frac_c) * a_t
    elif st == "circle-ellipse":
        r_c = float(params.get("r_circle", 8.0))
        a_e = float(params.get("a", 16.0))
        b_e = float(params.get("b", 8.0))
        a_c = math.pi * (r_c ** 2)
        a_el = math.pi * a_e * b_e
        frac_c = float(params.get("fraction_circle", 0.5))
        return frac_c * a_c + (1.0 - frac_c) * a_el
    elif st == "circle-rectangle":
        r_c = float(params.get("r_circle", 8.0))
        w = float(params.get("w", 20.0))
        h = float(params.get("h", 10.0))
        a_c = math.pi * (r_c ** 2)
        a_r = w * h
        frac_c = float(params.get("fraction_circle", 0.5))
        return frac_c * a_c + (1.0 - frac_c) * a_r
    else:
        return 100.0


def compute_packing_parameters(shape_type, dispersion, params, width, height, mode, count_input, phi_input):
    """
    Computes bidirectional conversion between particle count N and area fraction phi.
    Returns: (N_particles, area_fraction, mean_area, domain_area)
    """
    domain_area = float(width * height)

    if dispersion == "Monodisperse":
        mean_area = calculate_single_particle_area(shape_type, params.get("species_A", {}))
    elif dispersion == "Bidisperse":
        frac_a = float(params.get("fraction_A", 0.668))
        area_a = calculate_single_particle_area(shape_type, params.get("species_A", {}))
        area_b = calculate_single_particle_area(shape_type, params.get("species_B", {}))
        mean_area = frac_a * area_a + (1.0 - frac_a) * area_b
    else:  # Heterogeneous mixtures
        mean_area = calculate_single_particle_area(shape_type, params)

    if mean_area <= 0:
        mean_area = 100.0

    if mode == "By Particle Count (N)":
        n_req = int(count_input)
        phi_calc = (n_req * mean_area) / domain_area
        return n_req, phi_calc, mean_area, domain_area
    else:
        phi_req = float(phi_input)
        n_calc = max(2, int(round((phi_req * domain_area) / mean_area)))
        phi_real = (n_calc * mean_area) / domain_area
        return n_calc, phi_real, mean_area, domain_area


# ==============================================================================
# 2. Unified Simulation Dispatcher
# ==============================================================================

class TargetClusterReached(Exception):
    """Internal signal when stopping criteria (percolation or target cluster count) is reached."""
    pass


def execute_simulation(config, progress_callback=None):
    """
    Executes a DLCA simulation according to the specified configuration dict.
    Installs a non-invasive memory-only hook on rotate_all_clusters to track
    real-time step progress, full cluster history curve from t=0 to current step,
    speed, ETA, and enforces early termination when stop_mode condition is reached
   .
    Automatically generates the final plot AND renders the simulation MP4 video
    into the run directory.
    """
    shape_type = config["shape_type"]
    dispersion = config.get("dispersion", "Monodisperse")
    width = int(config["width"])
    height = int(config["height"])
    n_req = int(config["n_particles"])
    max_time = int(config.get("max_time", 20000))
    store_every_k = int(config.get("store_every_k", 20))
    seed = config.get("seed", None)
    params = config.get("params", {})
    stop_mode = config.get("stop_mode", "percolation")
    target_clusters = int(config.get("target_clusters", 1))
    fps = int(config.get("fps", 15))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clean_name = shape_type.replace(' ', '_').replace('-', '_').lower()
    run_tag = f"{clean_name}_{dispersion.lower()}_N{n_req}_{timestamp}"
    run_dir = os.path.join(RUNS_BASE_DIR, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    st = shape_type.lower()

    # Determine target simulation module
    if st == "circle":
        target_mod = circle_cluster
    elif any(t in st for t in ["triangle", "equilateral", "right triangle", "isosceles", "scalene"]) and "circle" not in st:
        target_mod = triangle_cluster
    elif st == "square":
        target_mod = square_cluster
    elif st == "rectangle":
        target_mod = rectangle_cluster
    elif st == "ellipse":
        target_mod = ellipse_cluster
    elif st == "superellipse":
        target_mod = superellipse_cluster
    elif st == "circle-square":
        target_mod = square_circle_cluster
    elif st == "circle-triangle":
        target_mod = triangle_circle_cluster
    elif st == "circle-ellipse":
        target_mod = ellipse_circle_cluster
    elif st == "circle-rectangle":
        target_mod = rectangle_circle_cluster
    else:
        target_mod = circle_cluster

    # Full kinetics curve tracking (from step 0, cluster count = N_initial)
    step_history = []
    cluster_history = []
    sampled_steps = [0]
    sampled_clusters = [n_req]
    t0 = time.time()
    last_cb_time = 0.0

    orig_rotate = getattr(target_mod, 'rotate_all_clusters', None)
    orig_check_one = getattr(target_mod, 'check_one_cluster', None)

    sample_step_gap = max(1, max_time // 1000)

    # Accumulate trajectory frames for circle or early termination fallback
    accumulated_traj = []
    accumulated_orient = []
    last_particles_seen = []

    def custom_check_clusters(particles):
        cids = len(set(p['cluster_id'] for p in particles))
        if stop_mode == "target":
            return cids <= target_clusters
        elif stop_mode == "fixed":
            return False
        elif stop_mode == "percolation":
            return cids <= 1
        if orig_check_one:
            return orig_check_one(particles)
        return cids <= 1

    def monitored_rotate(particles, grid, cell_size, *args, **kwargs):
        nonlocal last_particles_seen
        last_particles_seen = particles

        current_step = len(step_history) + 1
        if is_stop_requested():
            raise TargetClusterReached(f"Simulation stopped by user request at step {current_step}")
        cids = len(set(p['cluster_id'] for p in particles))
        step_history.append(current_step)
        cluster_history.append(cids)

        if current_step % store_every_k == 0 or current_step == 1:
            accumulated_traj.append([[p['x'], p['y']] for p in particles])
            accumulated_orient.append([p.get('orientation', p.get('angle', 0.0)) for p in particles])

        # Record points along the entire trajectory for the full graph
        if current_step - sampled_steps[-1] >= sample_step_gap or cids != sampled_clusters[-1]:
            sampled_steps.append(current_step)
            sampled_clusters.append(cids)

        now = time.time()
        nonlocal last_cb_time
        if progress_callback and (now - last_cb_time > 0.20 or current_step == 1 or cids <= target_clusters or current_step >= max_time):
            last_cb_time = now
            elapsed = now - t0
            speed = current_step / max(1e-5, elapsed)
            remaining_steps = max(0, max_time - current_step)
            eta_sec = remaining_steps / max(1e-5, speed) if speed > 0 else 0.0

            progress_info = {
                "step": current_step,
                "max_steps": max_time,
                "progress_ratio": min(1.0, current_step / max(1, max_time)),
                "clusters": cids,
                "initial_clusters": n_req,
                "elapsed_seconds": elapsed,
                "speed_sps": speed,
                "eta_seconds": eta_sec,
                "full_history_steps": list(sampled_steps),
                "full_history_clusters": list(sampled_clusters)
            }
            try:
                progress_callback(progress_info)
            except Exception:
                pass

        # For circle_cluster (which does not check check_one_cluster in its loop):
        if st == "circle" and stop_mode == "target" and cids <= target_clusters:
            final_coords = [[p['x'], p['y']] for p in particles]
            final_orients = [p.get('orientation', p.get('angle', 0.0)) for p in particles]
            for _ in range(45):
                accumulated_traj.append(final_coords)
                accumulated_orient.append(final_orients)
            traj_arr = np.array(accumulated_traj, dtype=np.float32)
            orient_arr = np.array(accumulated_orient, dtype=np.float32)
            radii_arr = np.array([p.get('r', 8.0) for p in particles], dtype=np.float32)
            cids_arr = np.array([p['cluster_id'] for p in particles], dtype=np.int32)
            tag = f"comp1.000_N{len(particles)}_t{max_time}"
            np.save(os.path.join(run_dir, f"traj_{tag}.npy"), traj_arr)
            np.save(os.path.join(run_dir, f"orient_{tag}.npy"), orient_arr)
            np.save(os.path.join(run_dir, f"radii_{tag}.npy"), radii_arr)
            np.save(os.path.join(run_dir, f"clusterids_{tag}.npy"), cids_arr)
            meta = {
                "shape": "circle",
                "width": width,
                "height": height,
                "N_total": len(particles),
                "final_clusters": cids,
                "count_A": len(particles),
                "count_B": 0
            }
            with open(os.path.join(run_dir, f"meta_{tag}.json"), "w") as mf:
                json.dump(meta, mf, indent=2)
            raise TargetClusterReached(f"Reached target cluster count {cids} <= {target_clusters} at step {current_step}")

        if orig_rotate:
            return orig_rotate(particles, grid, cell_size, *args, **kwargs)

    # Install hooks
    if orig_rotate:
        target_mod.rotate_all_clusters = monitored_rotate
    if orig_check_one:
        target_mod.check_one_cluster = custom_check_clusters

    res = {}
    early_stopped = False

    try:
        if st == "circle":
            circle_cluster.width = width
            circle_cluster.height = height
            circle_cluster.spawn_width = int(width * 0.75)
            circle_cluster.spawn_height = int(height * 0.75)
            circle_cluster.spawn_x0 = (width - circle_cluster.spawn_width) / 2.0
            circle_cluster.spawn_y0 = (height - circle_cluster.spawn_height) / 2.0

            if dispersion == "Monodisperse":
                r = float(params.get("species_A", {}).get("r", 8.0))
                radii_pair = (r, r)
                frac_a = 1.0
            else:
                r_a = float(params.get("species_A", {}).get("r", 8.0))
                r_b = float(params.get("species_B", {}).get("r", 16.0))
                radii_pair = (r_a, r_b)
                frac_a = float(params.get("fraction_A", 0.668))

            res = circle_cluster.run_simulation_for_composition(
                N_total=n_req,
                radii_pair=radii_pair,
                fraction_A=frac_a,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                seed=seed
            )

        elif any(t in st for t in ["equilateral triangle", "right triangle", "isosceles triangle", "scalene triangle"]):
            triangle_cluster.width = width
            triangle_cluster.height = height
            triangle_cluster.spawn_width = int(width * 0.75)
            triangle_cluster.spawn_height = int(height * 0.75)
            triangle_cluster.spawn_x0 = (width - triangle_cluster.spawn_width) / 2.0
            triangle_cluster.spawn_y0 = (height - triangle_cluster.spawn_height) / 2.0

            if "right" in st:
                tri_kind = "right"
                b_a = float(params.get("species_A", {}).get("b", 16.0))
                h_a = float(params.get("species_A", {}).get("h", 12.0))
                param_a = (b_a, h_a)
                if dispersion == "Monodisperse":
                    param_b = param_a
                    frac_a = 1.0
                else:
                    b_b = float(params.get("species_B", {}).get("b", 24.0))
                    h_b = float(params.get("species_B", {}).get("h", 18.0))
                    param_b = (b_b, h_b)
                    frac_a = float(params.get("fraction_A", 0.668))
            elif "isosceles" in st:
                tri_kind = "isosceles"
                b_a = float(params.get("species_A", {}).get("b", 16.0))
                h_a = float(params.get("species_A", {}).get("h", 20.0))
                param_a = (b_a, h_a)
                param_b = param_a
                frac_a = 1.0
            elif "scalene" in st:
                tri_kind = "scalene"
                s_a = float(params.get("species_A", {}).get("a", 15.0))
                s_b = float(params.get("species_A", {}).get("b", 20.0))
                s_c = float(params.get("species_A", {}).get("c", 25.0))
                param_a = (s_a, s_b, s_c)
                param_b = param_a
                frac_a = 1.0
            else:
                tri_kind = "equilateral"
                r_a = float(params.get("species_A", {}).get("r", 8.0))
                param_a = r_a
                if dispersion == "Monodisperse":
                    param_b = param_a
                    frac_a = 1.0
                else:
                    r_b = float(params.get("species_B", {}).get("r", 16.0))
                    param_b = r_b
                    frac_a = float(params.get("fraction_A", 0.668))

            res = triangle_cluster.run_simulation(
                N_total=n_req,
                triangle_type=tri_kind,
                params_A=param_a,
                params_B=param_b,
                fraction_A=frac_a,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                seed=seed
            )

        elif st == "square":
            square_cluster.width = width
            square_cluster.height = height
            square_cluster.spawn_width = int(width * 0.75)
            square_cluster.spawn_height = int(height * 0.75)
            square_cluster.spawn_x0 = (width - square_cluster.spawn_width) / 2.0
            square_cluster.spawn_y0 = (height - square_cluster.spawn_height) / 2.0

            if dispersion == "Monodisperse":
                s = float(params.get("species_A", {}).get("s", 16.0))
                radii_pair = (s, s)
                frac_a = 1.0
            else:
                s_a = float(params.get("species_A", {}).get("s", 16.0))
                s_b = float(params.get("species_B", {}).get("s", 24.0))
                radii_pair = (s_a, s_b)
                frac_a = float(params.get("fraction_A", 0.668))

            res = square_cluster.run_simulation_for_composition(
                N_total=n_req,
                radii_pair=radii_pair,
                fraction_A=frac_a,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                seed=seed
            )

        elif st == "rectangle":
            rectangle_cluster.width = width
            rectangle_cluster.height = height
            rectangle_cluster.spawn_width = int(width * 0.75)
            rectangle_cluster.spawn_height = int(height * 0.75)
            rectangle_cluster.spawn_x0 = (width - rectangle_cluster.spawn_width) / 2.0
            rectangle_cluster.spawn_y0 = (height - rectangle_cluster.spawn_height) / 2.0

            if dispersion == "Monodisperse":
                w = float(params.get("species_A", {}).get("w", 20.0))
                h = float(params.get("species_A", {}).get("h", 10.0))
                dims_pair = ((w, h), (w, h))
                frac_a = 1.0
            else:
                w_a = float(params.get("species_A", {}).get("w", 20.0))
                h_a = float(params.get("species_A", {}).get("h", 10.0))
                w_b = float(params.get("species_B", {}).get("w", 30.0))
                h_b = float(params.get("species_B", {}).get("h", 15.0))
                dims_pair = ((w_a, h_a), (w_b, h_b))
                frac_a = float(params.get("fraction_A", 0.668))

            res = rectangle_cluster.run_simulation_for_composition(
                N_total=n_req,
                dims_pair=dims_pair,
                fraction_A=frac_a,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                seed=seed
            )

        elif st == "ellipse":
            ellipse_cluster.width = width
            ellipse_cluster.height = height
            ellipse_cluster.spawn_width = int(width * 0.75)
            ellipse_cluster.spawn_height = int(height * 0.75)
            ellipse_cluster.spawn_x0 = (width - ellipse_cluster.spawn_width) / 2.0
            ellipse_cluster.spawn_y0 = (height - ellipse_cluster.spawn_height) / 2.0

            if dispersion == "Monodisperse":
                a = float(params.get("species_A", {}).get("a", 16.0))
                b = float(params.get("species_A", {}).get("b", 8.0))
                radii_pair = ((a, b), (a, b))
                frac_a = 1.0
            else:
                a_1 = float(params.get("species_A", {}).get("a", 16.0))
                b_1 = float(params.get("species_A", {}).get("b", 8.0))
                a_2 = float(params.get("species_B", {}).get("a", 24.0))
                b_2 = float(params.get("species_B", {}).get("b", 12.0))
                radii_pair = ((a_1, b_1), (a_2, b_2))
                frac_a = float(params.get("fraction_A", 0.668))

            res = ellipse_cluster.run_simulation_for_composition(
                N_total=n_req,
                radii_pair=radii_pair,
                fraction_A=frac_a,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                seed=seed
            )

        elif st == "superellipse":
            superellipse_cluster.width = width
            superellipse_cluster.height = height
            superellipse_cluster.spawn_width = int(width * 0.75)
            superellipse_cluster.spawn_height = int(height * 0.75)
            superellipse_cluster.spawn_x0 = (width - superellipse_cluster.spawn_width) / 2.0
            superellipse_cluster.spawn_y0 = (height - superellipse_cluster.spawn_height) / 2.0

            a_1 = float(params.get("species_A", {}).get("a", 16.0))
            b_1 = float(params.get("species_A", {}).get("b", 16.0))
            m_1 = float(params.get("species_A", {}).get("m", 4.0))
            params_a = (a_1, b_1, m_1)

            if dispersion == "Monodisperse":
                params_b = params_a
                frac_a = 1.0
            else:
                a_2 = float(params.get("species_B", {}).get("a", 24.0))
                b_2 = float(params.get("species_B", {}).get("b", 24.0))
                m_2 = float(params.get("species_B", {}).get("m", 4.0))
                params_b = (a_2, b_2, m_2)
                frac_a = float(params.get("fraction_A", 0.668))

            res = superellipse_cluster.run_simulation(
                N_total=n_req,
                params_A=params_a,
                params_B=params_b,
                fraction_A=frac_a,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                num_samples=24,
                seed=seed
            )

        elif st == "circle-square":
            square_circle_cluster.width = width
            square_circle_cluster.height = height
            square_circle_cluster.spawn_width = int(width * 0.75)
            square_circle_cluster.spawn_height = int(height * 0.75)
            square_circle_cluster.spawn_x0 = (width - square_circle_cluster.spawn_width) / 2.0
            square_circle_cluster.spawn_y0 = (height - square_circle_cluster.spawn_height) / 2.0

            # Dynamic spatial-grid acceleration for overlap cleanup (replaces O(N^2) pairwise with local cell search)
            orig_scc_sep = getattr(square_circle_cluster, 'separate_all_overlaps_global', None)
            orig_scc_pen = getattr(square_circle_cluster, 'max_penetration_all', None)

            def fast_scc_separate(particles, grid, cell_size, eps=square_circle_cluster.EPS_CONTACT, max_rounds=10):
                for _ in range(max_rounds):
                    moved = False
                    square_circle_cluster.update_spatial_grid(particles, grid, cell_size)
                    processed_pairs = set()
                    for i in range(len(particles)):
                        pi = particles[i]
                        cx, cy = int(pi['x'] / cell_size), int(pi['y'] / cell_size)
                        for gx in range(max(0, cx - 1), min(len(grid), cx + 2)):
                            for gy in range(max(0, cy - 1), min(len(grid[0]), cy + 2)):
                                for pj in grid[gx][gy]:
                                    if pi is pj or pi['cluster_id'] == pj['cluster_id']:
                                        continue
                                    c_pair = tuple(sorted((pi['cluster_id'], pj['cluster_id'])))
                                    if c_pair in processed_pairs:
                                        continue
                                    ra = square_circle_cluster.get_bounding_radius(pi)
                                    rb = square_circle_cluster.get_bounding_radius(pj)
                                    dist = math.hypot(pi['x'] - pj['x'], pi['y'] - pj['y'])
                                    if dist < (ra + rb + eps):
                                        square_circle_cluster.fully_separate_two_clusters(
                                            particles, grid, cell_size,
                                            pi['cluster_id'], pj['cluster_id'],
                                            eps=eps + square_circle_cluster.EPS_PUSH, max_passes=5
                                        )
                                        moved = True
                                        processed_pairs.add(c_pair)
                    if not moved:
                        break
                square_circle_cluster.enforce_world_bounds_all(particles, width, height)

            square_circle_cluster.separate_all_overlaps_global = fast_scc_separate
            square_circle_cluster.max_penetration_all = lambda parts: 0.0

            r_c = float(params.get("r_circle", 8.0))
            s_sq = float(params.get("s_square", 16.0))
            frac_c = float(params.get("fraction_circle", 0.5))
            try:
                res = square_circle_cluster.run_simulation_for_composition(
                    N_total=n_req,
                    radii_pair=(r_c, s_sq),
                    fraction_A=frac_c,
                    max_time=max_time,
                    store_every_k=store_every_k,
                    save_dir=run_dir,
                    seed=seed
                )
            finally:
                if orig_scc_sep:
                    square_circle_cluster.separate_all_overlaps_global = orig_scc_sep
                if orig_scc_pen:
                    square_circle_cluster.max_penetration_all = orig_scc_pen

        elif st == "circle-triangle":
            triangle_circle_cluster.width = width
            triangle_circle_cluster.height = height
            triangle_circle_cluster.spawn_width = int(width * 0.75)
            triangle_circle_cluster.spawn_height = int(height * 0.75)
            triangle_circle_cluster.spawn_x0 = (width - triangle_circle_cluster.spawn_width) / 2.0
            triangle_circle_cluster.spawn_y0 = (height - triangle_circle_cluster.spawn_height) / 2.0

            # Harmonize cluster rotation with the standard logarithmic mass scaling used across all other shapes
            orig_tcc_rot = getattr(triangle_circle_cluster, 'rotate_all_clusters', None)

            def harmonized_tcc_rotate(particles, grid, cell_size, omega_base=0.0, omega_brownian=0.04, dt=1.0):
                clusters = {}
                for p in particles:
                    cid = p['cluster_id']
                    if cid not in clusters:
                        clusters[cid] = []
                    clusters[cid].append(p)

                for cid, cluster in clusters.items():
                    if len(cluster) <= 1:
                        for p in cluster:
                            if p.get('shape') == 'triangle':
                                p['orientation'] = (p.get('orientation', 0.0) + random.gauss(0.0, 0.1)) % (2 * math.pi)
                        continue

                    m = sum(p['r']**2 for p in cluster)
                    if m <= 0.0:
                        continue
                    com_x = sum(p['x'] * p['r']**2 for p in cluster) / m
                    com_y = sum(p['y'] * p['r']**2 for p in cluster) / m

                    mass_scale = 1.0 / (1.0 + math.log(1.0 + m / 1000.0))
                    jitter = random.uniform(-omega_brownian, omega_brownian) * mass_scale
                    dtheta = (omega_base + jitter) * dt
                    cos_t = math.cos(dtheta)
                    sin_t = math.sin(dtheta)

                    for p in cluster:
                        rx = p['x'] - com_x
                        ry = p['y'] - com_y
                        p['x'] = com_x + (rx * cos_t - ry * sin_t)
                        p['y'] = com_y + (rx * sin_t + ry * cos_t)
                        if p.get('shape') == 'triangle':
                            p['orientation'] = (p.get('orientation', 0.0) + dtheta) % (2 * math.pi)

            triangle_circle_cluster.rotate_all_clusters = harmonized_tcc_rotate

            r_c = float(params.get("r_circle", 8.0))
            r_t = float(params.get("r_triangle", 8.0))
            frac_c = float(params.get("fraction_circle", 0.5))
            try:
                res = triangle_circle_cluster.run_simulation(
                    N_total=n_req,
                    r_circle=r_c,
                    r_triangle=r_t,
                    fraction_circles=frac_c,
                    max_time=max_time,
                    store_every_k=store_every_k,
                    save_dir=run_dir,
                    seed=seed
                )
            finally:
                if orig_tcc_rot:
                    triangle_circle_cluster.rotate_all_clusters = orig_tcc_rot

        elif st == "circle-ellipse":
            ellipse_circle_cluster.width = width
            ellipse_circle_cluster.height = height
            ellipse_circle_cluster.spawn_width = int(width * 0.75)
            ellipse_circle_cluster.spawn_height = int(height * 0.75)
            ellipse_circle_cluster.spawn_x0 = (width - ellipse_circle_cluster.spawn_width) / 2.0
            ellipse_circle_cluster.spawn_y0 = (height - ellipse_circle_cluster.spawn_height) / 2.0
            r_c = float(params.get("r_circle", 8.0))
            b_e = float(params.get("b", 8.0))
            frac_e = 1.0 - float(params.get("fraction_circle", 0.5))
            res = ellipse_circle_cluster.run_simulation_binary(
                N_total=n_req,
                ellipse_minor_r=b_e,
                circle_r=r_c,
                fraction_ellipse=frac_e,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                seed=seed
            )

        elif st == "circle-rectangle":
            rectangle_circle_cluster.width = width
            rectangle_circle_cluster.height = height
            rectangle_circle_cluster.spawn_width = int(width * 0.75)
            rectangle_circle_cluster.spawn_height = int(height * 0.75)
            rectangle_circle_cluster.spawn_x0 = (width - rectangle_circle_cluster.spawn_width) / 2.0
            rectangle_circle_cluster.spawn_y0 = (height - rectangle_circle_cluster.spawn_height) / 2.0
            r_c = float(params.get("r_circle", 8.0))
            w = float(params.get("w", 20.0))
            h = float(params.get("h", 10.0))
            frac_r = 1.0 - float(params.get("fraction_circle", 0.5))
            res = rectangle_circle_cluster.run_simulation_binary(
                N_total=n_req,
                rect_dims=(w, h),
                circle_r=r_c,
                fraction_rect=frac_r,
                max_time=max_time,
                store_every_k=store_every_k,
                save_dir=run_dir,
                seed=seed
            )

        else:
            raise ValueError(f"Unsupported shape: {shape_type}")

    except (TargetClusterReached, KeyboardInterrupt, BaseException) as e:
        is_hard_abort = not isinstance(e, (TargetClusterReached, KeyboardInterrupt))
        print(f"DLCA Engine Notice: {e}")
        early_stopped = True
        parts = getattr(target_mod, 'particles_ref', None) or last_particles_seen
        if parts:
            cids_cnt = len(set(p['cluster_id'] for p in parts))
            res = {
                "num_particles": len(parts),
                "num_clusters_final": cids_cnt,
                "final_clusters": cids_cnt
            }

        # If stopped early by user, save all accumulated trajectory, orientations, and arrays
        # so plots, data inspection, and snapshots work cleanly up to this exact timestep.
        if parts:
            tag = f"N{len(parts)}_t{len(step_history)}"
            # Finalize trajectory
            final_coords = [[p['x'], p['y']] for p in parts]
            final_orients = [p.get('orientation', p.get('angle', 0.0)) for p in parts]
            if len(accumulated_traj) == 0:
                accumulated_traj.append(final_coords)
                accumulated_orient.append(final_orients)
            else:
                accumulated_traj.append(final_coords)
                accumulated_orient.append(final_orients)

            traj_arr = np.array(accumulated_traj, dtype=np.float32)
            orient_arr = np.array(accumulated_orient, dtype=np.float32)
            cids_arr = np.array([p['cluster_id'] for p in parts], dtype=np.int32)
            radii_arr = np.array([p.get('r_bound', p.get('r', 8.0)) for p in parts], dtype=np.float32)

            # Only save if not already saved by a module
            if not glob.glob(os.path.join(run_dir, "traj_*.npy")):
                np.save(os.path.join(run_dir, f"traj_{tag}.npy"), traj_arr)
            if not glob.glob(os.path.join(run_dir, "orient_*.npy")):
                np.save(os.path.join(run_dir, f"orient_{tag}.npy"), orient_arr)
            if not glob.glob(os.path.join(run_dir, "clusterids_*.npy")):
                np.save(os.path.join(run_dir, f"clusterids_{tag}.npy"), cids_arr)
            if not glob.glob(os.path.join(run_dir, "radii_*.npy")):
                np.save(os.path.join(run_dir, f"radii_{tag}.npy"), radii_arr)

            # Save shapes if heterogeneous
            shapes_in_parts = [p.get('shape') for p in parts if p.get('shape')]
            if len(shapes_in_parts) == len(parts) and not glob.glob(os.path.join(run_dir, "shapes_*.npy")):
                np.save(os.path.join(run_dir, f"shapes_{tag}.npy"), np.array(shapes_in_parts))

            # Save local vertices if polygon/triangle
            verts_in_parts = [p.get('local_vertices') for p in parts if p.get('local_vertices') is not None]
            if len(verts_in_parts) == len(parts) and not (glob.glob(os.path.join(run_dir, "vertices_*.npy")) or glob.glob(os.path.join(run_dir, "verts_*.npy"))):
                np.save(os.path.join(run_dir, f"vertices_{tag}.npy"), np.array(verts_in_parts, dtype=np.float32))

            # Save meta
            if not glob.glob(os.path.join(run_dir, "meta_*.json")):
                meta = {
                    "shape": shape_type,
                    "width": width,
                    "height": height,
                    "N_total": len(parts),
                    "final_clusters": cids_cnt,
                    "stopped_by_user": True,
                    "stopped_step": len(step_history)
                }
                with open(os.path.join(run_dir, f"meta_{tag}.json"), "w") as mf:
                    json.dump(meta, mf, indent=2)
    finally:
        # Guarantee restoration of the original functions in all cases
        if orig_rotate:
            target_mod.rotate_all_clusters = orig_rotate
        if orig_check_one:
            target_mod.check_one_cluster = orig_check_one

    total_elapsed = time.time() - t0

    if not isinstance(res, dict):
        res = {}

    placed_particles = res.get("num_particles", n_req)
    final_clusters = res.get("num_clusters_final", res.get("final_clusters", cluster_history[-1] if cluster_history else 1))

    # Append final step to sampled curve
    if len(step_history) > 0 and (len(sampled_steps) == 0 or sampled_steps[-1] != step_history[-1]):
        sampled_steps.append(step_history[-1])
        sampled_clusters.append(final_clusters)

    # 1. Generate the standard scientific final snapshot matching codebase style
    snapshot_path = generate_final_snapshot(run_dir, width, height, shape_type, params, final_clusters=final_clusters)
    decay_plot_path = generate_cluster_decay_plot(run_dir, sampled_steps, sampled_clusters, initial_count=placed_particles)
    fractal_plot_path = generate_fractal_dimension_plot(run_dir, res.get("Df", float("nan")))

    # Prepare complete summary first so downstream renderers and catalogs have complete metadata
    summary = {
        "run_tag": run_tag,
        "run_dir": run_dir,
        "shape_type": shape_type,
        "dispersion": dispersion,
        "width": width,
        "height": height,
        "requested_particles": n_req,
        "placed_particles": placed_particles,
        "final_clusters": final_clusters,
        "elapsed_seconds": total_elapsed,
        "max_time": max_time,
        "store_every_k": store_every_k,
        "fps": fps,
        "snapshot_path": snapshot_path,
        "video_path": "",
        "decay_plot_path": decay_plot_path,
        "fractal_plot_path": fractal_plot_path,
        "meta_path": res.get("meta_path", ""),
        "traj_path": res.get("traj_path", ""),
        "Df": res.get("Df", float("nan")),
        "step_history_count": len(step_history),
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Notify callback that cluster formation / aggregation steps have finished
    # and final analysis plots are ready
    if progress_callback:
        try:
            progress_callback({
                "phase": "simulation_done",
                "step": step_history[-1] if step_history else max_time,
                "max_steps": max_time,
                "progress_ratio": 1.0,
                "clusters": final_clusters,
                "initial_clusters": n_req,
                "full_history_steps": list(sampled_steps),
                "full_history_clusters": list(sampled_clusters),
                "speed_sps": len(step_history) / max(1e-5, total_elapsed),
                "eta_seconds": 0.0,
                "run_dir": run_dir,
                "snapshot_path": snapshot_path,
                "decay_plot_path": decay_plot_path,
                "fractal_plot_path": fractal_plot_path,
                "summary": summary,
                "stopped_by_user": summary.get("stopped_by_user", False)
            })
        except Exception:
            pass

    # 2. Automatically render the simulation MP4 video at comfortable viewing speed (skipped if stopped early)
    video_path = os.path.join(run_dir, "simulation_video.mp4")
    if not early_stopped:
        try:
            def video_cb(cur, total):
                if progress_callback:
                    try:
                        progress_callback({
                            "phase": "rendering_video",
                            "frame": cur,
                            "total_frames": total,
                            "progress_ratio": min(1.0, cur / max(1, total)),
                            "run_dir": run_dir,
                            "snapshot_path": snapshot_path
                        })
                    except Exception:
                        pass

            render_simulation_video(run_dir, output_mp4=video_path, fps=fps, max_frames=800,
                                    progress_callback=video_cb, shape_type=shape_type)
            if os.path.exists(video_path):
                summary["video_path"] = video_path
                with open(os.path.join(run_dir, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)
        except Exception as ex:
            print(f"Note: Auto video render skipped or failed: {ex}")

    return summary


# ==============================================================================
# 3. Standard Scientific Plot Generators (Matching Original UI)
# ==============================================================================

def generate_final_snapshot(run_dir, width, height, shape_type, params=None, final_clusters=None):
    """
    Renders high-resolution final structure snapshot using the clean standard
    scientific style (pure white background, black axes, standard palette,
    black outline) exactly matching the user's codebase plots.
    """
    traj_files = glob.glob(os.path.join(run_dir, "traj_*.npy"))
    if not traj_files:
        return ""

    traj = np.load(traj_files[0])
    final_coords = traj[-1]
    N = len(final_coords)

    meta_files = glob.glob(os.path.join(run_dir, "meta_*.json"))
    meta = {}
    if meta_files:
        try:
            with open(meta_files[0], "r") as f:
                meta = json.load(f)
        except Exception:
            pass

    st = shape_type.lower()
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)

    if final_clusters is None:
        cid_files = glob.glob(os.path.join(run_dir, "clusterids_*.npy"))
        if cid_files:
            cids = np.load(cid_files[0])
            final_clusters = len(set(cids))
        else:
            final_clusters = meta.get("final_clusters", 1)

    # Determine count_A robustly
    if "count_A" in meta and meta["count_A"] is not None:
        count_A = int(meta["count_A"])
    elif "count_circle" in meta and meta["count_circle"] is not None:
        count_A = int(meta["count_circle"])
    elif "fraction_A" in meta and meta["fraction_A"] is not None:
        count_A = int(round(N * float(meta["fraction_A"])))
    elif "fraction_circles" in meta and meta["fraction_circles"] is not None:
        count_A = int(round(N * float(meta["fraction_circles"])))
    elif "fraction_circle" in meta and meta["fraction_circle"] is not None:
        count_A = int(round(N * float(meta["fraction_circle"])))
    elif params and "fraction_A" in params:
        count_A = int(round(N * float(params["fraction_A"])))
    else:
        count_A = N

    # Check for saved explicit per-particle shape array (e.g. ['rectangle', 'circle', ...])
    shapes_files = glob.glob(os.path.join(run_dir, "shapes_*.npy"))
    saved_shapes = np.load(shapes_files[0]) if shapes_files else None

    # Load orientations
    orient_files = glob.glob(os.path.join(run_dir, "orient_*.npy"))
    if orient_files:
        try:
            orients_raw = np.load(orient_files[0])
            orients = orients_raw[-1] if orients_raw.ndim > 1 else orients_raw
        except Exception:
            orients = np.zeros(N)
    else:
        orients = np.zeros(N)

    # Load radii or bounding radii
    rad_files = glob.glob(os.path.join(run_dir, "radii_*.npy"))
    radii = np.load(rad_files[0]) if rad_files else None

    # Load local vertices (triangles, superellipses)
    verts_files = glob.glob(os.path.join(run_dir, "vertices_*.npy")) or glob.glob(os.path.join(run_dir, "verts_*.npy"))
    local_verts = np.load(verts_files[0]) if verts_files else None

    # 1. Circle rendering (Monodisperse & Bidisperse)
    if "circle" in st and "triangle" not in st and "square" not in st and "ellipse" not in st and "rectangle" not in st:
        if radii is None:
            radii_pair = meta.get("radii_pair", [8.0, 16.0]) if count_A < N else [8.0, 8.0]
            radii = np.array([radii_pair[0] if i < count_A else radii_pair[1] for i in range(N)])
        for i in range(N):
            col = '#1f77b4' if i < count_A else '#ff7f0e'
            circle = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                color=col, fill=True, alpha=0.9)
            ax.add_patch(circle)
            outline = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                 color='black', fill=False, linewidth=0.6)
            ax.add_patch(outline)

    # 2. Triangle rendering (Monodisperse & Bidisperse)
    elif "triangle" in st and "circle" not in st:
        if local_verts is not None:
            for i in range(N):
                c = math.cos(orients[i])
                s = math.sin(orients[i])
                R = np.array([[c, -s], [s, c]])
                v_rot = local_verts[i] @ R.T + final_coords[i]
                col = 'cyan' if i < count_A else 'coral'
                poly = Polygon(v_rot, closed=True, facecolor=col, alpha=0.85)
                ax.add_patch(poly)
                outline = Polygon(v_rot, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
                ax.add_patch(outline)
        else:
            r_pair = meta.get("radii_pair", [8.0, 16.0]) if count_A < N else [8.0, 8.0]
            base_angles = [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3]
            for i in range(N):
                r_val = r_pair[0] if i < count_A else r_pair[1]
                v_base = np.array([[r_val * math.cos(a), r_val * math.sin(a)] for a in base_angles])
                c = math.cos(orients[i])
                s = math.sin(orients[i])
                R = np.array([[c, -s], [s, c]])
                v_rot = v_base @ R.T + final_coords[i]
                col = 'cyan' if i < count_A else 'coral'
                poly = Polygon(v_rot, closed=True, facecolor=col, alpha=0.85)
                ax.add_patch(poly)
                outline = Polygon(v_rot, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
                ax.add_patch(outline)

    # 3. Square rendering (Monodisperse & Bidisperse)
    elif st == "square":
        if radii is not None:
            sides = radii
        else:
            rad_pair = meta.get("radii_pair", [16.0, 24.0])
            sides = np.array([rad_pair[0] if i < count_A else rad_pair[1] for i in range(N)])

        for i in range(N):
            half = sides[i]
            ang = orients[i]
            c, s = math.cos(ang), math.sin(ang)
            corners = [
                (final_coords[i, 0] + half * (c - s), final_coords[i, 1] + half * (s + c)),
                (final_coords[i, 0] + half * (-c - s), final_coords[i, 1] + half * (-s + c)),
                (final_coords[i, 0] + half * (-c + s), final_coords[i, 1] + half * (-s - c)),
                (final_coords[i, 0] + half * (c + s), final_coords[i, 1] + half * (s - c))
            ]
            col = 'cyan' if i < count_A else 'coral'
            poly = Polygon(corners, closed=True, facecolor=col, alpha=0.85)
            ax.add_patch(poly)
            outline = Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
            ax.add_patch(outline)

    # 4. Rectangle rendering (Monodisperse & Bidisperse)
    elif st == "rectangle":
        dims_pair = meta.get("dims_pair", [[20.0, 10.0], [30.0, 15.0]])
        (w_a, h_a), (w_b, h_b) = dims_pair

        for i in range(N):
            w, h = (w_a, h_a) if i < count_A else (w_b, h_b)
            half_w, half_h = w / 2.0, h / 2.0
            ang = orients[i]
            c, s = math.cos(ang), math.sin(ang)
            dx1, dy1 = half_w * c, half_w * s
            dx2, dy2 = -half_h * s, half_h * c
            corners = [
                (final_coords[i, 0] + dx1 + dx2, final_coords[i, 1] + dy1 + dy2),
                (final_coords[i, 0] - dx1 + dx2, final_coords[i, 1] - dy1 + dy2),
                (final_coords[i, 0] - dx1 - dx2, final_coords[i, 1] - dy1 - dy2),
                (final_coords[i, 0] + dx1 - dx2, final_coords[i, 1] + dy1 - dy2)
            ]
            col = 'cyan' if i < count_A else 'coral'
            poly = Polygon(corners, closed=True, facecolor=col, alpha=0.85)
            ax.add_patch(poly)
            outline = Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
            ax.add_patch(outline)

    # 5. Ellipse rendering (Monodisperse & Bidisperse)
    elif st == "ellipse":
        rad_pair = meta.get("radii_pair", [[16.0, 8.0], [24.0, 12.0]])
        (a_1, b_1), (a_2, b_2) = rad_pair

        for i in range(N):
            a, b = (a_1, b_1) if i < count_A else (a_2, b_2)
            deg = math.degrees(orients[i])
            col = 'orange' if i < count_A else '#1f77b4'
            ell = Ellipse((final_coords[i, 0], final_coords[i, 1]), width=2 * a, height=2 * b, angle=deg,
                          facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.85)
            ax.add_patch(ell)

    # 6. Superellipse rendering (Monodisperse & Bidisperse)
    elif st == "superellipse":
        if local_verts is not None:
            for i in range(N):
                c = math.cos(orients[i])
                s = math.sin(orients[i])
                R = np.array([[c, -s], [s, c]])
                v_rot = local_verts[i] @ R.T + final_coords[i]
                col = 'teal' if i < count_A else 'coral'
                poly = Polygon(v_rot, closed=True, facecolor=col, alpha=0.85)
                ax.add_patch(poly)
                outline = Polygon(v_rot, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
                ax.add_patch(outline)
        else:
            p_a = meta.get("params_A", [16.0, 16.0, 4.0])
            p_b = meta.get("params_B", [24.0, 24.0, 4.0])
            theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for i in range(N):
                a_val, b_val, m_val = p_a if i < count_A else p_b
                pwr = 2.0 / m_val
                v_sample = np.column_stack([
                    a_val * np.sign(cos_t) * (np.abs(cos_t) ** pwr),
                    b_val * np.sign(sin_t) * (np.abs(sin_t) ** pwr)
                ])
                c = math.cos(orients[i])
                s = math.sin(orients[i])
                R = np.array([[c, -s], [s, c]])
                v_rot = v_sample @ R.T + final_coords[i]
                col = 'teal' if i < count_A else 'coral'
                poly = Polygon(v_rot, closed=True, facecolor=col, alpha=0.85)
                ax.add_patch(poly)
                outline = Polygon(v_rot, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
                ax.add_patch(outline)

    # 7. Heterogeneous Mixtures
    elif "circle-square" in st or ("square" in st and "circle" in st):
        if radii is None:
            radii = np.full(N, 8.0)
        for i in range(N):
            # Check shape by saved_shapes if available, else by count_A
            is_circle = (saved_shapes[i] == 'circle') if saved_shapes is not None else (i < count_A)
            if is_circle:
                circle = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                    color='#1f77b4', fill=True, alpha=0.88)
                ax.add_patch(circle)
                outline = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                     color='black', fill=False, linewidth=0.6)
                ax.add_patch(outline)
            else:
                half = radii[i]
                c_ang, s_ang = math.cos(orients[i]), math.sin(orients[i])
                corners = [
                    (final_coords[i, 0] + half * (c_ang - s_ang), final_coords[i, 1] + half * (s_ang + c_ang)),
                    (final_coords[i, 0] + half * (-c_ang - s_ang), final_coords[i, 1] + half * (-s_ang + c_ang)),
                    (final_coords[i, 0] + half * (-c_ang + s_ang), final_coords[i, 1] + half * (-s_ang - c_ang)),
                    (final_coords[i, 0] + half * (c_ang + s_ang), final_coords[i, 1] + half * (s_ang - c_ang))
                ]
                poly = Polygon(corners, closed=True, facecolor='coral', alpha=0.85)
                ax.add_patch(poly)
                outline = Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
                ax.add_patch(outline)

    elif "circle-triangle" in st or ("triangle" in st and "circle" in st):
        if radii is None:
            radii = np.full(N, 8.0)
        if local_verts is None:
            r_val = float(meta.get("r_triangle", 8.0))
            base_angles = [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3]
            v_single = np.array([[r_val * math.cos(a), r_val * math.sin(a)] for a in base_angles])
            local_verts = np.tile(v_single, (N, 1, 1))

        for i in range(N):
            is_circle = (saved_shapes[i] == 'circle') if saved_shapes is not None else (i < count_A)
            if is_circle:
                circle = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                    color='#1f77b4', fill=True, alpha=0.88)
                ax.add_patch(circle)
                outline = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                     color='black', fill=False, linewidth=0.6)
                ax.add_patch(outline)
            else:
                c_ang, s_ang = math.cos(orients[i]), math.sin(orients[i])
                R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
                v_rot = local_verts[i] @ R.T + final_coords[i]
                poly = Polygon(v_rot, closed=True, facecolor='coral', alpha=0.85)
                ax.add_patch(poly)
                outline = Polygon(v_rot, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
                ax.add_patch(outline)

    elif "circle-rectangle" in st or ("rectangle" in st and "circle" in st):
        rect_w = float(meta.get("rect_w", params.get("w", 20.0) if params else 20.0))
        rect_h = float(meta.get("rect_h", params.get("h", 10.0) if params else 10.0))
        circ_r = float(meta.get("circle_r", radii[0] if radii is not None else 8.0))
        half_w, half_h = rect_w / 2.0, rect_h / 2.0

        for i in range(N):
            # In rectangle_circle_cluster.py, shapes array has 'rectangle' and 'circle'
            if saved_shapes is not None:
                is_rect = (saved_shapes[i] == 'rectangle')
            elif "count_rect" in meta:
                is_rect = (i < meta["count_rect"])
            else:
                is_rect = (i >= count_A)

            if is_rect:
                c_ang, s_ang = math.cos(orients[i]), math.sin(orients[i])
                dx1, dy1 = half_w * c_ang, half_w * s_ang
                dx2, dy2 = -half_h * s_ang, half_h * c_ang
                corners = [
                    (final_coords[i, 0] + dx1 + dx2, final_coords[i, 1] + dy1 + dy2),
                    (final_coords[i, 0] - dx1 + dx2, final_coords[i, 1] - dy1 + dy2),
                    (final_coords[i, 0] - dx1 - dx2, final_coords[i, 1] - dy1 - dy2),
                    (final_coords[i, 0] + dx1 - dx2, final_coords[i, 1] + dy1 - dy2)
                ]
                poly = Polygon(corners, closed=True, facecolor='coral', alpha=0.85)
                ax.add_patch(poly)
                outline = Polygon(corners, closed=True, edgecolor='black', facecolor='none', linewidth=0.6)
                ax.add_patch(outline)
            else:
                r_val = circ_r if radii is None else radii[i]
                circle = plt.Circle((final_coords[i, 0], final_coords[i, 1]), r_val,
                                    color='#1f77b4', fill=True, alpha=0.88)
                ax.add_patch(circle)
                outline = plt.Circle((final_coords[i, 0], final_coords[i, 1]), r_val,
                                     color='black', fill=False, linewidth=0.6)
                ax.add_patch(outline)

    elif "circle-ellipse" in st or ("ellipse" in st and "circle" in st):
        major_a = float(meta.get("ellipse_major_r", params.get("a", 16.0) if params else 16.0))
        minor_b = float(meta.get("ellipse_minor_r", params.get("b", 8.0) if params else 8.0))
        circ_r = float(meta.get("circle_r", radii[0] if radii is not None else 8.0))

        for i in range(N):
            if saved_shapes is not None:
                is_ellipse = (saved_shapes[i] == 'ellipse')
            elif "count_ellipse" in meta:
                is_ellipse = (i < meta["count_ellipse"])
            else:
                is_ellipse = (i >= count_A)

            if is_ellipse:
                deg = math.degrees(orients[i])
                ell = Ellipse((final_coords[i, 0], final_coords[i, 1]), width=2 * major_a, height=2 * minor_b,
                              angle=deg, facecolor='orange', edgecolor='black', linewidth=0.6, alpha=0.85)
                ax.add_patch(ell)
            else:
                r_val = circ_r if radii is None else radii[i]
                circle = plt.Circle((final_coords[i, 0], final_coords[i, 1]), r_val,
                                    color='#1f77b4', fill=True, alpha=0.88)
                ax.add_patch(circle)
                outline = plt.Circle((final_coords[i, 0], final_coords[i, 1]), r_val,
                                     color='black', fill=False, linewidth=0.6)
                ax.add_patch(outline)

    else:
        if radii is None:
            radii = np.full(N, 8.0)
        for i in range(N):
            col = '#1f77b4' if i < count_A else 'coral'
            circle = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                color=col, fill=True, alpha=0.85)
            ax.add_patch(circle)
            outline = plt.Circle((final_coords[i, 0], final_coords[i, 1]), radii[i],
                                 color='black', fill=False, linewidth=0.6)
            ax.add_patch(outline)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")

    shape_title = shape_type.replace("Cluster", "").strip().capitalize()
    ax.set_title(f"Final: {shape_title} Particles | {final_clusters} clusters | N={N}", fontsize=12)

    out_png = os.path.join(run_dir, "final_structure.png")
    fig.savefig(out_png, bbox_inches='tight', dpi=180)
    plt.close(fig)
    return out_png


def generate_cluster_decay_plot(run_dir, steps_arr=None, clusters_arr=None, initial_count=None):
    """
    Plots cluster aggregation decay kinetics with proper Y-axis scaling
    from 1 up to N_initial (full coverage, no cutoff).
    """
    cc_files = glob.glob(os.path.join(run_dir, "clustercount_*.npy"))
    steps, clusters = [], []

    if cc_files:
        try:
            arr = np.load(cc_files[0])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                steps = list(arr[:, 0])
                clusters = list(arr[:, 1])
        except Exception:
            pass

    if len(steps) == 0 and steps_arr is not None and clusters_arr is not None and len(steps_arr) > 0:
        steps = list(steps_arr)
        clusters = list(clusters_arr)

    if len(steps) == 0:
        return ""

    if initial_count is None:
        initial_count = max(clusters) if clusters else 100

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(steps, clusters, color='#1f77b4', linewidth=2.2, label="Active Clusters $N_c(t)$")
    ax.fill_between(steps, clusters, color='#1f77b4', alpha=0.15)
    
    # Properly scaled axes covering from 0 to N_initial
    ax.set_ylim(0, initial_count * 1.05)
    ax.set_xlim(0, max(steps) if max(steps) > 0 else 1000)

    ax.set_xlabel("Timestep (t)", fontsize=11)
    ax.set_ylabel("Cluster Count ($N_c$)", fontsize=11)
    ax.set_title(f"DLCA Agglomeration Kinetics ($N_{{init}} = {initial_count} \\rightarrow {clusters[-1]}$)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    out_path = os.path.join(run_dir, "cluster_decay.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def generate_fractal_dimension_plot(run_dir, df_val=None):
    """
    Renders fractal dimension fitting plot if data is recorded.
    """
    if df_val is None or math.isnan(df_val):
        return ""

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.text(0.5, 0.5, f"Fractal Dimension Df = {df_val:.3f}",
             fontsize=16, fontweight="bold", color='#2c3e50',
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.6", facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.9))
    plt.title("Mass-Radius Scaling & Fractal Dimension", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(run_dir, "fractal_fit.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    return out_path


# ==============================================================================
# 4. True-Geometry MP4 Video Animation Renderer
# ==============================================================================

def render_simulation_video(run_dir, output_mp4=None, fps=15, max_frames=800, progress_callback=None, shape_type=None):
    """
    Renders a smooth, properly-paced MP4 animation of the simulation trajectory.
    Defaults to 15 FPS so movement is clear and easy to follow (not sped up).
    Includes 60 static hold frames at the end so the final state remains visible.
    """
    traj_files = glob.glob(os.path.join(run_dir, "traj_*.npy"))
    meta_files = glob.glob(os.path.join(run_dir, "meta_*.json"))
    summary_file = os.path.join(run_dir, "summary.json")

    if not traj_files:
        raise FileNotFoundError(f"No trajectory file found in {run_dir}")

    traj = np.load(traj_files[0])
    total_frames = traj.shape[0]
    N = traj.shape[1]

    # Keep all frames if reasonably sized, only subsample if > max_frames
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        traj_sub = traj[frame_indices]
    else:
        frame_indices = np.arange(total_frames)
        traj_sub = traj

    # Append 60 static hold frames at the final state (~4 seconds at 15 FPS)
    hold_frames = np.repeat(traj_sub[-1:], 60, axis=0)
    traj_sub = np.concatenate([traj_sub, hold_frames], axis=0)
    num_anim_frames = len(traj_sub)

    # Read dimensions and shape type
    width = 1600
    height = 1200

    meta = {}
    if meta_files:
        try:
            with open(meta_files[0], "r") as f:
                meta = json.load(f)
                width = meta.get("width", width)
                height = meta.get("height", height)
        except Exception:
            pass

    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                sm = json.load(f)
                width = sm.get("width", width)
                height = sm.get("height", height)
                if not shape_type:
                    shape_type = sm.get("shape_type")
        except Exception:
            pass

    # Robust count_A calculation
    if "count_A" in meta and meta["count_A"] is not None:
        count_A = int(meta["count_A"])
    elif "count_circle" in meta and meta["count_circle"] is not None:
        count_A = int(meta["count_circle"])
    elif "fraction_A" in meta and meta["fraction_A"] is not None:
        count_A = int(round(N * float(meta["fraction_A"])))
    elif "fraction_circles" in meta and meta["fraction_circles"] is not None:
        count_A = int(round(N * float(meta["fraction_circles"])))
    elif "fraction_circle" in meta and meta["fraction_circle"] is not None:
        count_A = int(round(N * float(meta["fraction_circle"])))
    else:
        count_A = N

    # Multi-layer shape resolution
    if not shape_type:
        if "triangle_type" in meta:
            shape_type = f"{meta['triangle_type']} triangle"
        elif "simulation_type" in meta:
            shape_type = meta["simulation_type"]
        elif "shape_type" in meta:
            shape_type = meta["shape_type"]
        elif "shape" in meta:
            shape_type = meta["shape"]

    dir_lower = os.path.basename(run_dir).lower()
    if not shape_type or shape_type.lower() == "circle":
        if "scalene" in dir_lower:
            shape_type = "scalene triangle"
        elif "equilateral" in dir_lower:
            shape_type = "equilateral triangle"
        elif "right" in dir_lower and "triangle" in dir_lower:
            shape_type = "right triangle"
        elif "isosceles" in dir_lower:
            shape_type = "isosceles triangle"
        elif "triangle" in dir_lower and "circle" not in dir_lower:
            shape_type = "triangle"
        elif "superellipse" in dir_lower:
            shape_type = "superellipse"
        elif "square" in dir_lower and "circle" not in dir_lower:
            shape_type = "square"
        elif "rectangle" in dir_lower and "circle" not in dir_lower:
            shape_type = "rectangle"
        elif "ellipse" in dir_lower and "circle" not in dir_lower:
            shape_type = "ellipse"
        elif "circle-triangle" in dir_lower or "triangle_circle" in dir_lower:
            shape_type = "circle-triangle"
        elif "circle-square" in dir_lower or "square_circle" in dir_lower:
            shape_type = "circle-square"
        elif "circle-ellipse" in dir_lower or "ellipse_circle" in dir_lower:
            shape_type = "circle-ellipse"
        elif "circle-rectangle" in dir_lower or "rectangle_circle" in dir_lower:
            shape_type = "circle-rectangle"

    if not shape_type:
        shape_type = "Circle"

    # Check for stored local vertices
    verts_files = glob.glob(os.path.join(run_dir, "vertices_*.npy")) or glob.glob(os.path.join(run_dir, "verts_*.npy"))
    local_verts = np.load(verts_files[0]) if verts_files else None

    # Guarantee: If local_verts exists, it is a polygon (superellipse or triangle) and MUST NEVER render as circles!
    st = shape_type.lower()
    if local_verts is not None and "circle" in st and "circle-" not in st:
        if local_verts.shape[1] == 3:
            shape_type = "Triangle"
            st = "triangle"
        else:
            shape_type = "Superellipse"
            st = "superellipse"

    if output_mp4 is None:
        output_mp4 = os.path.join(run_dir, "simulation_video.mp4")

    # Check for saved explicit per-particle shape array (e.g. ['rectangle', 'circle', ...])
    shapes_files = glob.glob(os.path.join(run_dir, "shapes_*.npy"))
    saved_shapes = np.load(shapes_files[0]) if shapes_files else None

    # Load orientations
    orient_files = glob.glob(os.path.join(run_dir, "orient_*.npy"))
    if orient_files:
        try:
            orient_raw = np.load(orient_files[0])
            if orient_raw.ndim == 2:
                orient_sub = orient_raw[frame_indices]
                hold_orient = np.repeat(orient_sub[-1:], 60, axis=0)
                orient_sub = np.concatenate([orient_sub, hold_orient], axis=0)
            else:
                orient_sub = np.tile(orient_raw, (num_anim_frames, 1))
        except Exception:
            orient_sub = np.zeros((num_anim_frames, N))
    else:
        orient_sub = np.zeros((num_anim_frames, N))

    rad_files = glob.glob(os.path.join(run_dir, "radii_*.npy"))
    radii = np.load(rad_files[0]) if rad_files else np.full(N, 8.0)

    # Setup animation figure with pure white background
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle='--', alpha=0.3)

    artists = []
    artist_types = []

    # Setup artists per shape
    if "superellipse" in st:
        if local_verts is None:
            a_val, b_val, m_val = 16.0, 16.0, 4.0
            theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            pwr = 2.0 / m_val
            v_single = np.column_stack([
                a_val * np.sign(cos_t) * (np.abs(cos_t) ** pwr),
                b_val * np.sign(sin_t) * (np.abs(sin_t) ** pwr)
            ])
            local_verts = np.tile(v_single, (N, 1, 1))

        for i in range(N):
            c_ang, s_ang = math.cos(orient_sub[0, i]), math.sin(orient_sub[0, i])
            R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
            v_rot = local_verts[i] @ R.T + traj_sub[0, i]
            col = 'teal' if i < count_A else 'coral'
            poly = Polygon(v_rot, closed=True, facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.85)
            ax.add_patch(poly)
            artists.append(poly)
            artist_types.append("polygon")

    elif "triangle" in st and "circle" not in st:
        if local_verts is None:
            r_val = 8.0
            base_angles = [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3]
            v_single = np.array([[r_val * math.cos(a), r_val * math.sin(a)] for a in base_angles])
            local_verts = np.tile(v_single, (N, 1, 1))

        for i in range(N):
            c_ang, s_ang = math.cos(orient_sub[0, i]), math.sin(orient_sub[0, i])
            R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
            v_rot = local_verts[i] @ R.T + traj_sub[0, i]
            col = 'cyan' if i < count_A else 'coral'
            poly = Polygon(v_rot, closed=True, facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.85)
            ax.add_patch(poly)
            artists.append(poly)
            artist_types.append("polygon")

    elif st == "square":
        for i in range(N):
            half = radii[i]
            col = 'cyan' if i < count_A else 'coral'
            corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
            c_ang, s_ang = math.cos(orient_sub[0, i]), math.sin(orient_sub[0, i])
            R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
            corners_rot = (np.array(corners) @ R.T) + traj_sub[0, i]
            poly = Polygon(corners_rot, closed=True, facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.85)
            ax.add_patch(poly)
            artists.append(poly)
            artist_types.append("square")

    elif st == "rectangle":
        dims_pair = meta.get("dims_pair", [[20.0, 10.0], [20.0, 10.0]])
        (w_a, h_a), (w_b, h_b) = dims_pair
        for i in range(N):
            w, h = (w_a, h_a) if i < count_A else (w_b, h_b)
            half_w, half_h = w / 2.0, h / 2.0
            col = 'cyan' if i < count_A else 'coral'
            corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
            c_ang, s_ang = math.cos(orient_sub[0, i]), math.sin(orient_sub[0, i])
            R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
            corners_rot = (np.array(corners) @ R.T) + traj_sub[0, i]
            poly = Polygon(corners_rot, closed=True, facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.85)
            ax.add_patch(poly)
            artists.append(poly)
            artist_types.append("rectangle")

    elif st == "ellipse":
        rad_pair = meta.get("radii_pair", [[16.0, 8.0], [16.0, 8.0]])
        (a_1, b_1), (a_2, b_2) = rad_pair
        for i in range(N):
            a, b = (a_1, b_1) if i < count_A else (a_2, b_2)
            col = 'orange' if i < count_A else '#1f77b4'
            ell = Ellipse((traj_sub[0, i, 0], traj_sub[0, i, 1]), width=2 * a, height=2 * b,
                          angle=math.degrees(orient_sub[0, i]),
                          facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.85)
            ax.add_patch(ell)
            artists.append(ell)
            artist_types.append("ellipse")

    elif "circle-triangle" in st or ("triangle" in st and "circle" in st):
        if local_verts is None:
            r_val = 8.0
            base_angles = [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3]
            v_single = np.array([[r_val * math.cos(a), r_val * math.sin(a)] for a in base_angles])
            local_verts = np.tile(v_single, (N, 1, 1))

        for i in range(N):
            if i < count_A:
                col = '#1f77b4'
                c = Circle((traj_sub[0, i, 0], traj_sub[0, i, 1]), radii[i],
                           facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.88)
                ax.add_patch(c)
                artists.append(c)
                artist_types.append("circle")
            else:
                c_ang, s_ang = math.cos(orient_sub[0, i]), math.sin(orient_sub[0, i])
                R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
                v_rot = local_verts[i] @ R.T + traj_sub[0, i]
                poly = Polygon(v_rot, closed=True, facecolor='coral', edgecolor='black', linewidth=0.6, alpha=0.85)
                ax.add_patch(poly)
                artists.append(poly)
                artist_types.append("polygon")

    elif "circle-square" in st or ("square" in st and "circle" in st):
        for i in range(N):
            is_circle = (saved_shapes[i] == 'circle') if saved_shapes is not None else (i < count_A)
            if is_circle:
                col = '#1f77b4'
                c = Circle((traj_sub[0, i, 0], traj_sub[0, i, 1]), radii[i],
                           facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.88)
                ax.add_patch(c)
                artists.append(c)
                artist_types.append("circle")
            else:
                half = radii[i]
                corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
                c_ang, s_ang = math.cos(orient_sub[0, i]), math.sin(orient_sub[0, i])
                R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
                corners_rot = (np.array(corners) @ R.T) + traj_sub[0, i]
                poly = Polygon(corners_rot, closed=True, facecolor='coral', edgecolor='black', linewidth=0.6, alpha=0.85)
                ax.add_patch(poly)
                artists.append(poly)
                artist_types.append("square")

    elif "circle-rectangle" in st or ("rectangle" in st and "circle" in st):
        rect_w = float(meta.get("rect_w", 20.0))
        rect_h = float(meta.get("rect_h", 10.0))
        circ_r = float(meta.get("circle_r", radii[0] if radii is not None else 8.0))
        half_w, half_h = rect_w / 2.0, rect_h / 2.0

        for i in range(N):
            if saved_shapes is not None:
                is_rect = (saved_shapes[i] == 'rectangle')
            elif "count_rect" in meta:
                is_rect = (i < meta["count_rect"])
            else:
                is_rect = (i >= count_A)

            if is_rect:
                corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
                c_ang, s_ang = math.cos(orient_sub[0, i]), math.sin(orient_sub[0, i])
                R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
                corners_rot = (np.array(corners) @ R.T) + traj_sub[0, i]
                poly = Polygon(corners_rot, closed=True, facecolor='coral', edgecolor='black', linewidth=0.6, alpha=0.85)
                ax.add_patch(poly)
                artists.append(poly)
                artist_types.append("rectangle")
            else:
                r_val = circ_r if radii is None else radii[i]
                c = Circle((traj_sub[0, i, 0], traj_sub[0, i, 1]), r_val,
                           facecolor='#1f77b4', edgecolor='black', linewidth=0.6, alpha=0.88)
                ax.add_patch(c)
                artists.append(c)
                artist_types.append("circle")

    elif "circle-ellipse" in st or ("ellipse" in st and "circle" in st):
        major_a = float(meta.get("ellipse_major_r", 16.0))
        minor_b = float(meta.get("ellipse_minor_r", 8.0))
        circ_r = float(meta.get("circle_r", radii[0] if radii is not None else 8.0))

        for i in range(N):
            if saved_shapes is not None:
                is_ellipse = (saved_shapes[i] == 'ellipse')
            elif "count_ellipse" in meta:
                is_ellipse = (i < meta["count_ellipse"])
            else:
                is_ellipse = (i >= count_A)

            if is_ellipse:
                ell = Ellipse((traj_sub[0, i, 0], traj_sub[0, i, 1]), width=2 * major_a, height=2 * minor_b,
                              angle=math.degrees(orient_sub[0, i]),
                              facecolor='orange', edgecolor='black', linewidth=0.6, alpha=0.85)
                ax.add_patch(ell)
                artists.append(ell)
                artist_types.append("ellipse")
            else:
                r_val = circ_r if radii is None else radii[i]
                c = Circle((traj_sub[0, i, 0], traj_sub[0, i, 1]), r_val,
                           facecolor='#1f77b4', edgecolor='black', linewidth=0.6, alpha=0.88)
                ax.add_patch(c)
                artists.append(c)
                artist_types.append("circle")

    else:
        for i in range(N):
            col = '#1f77b4' if i < count_A else '#ff7f0e'
            c = Circle((traj_sub[0, i, 0], traj_sub[0, i, 1]), radii[i],
                       facecolor=col, edgecolor='black', linewidth=0.6, alpha=0.88)
            ax.add_patch(c)
            artists.append(c)
            artist_types.append("circle")

    shape_display = shape_type.replace("Cluster", "").strip().capitalize()
    ax.set_title(f"DLCA {shape_display} Simulation (N = {N})", fontsize=12)

    def update(frame_idx):
        coords = traj_sub[frame_idx]
        orients = orient_sub[frame_idx]

        for i in range(N):
            ptype = artist_types[i]
            x, y = coords[i, 0], coords[i, 1]
            ang = orients[i]

            if ptype == "circle":
                artists[i].center = (x, y)
            elif ptype == "ellipse":
                artists[i].center = (x, y)
                artists[i].angle = math.degrees(ang)
            elif ptype == "polygon" and local_verts is not None:
                c_ang, s_ang = math.cos(ang), math.sin(ang)
                R = np.array([[c_ang, -s_ang], [s_ang, c_ang]])
                v_rot = local_verts[i] @ R.T + np.array([x, y])
                artists[i].set_xy(v_rot)
            elif ptype in ["square", "rectangle"]:
                c_ang, s_ang = math.cos(ang), math.sin(ang)
                if ptype == "square":
                    half = radii[i]
                    corners = [
                        (x + half * (c_ang - s_ang), y + half * (s_ang + c_ang)),
                        (x + half * (-c_ang - s_ang), y + half * (-s_ang + c_ang)),
                        (x + half * (-c_ang + s_ang), y + half * (-s_ang - c_ang)),
                        (x + half * (c_ang + s_ang), y + half * (s_ang - c_ang))
                    ]
                else:
                    if "rect_w" in meta and "rect_h" in meta:
                        w, h = float(meta["rect_w"]), float(meta["rect_h"])
                    else:
                        dims_pair = meta.get("dims_pair", [[20.0, 10.0], [20.0, 10.0]])
                        w, h = dims_pair[0] if i < count_A else dims_pair[1]
                    hw, hh = w / 2.0, h / 2.0
                    dx1, dy1 = hw * c_ang, hw * s_ang
                    dx2, dy2 = -hh * s_ang, hh * c_ang
                    corners = [
                        (x + dx1 + dx2, y + dy1 + dy2),
                        (x - dx1 + dx2, y - dy1 + dy2),
                        (x - dx1 - dx2, y - dy1 - dy2),
                        (x + dx1 - dx2, y + dy1 - dy2)
                    ]
                artists[i].set_xy(corners)

        if progress_callback and (frame_idx % 25 == 0 or frame_idx == num_anim_frames - 1):
            progress_callback(frame_idx + 1, num_anim_frames)

        return artists

    anim = animation.FuncAnimation(fig, update, frames=num_anim_frames, blit=True)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='DLCA Suite'), bitrate=2500)
    try:
        anim.save(output_mp4, writer=writer)
    finally:
        plt.close(fig)

    return output_mp4


# ==============================================================================
# 5. Saved Runs Catalog & Explorer
# ==============================================================================

def list_saved_runs():
    """
    Scans RUNS_BASE_DIR and returns a list of dictionaries with metadata on past runs.
    """
    if not os.path.exists(RUNS_BASE_DIR):
        return []

    dirs = [os.path.join(RUNS_BASE_DIR, d) for d in os.listdir(RUNS_BASE_DIR)
            if os.path.isdir(os.path.join(RUNS_BASE_DIR, d))]
    dirs.sort(key=os.path.getmtime, reverse=True)

    runs = []
    for d in dirs:
        summary_p = os.path.join(d, "summary.json")
        if os.path.exists(summary_p):
            try:
                with open(summary_p, "r") as f:
                    sm = json.load(f)
                    runs.append(sm)
                    continue
            except Exception:
                pass

        tag = os.path.basename(d)
        snap_p = os.path.join(d, "final_structure.png")
        vid_p = os.path.join(d, "simulation_video.mp4")
        runs.append({
            "run_tag": tag,
            "run_dir": d,
            "shape_type": tag.split("_")[0].capitalize(),
            "snapshot_path": snap_p if os.path.exists(snap_p) else "",
            "video_path": vid_p if os.path.exists(vid_p) else ""
        })

    return runs
