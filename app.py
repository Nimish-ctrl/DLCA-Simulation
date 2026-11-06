"""
app.py — DLCA Simulation Suite GUI
==============================================
A clean, responsive graphical user interface for running Diffusion-Limited
Cluster Aggregation (DLCA) simulations across multiple geometries, monitoring
live progress with real-time ETA, automatically generating publication plots,
and instantly viewing rendered simulation videos.
"""

import os
import glob
import json
import math
import time
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import streamlit.components.v1 as components

import dlca_gui_backend as backend

# Initialize stop signal port
STOP_PORT = backend.get_or_start_stop_server()

# ==============================================================================
# Page Configuration & Clean Adaptive Theme
# ==============================================================================

st.set_page_config(
    page_title="DLCA Studio [Experimental with Stop/Cancel]",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with 100% dark & light mode readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0984e3, #00cec9, #6c5ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.0rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding-top: 8px;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: CONFIGURATION
# ==============================================================================

st.sidebar.title("⚛️ Configuration")
st.sidebar.markdown("Configure particle geometry, packing density, and domain.")

# 1. Geometry Architecture
shape_options = [
    "Circle",
    "Equilateral Triangle",
    "Right Triangle",
    "Isosceles Triangle",
    "Scalene Triangle",
    "Square",
    "Rectangle",
    "Ellipse",
    "Superellipse",
    "Circle-Square",
    "Circle-Triangle",
    "Circle-Ellipse",
    "Circle-Rectangle"
]

shape_type = st.sidebar.selectbox("Particle Geometry", shape_options, index=0)

is_heterogeneous = shape_type in ["Circle-Square", "Circle-Triangle", "Circle-Ellipse", "Circle-Rectangle"]

if is_heterogeneous:
    dispersion = "Heterogeneous"
    st.sidebar.info(f"**{shape_type}** is inherently a 2-species heterogeneous mixture.")
else:
    dispersion = st.sidebar.radio("Dispersion Mode", ["Monodisperse", "Bidisperse"], horizontal=True)

# 2. Dynamic Geometric Dimensions
st.sidebar.subheader("📐 Particle Dimensions")

params = {"species_A": {}, "species_B": {}}

if shape_type == "Circle":
    if dispersion == "Monodisperse":
        r = st.sidebar.number_input("Radius (r)", min_value=1.0, max_value=100.0, value=8.0, step=1.0)
        params["species_A"]["r"] = r
    else:
        c1, c2 = st.sidebar.columns(2)
        r_a = c1.number_input("Radius A (r_A)", min_value=1.0, max_value=100.0, value=8.0, step=1.0)
        r_b = c2.number_input("Radius B (r_B)", min_value=1.0, max_value=100.0, value=16.0, step=1.0)
        params["species_A"]["r"] = r_a
        params["species_B"]["r"] = r_b
        params["fraction_A"] = st.sidebar.slider("Fraction of Species A", 0.05, 0.95, 0.668, 0.01)

elif shape_type == "Equilateral Triangle":
    if dispersion == "Monodisperse":
        r = st.sidebar.number_input("Circumradius (r)", min_value=1.0, max_value=100.0, value=8.0, step=1.0)
        params["species_A"]["r"] = r
    else:
        c1, c2 = st.sidebar.columns(2)
        r_a = c1.number_input("Circumradius A", min_value=1.0, max_value=100.0, value=8.0, step=1.0)
        r_b = c2.number_input("Circumradius B", min_value=1.0, max_value=100.0, value=16.0, step=1.0)
        params["species_A"]["r"] = r_a
        params["species_B"]["r"] = r_b
        params["fraction_A"] = st.sidebar.slider("Fraction of Species A", 0.05, 0.95, 0.668, 0.01)

elif shape_type == "Right Triangle":
    if dispersion == "Monodisperse":
        c1, c2 = st.sidebar.columns(2)
        b = c1.number_input("Base (b)", min_value=1.0, max_value=100.0, value=16.0, step=1.0)
        h = c2.number_input("Height (h)", min_value=1.0, max_value=100.0, value=12.0, step=1.0)
        params["species_A"]["b"] = b
        params["species_A"]["h"] = h
    else:
        c1, c2 = st.sidebar.columns(2)
        b_a = c1.number_input("Base A", value=16.0)
        h_a = c1.number_input("Height A", value=12.0)
        b_b = c2.number_input("Base B", value=24.0)
        h_b = c2.number_input("Height B", value=18.0)
        params["species_A"]["b"] = b_a
        params["species_A"]["h"] = h_a
        params["species_B"]["b"] = b_b
        params["species_B"]["h"] = h_b
        params["fraction_A"] = st.sidebar.slider("Fraction of Species A", 0.05, 0.95, 0.50, 0.01)

elif shape_type == "Isosceles Triangle":
    c1, c2 = st.sidebar.columns(2)
    b = c1.number_input("Base (b)", min_value=1.0, max_value=100.0, value=16.0, step=1.0)
    h = c2.number_input("Height (h)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    params["species_A"]["b"] = b
    params["species_A"]["h"] = h

elif shape_type == "Scalene Triangle":
    c1, c2, c3 = st.sidebar.columns(3)
    s_a = c1.number_input("Side a", min_value=1.0, max_value=100.0, value=15.0, step=1.0)
    s_b = c2.number_input("Side b", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    s_c = c3.number_input("Side c", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
    params["species_A"]["a"] = s_a
    params["species_A"]["b"] = s_b
    params["species_A"]["c"] = s_c

elif shape_type == "Square":
    if dispersion == "Monodisperse":
        s = st.sidebar.number_input("Side Length (s)", min_value=1.0, max_value=100.0, value=16.0, step=1.0)
        params["species_A"]["s"] = s
    else:
        c1, c2 = st.sidebar.columns(2)
        s_a = c1.number_input("Side A (s_A)", value=16.0)
        s_b = c2.number_input("Side B (s_B)", value=24.0)
        params["species_A"]["s"] = s_a
        params["species_B"]["s"] = s_b
        params["fraction_A"] = st.sidebar.slider("Fraction of Species A", 0.05, 0.95, 0.668, 0.01)

elif shape_type == "Rectangle":
    if dispersion == "Monodisperse":
        c1, c2 = st.sidebar.columns(2)
        w = c1.number_input("Width (w)", value=20.0)
        h = c2.number_input("Height (h)", value=10.0)
        params["species_A"]["w"] = w
        params["species_A"]["h"] = h
    else:
        c1, c2 = st.sidebar.columns(2)
        w_a = c1.number_input("Width A", value=20.0)
        h_a = c1.number_input("Height A", value=10.0)
        w_b = c2.number_input("Width B", value=30.0)
        h_b = c2.number_input("Height B", value=15.0)
        params["species_A"]["w"] = w_a
        params["species_A"]["h"] = h_a
        params["species_B"]["w"] = w_b
        params["species_B"]["h"] = h_b
        params["fraction_A"] = st.sidebar.slider("Fraction of Species A", 0.05, 0.95, 0.50, 0.01)

elif shape_type == "Ellipse":
    if dispersion == "Monodisperse":
        c1, c2 = st.sidebar.columns(2)
        a = c1.number_input("Semi-major Axis (a)", value=16.0)
        b = c2.number_input("Semi-minor Axis (b)", value=8.0)
        params["species_A"]["a"] = a
        params["species_A"]["b"] = b
    else:
        c1, c2 = st.sidebar.columns(2)
        a_1 = c1.number_input("Axis a_1", value=16.0)
        b_1 = c1.number_input("Axis b_1", value=8.0)
        a_2 = c2.number_input("Axis a_2", value=24.0)
        b_2 = c2.number_input("Axis b_2", value=12.0)
        params["species_A"]["a"] = a_1
        params["species_A"]["b"] = b_1
        params["species_B"]["a"] = a_2
        params["species_B"]["b"] = b_2
        params["fraction_A"] = st.sidebar.slider("Fraction of Species A", 0.05, 0.95, 0.668, 0.01)

elif shape_type == "Superellipse":
    if dispersion == "Monodisperse":
        c1, c2, c3 = st.sidebar.columns(3)
        a = c1.number_input("Axis a", value=16.0)
        b = c2.number_input("Axis b", value=16.0)
        m = c3.number_input("Exponent m", value=4.0, min_value=1.0, max_value=10.0, step=0.5)
        params["species_A"]["a"] = a
        params["species_A"]["b"] = b
        params["species_A"]["m"] = m
    else:
        st.sidebar.markdown("**Species A (Blue / Teal)**")
        c1, c2, c3 = st.sidebar.columns(3)
        a_1 = c1.number_input("Axis a_1", value=16.0)
        b_1 = c2.number_input("Axis b_1", value=16.0)
        m_1 = c3.number_input("Exponent m_1", value=4.0, min_value=1.0, max_value=10.0, step=0.5)
        params["species_A"]["a"] = a_1
        params["species_A"]["b"] = b_1
        params["species_A"]["m"] = m_1

        st.sidebar.markdown("**Species B (Coral / Amber)**")
        c4, c5, c6 = st.sidebar.columns(3)
        a_2 = c4.number_input("Axis a_2", value=24.0)
        b_2 = c5.number_input("Axis b_2", value=24.0)
        m_2 = c6.number_input("Exponent m_2", value=4.0, min_value=1.0, max_value=10.0, step=0.5)
        params["species_B"]["a"] = a_2
        params["species_B"]["b"] = b_2
        params["species_B"]["m"] = m_2

        params["fraction_A"] = st.sidebar.slider("Fraction of Species A", 0.05, 0.95, 0.668, 0.01)

elif shape_type == "Circle-Square":
    c1, c2 = st.sidebar.columns(2)
    r_c = c1.number_input("Circle Radius (r)", value=8.0)
    s_sq = c2.number_input("Square Side (s)", value=16.0)
    frac_c = st.sidebar.slider("Fraction Circles", 0.1, 0.9, 0.5, 0.05)
    params["r_circle"] = r_c
    params["s_square"] = s_sq
    params["fraction_circle"] = frac_c

elif shape_type == "Circle-Triangle":
    c1, c2 = st.sidebar.columns(2)
    r_c = c1.number_input("Circle Radius (r)", value=8.0)
    r_t = c2.number_input("Triangle Radius (r)", value=8.0)
    frac_c = st.sidebar.slider("Fraction Circles", 0.1, 0.9, 0.5, 0.05)
    params["r_circle"] = r_c
    params["r_triangle"] = r_t
    params["fraction_circle"] = frac_c

elif shape_type == "Circle-Ellipse":
    c1, c2, c3 = st.sidebar.columns(3)
    r_c = c1.number_input("Circle Radius", value=8.0)
    a_e = c2.number_input("Ellipse a", value=16.0)
    b_e = c3.number_input("Ellipse b", value=8.0)
    frac_c = st.sidebar.slider("Fraction Circles", 0.1, 0.9, 0.5, 0.05)
    params["r_circle"] = r_c
    params["a"] = a_e
    params["b"] = b_e
    params["fraction_circle"] = frac_c

elif shape_type == "Circle-Rectangle":
    c1, c2, c3 = st.sidebar.columns(3)
    r_c = c1.number_input("Circle Radius", value=8.0)
    w_r = c2.number_input("Rect Width", value=20.0)
    h_r = c3.number_input("Rect Height", value=10.0)
    frac_c = st.sidebar.slider("Fraction Circles", 0.1, 0.9, 0.5, 0.05)
    params["r_circle"] = r_c
    params["w"] = w_r
    params["h"] = h_r
    params["fraction_circle"] = frac_c

# 3. Domain Geometry
st.sidebar.subheader("📦 Arena Domain")
preset = st.sidebar.selectbox(
    "Arena Presets",
    ["1600 × 1200 (HD 4:3)", "1150 × 870 (Standard 4:3)", "1000 × 1000 (Square)", "3200 × 2400 (Large Arena)", "Custom"],
    index=0
)

if preset == "1600 × 1200 (HD 4:3)":
    box_w, box_h = 1600, 1200
elif preset == "1150 × 870 (Standard 4:3)":
    box_w, box_h = 1150, 870
elif preset == "1000 × 1000 (Square)":
    box_w, box_h = 1000, 1000
elif preset == "3200 × 2400 (Large Arena)":
    box_w, box_h = 3200, 2400
else:
    c1, c2 = st.sidebar.columns(2)
    box_w = c1.number_input("Width", min_value=200, max_value=8000, value=1600, step=100)
    box_h = c2.number_input("Height", min_value=200, max_value=8000, value=1200, step=100)

# 4. Packing Density Mode (Bidirectional Conversion)
st.sidebar.subheader("🎯 Packing Density")
pack_mode = st.sidebar.radio("Input Specification", ["By Area Fraction (φ)", "By Particle Count (N)"], horizontal=True)

if pack_mode == "By Area Fraction (φ)":
    phi_val = st.sidebar.slider("Target Area Fraction (φ)", min_value=0.005, max_value=0.45, value=0.05, step=0.005)
    count_val = 500
else:
    count_val = st.sidebar.number_input("Target Particles (N)", min_value=10, max_value=10000, value=400, step=25)
    phi_val = 0.05

calc_n, calc_phi, mean_part_area, domain_area = backend.compute_packing_parameters(
    shape_type, dispersion, params, box_w, box_h, pack_mode, count_val, phi_val
)

# Native theme-adaptive container (works 100% in dark & light modes, no white box clash)
with st.sidebar.container(border=True):
    st.markdown("**Live Packing Computation:**")
    st.markdown(f"• **Particles ($N$):** `{calc_n:,}`")
    st.markdown(f"• **Area Fraction (φ):** `{calc_phi:.5f}`")
    st.markdown(f"• **Single Particle Area:** `{mean_part_area:.2f} px²`")
    st.markdown(f"• **Domain Area:** `{domain_area:,.0f} px²`")

# 5. Simulation Duration & Stopping Criteria
st.sidebar.subheader("⏱️ Simulation Time & Video Pace")
stop_mode_choice = st.sidebar.selectbox(
    "Stop Condition",
    [
        "Percolation: When clusters = 1 (or max steps reached)",
        "Fixed Timesteps: Run full duration",
        "Target Cluster Count"
    ]
)

if stop_mode_choice.startswith("Percolation"):
    stop_mode = "percolation"
    target_k = 1
elif stop_mode_choice.startswith("Fixed"):
    stop_mode = "fixed"
    target_k = 1
else:
    stop_mode = "target"
    target_k = st.sidebar.number_input("Target Cluster Count (Stop when ≤ K)", min_value=1, max_value=500, value=5)

max_time_steps = st.sidebar.number_input("Max Time Steps", min_value=200, max_value=100000, value=10000, step=2000)
store_every_k = st.sidebar.number_input("Trajectory Store Interval (k)", min_value=1, max_value=500, value=20, step=5,
                                        help="Lower values save more frames, producing smoother and longer videos.")

# Video pace control
video_fps = st.sidebar.selectbox(
    "Video Playback Speed",
    [10, 15, 20, 30],
    index=1,
    format_func=lambda x: f"{x} FPS" + (" (Slow Motion)" if x == 10 else (" (Recommended • Smooth & Clear)" if x == 15 else (" (Fast)" if x == 30 else "")))
)

expected_frames = (max_time_steps // store_every_k) + 60
expected_duration_s = expected_frames / max(1, video_fps)
st.sidebar.caption(f"Estimated Video Duration: **~{expected_duration_s:.1f}s** ({expected_frames} frames at {video_fps} FPS)")

seed_input = st.sidebar.number_input("Random Seed (0 = Random)", min_value=0, max_value=999999, value=42)
chosen_seed = None if seed_input == 0 else int(seed_input)


# ==============================================================================
# MAIN PAGE DASHBOARD
# ==============================================================================

st.markdown('<div class="main-header">DLCA Simulation Studio <span style="font-size:1.1rem; vertical-align:middle; background:#0984e3; color:white; padding:3px 10px; border-radius:12px;">Experimental Cancel Mode</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Diffusion-Limited Cluster Aggregation Laboratory • Live Progress & Full Kinetics • Standard Scientific Output • Video Studio</div>', unsafe_allow_html=True)

if 'latest_run' not in st.session_state:
    st.session_state.latest_run = None

tab_sim, tab_results, tab_video, tab_history = st.tabs([
    "🚀 Simulation Studio",
    "🖼️ Structure & Plots",
    "🎬 Video Studio",
    "📁 Run History & Data"
])

# ------------------------------------------------------------------------------
# TAB 1: SIMULATION STUDIO (EXPERIMENTAL CANCEL & PRESERVATION MODE)
# ------------------------------------------------------------------------------
with tab_sim:
    st.subheader("Simulation Setup")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Geometry", f"{shape_type}", f"{dispersion}")
    c2.metric("Arena Size", f"{box_w} × {box_h}", f"{domain_area:,.0f} px²")
    c3.metric("Particle Count", f"{calc_n:,}")
    c4.metric("Area Fraction (φ)", f"{calc_phi:.4f}")

    st.markdown("---")

    # Session State tracking for cancellation
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    btn_col1, btn_col2, _ = st.columns([2, 1, 1])
    start_btn = btn_col1.button(
        "▶ Launch DLCA Simulation",
        type="primary",
        use_container_width=True
    )

    with btn_col2:
        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: transparent;
          }}
          .stop-btn {{
            width: 100%;
            height: 38px;
            background: linear-gradient(135deg, #ff4757, #ee5253);
            color: #ffffff;
            border: 1px solid #ee5253;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.15s ease-in-out;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            outline: none;
            user-select: none;
          }}
          .stop-btn:hover {{
            background: linear-gradient(135deg, #ee5253, #ff3838);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(255, 71, 87, 0.35);
          }}
          .stop-btn:active {{
            transform: translateY(1px);
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
          }}
          .stop-btn.stopping {{
            background: #57606f !important;
            border-color: #57606f !important;
            cursor: wait !important;
            opacity: 0.85;
          }}
        </style>
        </head>
        <body>
          <button id="stopBtn" class="stop-btn" onclick="doStop()">
            <span>⏹</span> <span id="btnTxt">Stop Simulation</span>
          </button>
          <script>
            function doStop() {{
              var b = document.getElementById('stopBtn');
              var t = document.getElementById('btnTxt');
              b.classList.add('stopping');
              t.innerText = 'Stopping & Saving...';
              fetch('http://127.0.0.1:{STOP_PORT}/stop', {{ method: 'POST' }})
                .then(function(r) {{ console.log('DLCA Stop Signal Delivered'); }})
                .catch(function(e) {{ console.error('DLCA Stop Signal Err:', e); }});
            }}
          </script>
        </body>
        </html>
        """, height=42)

    if start_btn:
        backend.reset_stop()
        st.session_state.is_running = True
        st.session_state.stop_requested = False
        st.session_state.latest_run = None  # Clean slate: clears previous graphs/data

        config = {
            "shape_type": shape_type,
            "dispersion": dispersion,
            "params": params,
            "width": box_w,
            "height": box_h,
            "n_particles": calc_n,
            "max_time": max_time_steps,
            "store_every_k": store_every_k,
            "seed": chosen_seed,
            "stop_mode": stop_mode,
            "target_clusters": target_k,
            "fps": video_fps
        }

        progress_bar = st.progress(0, text="Initializing arena and placing particles...")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        stat_step = m_col1.empty()
        stat_clusters = m_col2.empty()
        stat_speed = m_col3.empty()
        stat_eta = m_col4.empty()

        st.markdown("##### 📈 Live Cluster Agglomeration Kinetics")
        chart_placeholder = st.empty()
        status_box = st.empty()

        stat_step.metric("Timestep", f"0 / {max_time_steps:,}")
        stat_clusters.metric("Active Clusters", f"{calc_n:,}")
        stat_speed.metric("Speed", "Initializing...")
        stat_eta.metric("Estimated ETA", "Calculating...")

        last_seen_info = {}

        def live_callback(info):
            # 1. Instant Check: Has user clicked Stop button or aborted?
            if backend.is_stop_requested() or st.session_state.get("stop_requested", False):
                raise KeyboardInterrupt("Simulation stopped by user request.")

            phase = info.get("phase", "running")
            last_seen_info.update(info)

            if phase == "rendering_video":
                cur_f = info.get("frame", 0)
                tot_f = info.get("total_frames", 1)
                r_ratio = info.get("progress_ratio", 0.0)
                progress_bar.progress(
                    min(1.0, max(0.0, r_ratio)),
                    text=f"🎬 Rendering MP4 Video: Frame {cur_f}/{tot_f} ({r_ratio*100:.1f}%)... Final plots & data saved!"
                )
                status_box.info(f"🎬 **Encoding simulation video** (Frame {cur_f}/{tot_f}). High-resolution plots and data are already saved.")
            elif phase == "simulation_done":
                progress_bar.progress(100, text="Aggregation completed! Now rendering MP4 animation...")
                stat_speed.metric("Status", "Complete")
                stat_eta.metric("Estimated ETA", "0.0 s")
                status_box.info("✓ Aggregation completed. Generating final MP4 animation...")
            else:
                step = info["step"]
                max_s = info["max_steps"]
                ratio = info["progress_ratio"]
                cids = info["clusters"]
                speed = info["speed_sps"]
                eta = info["eta_seconds"]

                progress_bar.progress(
                    min(1.0, max(0.0, ratio)),
                    text=f"Simulating Brownian aggregation: Step {step:,}/{max_s:,} ({ratio*100:.1f}%)"
                )

                stat_step.metric("Timestep", f"{step:,} / {max_s:,}")
                stat_clusters.metric("Active Clusters", f"{cids:,}", delta=f"-{calc_n - cids} merged" if calc_n > cids else None)
                stat_speed.metric("Speed", f"{speed:.1f} step/s")
                
                if eta < 60:
                    stat_eta.metric("Estimated ETA", f"{eta:.1f} s")
                else:
                    stat_eta.metric("Estimated ETA", f"{eta / 60.0:.1f} min")

                # Properly scaled Altair kinetics chart covering 0 to N_initial on Y and 0 to max_time on X
                if "full_history_steps" in info and len(info["full_history_steps"]) >= 2:
                    df_trend = pd.DataFrame({
                        "Timestep": info["full_history_steps"],
                        "Active Clusters": info["full_history_clusters"]
                    })
                    chart = alt.Chart(df_trend).mark_line(color='#00cec9', strokeWidth=2.5).encode(
                        x=alt.X('Timestep:Q', scale=alt.Scale(domain=[0, max_s]), title='Timestep (t)'),
                        y=alt.Y('Active Clusters:Q', scale=alt.Scale(domain=[0, calc_n * 1.05]), title='Cluster Count (Nc)')
                    ).properties(height=220)
                    chart_placeholder.altair_chart(chart, use_container_width=True)

        try:
            status_box.info(f"Running DLCA physics engine for {shape_type} (N={calc_n:,}, φ={calc_phi:.4f})...")
            res = backend.execute_simulation(config, progress_callback=live_callback)
            
            st.session_state.latest_run = res
            if res.get("stopped_early") or res.get("stopped_by_user"):
                progress_bar.progress(100, text="Simulation stopped early! Data and scientific plots safely preserved.")
                status_box.warning("⏹ **Simulation stopped by user.** Intermediate particles, trajectory arrays, and scientific plots have been saved to disk.")
            else:
                progress_bar.progress(100, text="Simulation & Video Rendering Complete!")
                status_box.success(
                    f"✓ Simulation completed in {res['elapsed_seconds']:.2f}s! "
                    f"Placed: {res['placed_particles']} | Final Clusters: {res['final_clusters']} "
                    f"| Output saved to folder: {res['run_dir']}"
                )
                st.balloons()
        except (KeyboardInterrupt, Exception) as ex:
            is_cancel = isinstance(ex, KeyboardInterrupt) or "stopped by user" in str(ex).lower()
            if is_cancel:
                status_box.warning("⏹ **Simulation stopped by user.** Intermediate particles, trajectory, and scientific plots have been saved to disk.")
                # Retrieve the active run dir from last callback or scan latest folder
                run_dir_cand = last_seen_info.get("run_dir")
                if not run_dir_cand:
                    recent = backend.list_saved_runs()
                    run_dir_cand = recent[0]["run_dir"] if recent else ""
                
                if run_dir_cand and os.path.exists(run_dir_cand):
                    summary_path = os.path.join(run_dir_cand, "summary.json")
                    saved_summary = {}
                    if os.path.exists(summary_path):
                        with open(summary_path, "r") as sf:
                            saved_summary = json.load(sf)
                    else:
                        saved_summary = {
                            "run_tag": os.path.basename(run_dir_cand),
                            "run_dir": run_dir_cand,
                            "shape_type": shape_type,
                            "dispersion": dispersion,
                            "placed_particles": calc_n,
                            "final_clusters": last_seen_info.get("clusters", calc_n),
                            "elapsed_seconds": last_seen_info.get("elapsed_seconds", 0.0),
                            "snapshot_path": os.path.join(run_dir_cand, "final_structure.png"),
                            "decay_plot_path": os.path.join(run_dir_cand, "cluster_decay.png"),
                            "video_path": ""
                        }
                    saved_summary["stopped_early"] = True
                    st.session_state.latest_run = saved_summary
            else:
                status_box.error(f"Simulation error: {ex}")
                st.exception(ex)
        finally:
            st.session_state.is_running = False
            st.session_state.stop_requested = False

    # Latest Run Display: Plots & Video Side-by-Side
    if st.session_state.latest_run:
        r = st.session_state.latest_run
        st.markdown("### 🏆 Simulation Results & Artifacts")
        if r.get("stopped_early"):
            st.info("ℹ️ **Run was halted early.** Showing intermediate cluster structure and data preserved up to the stop point.")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Placed Particles", f"{r['placed_particles']:,} / {r.get('requested_particles', r['placed_particles']):,}")
        m2.metric("Final Clusters", f"{r['final_clusters']:,}")
        m3.metric("Runtime", f"{r['elapsed_seconds']:.2f} s")
        m4.metric("Fractal Dimension (Df)", f"{r['Df']:.3f}" if not math.isnan(r.get('Df', float('nan'))) else "N/A")

        st.info(f"📂 **Run artifacts folder:** `{r['run_dir']}`")

        # Instant display of the Final Plot and Video Player side-by-side
        col_plot, col_video = st.columns(2)

        snap_p = r.get("snapshot_path", "")
        vid_p = r.get("video_path", "")

        with col_plot:
            st.markdown("#### 🖼️ Final Aggregation Structure")
            if snap_p and os.path.exists(snap_p):
                st.image(snap_p, caption=f"Cluster State: {r['shape_type']} (N = {r['placed_particles']}, Clusters = {r['final_clusters']})", use_column_width=True)
                with open(snap_p, "rb") as sf:
                    st.download_button(
                        label="📥 Download Final Structure PNG",
                        data=sf,
                        file_name=f"dlca_{r['shape_type']}_structure.png",
                        mime="image/png",
                        use_container_width=True,
                        key="btn_dl_structure_tab1_exp"
                    )
            else:
                st.warning("Snapshot not found.")

        with col_video:
            st.markdown("#### 🎬 Simulation Video Playback")
            if vid_p and os.path.exists(vid_p):
                st.video(vid_p)
                with open(vid_p, "rb") as vf:
                    st.download_button(
                        label="📥 Download Simulation MP4 Video",
                        data=vf,
                        file_name=f"dlca_{r['shape_type']}_video.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key="btn_dl_video_tab1_exp"
                    )
            else:
                st.info("Video was not automatically rendered (or simulation stopped early).")
                if st.button("🎬 Generate Video for Saved Trajectory Now", key="gen_vid_main_exp"):
                    with st.spinner("Rendering MP4 video at 15 FPS from saved trajectory..."):
                        v_out = backend.render_simulation_video(r["run_dir"], fps=video_fps, shape_type=r.get("shape_type"))
                        r["video_path"] = v_out
                        st.session_state.latest_run = r
                        st.rerun()


# ------------------------------------------------------------------------------
# TAB 2: STRUCTURE & PLOTS
# ------------------------------------------------------------------------------
with tab_results:
    st.subheader("Scientific Plots & Analysis")

    if st.session_state.latest_run:
        r = st.session_state.latest_run
        snap_p = r.get("snapshot_path", "")
        decay_p = r.get("decay_plot_path", "")
        fractal_p = r.get("fractal_plot_path", "")

        c_p1, c_p2 = st.columns(2)

        with c_p1:
            st.markdown("#### Final Spatial Agglomerate")
            if snap_p and os.path.exists(snap_p):
                st.image(snap_p, caption=f"Final Cluster State: {r['shape_type']} | N={r['placed_particles']}", use_column_width=True)
                with open(snap_p, "rb") as sf:
                    st.download_button(
                        label="📥 Download Plot PNG",
                        data=sf,
                        file_name=f"dlca_{r['shape_type']}_structure.png",
                        mime="image/png",
                        use_container_width=True,
                        key="btn_dl_structure_tab2"
                    )

        with c_p2:
            st.markdown("#### Cluster Decay Kinetics")
            if decay_p and os.path.exists(decay_p):
                st.image(decay_p, caption="Active Clusters vs Timestep (Full Scale)", use_column_width=True)
                with open(decay_p, "rb") as df:
                    st.download_button(
                        label="📥 Download Decay PNG",
                        data=df,
                        file_name="cluster_decay.png",
                        mime="image/png",
                        use_container_width=True,
                        key="btn_dl_decay_tab2"
                    )

            if fractal_p and os.path.exists(fractal_p):
                st.image(fractal_p, caption="Fractal Dimension Fitting", use_column_width=True)

        st.markdown("---")
        meta_p = os.path.join(r["run_dir"], "summary.json")
        if os.path.exists(meta_p):
            with open(meta_p, "rb") as mf:
                st.download_button(
                    label="📥 Download Run Summary JSON",
                    data=mf,
                    file_name="summary.json",
                    mime="application/json",
                    key="btn_dl_summary_tab2"
                )
    else:
        st.info("Run a simulation in 'Simulation Studio' to inspect plots.")


# ------------------------------------------------------------------------------
# TAB 3: VIDEO STUDIO
# ------------------------------------------------------------------------------
with tab_video:
    st.subheader("Trajectory Animation & Video Studio")

    if st.session_state.latest_run:
        r = st.session_state.latest_run
        run_d = r["run_dir"]
        vid_p = r.get("video_path", os.path.join(run_d, "simulation_video.mp4"))

        c_fps, c_frames, c_render = st.columns([1, 1, 2])
        fps_choice = c_fps.selectbox("Framerate (FPS)", [10, 15, 20, 30], index=1,
                                     format_func=lambda x: f"{x} FPS" + (" (Slow Motion)" if x == 10 else (" (Recommended)" if x == 15 else "")))
        max_f_choice = c_frames.selectbox("Max Frames", [300, 500, 800, 1200], index=2)
        render_btn = c_render.button("🎬 Re-render MP4 Video", type="primary", use_container_width=True)

        if render_btn:
            prog_v = st.progress(0, text="Rendering video frames via FFmpeg...")
            t_v0 = time.time()
            try:
                def vid_cb(cur, total):
                    prog_v.progress(min(1.0, cur / total), text=f"Rendering video: Frame {cur}/{total}...")

                out_v = backend.render_simulation_video(
                    run_d, output_mp4=os.path.join(run_d, "simulation_video.mp4"),
                    fps=fps_choice, max_frames=max_f_choice, progress_callback=vid_cb,
                    shape_type=r.get("shape_type")
                )
                r["video_path"] = out_v
                st.session_state.latest_run = r
                prog_v.progress(100, text="Video rendered successfully!")
                st.success(f"✓ Video successfully rendered in {time.time() - t_v0:.1f}s!")
            except Exception as ex:
                st.error(f"Video render failed: {ex}")

        if os.path.exists(vid_p):
            st.markdown("#### Video Player")
            st.video(vid_p)

            with open(vid_p, "rb") as vf:
                st.download_button(
                    label="📥 Download Simulation MP4 Video",
                    data=vf,
                    file_name=f"dlca_{r['shape_type']}_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key="btn_dl_video_tab3"
                )
        else:
            st.info("No video rendered yet. Click 'Re-render MP4 Video' above.")
    else:
        st.info("Run a simulation first to generate trajectory data for video rendering.")


# ------------------------------------------------------------------------------
# TAB 4: RUN HISTORY & SAVED RUNS
# ------------------------------------------------------------------------------
with tab_history:
    st.subheader("Saved Simulation Runs Explorer")

    saved_runs = backend.list_saved_runs()

    if saved_runs:
        run_tags = [r["run_tag"] for r in saved_runs]
        selected_tag = st.selectbox("Select Past Simulation Run", run_tags, index=0)

        sel_run = next(r for r in saved_runs if r["run_tag"] == selected_tag)

        st.markdown(f"**Selected Run Folder:** `{sel_run['run_dir']}`")

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Geometry", sel_run.get("shape_type", "Unknown"))
        c_b.metric("Particles", f"{sel_run.get('placed_particles', 'N/A')}")
        c_c.metric("Final Clusters", f"{sel_run.get('final_clusters', 'N/A')}")

        hist_snap = sel_run.get("snapshot_path", os.path.join(sel_run["run_dir"], "final_structure.png"))
        hist_vid = sel_run.get("video_path", os.path.join(sel_run["run_dir"], "simulation_video.mp4"))
        hist_decay = os.path.join(sel_run["run_dir"], "cluster_decay.png")

        col1, col2 = st.columns(2)
        if os.path.exists(hist_snap):
            col1.image(hist_snap, caption="Final Structure Snapshot", use_column_width=True)
        if os.path.exists(hist_decay):
            col2.image(hist_decay, caption="Cluster Decay Kinetics", use_column_width=True)

        if os.path.exists(hist_vid):
            st.markdown("#### Stored Simulation Video")
            st.video(hist_vid)
    else:
        st.info("No saved runs found in `gui_runs/` yet.")

    st.markdown("---")
    st.subheader("Master Sweep Manifests")
    sweep_sel = st.selectbox("Select Master Dataset", ["Equilateral Triangles Sweep", "Circles Sweep"])

    csv_path = os.path.join("Data points", "triangles" if "Triangles" in sweep_sel else "circles", "manifest.csv")
    if os.path.exists(csv_path):
        df_sw = pd.read_csv(csv_path)
        st.dataframe(df_sw, use_container_width=True)
        with open(csv_path, "rb") as cf:
            st.download_button("📥 Download Manifest CSV", data=cf, file_name="manifest.csv", mime="text/csv", key="btn_dl_manifest_tab4")
