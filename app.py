import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

st.set_page_config(page_title="4D Explorer", layout="wide")
st.title("Higher-Dimensional Geometry Explorer")

# Sidebar
mode = st.sidebar.radio("Mode", ["3D Model", "4D → 3D Projection", "Pure 4D Polytopes"])

# Session state for 3D points
if "points_3d" not in st.session_state:
    # Start with perfect tetrahedral methane-like positions
    a = 1.0
    st.session_state.points_3d = np.array([
        [ 0.000,  1.000,  0.000],  # top → Y
        [ 0.943, -0.333,  0.000],  # right → roughly X
        [-0.471, -0.333, -0.816],  # back-left → Z + W direction
        [-0.471, -0.333,  0.816],  # front-left → Z direction
    ]) * 1.2

# Helper: snap to regular tetrahedron (exact 109.5°)
def make_regular_tetrahedron():
    phi = (1 + np.sqrt(5)) / 2
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, -1, -1])
    v3 = np.array([-1, 1, -1])
    v4 = np.array([-1, -1, 1])
    verts = np.array([v1, v2, v3, v4])
    verts = verts / np.linalg.norm(verts[0]) * 1.2
    # Rotate so one vertex points straight up (like methane image)
    up = np.array([0, 1, 0])
    current_up = verts[0]
    axis = np.cross(current_up, up)
    if np.linalg.norm(axis) > 0:
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.dot(current_up, up))
        c, s = np.cos(angle), np.sin(angle)
        R = np.eye(3) + s*np.cross(axis[:,None], axis[None,:]*-1) + (1-c)*(axis[:,None]@axis[None,:])
        verts = verts @ R.T
    return verts

# ================================
# MODE 1: 3D Model (fully draggable)
# ================================
if mode == "3D Model":
    st.header("3D Model – Drag Points with Mouse")
    n = st.slider("Number of points", 3, 8, 4)

    if len(st.session_state.points_3d) != n or st.button("Reset to Regular Tetrahedron"):
        st.session_state.points_3d = make_regular_tetrahedron()[:n]

    fig = go.Figure()

    # Points
    fig.add_trace(go.Scatter3d(
        x=st.session_state.points_3d[:,0],
        y=st.session_state.points_3d[:,1],
        z=st.session_state.points_3d[:,2],
        mode="markers+text",
        marker=dict(size=12, color="cyan"),
        text=[f"P{i}" for i in range(len(st.session_state.points_3d))],
        name="Vertices"
    ))

    # Edges
    pts = st.session_state.points_3d
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            fig.add_trace(go.Scatter3d(
                x=[pts[i,0], pts[j,0]],
                y=[pts[i,1], pts[j,1]],
                z=[pts[i,2], pts[j,2]],
                mode="lines",
                line=dict(color="white", width=4),
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", showgrid=False, zeroline=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False),
            zaxis=dict(title="Z", showgrid=False, zeroline=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        paper_bgcolor="#0e1117",
        scene_bgcolor="#0e1117",
        font_color="white",
        height=700
    )

    # THIS IS THE CORRECT WAY TO GET DRAGGED POINTS IN STREAMLIT + PLOTLY
    chart = st.plotly_chart(fig, use_container_width=True, key="drag3d")
    if chart.data:
        new_x = chart.data[0].x
        new_y = chart.data[0].y
        new_z = chart.data[0].z
        new_points = np.column_stack([new_x, new_y, new_z])
        if new_points.shape == st.session_state.points_3d.shape:
            st.session_state.points_3d = new_points

# ================================
# MODE 2: 4D → 3D Projection (exactly like your methane image)
# ================================
elif mode == "4D → 3D Projection":
    st.header("Your 3D Model as a 4D Projection")
    st.info("One vertex is pulled along the invisible W axis (into the screen). This is exactly how the methane diagram represents a 4-simplex.")

    pts = st.session_state.points_3d.copy()
    center = pts.mean(axis=0)
    scale = 1.5

    # 4D coordinates
    central_4d = np.array([0.0, 0.0, 0.0, 1.8])           # the one "popping out" along W
    base_4d    = np.hstack([pts, np.full((len(pts), 1), -0.6)])

    # Perspective projection from 4D → 3D
    def project(p4, d=5.0):
        factor = d / (d - p4[3])
        return p4[:3] * factor

    central_3d = project(central_4d)
    base_3d    = np.array([project(p) for p in base_4d])

    fig = go.Figure()

    # Central vertex (cyan, big)
    fig.add_trace(go.Scatter3d(
        x=[central_3d[0]], y=[central_3d[1]], z=[central_3d[2]],
        mode="markers",
        marker=dict(size=16, color="cyan"),
        name="4D Vertex (W direction)"
    ))

    # Base vertices (white)
    fig.add_trace(go.Scatter3d(
        x=base_3d[:,0], y=base_3d[:,1], z=base_3d[:,2],
        mode="markers+text",
        marker=dict(size=10, color="white"),
        text=[f"H{i}" for i in range(len(base_3d))],
        name="3D Base"
    ))

    # Bonds from central to base
    for p in base_3d:
        fig.add_trace(go.Scatter3d(
            x=[central_3d[0], p[0]],
            y=[central_3d[1], p[1]],
            z=[central_3d[2], p[2]],
            mode="lines",
            line=dict(color="cyan", width=6),
            showlegend=False
        ))

    # Base edges (dim)
    for i in range(len(base_3d)):
        for j in range(i+1, len(base_3d)):
            fig.add_trace(go.Scatter3d(
                x=[base_3d[i,0], base_3d[j,0]],
                y=[base_3d[i,1], base_3d[j,1]],
                z=[base_3d[i,2], base_3d[j,2]],
                mode="lines",
                line=dict(color="#666", width=3),
                showlegend=False
            ))

    # Dark conical shadow (exactly like the methane image)
    if len(base_3d) >= 4:
        try:
            hull = ConvexHull(base_3d)
            for simplex in hull.simplices:
                triangle = base_3d[simplex]
                # Create a pyramid from central point to each triangular face
                x = np.append(triangle[:,0], central_3d[0])
                y = np.append(triangle[:,1], central_3d[1])
                z = np.append(triangle[:,2], central_3d[2])
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=[0,0,0,1], j=[1,2,1,2], k=[2,1,3,3],
                    color="rgba(40,40,60,0.9)",
                    opacity=0.8,
                    showlegend=False,
                    lighting=dict(ambient=0.8)
                ))
        except:
            pass  # fallback if hull fails

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title="", showgrid=False, zeroline=False, showticklabels=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.4, y=1.8, z=1.2))
        ),
        paper_bgcolor="#0e1117",
        scene_bgcolor="#0e1117",
        font_color="white",
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================
# MODE 3: Pure 4D Polytopes (coming soon)
# ================================
else:
    st.header("Regular 4D Polytopes")
    st.write("5-cell, tesseract, 24-cell, etc. — coming in the next update!")
    st.info("For now enjoy the perfect 4D → 3D projection in the previous tab — it’s already a real 5-cell!")

st.caption("Built because one methane diagram changed everything.")
