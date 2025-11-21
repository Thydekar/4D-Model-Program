import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

st.set_page_config(page_title="4D Explorer", layout="wide")
st.title("Higher-Dimensional Geometry Explorer")

# Sidebar mode selection
mode = st.sidebar.radio(
    "Mode",
    ["3D Model", "4D Model of 3D", "4D Model"],
    index=0
)

# Shared state for 3D points
if 'points_3d' not in st.session_state:
    # Start with regular tetrahedron
    t = (1 + np.sqrt(5)) / 2
    st.session_state.points_3d = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / 2.0

# Helper: make points equidistant (regular simplex projection)
def make_equidistant(points):
    n = len(points)
    center = points.mean(axis=0)
    centered = points - center
    dists = np.linalg.norm(centered, axis=1)
    target = dists.mean()
    centered = centered / dists[:, None] * target
    return centered + center

# ================ MODE 1: 3D Model ================
if mode == "3D Model":
    st.header("3D Model Builder")
    n_points = st.slider("Number of vertices", 3, 8, 4)

    if len(st.session_state.points_3d) != n_points:
        # Generate reasonable starting positions
        if n_points == 4:
            st.session_state.points_3d = np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
            ]) / 2.0
        elif n_points == 6:
            st.session_state.points_3d = np.array([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ])
        else:
            np.random.seed(42)
            st.session_state.points_3d = np.random.randn(n_points, 3) * 0.8

    # Snap to equidistant
    if st.button("Snap to Equidistant (Regular Simplex Projection)"):
        st.session_state.points_3d = make_equidistant(st.session_state.points_3d)
        st.success("Snapped to equidistant configuration!")

    # Interactive 3D scatter
    fig = go.Figure()
    pts = st.session_state.points_3d

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers+text',
        marker=dict(size=10, color=list(range(len(pts))), colorscale='Viridis'),
        text=[f"P{i}" for i in range(len(pts))],
        textposition="top center"
    ))

    # Connect all edges
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            fig.add_trace(go.Scatter3d(
                x=[pts[i, 0], pts[j, 0]],
                y=[pts[i, 1], pts[j, 1]],
                z=[pts[i, 2], pts[j, 2]],
                mode='lines',
                line=dict(color='white', width=3),
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        paper_bgcolor="#0e1117",
        scene_bgcolor="#0e1117",
        font_color="white",
        height=700
    )

    # Let user drag points!
    edited = st.plotly_chart(fig, use_container_width=True, key="editable_3d")
    if edited and edited["data"]:
        new_pts = []
        for trace in edited["data"]:
            if trace["mode"] == "markers+text":
                new_pts = np.array([trace["x"], trace["y"], trace["z"]]).T
                break
        if len(new_pts) == len(pts):
            st.session_state.points_3d = new_pts

# ================ MODE 2: 4D Model of 3D ================
elif mode == "4D Model of 3D":
    st.header("4D Projection of Your 3D Model")
    st.info("Your 3D shape is now one vertex pulled forward in the 4th dimension. The rest form a tetrahedral 'shadow' behind.")

    pts3d = st.session_state.points_3d
    center = pts3d.mean(axis=0)
    scale = np.linalg.norm(pts3d - center, axis=1).mean() * 1.5

    # 4D coordinates: one point at (0,0,0,1), others at (x,y,z,-0.3)
    central_4d = np.array([0, 0, 0, 1.2])
    base_4d = np.hstack([pts3d, -0.3 * np.ones((len(pts3d), 1))])

    # Perspective projection from 4D → 3D (w=4)
    def project_4d_to_3d(p4, distance=4.0):
        x, y, z, w = p4
        factor = distance / (distance - w)
        return np.array([x * factor, y * factor, z * factor])

    central_3d = project_4d_to_3d(central_4d)
    base_3d_proj = np.array([project_4d_to_3d(p) for p in base_4d])

    fig = go.Figure()

    # Central point (popping out)
    fig.add_trace(go.Scatter3d(
        x=[central_3d[0]], y=[central_3d[1]], z=[central_3d[2]],
        mode='markers',
        marker=dict(size=14, color='cyan'),
        name="4D Vertex"
    ))

    # Base tetrahedron
    fig.add_trace(go.Scatter3d(
        x=base_3d_proj[:, 0], y=base_3d_proj[:, 1], z=base_3d_proj[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='orange'),
        text=[f"H{i}" for i in range(len(base_3d_proj))],
        name="3D Shadow"
    ))

    # Connect central to all
    for p in base_3d_proj:
        fig.add_trace(go.Scatter3d(
            x=[central_3d[0], p[0]],
            y=[central_3d[1], p[1]],
            z=[central_3d[2], p[2]],
            mode='lines',
            line=dict(color='cyan', width=5),
            showlegend=False
        ))

    # Connect base edges
    for i in range(len(base_3d_proj)):
        for j in range(i + 1, len(base_3d_proj)):
            fig.add_trace(go.Scatter3d(
                x=[base_3d_proj[i, 0], base_3d_proj[j, 0]],
                y=[base_3d_proj[i, 1], base_3d_proj[j, 1]],
                z=[base_3d_proj[i, 2], base_3d_proj[j, 2]],
                mode='lines',
                line=dict(color='white', width=2),
                showlegend=False
            ))

    # Dark cone shadow
    hull = ConvexHull(base_3d_proj[:, :3])
    for simplex in hull.simplices:
        tri = base_3d_proj[simplex]
        fig.add_trace(go.Mesh3d(
            x=np.concatenate([tri[:, 0], [central_3d[0]]]),
            y=np.concatenate([tri[:, 1], [central_3d[1]]]),
            z=np.concatenate([tri[:, 2], [central_3d[2]]]),
            i=[0, 0, 0, 1], j=[1, 2, 1, 2], k=[2, 1, 3, 3],
            color='rgba(50,50,70,0.8)',
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
        ),
        paper_bgcolor="#0e1117",
        scene_bgcolor="#0e1117",
        font_color="white",
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

# ================ MODE 3: 4D Model ================
else:
    st.header("Regular 4D Polytopes")
    polytope = st.selectbox("Choose a 4D shape", [
        "5-cell (Pentachoron)",
        "8-cell (Tesseract)",
        "16-cell (Hexadecachoron)",
        "24-cell (Octacube)",
        "120-cell (Projected)",
        "600-cell (Projected)"
    ])

    # Simple 5-cell
    if "5-cell" in polytope:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [1, 1, 1, -1/np.sqrt(5)],
            [1, -1, -1, -1/np.sqrt(5)],
            [-1, 1, -1, -1/np.sqrt(5)],
            [-1, -1, 1, -1/np.sqrt(5)],
            [0, 0, 0, np.sqrt(5)-1]
        ]) / 2.0
        verts[:, :3] *= 0.8

        distance = st.slider("Projection distance", 2.0, 10.0, 5.0)

        def proj(p):
            factor = distance / (distance - p[3])
            return p[:3] * factor

        proj_verts = np.array([proj(v) for v in verts])

        fig = go.Figure()
        colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange']
        for i, pt in enumerate(proj_verts):
            fig.add_trace(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode='markers',
                marker=dict(size=12, color=colors[i]),
                name=f"V{i}"
            ))

        # All edges in 5-cell
        edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        for i, j in edges:
            fig.add_trace(go.Scatter3d(
                x=[proj_verts[i,0], proj_verts[j,0]],
                y=[proj_verts[i,1], proj_verts[j,1]],
                z=[proj_verts[i,2], proj_verts[j,2]],
                mode='lines',
                line=dict(color='white', width=4),
                showlegend=False
            ))

        fig.update_layout(
            scene=dict(aspectmode='cube', camera=dict(eye=dict(x=2, y=2, z=1.5))),
            paper_bgcolor="#0e1117", scene_bgcolor="#0e1117", font_color="white",
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)

    # You can expand this with tesseract, etc. later!
    else:
        st.info("More 4D polytopes coming soon! 5-cell is fully interactive for now.")
