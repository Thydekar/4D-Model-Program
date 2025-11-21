import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Tetrahedral 4D Explorer", layout="wide")
st.title("🔮 Tetrahedral 4D Coordinate System")

# ====================== TETRAHEDRAL BASIS (4 axes at 109.5°) ======================
@st.cache_data
def tetrahedral_basis():
    # Four vectors from center to tetrahedron vertices, all unit length, 109.5° apart
    a = 1.0
    v1 = np.array([ 1,  1,  1])
    v2 = np.array([ 1, -1, -1])
    v3 = np.array([-1,  1, -1])
    v4 = np.array([-1, -1,  1])
    basis = np.array([v1, v2, v3, v4])
    basis = basis / np.linalg.norm(basis[0])
    return basis  # shape (4,3)

basis = tetrahedral_basis()
axis_colors = ["cyan", "magenta", "yellow", "lime"]

# ====================== 4D → 3D via Tetrahedral Embedding ======================
def embed_4d_to_3d(point_4d, scale=1.0):
    """point_4d: [x,y,z,w] → 3D point using tetrahedral basis"""
    return scale * (point_4d[0]*basis[0] + point_4d[1]*basis[1] + 
                    point_4d[2]*basis[2] + point_4d[3]*basis[3])

# ====================== Generate Polytopes ======================
@st.cache_data
def get_polytope(name):
    if name == "5-cell (Pentachoron)":
        # Standard 5-cell in 4D
        phi = (1 + np.sqrt(5))/2
        s = 1 / np.sqrt(8)
        verts = np.array([
            [ s,  s,  s, -1.25*s],
            [ s, -s, -s, -1.25*s],
            [-s,  s, -s, -1.25*s],
            [-s, -s,  s, -1.25*s],
            [0,  0,  0,  np.sqrt(5)*s*2]
        ])
        edges = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        return verts, edges

    elif name == "Tesseract (4D Cube)":
        verts = np.array([[x,y,z,w] for x in [-1,1] for y in [-1,1] 
                                        for z in [-1,1] for w in [-1,1]]) * 0.6
        edges = []
        for i in range(16):
            p = verts[i]
            for j in range(i+1,16):
                if np.sum(np.abs(p - verts[j])) == 2:
                    edges.append((i,j))
        return verts, edges

    elif name == "16-cell":
        verts = np.array([[±1,0,0,0], [0,±1,0,0], [0,0,±1,0], [0,0,0,±1]]) * 1.2
        verts = verts.reshape(-1,4)
        edges = [(i,j) for i in range(8) for j in range(i+1,8) 
                        if np.sum(np.abs(verts[i] - verts[j])) == 2]
        return verts, edges

    elif name == "24-cell":
        verts = np.concatenate([
            np.array([[±1,±1,0,0] for _ in range(4)] + 
                     [[±1,0,±1,0] for _ in range(4)] + 
                     [[±1,0,0,±1] for _ in range(4)] +
                     [[0,±1,±1,0] for _ in range(4)] +
                     [[0,±1,0,±1] for _ in range(4)] +
                     [[0,0,±1,±1] for _ in range(4)]) * 0.7,
            np.array([[±0.5,±0.5,±0.5,±0.5]]) * 8
        ])
        # Simplified edges: connect if distance == √2
        edges = []
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if 1.9 < np.linalg.norm(verts[i] - verts[j]) < 2.1:
                    edges.append((i,j))
        return verts, edges

    else:  # Random points on 4-sphere
        np.random.seed(42)
        verts = np.random.randn(20, 4)
        verts /= np.linalg.norm(verts, axis=1)[:, None]
        verts *= 0.8
        edges = []
        return verts, edges

# ====================== SIDEBAR ======================
mode = st.sidebar.radio("Mode", ["Tetrahedral 4D Graph", "3D Editor → 4D"])

if mode == "3D Editor → 4D":
    st.header("Drag 3D Points → See in 4D Tetrahedral View")
    if "pts3d" not in st.session_state:
        st.session_state.pts3d = basis[:4].copy() * 1.2

    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=st.session_state.pts3d[:,0], y=st.session_state.pts3d[:,1], z=st.session_state.pts3d[:,2],
                                 mode="markers+text", marker=dict(size=12, color="white"), text=[f"P{i}" for i in range(4)]))
    for i in range(4):
        for j in range(i+1,4):
            fig3d.add_trace(go.Scatter3d(x=[st.session_state.pts3d[i,0], st.session_state.pts3d[j,0]],
                                         y=[st.session_state.pts3d[i,1], st.session_state.pts3d[j,1]],
                                         z=[st.session_state.pts3d[i,2], st.session_state.pts3d[j,2]],
                                         mode="lines", line=dict(color="gray", width=3), showlegend=False))
    fig3d.update_layout(scene=dict(aspectmode='cube', camera=dict(eye=dict(x=2,y=2,z=1))),
                        paper_bgcolor="#111", scene_bgcolor="#111", font_color="white", height=500)
    chart = st.plotly_chart(fig3d, use_container_width=True, key="drag")
    if chart.data:
        new_pts = np.column_stack([chart.data[0].x, chart.data[0].y, chart.data[0].z])
        if new_pts.shape == (4,3):
            st.session_state.pts3d = new_pts

    # Show in 4D view
    pts4d = np.zeros((4,4))
    pts4d[:,:3] = st.session_state.pts3d
    pts4d[:,3] = 0
else:
    st.header("True 4D Graph with Tetrahedral Axes")
    polytope_name = st.selectbox("Select 4D Object", [
        "5-cell (Pentachoron)",
        "Tesseract (4D Cube)",
        "16-cell",
        "24-cell",
        "Random 4D Points"
    ])
    verts4, edges = get_polytope(polytope_name)
    scale = st.slider("Scale", 0.5, 3.0, 1.2, 0.1)
    show_grid = st.checkbox("Show Tetrahedral Grid", True)
    show_axes = st.checkbox("Show 4D Axes", True)

# ====================== MAIN 4D PLOT ======================
fig = go.Figure()

# --- Grid lines (contour-like in tetrahedral directions) ---
if show_grid:
    grid_size = 5
    for i in range(4):
        for val in np.linspace(-1, 1, grid_size):
            if abs(val) < 1e-6: continue
            line = np.zeros((100,4))
            line[:, i] = np.linspace(-1.2, 1.2, 100) * scale
            line[:, (i+1)%4] = val * scale
            pts3d = np.array([embed_4d_to_3d(p, scale) for p in line])
            fig.add_trace(go.Scatter3d(x=pts3d[:,0], y=pts3d[:,1], z=pts3d[:,2],
                                       mode="lines", line=dict(color="#333", width=1), showlegend=False))

# --- Four bent axes ---
if show_axes:
    for i in range(4):
        axis_4d = np.zeros((2,4))
        axis_4d[1,i] = 1.8 * scale
        pts = np.array([embed_4d_to_3d(p, scale) for p in axis_4d])
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                   mode="lines", line=dict(color=axis_colors[i], width=8),
                                   name=f"Axis {i}", showlegend=True))
        # Label
        fig.add_trace(go.Scatter3d(x=[pts[1,0]], y=[pts[1,1]], z=[pts[1,2]],
                                   mode="text", text=[f"<b>w{i}</b>"], textposition="top center",
                                   textfont=dict(color=axis_colors[i], size=16), showlegend=False))

# --- Plot the 4D object ---
if mode == "3D Editor → 4D":
    verts_to_plot = pts4d
    edges_to_plot = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
else:
    verts_to_plot = verts4
    edges_to_plot = edges

embedded = np.array([embed_4d_to_3d(v, scale) for v in verts_to_plot])

# Vertices
fig.add_trace(go.Scatter3d(x=embedded[:,0], y=embedded[:,1], z=embedded[:,2],
                           mode="markers", marker=dict(size=8, color="white"), name="Vertices"))

# Edges
for i, j in edges_to_plot:
    if i >= len(embedded) or j >= len(embedded): continue
    fig.add_trace(go.Scatter3d(x=[embedded[i,0], embedded[j,0]],
                               y=[embedded[i,1], embedded[j,1]],
                               z=[embedded[i,2], embedded[j,2]],
                               mode="lines", line=dict(color="white", width=3), showlegend=False))

fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.7, y=1.7, z=1.3))
    ),
    paper_bgcolor="#0a0a0a",
    scene_bgcolor="#0a0a0a",
    font_color="white",
    height=900,
    legend=dict(y=0.9, x=0.8)
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Four axes at exact 109.5° — the true geometry of 4D space. Drag any 3D shape in → see it in real 4D coordinates.")
