import numpy as np
from scipy.stats import multivariate_normal
import plotly.graph_objects as go
import webbrowser
from pathlib import Path

"""
3D Visualization of reverse diffusion process.
"""

def random_covs(std_min, std_max, num_covs=5):
    covs = []
    for _ in range(num_covs):
        # Random eigenvalues (variances)
        eigvals = np.diag([rand(std_min, std_max), rand(std_min, std_max)])
        # Random rotation matrix
        theta = np.random.uniform(0, 2*np.pi)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        # Construct covariance
        cov = R @ eigvals @ R.T
        covs.append(cov)
    return covs

def p(x, y):
    pos = np.dstack((x, y))
    return sum(c.pdf(pos) for c in components) / len(components)

def score(x, y, eps=1e-4):
    dpdx = (np.log(p(x + eps, y)) - np.log(p(x - eps, y))) / (2 * eps)
    dpdy = (np.log(p(x, y + eps)) - np.log(p(x, y - eps))) / (2 * eps)
    return np.array([dpdx, dpdy])

def generate_trajectory(steps=100, eta=1.0, sigma=0.1):
    point = np.random.uniform(mu_min, mu_max, size=2)
    traj = [point.copy()]
    for _ in range(steps):
        grad = score(point[0], point[1])
        point = point + eta * grad + sigma * np.random.randn(2)
        traj.append(point.copy())
    traj = np.array(traj)
    Z_traj = p(traj[:, 0], traj[:, 1])
    return traj, Z_traj

# Defining possible centers and spreads of distributions
mu_min, mu_max = -10, 10
std_min, std_max = 3, 6

rand = np.random.randint

# Generate 5 random centers and spreads
n_dists = 5
means = [np.array([rand(mu_min, mu_max), rand(mu_min, mu_max)]) for _ in range(n_dists)]
covs = random_covs(std_min, std_max, num_covs=n_dists)

# Create 5 multivariate distributions
components = [
    multivariate_normal(mean, cov)
    for mean, cov in zip(means, covs)
]

# Initialize 2D input space (X, Y) and PDF surface Z
x = np.linspace(mu_min * 1.5, mu_max * 1.5, 100)
y = np.linspace(mu_min * 1.5, mu_max * 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = p(X, Y)

steps=150
eta=0.2 # controls gradient step strength
sigma=0.1 # controls random exploration strength

# Generate 1 random particle trajectory
traj, Z_traj = generate_trajectory(steps=steps, eta=eta, sigma=sigma)

frames = []
for t in range(1, len(traj)):
    frames.append(
        go.Frame(
            name=str(t),
            data=[
                # update trace 1 (trajectory line)
                go.Scatter3d(
                    x=traj[:t+1, 0],
                    y=traj[:t+1, 1],
                    z=Z_traj[:t+1]
                ),

                # update trace 2 (current point)
                go.Scatter3d(
                    x=[traj[t, 0]],
                    y=[traj[t, 1]],
                    z=[Z_traj[t]]
                )
            ],
            traces=[1, 2]
        )
    )


fig = go.Figure(
    data=[
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Viridis",
            opacity=0.85,
            showscale=False
        ),

        go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[Z_traj[0]],
            mode="lines",
            line=dict(color="red", width=6),
            name="Trajectory"
        ),

        go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[Z_traj[0]],
            mode="markers",
            marker=dict(size=7, color="yellow"),
            name="Current Position"
        )
    ]
)

fig.frames = frames

sliders = [{
    "active": 0,
    "currentvalue": {"prefix": "Frame: "},
    "pad": {"t": 50},
    "steps": [
        {
            "label": str(t),
            "method": "animate",
            "args": [
                [str(t)],
                {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}
            ]
        }
        for t in range(len(frames))
    ]
}]

play_button = dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=1, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )
                    ]
                )

pause_button = dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [],
                        dict(mode="immediate")
                    ]
                )

fig.update_layout(
    title="Reverse Diffusion Process",
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="p(x,y)",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        zaxis=dict(tickformat=".6f")
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0, y=1.05,
            buttons=[play_button, pause_button]
        )
    ]
)

fig.update_layout(
    sliders=sliders,
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "buttons": [play_button, pause_button]
    }]
)

# Trick: trigger first frame automatically
#fig.frames[0].layout = dict(play=True)

# ===============================
# Write HTML and open browser
# ===============================
out = Path("trajectory.html")
fig.write_html(out, auto_open=True)
