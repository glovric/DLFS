import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.graph_objects as go

def plot_2d_clf_problem(X, y, h=None):
    '''
    Plots a two-dimensional labeled dataset (X,y) and, if function h(x) is given, 
    the decision surfaces.
    '''
    assert X.shape[1] == 2, "Dataset is not two-dimensional"
    if h!=None : 
        # Create a mesh to plot in
        r = 0.03  # mesh resolution
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, r),
                             np.arange(y_min, y_max, r))
        XX=np.c_[xx.ravel(), yy.ravel()]
        try:
            Z_test = h(XX)
            if Z_test.shape == ():
                # h returns a scalar when applied to a matrix; map explicitly
                Z = np.array(list(map(h,XX)))
            else :
                Z = Z_test
        except ValueError:
            # can't apply to a matrix; map explicitly
            Z = np.array(list(map(h,XX)))
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)

    # Plot the dataset
    plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.tab20b, marker='o', s=50);

def tab20b_colors_to_plotly(n):
    cmap = plt.cm.tab20b
    values = np.linspace(0, 1, n)

    mpl_colors = [cmap(v) for v in values]  # list of RGBA tuples from matplotlib
    plotly_colors = [
        f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        for r, g, b, _ in cmap(values)
    ]

    return plotly_colors, mpl_colors

def create_meshgrid(X, resolution):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    X1, X2 = np.meshgrid(x1, x2)
    return X1, X2

def plot_3d_classification_data(X, y, title):
    fig3d = go.Figure()

    classes = np.unique(y)
    colors, _ = tab20b_colors_to_plotly(len(classes))

    # map class -> color index
    color_map = {cls: i for i, cls in enumerate(classes)}
    color_indices = np.vectorize(color_map.get)(y)

    fig3d.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color_indices,
            colorscale=colors,
            opacity=0.9,
            symbol='circle'
        )
    ))

    fig3d.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Neuron 1',
            yaxis_title='Neuron 2',
            zaxis_title='Neuron 3'
        )
    )

    fig3d.show()

def plot_1d_classification_data(X, y, title, logits=False):
    # Assuming X is shape (num_samples, 1)
    X = X.flatten()  # flatten to 1D for plotting

    classes = np.unique(y)
    _, colors = tab20b_colors_to_plotly(len(classes))
    cmap = ListedColormap(colors)
    bounds = np.append(classes - 0.5, classes[-1] + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8,5))

    # Scatter plot: color by true label
    scatter = plt.scatter(
        range(len(X)),
        X,
        c=y,
        cmap=cmap,
        norm=norm,
        s=60,
        edgecolor='k',
        alpha=0.9
    )

    # Add horizontal line at 0 or 0.5 (decision boundary)
    if logits:
        plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, label='Decision boundary (logits=0)')
        plt.ylabel('Logits', fontsize=12)
    else:
        plt.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision boundary (probability=0.5)')
        plt.ylabel('Probability', fontsize=12)

    # Labels & title
    plt.xlabel('Sample index', fontsize=12)
    plt.title(title, fontsize=14)

    # Grid and style
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Color legend
    cbar = plt.colorbar(scatter, boundaries=bounds, ticks=classes)
    cbar.set_label('Class label', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_2d_classification_data(X, y, title):

    classes = np.unique(y)
    _, colors = tab20b_colors_to_plotly(len(classes))
    cmap = ListedColormap(colors)
    bounds = np.append(classes - 0.5, classes[-1] + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8,5))

    # Scatter plot: color by true label
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap,
        norm=norm,
        s=60,
        edgecolor='k',
        alpha=0.9
    )

    plt.title(title, fontsize=14)

    plt.xlabel("Neuron 1")
    plt.ylabel("Neuron 2")

    # Color legend
    cbar = plt.colorbar(scatter, boundaries=bounds, ticks=classes)
    cbar.set_label('Class label', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_1d_regression_data(X, y, title):
    """
    Plots elements of X onto a horizontal number line (x axis).
    X is 1D array, y is 1D array used for coloring.
    """
    plt.scatter(X, np.zeros_like(X), c=y, edgecolors="k", alpha=0.7)
    plt.title(title)
    plt.show()

def plot_2d_regression_data(X, y, title, axis1, axis2):
    """X is 2D, y is 1D used for coloring."""
    plt.figure(figsize=(8,5))

    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=60,
        edgecolor='k',
        alpha=0.9
    )

    plt.title(title, fontsize=14)

    plt.xlabel(axis1)
    plt.ylabel(axis2)

    plt.tight_layout()
    plt.show()

def plot_3d_regression_data(X, y, title):
    """
    X is 2D array, y is 1d array
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=y,
            colorscale='Viridis',
            opacity=0.9,
            symbol='circle'
        )
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Neuron 1',
            yaxis_title='Neuron 2',
            zaxis_title='Neuron 3'
        )
    )

    fig.show()

def plot_2d_reg_problem(X, y, curve=None, f=None, title="", curve_label="Curve", resolution=100):

    input_range = None
    plt.scatter(X, y, c=y, edgecolors="k", alpha=0.7)

    if f is not None and curve is None:
        input_range = np.linspace(X.min(), X.max(), resolution)
        with np.errstate(invalid='ignore', divide='ignore'):
            curve = f(input_range)
        curve[~np.isfinite(curve)] = np.nan
    
    if curve is not None:
        if input_range is None:
            input_range = np.linspace(X.min(), X.max(), resolution)
        plt.plot(input_range, curve, color="red", label=curve_label, linewidth=2)
        plt.legend()


    plt.title(title)
    plt.show()

def plot_3d_reg_problem(X, y, surface=None, f=None, title="", surface_label="Surface", resolution=80):

    fig = go.Figure()
    meshgrid = None

    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y, mode='markers',
        marker=dict(size=3, color=y, colorscale='Viridis', opacity=0.7,
                    colorbar=dict(title='y')),
        name='Data',
        hovertemplate='x1:%{x:.2f}<br>x2:%{y:.2f}<br>y:%{z:.2f}<extra></extra>'
    ))

    if surface is None and f is not None:
        meshgrid = create_meshgrid(X, resolution)
        X1, X2 = meshgrid
        with np.errstate(invalid='ignore', divide='ignore'):
            surface = f(X1, X2)
        surface[~np.isfinite(surface)] = np.nan

    if surface is not None:

        if meshgrid is None:
            meshgrid = create_meshgrid(X, resolution)

        X1, X2 = meshgrid

        fig.add_trace(go.Surface(
            x=X1, y=X2, z=surface, opacity=0.45, colorscale='Blues',
            showscale=False, name=surface_label,
            hovertemplate='x1:%{x:.2f}<br>x2:%{y:.2f}<br>ŷ:%{z:.2f}<extra></extra>'

        ))

        fig.update_layout(
            title=title, height=700,
            scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='y',
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="down",
                    buttons=[
                        dict(
                            label=f"Show {surface_label}",
                            method="restyle",
                            args=[{"visible": [True]}, [1]]
                        ),
                        dict(
                            label=f"Hide {surface_label}",
                            method="restyle",
                            args=[{"visible": [False]}, [1]]
                        ),
                    ],
                )
            ]
        )

    fig.show()