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


def plot_3d_output(X, y, title):
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


def plot_1d_output(X, y, title, logits=False):
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

def plot_2d_output(X, y, title):

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

    # Color legend
    cbar = plt.colorbar(scatter, boundaries=bounds, ticks=classes)
    cbar.set_label('Class label', fontsize=12)

    plt.tight_layout()
    plt.show()