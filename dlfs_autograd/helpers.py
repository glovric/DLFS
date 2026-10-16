import numpy as np
import matplotlib.pyplot as plt
from .tensor import Tensor

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

def get_random_batch(*data: tuple[Tensor, ...], batch_size: int):
    """
    Sample a random batch of data from the provided input arrays.

    Parameters
    ----------
    *data : tuple[np.ndarray, ...]
        One or more input arrays from which to sample the batch. Each array must 
        have the same number of samples, `(num_samples, d1, ..., dN)`
        
    batch_size : int
        The number of samples to include in the batch.

    Returns
    -------
    tuple : tuple[np.ndarray, ...]
        A tuple containing the random batch of samples from each input array of shape `(batch_size, d1, ..., dN)`
    """
    # Generate random sample indices
    idx = np.random.randint(len(data[0]), size=(batch_size, ))
    return tuple(d[idx] for d in data)