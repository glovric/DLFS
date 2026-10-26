import numpy as np
import matplotlib.pyplot as plt
from .tensor.base import Tensor

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
    idx = np.random.randint(len(data[0].data), size=(batch_size, ))
    return tuple(
        Tensor(d.data[idx]) for d in data
    )

def im2col_strided(X, kernel_shape, stride=1, padding=(0, 0)):
    B, C, H, W = X.shape
    kH, kW = kernel_shape

    # Handle padding argument
    if isinstance(padding, tuple):
        pad_H, pad_W = padding
    else:
        pad_H = pad_W = padding

    # Pad input
    X_padded = np.pad(
        X,
        ((0, 0), (0, 0), (pad_H, pad_H), (pad_W, pad_W)),
        mode="constant"
    )

    _, _, H_p, W_p = X_padded.shape

    # Compute output spatial size
    out_H = (H_p - kH) // stride + 1
    out_W = (W_p - kW) // stride + 1
    L = out_H * out_W  # number of sliding windows

    # Original strides
    sB, sC, sH, sW = X_padded.strides

    # as_strided shape: (B, C, out_H, out_W, kH, kW)
    shape = (B, C, out_H, out_W, kH, kW)

    # as_strided strides
    strides = (
        sB,                   # batch dim
        sC,                   # channel dim
        sH * stride,          # move down by stride
        sW * stride,          # move right by stride
        sH,                   # kernel H step
        sW                    # kernel W step
    )

    # Extract sliding windows
    windows = np.lib.stride_tricks.as_strided(
        X_padded, shape=shape, strides=strides, writeable=False
    )
    # windows shape: (B, C, out_H, out_W, kH, kW)

    # Rearrange into im2col format:
    # 1. move kernel dims next to channels
    cols = windows.reshape(B, C, out_H * out_W, kH * kW)
    # shape: (B, C, L, kH*kW)

    # 2. merge C and kH*kW
    cols = cols.transpose(0, 1, 3, 2).reshape(B, C * kH * kW, L)
    # final shape: (B, C*kH*kW, L)

    return cols, out_H, out_W

def col2im_strided(cols, output_shape, kernel_shape, stride=1, padding=0):
    B, C_k, L = cols.shape  # C_k = C * kH * kW
    H, W = output_shape
    kH, kW = kernel_shape
    C = C_k // (kH * kW)

    # padded spatial dimensions
    H_p, W_p = H + 2*padding, W + 2*padding

    # output tensor (padded)
    X_padded = np.zeros((B, C, H_p, W_p), dtype=cols.dtype)

    # compute output height/width
    out_H = (H_p - kH) // stride + 1
    out_W = (W_p - kW) // stride + 1

    # reshape cols into (B, C, out_H, out_W, kH, kW)
    cols_reshaped = cols.reshape(B, C, kH * kW, out_H * out_W)
    cols_reshaped = cols_reshaped.transpose(0, 1, 3, 2)
    cols_reshaped = cols_reshaped.reshape(B, C, out_H, out_W, kH, kW)

    # create strided view of X_padded
    sB, sC, sH, sW = X_padded.strides
    shape = (B, C, out_H, out_W, kH, kW)
    strides = (sB, sC, sH*stride, sW*stride, sH, sW)
    X_strided = np.lib.stride_tricks.as_strided(X_padded, shape=shape, strides=strides)

    # accumulate all patches into X_strided
    np.add.at(X_strided, (...,), cols_reshaped)

    # remove padding if needed
    if padding > 0:
        X_padded = X_padded[:, :, padding:-padding, padding:-padding]

    return X_padded