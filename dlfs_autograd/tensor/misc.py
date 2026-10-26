import numpy as np
from .base import Tensor, ensure_tensor

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

def im2col_strided_tensor(
    img: Tensor, 
    kernel_shape: tuple[int, int], 
    stride: int = 1, 
    padding: tuple[int, int] | int = (0, 0)
):

    img = ensure_tensor(img)

    B, C_in, H_in, W_in = img.data.shape

    cols, out_H, out_W = im2col_strided(
        img.data,
        kernel_shape,
        stride=stride,
        padding=padding
    )

    result = Tensor(
        cols,
        _children=(img, ),
        _op="im2col"
    )

    def _backward():

        img_padded = col2im_strided(
            result.data,
            output_shape=(H_in, W_in),
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding
        )

        img.grad = img_padded if img.grad is None else img.grad + img_padded

    result._backward = _backward

    return result, out_H, out_W

def maxpool(x: Tensor, pool_size: int | tuple, stride: int = 1, padding: int = 0):

    if isinstance(pool_size, tuple):
        pool_size = pool_size
    else:
        pool_size = (pool_size, pool_size)

    x = ensure_tensor(x)

    B, C, H_in, W_in = x.data.shape
    kH, kW = pool_size

    cols, out_H, out_W = im2col_strided(
        x.data, 
        kernel_shape=pool_size,
        stride=stride, 
        padding=padding
    )

    cols_reshaped = cols.reshape(B, C, kH*kW, out_H*out_W)

    max_vals = np.max(cols_reshaped, axis=2)
    max_idx = np.argmax(cols_reshaped, axis=2)   # <--- NEEDED FOR BACKWARD

    pooled = max_vals.reshape(B, C, out_H, out_W)

    result = Tensor(
        pooled,
        _children=(x, ),
        _op="maxpool"
    )

    def _backward():

        if x.use_grad:

            out_HW = out_H * out_W

            delta_flat = result.grad.reshape(B, C, out_HW)
            
            dcols = np.zeros_like(cols_reshaped)

            # Scatter dout to positions of max values
            # max_idx has shape (B,C,out_HW); use advanced indexing
            b_idx = np.arange(B)[:, None, None]
            c_idx = np.arange(C)[None, :, None]
            hw_idx = np.arange(out_HW)[None, None, :]

            dcols[b_idx, c_idx, max_idx, hw_idx] = delta_flat

            # reshape back to original im2col shape
            dcols = dcols.reshape(B, C*kH*kW, out_HW)

            # convert col-gradients back to image
            dX = col2im_strided(dcols, output_shape=(H_in, W_in), kernel_shape=pool_size,
                                stride=stride, padding=padding)

            x.grad = dX if x.grad is None else x.grad + dX

    result._backward = _backward
    return result