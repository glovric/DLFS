import numpy as np
import time
import struct

def dilate(arr: np.ndarray, stride: int) -> np.ndarray:
    """
    Expands boundaries of an array by adding rows and columns of zeros between array elements.

    Parameters
    ----------
    arr : np.ndarray
        Array to dilate.

    stride : int
        Number of zeroes added between a pair of elements.
        NOTE: stride - 1 zeros are added between elements.

    Returns
    -------
    dilated_arr : np.ndarray
    """
    # Create a new array with appropriate size for dilation
    dilated_shape = (arr.shape[0] - 1) * stride + 1, (arr.shape[1] - 1) * stride + 1
    dilated = np.zeros(dilated_shape)
    
    # Place the original array elements into the dilated array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            dilated[i * stride, j * stride] = arr[i, j]
    
    return dilated

def pad_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Adds padding to array so it matches target shape.

    Parameters
    ----------
    arr : np.ndarray
        Array to pad.

    target_shape : tuple
        Shape of the array after padding.

    Returns
    -------
    padded_arr : np.ndarray
    """
    # Calculate padding needed
    pad_height = target_shape[0] - arr.shape[0]
    pad_width = target_shape[1] - arr.shape[1]
    
    if pad_height < 0 or pad_width < 0:
        raise ValueError("Target shape must be larger than the array shape.")
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Apply padding
    padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded

def timeit(name):

    def timed(func):

        def decorator(*args, **kwargs):

            ts = time.time()
            func(*args, **kwargs)
            te = time.time()
            print('Function', func.__module__, name, func.__name__, 'time:', round((te - ts) * 1000, 1), 'ms')

        return decorator
    
    return timed

def get_random_batch(*data: tuple[np.ndarray, ...], batch_size: int):
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

def im2col(X, kernel_shape, stride=1, padding=(0, 0)):
    B, C = X.shape[:2]
    kH, kW = kernel_shape

    if isinstance(padding, tuple):
        pad_H, pad_W = padding
    else:
        pad_H = pad_W = padding

    X_padded = np.pad(X, ( (0, 0), (0, 0),(pad_H, pad_H), (pad_W, pad_W) ), mode='constant')

    H_p, W_p = X_padded.shape[2:]

    out_H = (H_p- kH) // stride + 1
    out_W = (W_p- kW) // stride + 1

    L = out_H * out_W

    cols = np.zeros((B, C * kH * kW, L))

    patch_idx = 0
    for i in range(out_H):
        for j in range(out_W):

            h_start = i * stride
            w_start = j * stride

            # slice patch for ALL channels
            patch = X_padded[:, :, h_start:h_start + kH, w_start:w_start + kW]

            # flatten channels + kernel dims
            patch = patch.reshape(B, -1)    # → (B, C*kH*kW)

            cols[:, :, patch_idx] = patch
            patch_idx += 1
    
    return cols, out_H, out_W

def col2im(cols, output_shape, kernel_shape, stride=1, padding=0):
    B, C_k, num_cols = cols.shape   # C_k = C * kH * kW
    H, W = output_shape
    kH, kW = kernel_shape

    # padded spatial dims
    H_p, W_p = H + 2*padding, W + 2*padding

    # output tensor
    X_padded = np.zeros((B, C_k // (kH*kW), H_p, W_p))

    # how many sliding positions?
    out_H = (H_p - kH) // stride + 1
    out_W = (W_p - kW) // stride + 1

    idx = 0
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride

            # reshape column back into (B, C, kH, kW)
            patch = cols[:, :, idx].reshape(B, -1, kH, kW)

            # scatter-add patch into spatial tensor
            X_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += patch

            idx += 1

    # remove padding
    if padding > 0:
        X_padded = X_padded[:, :, padding:-padding, padding:-padding]

    return X_padded

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
        X_padded, shape=shape, strides=strides
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

#### Dataset helpers ####

class DLFSData:

    @staticmethod
    def load_MNIST_images(file_path):
        with open(file_path, 'rb') as f:
            # Read the header information
            magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read the image data
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
            return images
        
    @staticmethod
    def load_MNIST_labels(file_path):
        with open(file_path, 'rb') as f:
            # Read the header information
            magic_number, num_labels = struct.unpack(">II", f.read(8))
            # Read the label data
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    @staticmethod
    def load_MNIST(folder_path):
        path_train_data = folder_path + "/train-images.idx3-ubyte"
        path_train_labels = folder_path + "/train-labels.idx1-ubyte"
        path_test_data = folder_path + "/t10k-images.idx3-ubyte"
        path_test_labels = folder_path + "/t10k-labels.idx1-ubyte"

        X_train = DLFSData.load_MNIST_images(path_train_data)
        y_train = DLFSData.load_MNIST_labels(path_train_labels)

        X_test = DLFSData.load_MNIST_images(path_test_data)
        y_test = DLFSData.load_MNIST_labels(path_test_labels)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def normalize_MNIST(x, range: tuple = (0, 1)):
        if range == (0, 1):
            x = x.astype("float32") / 255
        elif range == (-1, 1):
            x = x.astype("float32") / 127.5 - 1
        return x
    
    @staticmethod
    def select_MNIST_labels(x, y, labels: list, limit=None, shuffle=True):
        label_indices = []

        for l in labels:
            if limit is not None:
                label_index = np.where(y == l)[0][:limit]
            else:
                label_index = np.where(y == l)[0]

            label_indices.append(label_index)

        all_indices = np.hstack(label_indices)

        if shuffle:
            all_indices = np.random.permutation(all_indices)

        x, y = x[all_indices], y[all_indices]
        return x, y