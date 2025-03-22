import numpy as np
import time

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