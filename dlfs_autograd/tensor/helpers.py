import numpy as np

def sum_to_shape(grad: np.ndarray, shape: tuple):
    """Sum gradient axes introduced by NumPy broadcasting."""

    # Remove leading dimensions added during broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum dimensions where the original shape was 1
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad

def unbroadcast(grad: np.ndarray, original_shape: tuple):
    # Remove extra leading dimensions
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)

    # Sum dimensions that were broadcast from size 1
    for axis, size in enumerate(original_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(original_shape)