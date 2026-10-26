import numpy as np
from .base import Tensor, ensure_tensor

def exp(x: Tensor):
    x = ensure_tensor(x)
    y = np.exp(x.data)

    result = Tensor(
        y,
        _children=(x, ),
        _op="exp"
    )

    def _backward():
        if x.use_grad:
            up_grad = result.grad * y
            x.grad = up_grad if x.grad is None else x.grad + up_grad

    result._backward = _backward
    return result

def sigmoid(x: Tensor):
    x = ensure_tensor(x)
    s = 1 / (1 + np.exp(-x.data))

    result = Tensor(
        s,
        _children=(x, ),
        _op="sigmoid"
    )

    def _backward():
        if x.use_grad:
            up_grad = s * (1 - s) * result.grad
            x.grad = up_grad if x.grad is None else x.grad + up_grad

    result._backward = _backward
    return result

def relu(x: Tensor):
    x = ensure_tensor(x)
    r = np.maximum(0, x.data)

    result = Tensor(
        r,
        _children=(x, ),
        _op="relu"
    )

    def _backward():
        if x.use_grad:
            relu_grad = np.where(x.data < 0, 0, 1)
            up_grad = relu_grad * result.grad
            x.grad = up_grad if x.grad is None else x.grad + up_grad

    result._backward = _backward
    return result

def tanh(x: Tensor):
    x = ensure_tensor(x)
    t = np.tanh(x.data)

    result = Tensor(
        t,
        _children=(x, ),
        _op="tanh"
    )

    def _backward():
        if x.use_grad:
            up_grad = result.grad * (1 - t**2)
            x.grad = up_grad if x.grad is None else x.grad + up_grad

    result._backward = _backward
    return result

def softmax(x: Tensor):
    x = ensure_tensor(x)

    exp = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    softm = exp / np.sum(exp, axis=-1, keepdims=True)
    result = Tensor(
        softm,
        _children=(x, ),
        _op="softmax"
    )

    def _backward():

        if x.use_grad:
            # Get the shape of the input (abstract the first N-1 dimensions)
            *dims, num_classes = result.grad.shape
            
            # Reshape delta and output to (batch_size * other_dims, num_classes)
            delta_reshaped = result.grad.reshape(-1, num_classes)  # Flatten the first N-1 dimensions
            output_reshaped = softm.reshape(-1, num_classes)  # Same reshape for output

            # Now, for each sample (in the flattened batch), compute the Jacobian matrix.
            # We are going to compute this efficiently by using matrix operations.
            dinputs = np.empty_like(delta_reshaped)

            for i in range(delta_reshaped.shape[0]):
                single_output = output_reshaped[i].reshape(-1, 1)  # Shape: (num_classes, 1)
                jacobian_matrix = np.diagflat(single_output) - np.matmul(single_output, single_output.T)  # Jacobian matrix for softmax

                # Now compute the gradient for this particular sample
                dinputs[i] = np.dot(delta_reshaped[i], jacobian_matrix)

            # Reshape back to the original shape (same as delta)
            dinputs = dinputs.reshape(*dims, num_classes)
            x.grad = dinputs if x.grad is None else dinputs + x.grad

    result._backward = _backward
    return result