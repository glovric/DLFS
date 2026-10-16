import numpy as np

class Tensor:

    def __init__(self, data, use_grad=True, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.use_grad = use_grad

        if use_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

        self.T = self.data.T

        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data + other.data,
            _children=(self, other),
            _op="+"
        )

        def _backward():
            if self.use_grad:
                self.grad += sum_to_shape(
                    result.grad, self.data.shape
                )
            if other.use_grad:
                other.grad += sum_to_shape(
                    result.grad, other.data.shape
                )

        result._backward = _backward
        return result

    def __pow__(self, exponent):
        result = Tensor(
            self.data ** exponent,
            _children=(self,),
            _op=f"**{exponent}"
        )

        def _backward():
            if self.use_grad:
                self.grad += (
                    exponent
                    * self.data ** (exponent - 1)
                    * result.grad
                )

        result._backward = _backward
        return result    

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data * other.data,
            _children=(self, other),
            _op="*"
        )

        def _backward():
            if self.use_grad:
                self.grad += result.grad * other.data
            if other.use_grad:
                other.grad += result.grad * self.data

        result._backward = _backward
        return result

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data @ other.data,
            _children=(self, other),
            _op="@"
        )

        def _backward():
            *dims, n_features = result.grad.shape
            grad_reshaped = result.grad.reshape(-1, n_features)

            *data_dims, data_features = self.data.shape
            data_reshaped = self.data.reshape(-1, data_features)

            *other_dims, other_features = other.data.shape
            other_reshaped = other.data.reshape(-1, other_features)

            if other.use_grad:
                other.grad += np.matmul(data_reshaped.T, grad_reshaped)
            if self.use_grad:
                self.grad += np.matmul(grad_reshaped, other_reshaped.T)

        result._backward = _backward
        return result

    def backward(self):
        # Build topological ordering of graph
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        # d(output)/d(output) = 1
        self.grad = np.ones_like(self.data)

        #print(f'Kolicina ekipe u topo: {len(topo)}')

        # Reverse topological order
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        if self.use_grad:
            self.grad = np.zeros_like(self.data)

    def sum(self):
        result = Tensor(self.data.sum(), (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * result.grad

        result._backward = _backward
        return result

def sum_to_shape(grad, shape):
    """Sum gradient axes introduced by NumPy broadcasting."""

    # Remove leading dimensions added during broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum dimensions where the original shape was 1
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad