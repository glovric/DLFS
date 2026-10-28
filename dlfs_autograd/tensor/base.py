import numpy as np
import gc
from .helpers import sum_to_shape, unbroadcast

class Tensor:

    def __init__(self, data, use_grad=True, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.use_grad = use_grad
        self.grad = None

        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):

        other = ensure_tensor(other)

        result = Tensor(
            self.data + other.data,
            _children=(self, other),
            _op="+"
        )

        def _backward():
            if self.use_grad:
                up_grad = sum_to_shape(result.grad, self.data.shape)
                self.grad = up_grad if self.grad is None else self.grad + up_grad
            if other.use_grad:
                up_grad = sum_to_shape(result.grad, other.data.shape)
                other.grad = up_grad if other.grad is None else other.grad + up_grad

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
                up_grad = exponent * self.data ** (exponent - 1) * result.grad
                self.grad = up_grad if self.grad is None else self.grad + up_grad

        result._backward = _backward
        return result    

    def __mul__(self, other):
        other = ensure_tensor(other)

        result = Tensor(
            self.data * other.data,
            _children=(self, other),
            _op="*"
        )

        def _backward():
            if self.use_grad:
                up_grad = result.grad * other.data
                self.grad = up_grad if self.grad is None else self.grad + up_grad
            if other.use_grad:
                up_grad = result.grad * self.data
                other.grad = up_grad if other.grad is None else other.grad + up_grad

        result._backward = _backward
        return result

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = ensure_tensor(other)

        result = Tensor(
            self.data @ other.data,
            _children=(self, other),
            _op="@"
        )

        def _backward():   
            """                 
            *dims, n_features = result.grad.shape
            grad_reshaped = result.grad.reshape(-1, n_features)

            *data_dims, data_features = self.data.shape
            data_reshaped = self.data.reshape(-1, data_features)

            *other_dims, other_features = other.data.shape
            other_reshaped = other.data.reshape(-1, other_features)

            if other.use_grad:
                if other.grad is None:
                    other.init_grad()         
                other.grad += np.matmul(data_reshaped.T, grad_reshaped)
            if self.use_grad:
                if self.grad is None:
                    self.init_grad()
                self.grad += np.matmul(grad_reshaped, other_reshaped.T)
            """

                     
            if other.use_grad:    
                up_grad = np.matmul(np.swapaxes(self.data, -1, -2), result.grad)
                up_grad = unbroadcast(up_grad, other.data.shape)
                other.grad = up_grad if other.grad is None else other.grad + up_grad

            if self.use_grad:
                up_grad = np.matmul(result.grad, np.swapaxes(other.data, -1, -2))
                up_grad = unbroadcast(up_grad, self.data.shape)
                self.grad = up_grad if self.grad is None else self.grad + up_grad            
            
        result._backward = _backward
        return result

    def __getitem__(self, idx):
        result = Tensor(
            self.data[idx],
            _children=(self,),
            _op="getitem",
        )

        def _backward():
            if self.grad:
                np.add.at(self.grad, idx, result.grad)

        result._backward = _backward
        return result

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def backward(self):
        # Build topological ordering of graph
        topo = []
        visited = set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

        for v in topo:
            v._prev.clear()
            v._backward = lambda: None

        # Slows down training, but keeps memory usage stable
        #gc.collect() # This helps tremendously because Python and memory

    def zero_grad(self):
        self.grad = None

    def sum(self):
        result = Tensor(
            self.data.sum(),
            _children=(self,), 
            _op="sum"
        )

        def _backward():
            if self.use_grad:
                up_grad = np.ones_like(self.data, dtype=np.float32) * result.grad
                self.grad = up_grad if self.grad is None else self.grad + up_grad

        result._backward = _backward
        return result

    def reshape(self, *new_shape):
        old_shape = self.data.shape
        new_data = self.data.reshape(*new_shape)

        result = Tensor(
            new_data,
            use_grad=self.use_grad,
            _children = (self, ),
            _op="reshape"
        )

        def _backward():
            if self.use_grad:
                up_grad = result.grad.reshape(old_shape)
                self.grad = up_grad if self.grad is None else self.grad + up_grad

        result._backward = _backward
        return result

class Parameter(Tensor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def ensure_tensor(obj):
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)