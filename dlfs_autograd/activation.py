import numpy as np
from .tensor import Tensor
from .base import Activation

class ReLU(Activation):

    def forward(self, inputs, training=False):
        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        data = inputs.data

        result = Tensor(
            np.maximum(0, data),
            _children=(inputs, ),
            _op="relu"
        )

        def _backward():
            relu_grad = data.copy()
            relu_grad[data < 0] = 0
            inputs.grad += relu_grad * result.grad

        result._backward = _backward
        self.output = result

class Sigmoid(Activation):

    def forward(self, inputs, training=False):
        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        data = inputs.data

        result = Tensor(
            1 / (1 + np.exp(-1 * data)),
            _children=(inputs, ),
            _op="sigmoid"
        )

        def _backward():
            sig_out = result.data.copy()
            inputs.grad += sig_out * (1 - sig_out) * result.grad

        result._backward = _backward
        self.output = result