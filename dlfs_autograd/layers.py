import numpy as np
from .base import Module
from .tensor import Tensor, Parameter

class DenseLayer(Module):

    def __init__(self, n_inputs, n_neurons):
        lim = np.sqrt(1 / n_inputs)
        self.weights = Parameter(
            np.random.uniform(-lim, lim, size=(n_inputs, n_neurons))
        )
        self.biases = Parameter(
            np.zeros(n_neurons)
        )

    def forward(self, inputs, training=False):
        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        self.output = inputs @ self.weights + self.biases