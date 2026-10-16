import numpy as np
from .base import Layer
from .tensor import Tensor

class DenseLayer(Layer):

    def __init__(self, n_inputs, n_neurons):
        lim = np.sqrt(1 / n_inputs)
        self.weights = Tensor(
            np.random.uniform(-lim, lim, size=(n_inputs, n_neurons))
        )
        self.biases = Tensor(
            np.zeros(n_neurons)
        )

    def forward(self, inputs, training=False):
        self.inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        self.output = self.inputs @ self.weights + self.biases

    def get_parameters(self):
        param_names = ["weights", "biases"]
        return super()._filter_parameters(param_names)