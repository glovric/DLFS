import numpy as np
from .base import Module
from .tensor.base import Tensor, Parameter, ensure_tensor
from .tensor.misc import im2col_strided_tensor, maxpool

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
        self.output = inputs @ self.weights + self.biases

class ConvLayer(Module):

    def __init__(self, input_channels: tuple, output_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = input_channels * kernel_size**2
        std = np.sqrt(1.0 / fan_in)

        # Initialize layer parameters
        self.kernels = Parameter(
            np.random.uniform(-std, std, size=(output_channels, input_channels*kernel_size*kernel_size))
        )
        self.biases = Parameter(
            np.zeros(output_channels)
        )

    def forward(self, inputs: Tensor, training=False):

        inputs = ensure_tensor(inputs)

        B = inputs.data.shape[0]

        C_out, kH, kW = self.output_channels, self.kernel_size, self.kernel_size

        inputs_col, H_out, W_out = im2col_strided_tensor( # (B, C_in*kH*kW, H_out*W_out)
            inputs, 
            (kH, kW), 
            stride=self.stride,
            padding=self.padding
        )

        result_col = self.kernels @ inputs_col + self.biases.reshape(1, -1, 1)
        result = result_col.reshape(B, C_out, H_out, W_out) # (B, C_out, H_out, W_out)
        self.output = result

class MaxPoolLayer(Module):

    def __init__(self, pool_size: int | tuple, stride: int = 1, padding: int = 0):
        self.stride = stride
        self.padding = padding

        if isinstance(pool_size, tuple):
            self.pool_size = pool_size
        else:
            self.pool_size = (pool_size, pool_size)

    def forward(self, inputs: Tensor, training=False):
        self.output = maxpool(inputs, self.pool_size, self.stride, self.padding)

class ReshapeLayer:

    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs: Tensor, training=False) -> None:
        # Store number of samples, first dimension
        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        batch_size = inputs.data.shape[0]
        result = inputs.reshape(batch_size, *self.output_shape)
        self.output = result