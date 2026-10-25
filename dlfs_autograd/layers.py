import numpy as np
from .base import Module
from .tensor import Tensor, Parameter
from .helpers import im2col_strided, col2im_strided
from .functions import im2col_strided_tensor

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

        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)

        B, _, self.H_in, self.W_in = inputs.data.shape

        C_out, kH, kW = self.output_channels, self.kernel_size, self.kernel_size

        inputs_col, H_out, W_out = im2col_strided_tensor( # (B, C_in*kH*kW, H_out*W_out)
            inputs, 
            (kH, kW), 
            stride=self.stride,
            padding=self.padding
        )

        result_col = self.kernels @ inputs_col + self.biases.reshape(1, -1, 1)
        result = result_col.reshape(B, C_out, H_out, W_out) # (B, C_out, H_out, W_out)

        """   
        def _backward():

            if inputs.grad is None:
                inputs.init_grad()
            
            if self.kernels.grad is None:
                self.kernels.init_grad()

            if self.biases.grad is None:
                self.biases.init_grad()

            B, _, _, _ = result.grad.shape
            C_out, C_in, kH, kW = self.output_channels, self.input_channels, self.kernel_size, self.kernel_size

            delta_col = result.grad.reshape(B, C_out, -1) # (B, C_out, out_H*out_W)

            dkernels = delta_col @ inputs_col.transpose(0, 2, 1) # (B, C_out, C_in*kH*kW)
            dkernels = np.sum(dkernels, axis=0)
            dkernels = dkernels.reshape(self.kernels.data.shape)
            self.kernels.grad += dkernels

            dinputs = K_col.T @ delta_col # (B, C_in*kH*kW, out_H*out_W)
            dinputs = col2im_strided(
                dinputs, 
                output_shape=(self.H_in, self.W_in), 
                kernel_shape=(kH, kW), 
                stride=self.stride, 
                padding=self.padding
            ) # (B, C_in, H_in, W_in)
            inputs.grad += dinputs

            dbiases = np.sum(delta_col, axis=(0, 2)) # (C_out)
            self.biases.grad += dbiases
        """
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

        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)

        self.H_in, self.W_in = inputs.data.shape[-2:]
        B, C = inputs.data.shape[:2]
        kH, kW = self.pool_size

        cols, out_H, out_W = im2col_strided(
            inputs.data, 
            kernel_shape=self.pool_size,
            stride=self.stride, 
            padding=self.padding
        )

        cols_reshaped = cols.reshape(B, C, kH*kW, out_H*out_W)

        max_vals = np.max(cols_reshaped, axis=2)
        max_idx = np.argmax(cols_reshaped, axis=2)   # <--- NEEDED FOR BACKWARD

        pooled = max_vals.reshape(B, C, out_H, out_W)

        result = Tensor(
            pooled,
            _children=(inputs, ),
            _op="maxpool"
        )

        def _backward():

            if inputs.grad is None:
                inputs.init_grad()

            B, C, _, out_HW = cols_reshaped.shape
            kH, kW = self.pool_size

            delta_flat = result.grad.reshape(B, C, out_HW)
            
            dcols = np.zeros_like(cols_reshaped)

            # Scatter dout to positions of max values
            # max_idx has shape (B,C,out_HW); use advanced indexing
            b_idx = np.arange(B)[:, None, None]
            c_idx = np.arange(C)[None, :, None]
            hw_idx = np.arange(out_HW)[None, None, :]

            dcols[b_idx, c_idx, max_idx, hw_idx] = delta_flat

            # reshape back to original im2col shape
            dcols = dcols.reshape(B, C*kH*kW, out_HW)

            # convert col-gradients back to image
            dX = col2im_strided(dcols, output_shape=(self.H_in, self.W_in), kernel_shape=self.pool_size,
                                stride=self.stride, padding=self.padding)

            inputs.grad += dX

        result._backward = _backward

        self.output = result

class ReshapeLayer:

    def __init__(self, input_shape: tuple, output_shape: tuple) -> None:
        """
        Layer used to reshape (flatten) an array.

        Parameters
        ----------
        input_shape : tuple
            Input shape of a single sample. For images it's (channels, height, width).

        output_shape : int
            Output shape of a single sample.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs: Tensor, training=False) -> None:
        """
        Reshapes input array to output shape. Creates output attribute.

        Parameters
        ----------
        inputs : np.ndarray
            Array to reshape.

        Returns
        -------
        None
        """
        # Store number of samples, first dimension
        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        batch_size = inputs.data.shape[0]
        result = inputs.reshape(batch_size, *self.output_shape)
        self.output = result