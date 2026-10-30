import numpy as np
from .base import Module
from .tensor.base import Tensor, Parameter, ensure_tensor
from .tensor.misc import im2col_strided_tensor, col2im_strided_tensor, maxpool, stack_tensors, concatenate_tensors
from .tensor.math import tanh, sigmoid

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
        inputs = ensure_tensor(inputs)
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

        # Kernels is usually (output_channels, input_channels, kernel_size, kernel_size)
        # To save some computation graph without reshaping lets store it as (output_channels, input_channels*kernel_size*kernel_size)
        # This implementation differs from PyTorch kernel weights
        # For comparison we need torch_conv_kernel.reshape(C_out, C_in*kernel_size*kernel_size)
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

class ConvTransposeLayer(Module):

    def __init__(self, input_channels: tuple, output_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding=0) -> None:

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        fan_in = input_channels * kernel_size**2
        std = np.sqrt(1.0 / fan_in)

        # Kernels is usually (input_channels, output_channels, kernel_size, kernel_size)
        # To save some computation graph without reshaping lets store it as (output_channels*kernel_size*kernel_size, input_channels)
        # This implementation differs from PyTorch kernel weights
        # For comparison we need torch_convt_kernel.transpose(1, 2, 3, 0).reshape(C_in*kernel_size*kernel_size, C_out)
        self.kernels = Parameter(
            np.random.uniform(-std, std, size=(output_channels*kernel_size*kernel_size, input_channels))
        )
        self.biases = Parameter(
            np.zeros(output_channels)
        )

    def forward(self, inputs: np.ndarray, training=False) -> None:

        inputs = ensure_tensor(inputs)

        C_in, kH, kW = self.input_channels, self.kernel_size, self.kernel_size
        B, _, self.H_in, self.W_in = inputs.shape

        Y_col = inputs.reshape(B, C_in, -1) # (B, C_in, H_in*W_in)
        X_col = self.kernels @ Y_col # (C_out*kH*kW, C_in) x (B, C_in, H_in*W_in) = (B, C_out*kH*kW, H_in*W_in)

        H_out = (self.H_in - 1) * self.stride - 2 * self.padding + kH + self.output_padding
        W_out = (self.W_in - 1) * self.stride - 2 * self.padding + kW + self.output_padding

        X = col2im_strided_tensor(
            X_col, 
            output_shape=(H_out, W_out), 
            kernel_shape=(kH, kW), 
            stride=self.stride, 
            padding=self.padding
        ) # (B, C_out, H_out, W_out)
    
        self.output = X + self.biases.reshape(1, -1, 1, 1)

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

class RecurrentLayer(Module):

    def __init__(self, n_inputs: int, n_hidden: int, predict_sequence: bool = False) -> None:
        self.predict_sequence = predict_sequence

        # Initialize parameters
        k = 1 / np.sqrt(n_hidden)
        self.n_hidden = n_hidden
        self.input_weights = Parameter(
            np.random.uniform(-k, k, (n_inputs, n_hidden))
        )
        self.hidden_weights = Parameter(
            np.random.uniform(-k, k, (n_hidden, n_hidden))
        )
        self.input_bias = Parameter(
            np.random.uniform(-k, k, (n_hidden))
        )
      
    def forward(self, inputs: np.ndarray, training=False) -> None:

        inputs = ensure_tensor(inputs)

        n_samples, timestamps = inputs.data.shape[:2]

        hidden_states_list = [Tensor(np.zeros((n_samples, self.n_hidden)))]

        for t in range(timestamps):

            prev_hidden = hidden_states_list[-1]

            products = inputs[:, t, :] @ self.input_weights \
                       + prev_hidden @ self.hidden_weights \
                       + self.input_bias

            hidden_states_list.append(tanh(products))

        hidden_states_stacked = stack_tensors(hidden_states_list[1:], axis=1)

        if self.predict_sequence:
            self.output = hidden_states_stacked
        else:
            self.output = hidden_states_list[-1]

        self._hidden_states_list = hidden_states_list  # Keep the list!
        self.hidden_states = hidden_states_stacked  # Stacked version for external use

class LSTMLayer(Module):

    def __init__(self, n_inputs: int, n_hidden: int, predict_sequence: bool = False) -> None:
        self.predict_sequence = predict_sequence

        k = 1 / np.sqrt(n_hidden)
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs

        self.input_weights = Parameter(
            np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        )
        self.input_bias = Parameter(
            np.random.uniform(-k, k, (n_hidden))
        )

        self.forget_weights = Parameter(
            np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        )
        self.forget_bias = Parameter(
            np.random.uniform(-k, k, (n_hidden))
        )

        self.candidate_weights = Parameter(
            np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        )
        self.candidate_bias = Parameter(
            np.random.uniform(-k, k, (n_hidden))
        ) 

        self.output_weights = Parameter(
            np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        )
        self.output_bias = Parameter(
            np.random.uniform(-k, k, (n_hidden))
        ) 
             
    def forward(self, inputs: np.ndarray, cell_state: Tensor = None, training=False) -> None:

        inputs = ensure_tensor(inputs)

        assert inputs.ndim == 3, "LSTM input array must be of shape (B, T, N)"

        n_samples, timestamps = inputs.shape[:2]

        hidden_states_list = [Tensor(np.zeros((n_samples, self.n_hidden)))]
        cell_states_list = [Tensor(np.zeros((n_samples, self.n_hidden)))] if cell_state is None else [cell_state]

        for t in range(timestamps):

            inputs_hidden_concatenated = concatenate_tensors(
                                            [inputs[:, t, :], hidden_states_list[max(0, t-1)]], 
                                            axis=1
                                        )

            # Calculate current forget state
            f = inputs_hidden_concatenated @ self.forget_weights + self.forget_bias
            f_t = sigmoid(f)

            # Calculate current input state
            i = inputs_hidden_concatenated @ self.input_weights + self.input_bias
            i_t = sigmoid(i)

            # Calculate current candidate state
            cc = inputs_hidden_concatenated @ self.candidate_weights + self.candidate_bias
            cc_t = tanh(cc)

            # Calculate current output state
            o = inputs_hidden_concatenated @ self.output_weights + self.output_bias
            o_t = sigmoid(o)

            # Calculate current cell state
            c_t = cell_states_list[max(0, t-1)] * f_t + i_t * cc_t
            cell_states_list.append(c_t)

            # Calculate current hidden state
            h_t = o_t * tanh(c_t)
            hidden_states_list.append(h_t)

        hidden_states_stacked = stack_tensors(hidden_states_list[1:], axis=1)
       
        if self.predict_sequence:
            self.output = hidden_states_stacked
        else:
            self.output = hidden_states_list[-1]

        self._hidden_states_list = hidden_states_list
        self.hidden_states = hidden_states_stacked
        self.last_cell_state = cell_states_list[-1]