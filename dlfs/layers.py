import numpy as np
from .base import Layer
from .helpers import im2col_strided, col2im_strided

class DenseLayer(Layer):
    
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        """
        Fully connected dense layer of neurons.

        Parameters
        ----------
        n_inputs : int
            Number of inputs that connect to the layer.

        n_neurons : int
            Number of neurons the layer consists of.

        Attributes
        ----------
        weights : numpy.ndarray
            Matrix of weight coefficients.

        biases : numpy.ndaray
            Vector of bias coefficients.
        """

        # Weights are randomly initialized, small random numbers seem to work well
        lim = np.sqrt(2 / n_inputs)
        self.weights = lim * np.random.randn(n_inputs, n_neurons)
        # Bias vector is initialized to a zero vector
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs: np.ndarray, training=False) -> None:
        """
        Forward pass using the dense layer. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Store inputs for later use (backpropagation)
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the dense layer. Creates gradient attributes with respect to layer weights, biases and inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """

        # Way faster this way
        *dims, n_features = delta.shape
        delta_reshaped = delta.reshape(-1, n_features)

        *input_dims, input_features = self.inputs.shape
        inputs_reshaped = self.inputs.reshape(-1, input_features)

        self.dweights = np.matmul(inputs_reshaped.T, delta_reshaped)
        self.dbiases = np.sum(delta_reshaped, axis=0)
        self.dinputs = np.matmul(delta_reshaped, self.weights.T)

        self.dinputs = self.dinputs.reshape(*input_dims, input_features)

        # 2D case (n_samples, n_inputs)
        """
        This is the reminder of the old slow way.

        if len(delta.shape) == 2:
            self.dweights = np.dot(self.inputs.T, delta)
            self.dbiases = np.sum(delta, axis=0)
            self.dinputs = np.dot(delta, self.weights.T)
        
        # 3D case (n_samples, n_timestamps, n_inputs), used with RNN sequences
        if len(delta.shape) == 3:
            self.dweights = np.zeros_like(self.weights)
            self.dbiases = np.zeros_like(self.biases)
            self.dinputs = np.zeros_like(self.inputs)

            for i in range(delta.shape[0]):
                self.dweights += np.dot(self.inputs[i].T, delta[i])
                self.dbiases += np.sum(delta[i], axis=0)
                self.dinputs += np.dot(delta[i], self.weights.T)"""

    def get_parameters(self):
        param_names = ["weights", "biases"]
        return super()._filter_parameters(param_names)

class ConvLayer(Layer):

    def __init__(self, input_channels: tuple, output_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = input_channels * kernel_size**2
        std = np.sqrt(2.0 / fan_in)

        # Initialize layer parameters
        self.kernels = std * np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.biases = np.zeros(output_channels)

    def forward(self, inputs, training=False):

        B, _, self.H_in, self.W_in = inputs.shape

        C_out, C_in, kH, kW = self.kernels.shape

        self.X_col, H_out, W_out = im2col_strided(inputs, (kH, kW), stride=self.stride, padding=self.padding) # (B, C_in*kH*kW, H_out*W_out)

        self.K_col = self.kernels.reshape(C_out, -1) # (C_out, C_in*kH*kW)

        Y_col = self.K_col @ self.X_col + self.biases[None, :, None] # (B, C_out, H_out*W_out)

        Y = Y_col.reshape(B, C_out, H_out, W_out) # (B, C_out, H_out, W_out)

        self.output = Y

    def backward(self, delta):

        B, C_out, out_H, out_W = delta.shape
        C_out, C_in, kH, kW = self.kernels.shape

        delta_col = delta.reshape(B, C_out, -1) # (B, C_out, out_H*out_W)

        self.dkernels = delta_col @ self.X_col.transpose(0, 2, 1) # (B, C_out, C_in*kH*kW)
        self.dkernels = np.sum(self.dkernels, axis=0)
        self.dkernels = self.dkernels.reshape(self.kernels.shape)

        self.dinputs = self.K_col.T @ delta_col # (B, C_in*kH*kW, out_H*out_W)
        self.dinputs = col2im_strided(self.dinputs, 
                              output_shape=(self.H_in, self.W_in), 
                              kernel_shape=(kH, kW), 
                              stride=self.stride, 
                              padding=self.padding) # (B, C_in, H_in, W_in)

        self.dbiases = np.sum(delta_col, axis=(0, 2)) # (C_out)

    def get_parameters(self):
        param_names = ["kernels", "biases"]
        return super()._filter_parameters(param_names)

class ConvTransposeLayer(Layer):

    def __init__(self, input_channels: tuple, output_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding=0) -> None:

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        fan_in = input_channels * kernel_size**2
        std = np.sqrt(2.0 / fan_in)

        # Initialize layer parameters
        self.kernels = std * np.random.randn(input_channels, output_channels, kernel_size, kernel_size)
        self.biases = np.zeros(output_channels)

    def forward(self, inputs: np.ndarray, training=False) -> None:
        C_in, C_out, kH, kW = self.kernels.shape
        B, _, self.H_in, self.W_in = inputs.shape

        self.Y_col = inputs.reshape(B, C_in, -1) # (B, C_in, H_in*W_in)
        self.K_col = self.kernels.reshape(C_in, -1)  # (C_in, C_out*kH*kW)

        X_col = self.K_col.T @ self.Y_col # (B, C_out*kH*kW, H_in*W_in)

        H_out = (self.H_in - 1) * self.stride - 2 * self.padding + kH
        W_out = (self.W_in - 1) * self.stride - 2 * self.padding + kW

        X = col2im_strided(X_col, 
                   output_shape=(H_out, W_out), 
                   kernel_shape=(kH, kW), 
                   stride=self.stride, 
                   padding=self.padding) # (B, C_out, H_out, W_out)
    
        self.output = X + self.biases[None, :, None, None]

    def backward(self, delta: np.ndarray) -> None:
        
        B, C_out, out_H, out_W = delta.shape
        C_in, _, kH, kW = self.kernels.shape
        delta_col, _, _ = im2col_strided(delta, kernel_shape=(kH, kW), stride=self.stride, padding=self.padding) # (B, C_out*kH*kW, H_out*W_out)

        self.dkernels = self.Y_col @ delta_col.transpose(0, 2, 1) # # (B, C_in, H_in*W_in) x (B, H_in*W_in, C_out*kH*kW) = (B, C_in, C_out*kH*kW)
        self.dkernels = np.sum(self.dkernels, axis=0)
        self.dkernels = self.dkernels.reshape(C_in, C_out, kH, kW)
        
        self.dinputs = self.K_col @ delta_col
        self.dinputs = self.dinputs.reshape(B, C_in, self.H_in, self.W_in)

        self.dbiases = np.sum(delta, axis=(0, 2, 3))

    def get_parameters(self):
        param_names = ["kernels", "biases"]
        return super()._filter_parameters(param_names)

class MaxPoolLayer(Layer):

    def __init__(self, pool_size: int | tuple, stride: int = 1, padding: int = 0):
        self.stride = stride
        self.padding = padding

        if isinstance(pool_size, tuple):
            self.pool_size = pool_size
        else:
            self.pool_size = (pool_size, pool_size)

    def forward(self, inputs, training=False):
        self.H_in, self.W_in = inputs.shape[-2:]
        B, C = inputs.shape[:2]
        kH, kW = self.pool_size

        cols, out_H, out_W = im2col_strided(inputs, kernel_shape=self.pool_size,
                                            stride=self.stride, padding=self.padding)

        self.cols_reshaped = cols.reshape(B, C, kH*kW, out_H*out_W)

        max_vals = np.max(self.cols_reshaped, axis=2)
        self.max_idx = np.argmax(self.cols_reshaped, axis=2)   # <--- NEEDED FOR BACKWARD

        pooled = max_vals.reshape(B, C, out_H, out_W)

        self.output = pooled

    def backward(self, delta):
        B, C, _, out_HW = self.cols_reshaped.shape
        kH, kW = self.pool_size

        delta_flat = delta.reshape(B, C, out_HW)
        
        dcols = np.zeros_like(self.cols_reshaped)

        # Scatter dout to positions of max values
        # max_idx has shape (B,C,out_HW); use advanced indexing
        b_idx = np.arange(B)[:, None, None]
        c_idx = np.arange(C)[None, :, None]
        hw_idx = np.arange(out_HW)[None, None, :]

        dcols[b_idx, c_idx, self.max_idx, hw_idx] = delta_flat

        # reshape back to original im2col shape
        dcols = dcols.reshape(B, C*kH*kW, out_HW)

        # convert col-gradients back to image
        dX = col2im_strided(dcols, output_shape=(self.H_in, self.W_in), kernel_shape=self.pool_size,
                            stride=self.stride, padding=self.padding)

        self.dinputs = dX

class ReshapeLayer(Layer):

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

    def forward(self, inputs: np.ndarray, training=False) -> None:
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
        batch_size = inputs.shape[0]
        self.output = np.reshape(inputs, (batch_size, *self.output_shape))

    def backward(self, delta: np.ndarray) -> None:
        """
        Reshapes input array to input shape. Creates gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient to reshape.

        Returns
        -------
        None
        """
        # Store number of samples, first dimension
        batch_size = delta.shape[0]
        self.dinputs = np.reshape(delta, (batch_size, *self.input_shape))

class RecurrentLayer(Layer):

    def __init__(self, n_inputs: int, n_hidden: int, predict_sequence: bool = False) -> None:
        """
        Recurrent layer. Takes 3D arrays of shape (n_samples, n_timestamps, n_features) as input.

        Parameters
        ----------
        n_inputs : int
            Number of input features.

        n_hidden : int
            Number of hidden features.

        predict_sequence : bool, default=False
            Whether a sequence or a single element is returned as output.

        Attributes
        ----------
        input_weights : numpy.ndarray
            Matrix of input weight coefficients.

        hidden_weights : numpy.ndarray
            Matrix of hidden weight coefficients.

        input_bias : numpy.ndaray
            Vector of input bias coefficients.
        """
        self.predict_sequence = predict_sequence

        # Initialize parameters
        k = 1 / np.sqrt(n_hidden)
        self.n_hidden = n_hidden
        self.input_weights = np.random.uniform(-k, k, (n_inputs, n_hidden))
        self.hidden_weights = np.random.uniform(-k, k, (n_hidden, n_hidden))
        self.input_bias = np.random.uniform(-k, k, (n_hidden))
      
    def forward(self, inputs: np.ndarray, training=False) -> None:
        """
        Forward pass using the recurrent layer. Creates hidden states and output attributes.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Store inputs for backpropagation
        self.inputs = inputs

        # Store number of samples
        self.n_samples, self.timestamps = inputs.shape[:2]

        # Initialize output
        if self.predict_sequence:
            self.output = np.zeros((self.n_samples, self.timestamps, self.n_hidden))
        else:
            self.output = np.zeros((self.n_samples, self.n_hidden))

        # Initialize hidden states
        self.hidden_states = np.zeros((self.n_samples, self.timestamps, self.n_hidden))

        # Loop through timestamps
        for t in range(self.timestamps):
            # Compute current hidden states
            hidden_states_t = np.tanh(np.dot(inputs[:, t, :], self.input_weights) + np.dot(self.hidden_states[:, max(0, t-1), :], self.hidden_weights) + self.input_bias)
            # Store current hidden states
            self.hidden_states[:, t, :] = hidden_states_t.copy()

        if self.predict_sequence:
            # Hidden states of the current sequence are the predicted sequence
            self.output = self.hidden_states.copy()
        else:
            # Last hidden state of the current sequence is the predicted element
            self.output = self.hidden_states[:, -1, :].copy()

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the recurrent layer. 
        Creates gradient attributes with respect to input weights, hidden weights, input bias, and inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Initialize gradient attributes
        self.dinput_weights = np.zeros_like(self.input_weights)
        self.dhidden_weights = np.zeros_like(self.hidden_weights)
        self.dinput_bias = np.zeros_like(self.input_bias)
        self.dinputs = np.zeros_like(self.inputs, dtype=np.float64)

        # Initialize next hidden gradient
        next_hidden_gradient = None

        # Loop through timestamps in reversed order
        for t in range(self.timestamps - 1, -1, -1):

            if len(delta.shape) == 2:
                hidden_gradient = delta.copy()
            elif len(delta.shape) == 3:
                hidden_gradient = delta[:, t, :].copy()

            if next_hidden_gradient is not None:
                hidden_gradient += np.dot(next_hidden_gradient, self.hidden_weights)

            dtanh = 1 - self.hidden_states[:, t, :]**2
            hidden_gradient *= dtanh

            next_hidden_gradient = hidden_gradient.copy()

            if t > 0:
                self.dhidden_weights += np.dot(self.hidden_states[:, t-1, :].T, hidden_gradient)

            self.dinput_weights += np.dot(self.inputs[:, t, :].T, hidden_gradient)
            self.dinput_bias += hidden_gradient.sum(axis=0)

            self.dinputs[:, t, :] += np.dot(self.input_weights, hidden_gradient.T).T

    def get_parameters(self):
        param_names = ["input_weights", "hidden_weights", "input_bias"]
        return super()._filter_parameters(param_names)

class LSTMLayer(Layer):

    def __init__(self, n_inputs: int, n_hidden: int, predict_sequence: bool = False) -> None:
        """
        Long short term memory layer. Takes 3D arrays of shape (n_samples, n_timestamps, n_features) as input.

        Parameters
        ----------
        n_inputs : int
            Number of input features.

        n_hidden : int
            Number of hidden features.

        predict_sequence : bool, default=False
            Whether a sequence or a single element is returned as output.

        Attributes
        ----------
        input_weights : numpy.ndarray
            Matrix of input weight coefficients.

        input_bias : numpy.ndarray
            Vector of input bias coefficients.

        forget_weights : numpy.ndarray
            Matrix of forget weight coefficients.

        forget_bias : numpy.ndarray
            Vector of forget bias coefficients.

        candidate_weights : numpy.ndarray
            Matrix of candidate weight coefficients.

        candidate_bias : numpy.ndarray
            Vector of candidate bias coefficients.

        output_weights : numpy.ndarray
            Matrix of output weight coefficients.

        output_bias : numpy.ndarray
            Vector of output bias coefficients.
        """
        self.predict_sequence = predict_sequence

        # Initialize parameters
        k = 1 / np.sqrt(n_hidden)
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs

        self.input_weights = np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        self.input_bias = np.random.uniform(-k, k, (n_hidden))

        self.forget_weights = np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        self.forget_bias = np.random.uniform(-k, k, (n_hidden))

        self.candidate_weights = np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        self.candidate_bias = np.random.uniform(-k, k, (n_hidden)) 

        self.output_weights = np.random.uniform(-k, k, (n_inputs + n_hidden, n_hidden))
        self.output_bias = np.random.uniform(-k, k, (n_hidden)) 
             
    def forward(self, inputs: np.ndarray, training=False) -> None:
        """
        Forward pass using the LSTM layer. 
        Creates hidden, candidate, cell, forget, input, output states and output attributes.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """

        # Store number of samples and timestamps
        self.n_samples, self.timestamps = inputs.shape[:2]

        # Initialize concatenated inputs matrix
        self.concat_inputs = np.zeros((self.n_samples, self.timestamps, self.n_hidden + self.n_inputs))

        # Store input shape
        self.input_shape = inputs.shape

        # Initialize states
        self.hidden_states = np.zeros((self.n_samples, self.timestamps, self.n_hidden))
        self.candidate_states = np.zeros((self.n_samples, self.timestamps, self.n_hidden))
        self.cell_states = np.zeros((self.n_samples, self.timestamps, self.n_hidden))
        self.forget_states = np.zeros((self.n_samples, self.timestamps, self.n_hidden))
        self.input_states = np.zeros((self.n_samples, self.timestamps, self.n_hidden))
        self.output_states = np.zeros((self.n_samples, self.timestamps, self.n_hidden))

        # Loop through timestamps
        for t in range(self.timestamps):

            # Concatenate inputs and previous hidden states
            inputs_hidden_concatenated = np.concatenate((inputs[:, t, :], self.hidden_states[:, max(0, t-1), :]), axis=1)
            self.concat_inputs[:, t, :] = inputs_hidden_concatenated

            # Calculate current forget state
            f_t = np.dot(inputs_hidden_concatenated, self.forget_weights) + self.forget_bias
            f_t = self._sigmoid(f_t)
            self.forget_states[:, t, :] = f_t

            # Calculate current input state
            i_t = np.dot(inputs_hidden_concatenated, self.input_weights) + self.input_bias
            i_t = self._sigmoid(i_t)
            self.input_states[:, t, :] = i_t

            # Calculate current candidate state
            cc_t = np.dot(inputs_hidden_concatenated, self.candidate_weights) + self.candidate_bias
            cc_t = np.tanh(cc_t)
            self.candidate_states[:, t, :] = cc_t

            # Calculate current output state
            o_t = np.dot(inputs_hidden_concatenated, self.output_weights) + self.output_bias
            o_t = self._sigmoid(o_t)
            self.output_states[:, t, :] = o_t

            # Calculate current cell state
            c_t = self.cell_states[:, max(0, t-1), :] * f_t + i_t * cc_t
            self.cell_states[:, t, :] = c_t

            # Calculate current hidden state
            self.hidden_states[:, t, :] = o_t * np.tanh(c_t)
       
        if self.predict_sequence:
            # Hidden states of the current sequence are the predicted sequence
            self.output = self.hidden_states
        else:
            # Last hidden state of the current sequence is the predicted element
            self.output = self.hidden_states[:, -1, :]

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the LSTM layer. 
        Creates gradient attributes with respect to input, forget, candidate, output weights and biases, and inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Initialize gradient attributes
        self.dforget_weights = np.zeros_like(self.forget_weights)
        self.dforget_bias = np.zeros_like(self.forget_bias)

        self.dinput_weights = np.zeros_like(self.input_weights)
        self.dinput_bias = np.zeros_like(self.input_bias)

        self.dcandidate_weights = np.zeros_like(self.candidate_weights)
        self.dcandidate_bias = np.zeros_like(self.candidate_bias)

        self.doutput_weights = np.zeros_like(self.output_weights)
        self.doutput_bias = np.zeros_like(self.output_bias)

        self.dinputs = np.zeros(self.input_shape, dtype=np.float64)

        # Initialize next cell state gradient
        next_cell_state_grad = None

        # Loop through timestamps in reversed order
        for t in range(self.timestamps - 1, -1, -1):

            # Get current states
            c_t = self.cell_states[:, t, :]
            cc_t = self.candidate_states[:, t, :]
            o_t = self.output_states[:, t, :]
            f_t = self.forget_states[:, t, :]
            i_t = self.input_states[:, t, :]
            x_t = self.concat_inputs[:, t, :]

            if len(delta.shape) == 2:
                delta_t = delta
            elif len(delta.shape) == 3:
                delta_t = delta[:, t, :]

            output_grad = delta_t * np.tanh(c_t) * (1 - o_t) * o_t
            self.doutput_weights += np.dot(x_t.T, output_grad)
            self.doutput_bias += output_grad.sum(axis=0)

            cell_state_grad = delta_t * o_t * (1 - np.tanh(c_t)**2)
            if next_cell_state_grad is not None:
                cell_state_grad += next_cell_state_grad * self.forget_states[:, t+1, :]

            next_cell_state_grad = cell_state_grad

            candidate_grad = cell_state_grad *  i_t * (1 - cc_t**2)

            self.dcandidate_weights += np.dot(x_t.T, candidate_grad)
            self.dcandidate_bias += candidate_grad.sum(axis=0)

            input_grad = cell_state_grad * cc_t * (1 - i_t) * i_t
            self.dinput_weights += np.dot(x_t.T, input_grad)
            self.dinput_bias += input_grad.sum(axis=0)

            if t > 0:
                forget_grad = cell_state_grad * self.candidate_states[:, t-1, :] * ((1 - f_t) * f_t)
                self.forget_weights += np.dot(x_t.T, forget_grad)
                self.forget_bias += forget_grad.sum(axis=0)

            self.dinputs[:, t, :] = np.dot(output_grad, self.output_weights[:self.n_inputs, :].T)  + \
                                    np.dot(input_grad, self.input_weights[:self.n_inputs, :].T) + \
                                    np.dot(forget_grad, self.forget_weights[:self.n_inputs, :].T) + \
                                    np.dot(candidate_grad, self.candidate_weights[:self.n_inputs, :].T)
   
    def get_parameters(self):
        param_names = ["input_weights", "forget_weights", "candidate_weights", "output_weights",
                       "input_bias", "forget_bias", "candidate_bias", "output_bias"]
        return super()._filter_parameters(param_names)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        sigmoid_output : np.ndarray
        """
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))
    
class DropoutLayer(Layer):

    def __init__(self, rate: float) -> None:
        """
        Dropout Layer is a regularization technique used to 
        decrease overfitting by randomly setting a fraction of the input units to 
        zero at each training epoch.

        Parameters
        ----------
        rate : float
            Float between 0 and 1 which represents the fraction 
            of input units to drop. 
        """
        self.rate = 1 - rate

    def forward(self, inputs: np.ndarray, training: bool = False) -> None:
        """
        Forward pass using the Dropout layer. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        training : bool, default=False
            Flag indicating whether dropout is applied or not.

        Returns
        -------
        None
        """
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the Dropout layer. Creates dinputs gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        self.dinputs = delta * self.binary_mask

class LayerNorm(Layer):
    
    def __init__(self, num_features: int, epsilon: float = 1e-5) -> None:
        """
        Layer Normalization (LayerNorm) is a technique for normalizing the input across the features 
        of a layer (sample-wise, normalization is done independently for each sample). 
        It's primary use is stabilizing training in deep models. 

        Parameters
        ----------
        num_features : int
            The number of features of the input array. 
        
        epsilon : float, default=1e-5
            A small constant added to the denominator during normalization to prevent division by zero 
            and to maintain numerical stability.
        """
        self.epsilon = epsilon
        
        # Initialize the scale (gamma) and shift (beta) parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
    def forward(self, inputs: np.ndarray, training: bool = False):
        """
        Forward pass using LayerNorm. Creates output attribute.
        
        Parameters
        ----------
        inputs : np.ndarray
            Input array to be normalized.

        training : bool, default=False
            Flag indicating whether dropout is applied or not (used for API consistency).

        Returns
        -------
        None
        """
        self.inputs = inputs

        mean = np.mean(inputs, axis=-1, keepdims=True)
        variance = np.var(inputs, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(variance + self.epsilon)

        self.centered = self.inputs - mean
        self.normalized = self.centered * self.std_inv
        self.output = self.gamma * self.normalized + self.beta
    
    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using LayerNorm. Creates dinputs gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Assumes delta has shape (B, ..., F) matching self.output
        N = self.inputs.shape[-1]  # num_features

        # Gradients w.r.t gamma and beta
        self.dgamma = np.sum(delta * self.normalized, axis=tuple(range(delta.ndim - 1)), keepdims=False)
        self.dbeta = np.sum(delta, axis=tuple(range(delta.ndim - 1)), keepdims=False)

        # Gradient w.r.t normalized input
        dx_hat = delta * self.gamma

        # Gradients w.r.t input using LayerNorm formula
        dvar = np.sum(dx_hat * self.centered * -0.5 * self.std_inv**3, axis=-1, keepdims=True)
        dmean = np.sum(-dx_hat * self.std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * self.centered, axis=-1, keepdims=True)

        self.dinputs = dx_hat * self.std_inv + dvar * 2.0 * self.centered / N + dmean / N

    def get_parameters(self):
        param_names = ["gamma", "beta"]
        return super()._filter_parameters(param_names)

class EmbeddingLayer(Layer):

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01  # Initialize with small random values

    def forward(self, input_indices, training=False):
        """
        Forward pass: Given input indices, retrieve corresponding embeddings.
        """
        self.input_indices = input_indices
        self.output = self.embeddings[input_indices]

    def backward(self, delta):

        self.dembeddings = np.zeros_like(self.embeddings)

        batch_size, sequence_length, embedding_dim = delta.shape
        
        # Flatten the batch and sequence dimensions to get the word indices as a 1D array
        batch_sequence_flat = np.reshape(self.input_indices, -1)  # Shape: (batch_size * sequence_length,)
        delta_flat = np.reshape(delta, (-1, embedding_dim))  # Shape: (batch_size * sequence_length, embedding_dim)

        for idx, grad in zip(batch_sequence_flat, delta_flat):
            self.dembeddings[idx] += grad

    def get_parameters(self):
        param_names = ["embeddings"]
        return super()._filter_parameters(param_names)

class PositionalEncoding(Layer):

    def __init__(self, sequence_length, n_embed):
        self.sequence_length = sequence_length
        self.n_embed = n_embed

    def _positional_encode(self):
        P = np.zeros((self.sequence_length, self.n_embed))
        for k in range(self.sequence_length):
            for i in np.arange(int(self.n_embed/2)):
                denominator = 10000**(2*i / self.n_embed)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def forward(self, inputs, training=False):
        self.sequence_length = inputs.shape[1]
        self.output = inputs + self._positional_encode()

    def backward(self, delta):
        self.dinputs = delta
