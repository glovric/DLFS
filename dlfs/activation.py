import numpy as np
from .base import Activation

class Linear(Activation):

    def __init__(self) -> None:
        """
        Linear activation function.
        """
        pass

    def forward(self, inputs: np.ndarray, training = False) -> None:
        """
        Forward pass using Linear activation. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        self.inputs = inputs
        self.output = inputs

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using Linear activation. Creates gradient attribute with respect to inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Derivative of Linear activation
        self.dinputs = delta

class ReLU(Activation):

    def __init__(self) -> None:
        """
        Rectified Linear Unit activation function.
        """
        pass

    def forward(self, inputs: np.ndarray, training = False) -> None:
        """
        Forward pass using ReLU. Creates output attribute.

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
        self.output = np.maximum(0, inputs)

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using ReLU. Creates gradient attribute with respect to inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        self.dinputs = delta.copy()
        # Derivative of ReLU
        self.dinputs[self.inputs < 0] = 0

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        """
        Leaky ReLU activation function.

        Parameters
        ----------
        alpha : float
            Slope for negative inputs (default 0.01).
        """
        self.alpha = alpha

    def forward(self, inputs: np.ndarray, training=False) -> None:
        """
        Forward pass using Leaky ReLU.
        """
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass through Leaky ReLU.
        """
        self.dinputs = delta.copy()
        self.dinputs[self.inputs < 0] *= self.alpha

class GELU(Activation):
    def __init__(self) -> None:
        """
        Gaussian Error Linear Unit (GELU) activation function.
        Approximate version used for performance.
        """
        pass

    def forward(self, inputs: np.ndarray, training=False) -> None:
        """
        Forward pass using GELU activation (approximation).
        Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.
        """
        self.inputs = inputs
        self.output = 0.5 * inputs * (
            1 + np.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * np.power(inputs, 3)))
        )

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using GELU activation (approximate derivative).
        Creates dinputs attribute (gradient of the loss with respect to inputs).

        Parameters
        ----------
        delta : numpy.ndarray
            Accumulated gradient from upstream layers.
        """
        x = self.inputs
        tanh_out = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
        sech2 = 1 - tanh_out**2

        term1 = 0.5 * tanh_out
        term2 = (
            (0.5 * x * sech2)
            * (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2))
        )
        grad = term1 + term2 + 0.5

        self.dinputs = delta * grad

class Sigmoid(Activation):

    def __init__(self) -> None:
        """
        Sigmoid activation function.
        """
        pass

    def forward(self, inputs: np.ndarray, training = False) -> None:
        """
        Forward pass using Sigmoid. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using Sigmoid. Creates gradient attribute with respect to inputs.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Derivative of Sigmoid
        self.dinputs = delta * (1 - self.output) * self.output

class Tanh(Activation):

    def forward(self, inputs):
        self.output = np.tanh(inputs)

    def backward(self, delta):
        self.dinputs = 1 - self.output**2

class SiLU(Activation):

    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.sigma = np.where(inputs >= 0, 1 / (1 + np.exp(-inputs)), np.exp(inputs) / (1 + np.exp(inputs)))
        self.output = inputs * self.sigma

    def backward(self, delta):
        self.dinputs = delta * self.sigma * (1 + self.inputs * (1 - self.sigma))

class Softmax(Activation):

    def __init__(self) -> None:
        """
        Softmax activation function.
        """
        pass

    def forward(self, inputs: np.ndarray, training=False) -> None:
        """
        Forward pass using Softmax. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        exp = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def calculate(inputs: np.ndarray) -> np.ndarray:
        exp = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def backward(self, delta: np.ndarray) -> None:
        """
        Perform the backward pass for softmax.
        """
        # Initialize dinputs to be of the same shape as delta
        self.dinputs = np.empty_like(delta)

        # Get the shape of the input (abstract the first N-1 dimensions)
        *dims, num_classes = delta.shape
        
        # Reshape delta and output to (batch_size * other_dims, num_classes)
        delta_reshaped = delta.reshape(-1, num_classes)  # Flatten the first N-1 dimensions
        output_reshaped = self.output.reshape(-1, num_classes)  # Same reshape for output

        # Now, for each sample (in the flattened batch), compute the Jacobian matrix.
        # We are going to compute this efficiently by using matrix operations.
        self.dinputs = np.empty_like(delta_reshaped)

        for i in range(delta_reshaped.shape[0]):
            single_output = output_reshaped[i].reshape(-1, 1)  # Shape: (num_classes, 1)
            jacobian_matrix = np.diagflat(single_output) - np.matmul(single_output, single_output.T)  # Jacobian matrix for softmax

            # Now compute the gradient for this particular sample
            self.dinputs[i] = np.dot(delta_reshaped[i], jacobian_matrix)

        # Reshape back to the original shape (same as delta)
        self.dinputs = self.dinputs.reshape(*dims, num_classes)
