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

        self.dinputs = np.empty_like(delta) 

        # 3D batch processing
        if len(delta.shape) == 3:

            for outer_index in range(len(delta)):

                for index, (single_output, single_delta) in enumerate(zip(self.output[outer_index], delta[outer_index])):

                    single_output = single_output.reshape(-1, 1)

                    jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

                    self.dinputs[outer_index][index] = np.dot(jacobian_matrix, single_delta) 

        elif len(delta.shape) == 2:

            for index, (single_output, single_delta) in enumerate(zip(self.output, delta)):

                single_output = single_output.reshape(-1, 1)

                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

                self.dinputs[index] = np.dot(jacobian_matrix, single_delta) 