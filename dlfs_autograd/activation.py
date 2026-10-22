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
            if inputs.grad is None:
                inputs.init_grad()
            relu_grad = np.where(data < 0, 0, 1)
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
            if inputs.grad is None:
                inputs.init_grad()
            sig_out = result.data.copy()
            inputs.grad += sig_out * (1 - sig_out) * result.grad

        result._backward = _backward
        self.output = result

class Softmax(Activation):


    def forward(self, inputs: Tensor, training=False) -> None:
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
        inputs = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        exp = np.exp(inputs.data - np.max(inputs.data, axis=-1, keepdims=True))
        result = Tensor(
            exp / np.sum(exp, axis=-1, keepdims=True),
            _children=(inputs, ),
            _op="softmax"
        )

        def _backward():
            if inputs.grad is None:
                inputs.init_grad()
            # Initialize dinputs to be of the same shape as delta
            #self.dinputs = np.empty_like(result.grad)

            # Get the shape of the input (abstract the first N-1 dimensions)
            *dims, num_classes = result.grad.shape
            
            # Reshape delta and output to (batch_size * other_dims, num_classes)
            delta_reshaped = result.grad.reshape(-1, num_classes)  # Flatten the first N-1 dimensions
            output_reshaped = self.output.data.reshape(-1, num_classes)  # Same reshape for output

            # Now, for each sample (in the flattened batch), compute the Jacobian matrix.
            # We are going to compute this efficiently by using matrix operations.
            dinputs = np.empty_like(delta_reshaped)

            for i in range(delta_reshaped.shape[0]):
                single_output = output_reshaped[i].reshape(-1, 1)  # Shape: (num_classes, 1)
                jacobian_matrix = np.diagflat(single_output) - np.matmul(single_output, single_output.T)  # Jacobian matrix for softmax

                # Now compute the gradient for this particular sample
                dinputs[i] = np.dot(delta_reshaped[i], jacobian_matrix)

            # Reshape back to the original shape (same as delta)
            dinputs = dinputs.reshape(*dims, num_classes)
            inputs.grad += dinputs

        result._backward = _backward
        self.output = result

    """    
    @staticmethod
    def calculate(inputs: np.ndarray) -> np.ndarray:
        exp = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
    """