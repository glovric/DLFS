from .config import USE_CUPY

if USE_CUPY:
    import cupy as np
else:
    import numpy as np

from typing import Union
#import numpy as np
from .base import Layer, Activation, Loss, Optimizer
from .layers import RNN, LSTM

class Model:

    def __init__(self, layers: list[Union[Layer, Activation]] = None, loss_function: Loss = None, optimizer: Optimizer = None) -> None:
        """
        Neural network model.

        Parameters
        ----------
        layers : list[Union[Layer, Activation]], default=None
            List of layers and activation functions.

        loss_function : Loss, default=None
            Loss function.
        
        optimizer : Optimizer, default=None
            Optimizer algorithm.
        """
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer

    def _forward(self, X: np.ndarray, tgt: np.ndarray = None, training: bool = False) -> None:
        """
        Forward pass.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        Returns
        -------
        None
        """

        # Pass data to the input layer
        self.layers[0].forward(X, training)

        # Forward data through all the layers
        for idx, layer in enumerate(self.layers[1:], start=1):
            layer.forward(self.layers[idx - 1].output, training)

        # Output of the model is the output of the last layer
        self.output = self.layers[-1].output

    def _backward(self, y: np.ndarray) -> None:
        """
        Backward pass.
        
        Parameters
        ----------
        y : np.ndarray
            Output values.

        Returns
        -------
        None
        """

        # Backward pass starts with loss function gradient calculation
        self.loss_function.backward(self.output, y)
        self.layers[-1].backward(self.loss_function.dinputs)

        # Pass gradients backwards to all layers
        for idx, layer in reversed(list(enumerate(self.layers[:-1]))):
            layer.backward(self.layers[idx + 1].dinputs)

    def _update_model_parameters(self) -> None:
        """
        Method for updating model parameters. 
        Depending on the model architecture parameters are updated in different manners.

        Returns
        -------
        None
        """

        self.optimizer.pre_update_parameters()

        # Loop through all layers
        for layer in self.layers:

            self.optimizer.update_layer_parameters(layer)

        self.optimizer.post_update_parameters()

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, batch_size: int = None, print_every: int = None) -> None:
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        y : np.ndarray
            Output values.

        epochs : int, default=1000
            Number of training epochs.

        batch_size : int, default=None
            Number of samples in a data subset. Batch size of None results in using all samples.

        print_every : int, default=None
            If given an integer, prints loss value.

        Returns
        -------
        None
        """

        for i in range(epochs + 1):

            if batch_size is None:

                # Forward pass
                self._forward(X, training=True)

                # Backward pass
                self._backward(y)

                # Update parameters
                self._update_model_parameters()

                if print_every is not None:
                    if not i % print_every:
                        print(f'===== EPOCH : {i} ===== LOSS : {self.loss_function.calculate(self.output, y):.5f} =====')

            else:

                batch_loss = 0

                for j in range(0, len(X), batch_size):

                    # Subset data into a batch
                    batch_X = X[j:j+batch_size, :]
                    batch_y = y[j:j+batch_size]

                    # Forward pass
                    self._forward(batch_X)

                    batch_loss += self.loss_function.calculate(self.output, batch_y)

                    # Backward pass
                    self._backward(batch_y)

                    # Update parameters
                    self._update_model_parameters()

                if print_every is not None:
                    if not i % print_every:
                        print(f'===== EPOCH : {i} ===== LOSS : {batch_loss:.5f} =====')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        Returns
        -------
        prediction : np.ndarray
        """
        self._forward(X, training=False)
        return self.output
    
    def add(self, layer : Layer | Activation) -> None:
        """
        Add layer to the network.

        Parameters
        ----------
        layer : Layer | Activation_Function
            Layer to add.

        Returns
        -------
        None
        """
        self.layers.append(layer)