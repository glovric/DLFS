import numpy as np
from .base import Layer, Module, Activation, Loss, Optimizer
from .modules import SequentialWrapper
from .helpers import get_random_batch

class SequentialModel(Module):

    def __init__(self, layers: list[Layer | Module | Activation] = None, loss_function: Loss = None, optimizer: Optimizer = None) -> None:
        """
        Neural network model.

        Parameters
        ----------
        layers : list[Layer | Module | Activation], default=None
            List of layers and activation functions.

        loss_function : Loss, default=None
            Loss function.
        
        optimizer : Optimizer, default=None
            Optimizer algorithm.
        """
        self.wrapper = SequentialWrapper(layers)
        self.loss_function = loss_function
        self.optimizer = optimizer

    def forward(self, X: np.ndarray, training: bool = False) -> None:
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

        # Pass data through SequentialWrapper
        self.wrapper.forward(X)

        # Output of the model is the output of SequentialWrapper
        self.output = self.wrapper.output

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

            if batch_size is not None:
                batch_X, batch_y = get_random_batch(X, y, batch_size=batch_size)
            else:
                # If batch size is not specified take the whole dataset
                batch_X, batch_y = X, y

            # Forward pass
            self.forward(batch_X, training=True)

            loss = self.loss_function.calculate(self.output, batch_y)

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.pre_update_parameters()
            self.optimizer.update_parameters(self.wrapper)
            self.optimizer.post_update_parameters()

            self.optimizer.zero_grad(self.wrapper)

            if print_every is not None:
                if not i % print_every:
                    print(f'===== EPOCH : {i} ===== LOSS : {loss.data} =====')

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
        self.forward(X, training=False)
        return self.output.data