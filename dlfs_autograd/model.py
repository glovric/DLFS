import numpy as np
from .base import Module, Activation, Loss, Optimizer
from .modules import SequentialWrapper
from .helpers import get_random_batch
from .tensor.base import Tensor, ensure_tensor

class SequentialModel(Module):

    def __init__(self, layers: list[Module | Activation] = None, loss_function: Loss = None, optimizer: Optimizer = None) -> None:
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
        self.optimizer.set_parameters(self.parameters())

    def forward(self, X: Tensor, training: bool = False) -> None:
        """
        Forward pass.

        Parameters
        ----------
        X : Tensor
            Input values.

        Returns
        -------
        None
        """

        # Pass data through SequentialWrapper
        self.wrapper.forward(X, training=training)

        # Output of the model is the output of SequentialWrapper
        self.output = self.wrapper.output

    def train(self, X: Tensor, y: Tensor, epochs: int = 1000, batch_size: int = None, print_every: int = None) -> None:
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
        X, y = ensure_tensor(X), ensure_tensor(y)

        for i in range(epochs + 1):

            self.optimizer.zero_grad()

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
            self.optimizer.update_parameters()
            self.optimizer.post_update_parameters()

            if print_every is not None:
                if not i % print_every:
                    print(f'===== EPOCH : {i} ===== LOSS : {loss.data} =====')

    def predict(self, X: Tensor) -> np.ndarray:
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