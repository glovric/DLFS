import numpy as np
from .base import Module, Activation
from .tensor.base import Tensor

class SequentialWrapper(Module):

    def __init__(self, layers: list[Module | Activation] = []) -> None:
        """
        SequentialWrapper is module which sequentially processes input data, forwarding it through every layer.

        Parameters
        ----------
        layers: list[Layer | Module | Activation], default=[]
            List of network components.
        """
        self.layers = layers

    def forward(self, x: Tensor, training: bool = False) -> None:
        """
        Forward pass for the SequentialWrapper. Creates output attribute.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied.

        Returns
        -------
        None
        """
        self.layers[0].forward(x, training)

        for idx, layer in enumerate(self.layers[1:], start=1):
            layer.forward(self.layers[idx - 1].output, training)

        self.output = self.layers[-1].output