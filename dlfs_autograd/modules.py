import numpy as np
from .base import Module, Activation
from .tensor.base import Tensor
from .layers import RecurrentLayer, LSTMLayer

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
        self.layers[0].forward(x, training=training)

        for idx, layer in enumerate(self.layers[1:], start=1):
            layer.forward(self.layers[idx - 1].output, training=training)

        self.output = self.layers[-1].output

class RNN(Module):

    def __init__(self, n_inputs: int, n_hidden: int, n_layers: int = 1, predict_sequence: bool = False) -> None:

        if n_layers == 1:
            self.recurrent_layers = [RecurrentLayer(n_inputs, n_hidden, predict_sequence)]
        else:
            self.recurrent_layers = [RecurrentLayer(n_inputs, n_hidden)]
            if predict_sequence:
                for i in range(n_layers - 1):
                    if i == n_layers - 2:
                        self.recurrent_layers.append(RecurrentLayer(n_hidden, n_hidden, predict_sequence=True))
                    else:
                        self.recurrent_layers.append(RecurrentLayer(n_hidden, n_hidden))
            else:
                for i in range(n_layers - 1):
                    self.recurrent_layers.append(RecurrentLayer(n_hidden, n_hidden))

    def forward(self, inputs: np.ndarray, training=False) -> None:
        # Pass data to the first recurrent layer
        self.recurrent_layers[0].forward(inputs)

        # Forward hidden states of the previous recurrent layer to the current one
        for idx, layer in enumerate(self.recurrent_layers[1:], start=1):
            layer.forward(self.recurrent_layers[idx - 1].hidden_states, training)

        # Output of the RNN is the final recurrent layer's output
        self.output = self.recurrent_layers[-1].output

class LSTM(Module):

    def __init__(self, n_inputs: int, n_hidden: int, n_layers: int = 1, predict_sequence: bool = False) -> None:

        if n_layers == 1:
            self.lstm_layers = [LSTMLayer(n_inputs, n_hidden, predict_sequence)]
        else:
            self.lstm_layers = [LSTMLayer(n_inputs, n_hidden)]
            if predict_sequence:
                for i in range(n_layers - 1):
                    if i == n_layers - 2:
                        self.lstm_layers.append(LSTMLayer(n_hidden, n_hidden, predict_sequence=True))
                    else:
                        self.lstm_layers.append(LSTMLayer(n_hidden, n_hidden))
            else:
                for i in range(n_layers - 1):
                    self.lstm_layers.append(LSTMLayer(n_hidden, n_hidden))

    def forward(self, inputs: np.ndarray, training=False) -> None:
        self.lstm_layers[0].forward(inputs)

        for idx, layer in enumerate(self.lstm_layers[1:], start=1):
            layer.forward(
                self.lstm_layers[idx - 1].hidden_states, 
                cell_state=self.lstm_layers[idx - 1].last_cell_state,
                training=training
            )

        self.output = self.lstm_layers[-1].output