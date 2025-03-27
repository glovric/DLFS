import numpy as np
from .base import Module
from .layers import *
from .activation import Softmax, ReLU

class RNN(Module):

    def __init__(self, n_inputs: int, n_hidden: int, n_layers: int = 1, predict_sequence: bool = False) -> None:
        """
        Recurrent neural network. Takes 3D arrays of shape (n_samples, n_timestamps, n_features) as input.

        Parameters
        ----------
        n_inputs : int
            Number of input features.

        n_hidden : int
            Number of hidden features.

        n_layers : int, default=1
            Number of recurrent layers.

        predict_sequence : bool, default=False
            Whether a sequence or a single element is returned as output.

        Attributes
        ----------
        recurrent_layers : list[RecurrentLayer]
            List containing recurrent layers.
        """
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
        """
        Forward pass using the RNN. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Pass data to the first recurrent layer
        self.recurrent_layers[0].forward(inputs)

        # Forward hidden states of the previous recurrent layer to the current one
        for idx, layer in enumerate(self.recurrent_layers[1:], start=1):
            layer.forward(self.recurrent_layers[idx - 1].hidden_states, training)

        # Output of the RNN is the final recurrent layer's output
        self.output = self.recurrent_layers[-1].output.copy()

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the RNN.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Pass gradient to the final recurrent layer
        self.recurrent_layers[-1].backward(delta)

        # Backpropagate gradient
        for idx, layer in reversed(list(enumerate(self.recurrent_layers[:-1]))):
            layer.backward(self.recurrent_layers[idx + 1].dinputs)

        self.dinputs = self.recurrent_layers[0].dinputs

class LSTM(Module):

    def __init__(self, n_inputs: int, n_hidden: int, n_layers: int = 1, predict_sequence: bool = False) -> None:
        """
        LSTM neural network. Takes 3D arrays of shape (n_samples, n_timestamps, n_features) as input.

        Parameters
        ----------
        n_inputs : int
            Number of input features.

        n_hidden : int
            Number of hidden features.

        n_layers : int, default=1
            Number of LSTM layers.

        predict_sequence : bool, default=False
            Whether a sequence or a single element is returned as output.

        Attributes
        ----------
        lstm_layers : list[LSTMLayer]
            List containing LSTM layers.
        """
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
        """
        Forward pass using the LSTM. Creates output attribute.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input matrix.

        Returns
        -------
        None
        """
        # Pass data to the first LSTM layer
        self.lstm_layers[0].forward(inputs)

        # Forward hidden states of the previous LSTM layer to the current one
        for idx, layer in enumerate(self.lstm_layers[1:], start=1):
            layer.forward(self.lstm_layers[idx - 1].hidden_states, training)

        # Output of the LSTM is the final LSTM layer's output
        self.output = self.lstm_layers[-1].output.copy()

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass using the LSTM.

        Parameters
        ----------
        delta : np.ndarray
            Accumulated gradient obtained by backpropagation.

        Returns
        -------
        None
        """
        # Pass gradient to the final LSTM layer
        self.lstm_layers[-1].backward(delta)

        # Backpropagate gradient
        for idx, layer in reversed(list(enumerate(self.lstm_layers[:-1]))):
            layer.backward(self.lstm_layers[idx + 1].dinputs)

        self.dinputs = self.lstm_layers[0].dinputs

class SingleAttentionHead(Module):

    def __init__(self, n_embed, head_size, block_size, dropout=0.1):
        self.key = DenseLayer(n_embed, head_size)
        self.query = DenseLayer(n_embed, head_size)
        self.value = DenseLayer(n_embed, head_size)
        self.softmax = Softmax()
        self.dropout = DropoutLayer(dropout)

        self.tril = np.tril(np.ones((block_size, block_size)))
        self.normalize_factor = head_size**0.5

    def forward(self, x, training=False):
        B, T, C = x.shape

        self.key.forward(x)
        self.query.forward(x)
        self.value.forward(x)

        self.k = self.key.output
        self.q = self.query.output
        self.v = self.value.output
        
        self.w = np.matmul(self.q, self.k.swapaxes(-2, -1)) / self.normalize_factor
        mask_condition = self.tril[:T, :T] == 0
        self.w[:,mask_condition] = -np.inf
        self.softmax.forward(self.w)
        self.w = self.softmax.output
        self.dropout.forward(self.w, training)
        self.w = self.dropout.output
        self.output = np.matmul(self.w, self.v)

    def backward(self, delta):

        # Step 1: Gradient of the loss with respect to w (attention weights)
        d_w = np.matmul(delta, self.v.swapaxes(-2, -1))

        self.dropout.backward(d_w)  # This will apply the dropout mask to the gradients
        d_w = self.dropout.dinputs

        # Step 2: Gradient of the loss with respect to softmax input (logits)
        self.softmax.backward(d_w)  # Softmax backward pass
        d_w = self.softmax.dinputs

        # Step 3: Gradient of the loss with respect to w (before softmax)
        d_w = d_w * (self.w > 0).astype(float)  # Masking out invalid values from softmax

        # Step 4: Gradients w.r.t. key and query using the chain rule
        d_q = np.matmul(d_w, self.k)  # shape: (B, T, head_size)
        d_k = np.matmul(d_w.swapaxes(-2, -1), self.q)  # shape: (B, T, head_size)

        # Step 5: Update the key, query, and value parameters using the gradients
        # Gradient for the key (d_k) and query (d_q) go through the dense layers
        self.key.backward(d_k)
        self.query.backward(d_q)
        self.value.backward(np.matmul(d_w, self.v))

        self.dinputs = self.key.dinputs + self.query.dinputs + self.value.dinputs

class MultiHeadAttention(Module):

    def __init__(self, n_embed, n_heads, head_size, block_size, dropout=0.1):
        self.n_heads = n_heads
        self.head_size = head_size

        # List to store each individual attention head
        self.attention_heads = [
            SingleAttentionHead(n_embed, head_size, block_size, dropout)
            for _ in range(n_heads)
        ]

        # Output Dense layer to combine the heads
        self.output_dense = DenseLayer(n_embed, n_embed)

        self.dropout = DropoutLayer(dropout)

    def forward(self, x, training=False):

        # Store outputs of all attention heads
        head_outputs = []

        for head in self.attention_heads:
            head.forward(x, training)  # Compute attention for this head
            head_outputs.append(head.output)  # Store the output of each head

        # Concatenate the outputs of all heads along the last dimension (features)
        concatenated_output = np.concatenate(np.array(head_outputs), axis=-1) 

        # Pass the concatenated output through the output dense layer
        self.output_dense.forward(concatenated_output)

        self.dropout.forward(self.output_dense.output, training)

        # Final output
        self.output = self.dropout.output

    def backward(self, delta):

        self.dropout.backward(delta)

        self.output_dense.backward(self.dropout.dinputs)

        d_concatenated_output = self.output_dense.output

        # Step 2: Split the gradient back into the individual heads
        d_head_outputs = np.split(d_concatenated_output, self.n_heads, axis=-1)

        # Step 3: Backpropagate through each attention head
        for i, head in enumerate(self.attention_heads):
            head.backward(d_head_outputs[i])  # Backprop through each head

        self.dinputs = self.attention_heads[0].dinputs

class FeedForward(Module):

    def __init__(self, n_embed, dropout=0.1):
        self.fc1 = DenseLayer(n_embed, 4*n_embed)
        self.relu1 = ReLU()
        self.fc2 = DenseLayer(4*n_embed, n_embed)
        self.relu2 = ReLU()
        self.dropout = DropoutLayer(dropout)

    def forward(self, inputs, training=False):
        self.fc1.forward(inputs)
        self.relu1.forward(self.fc1.output)
        self.fc2.forward(self.relu1.output)
        self.relu2.forward(self.fc2.output)
        self.dropout.forward(self.relu2.output, training)
        self.output = self.dropout.output

    def backward(self, delta):
        self.dropout.backward(delta)
        self.relu2.backward(self.dropout.dinputs)
        self.fc2.backward(self.relu2.dinputs)
        self.relu1.backward(self.fc2.dinputs)
        self.fc1.backward(self.relu1.dinputs)
        self.dinputs = self.fc1.dinputs

class Block(Module):
    def __init__(self, n_embed, n_head, block_size, dropout=0.1):
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_heads=n_head, head_size=head_size, n_embed=n_embed, block_size=block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)
    def forward(self, x, training=False):
        self.ln1.forward(x)
        self.sa.forward(self.ln1.output, training)
        x = x + self.sa.output
        self.ln2.forward(x)
        self.ffwd.forward(self.ln2.output, training)
        x = x + self.ffwd.output
        self.output = x

    def backward(self, delta):

        dx = delta
        dffwd = dx  # Gradient to pass to the FeedForward layer
        
        self.ffwd.backward(dffwd)

        self.ln2.backward(dx)
        dln2 = self.ln2.dinputs
        
        dsa = dln2  # Gradient to pass to MultiHeadAttention
        
        self.sa.backward(dsa)
        
        self.ln1.backward(dsa)

        self.dinputs = self.ln1.dinputs

class TransformerDecoder(Module):

    def __init__(self, n_embed, n_head, block_size, n_layers: int = 1, dropout=0.1) -> None:
        if n_layers == 1:
            self.blocks = [Block(n_embed, n_head, block_size, dropout)]
        else:
            self.blocks = [Block(n_embed, n_head, block_size, dropout) for _ in range(n_layers)]

    def forward(self, inputs: np.ndarray, training=False) -> None:

        # Pass data to the first LSTM layer
        self.blocks[0].forward(inputs, training)

        # Forward hidden states of the previous LSTM layer to the current one
        for idx, layer in enumerate(self.blocks[1:], start=1):
            layer.forward(self.blocks[idx - 1].output, training)

        # Output of the LSTM is the final LSTM layer's output
        self.output = self.blocks[-1].output.copy()

    def backward(self, delta: np.ndarray) -> None:

        self.blocks[-1].backward(delta)

        for idx, layer in reversed(list(enumerate(self.blocks[:-1]))):
            layer.backward(self.blocks[idx + 1].dinputs)

        self.dinputs = self.blocks[0].dinputs