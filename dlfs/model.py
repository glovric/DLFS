import numpy as np
from .base import Layer, Activation, Loss, Optimizer, Module
from .activation import ReLU, Softmax
from .layers import EmbeddingLayer, PositionalEncoding, LayerNorm, DenseLayer
from .modules import TransformerDecoder, SequentialWrapper
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

    def backward(self, y: np.ndarray) -> None:
        """
        Backward pass.
        
        Parameters
        ----------
        y : np.ndarray
            True output values.

        Returns
        -------
        None
        """

        # Backward pass starts with loss function gradient calculation
        self.loss_function.backward(self.output, y)
        self.wrapper.backward(self.loss_function.dinputs)

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
                self.forward(X, training=True)

                # Backward pass
                self.backward(y)

                # Update parameters
                self.optimizer.pre_update_parameters()
                self.optimizer.update_parameters(self.wrapper)
                self.optimizer.post_update_parameters()

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
                    self.forward(batch_X)

                    batch_loss += self.loss_function.calculate(self.output, batch_y)

                    # Backward pass
                    self.backward(batch_y)

                    # Update parameters
                    self.optimizer.pre_update_parameters()
                    self.optimizer.update_parameters(self.wrapper)
                    self.optimizer.post_update_parameters()

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
        self.forward(X, training=False)
        return self.output

class TransformerDecoderModel(Module):

    def __init__(self, vocab_size: int, seq_len: int, 
                       n_embed: int = 512, n_head: int = 4, n_dec_layers: int = 2, dim_ff: int = 2048, 
                       dropout: float = 0.1, activation: Activation = ReLU, layer_norm_eps: float = 1e-5, 
                       loss_function: Loss = None, optimizer: Optimizer = None) -> None:
        """
        Decoder only model similar to GPT.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary, i.e., the total number of unique tokens (words/characters) in the dataset.

        seq_len : int
            The maximum sequence length of the input/output sequences. The model will be trained on sequences of this length.

        n_embed : int, default=512
            The dimensionality of the embedding vectors for both input and output.

        n_head : int, default=4
            The number of attention heads in each attention layer.

        n_dec_layers : int, default=2
            The number of decoder layers.

        dim_ff : int, default=2048
            The dimensionality of the feedforward layers in the encoder and decoder.

        dropout : float, default=0.1
            The dropout rate applied to layers during training to help prevent overfitting.

        activation : Activation, default=ReLU
            The activation function used in the feedforward layers.

        layer_norm_eps : float, default=1e-5
            A small value added to the denominator for numerical stability when performing layer normalization.

        loss_function : Loss, default=None
            The loss function used during training.

        optimizer : Optimizer, default=None
            The optimizer used for training.

        Returns
        -------
        None
        """
        
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.embed = EmbeddingLayer(vocab_size, n_embed)
        self.pos_enc = PositionalEncoding(seq_len, n_embed)

        self.decoder = TransformerDecoder(d_model=n_embed, n_head=n_head, n_dec_layers=n_dec_layers,
                                          dim_ff=dim_ff, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps)

        self.ln = LayerNorm(n_embed, epsilon=layer_norm_eps)

        self.linear = DenseLayer(n_embed, vocab_size)

        # Used in inference only
        self.softm = Softmax()

    def forward(self, inputs: np.ndarray, training: bool = False) -> None:
        """
        Forward pass for the Transformer Decoder model. Creates output attribute (logits) of shape `(batch_size, seq_len, vocab_size)`.

        Parameters
        ----------
        inputs : np.ndarray
            The input sequence for the decoder of shape `(batch_size, seq_len)`.

        training : bool, default=False
            A flag indicating whether the model is in training mode (`True`) or inference mode (`False`).
            This controls dropout, which is only applied during training.

        Returns
        -------
        None

        Notes
        -----
        - `inputs` are expected to be token indices for the decoder sequences.
        - The output is typically passed through a softmax function externally for final predictions (e.g., during inference).
        """

        # Embed decoder input, outputs (batch_size, seq_len, n_embed)
        self.embed.forward(inputs, training)

        # Add positional encoding to embeddings, outputs (batch_size, seq_len, n_embed)
        self.pos_enc.forward(self.embed.output, training)

        # Pass embedded inputs through the Decoder, outputs (batch_size, seq_len, n_embed)
        self.decoder.forward(self.pos_enc.output, training=training)

        # Pass decoder output to LayerNorm (optional in frameworks like PyTorch), outputs (batch_size, seq_len, n_embed)
        self.ln.forward(self.decoder.output, training)

        # Map LayerNorm output to raw logits, outputs (batch_size, seq_len, vocab_size)
        self.linear.forward(self.ln.output, training)

        # Create output attribute to be the final layer's output
        self.output = self.linear.output

    def backward(self, y: np.ndarray) -> None:
        """
        Backward pass for the Transformer Decoder model. Computes gradients for all Decoder modules.

        Parameters
        ----------
        y : np.ndarray
            The true labels (indices) for the model's output of shape `(batch_size, seq_len)`.

        Returns
        -------
        None
        """

        # Reshape y_pred and y_true for CCE
        B, T, C = self.output.shape
        y_pred = self.output.reshape(B*T, C)
        y_true = y.reshape(B*T, )

        # Compute loss gradient
        self.loss_function.backward(y_pred, y_true)
        
        # Backprop through linear
        self.linear.backward(self.loss_function.dinputs.reshape(B, T, C))

        # Backprop through LayerNorm
        self.ln.backward(self.linear.dinputs)

        # Backprop through decoder, positional encoding, and embedding
        self.decoder.backward(self.ln.dinputs)
        self.pos_enc.backward(self.decoder.dinputs)
        self.embed.backward(self.pos_enc.dinputs)

    def train(self, X: np.ndarray, y: np.ndarray, epochs = 1000, batch_size: int = None, print_every: int = None) -> None:
        """
        Train the model for a specified number of epochs, updating parameters using backpropagation.

        Parameters
        ----------
        X : np.ndarray
            Decoder input data (shifted targets) of shape `(batch_size, seq_len)`.

        y : np.ndarray
            True target labels (indices) of shape `(batch_size, seq_len)`.

        epochs : int, default=1000
            The number of epochs to train the model.

        batch_size : int, default=None
            The size of the batch to use for each training step. If `None`, the entire dataset is used
            for each forward/backward pass.

        print_every : int, default=None
            If provided, will print the current loss value every 
            `print_every` epochs.

        Returns
        -------
        None
        """

        # Store loss values for plotting
        self.loss_vals = []

        for i in range(epochs + 1):

            if batch_size is not None:
                batch_X, batch_y = get_random_batch(X, y, batch_size=batch_size)
                batch_X, batch_y = batch_X.astype(int), batch_y.astype(int)
            else:
                # If batch size is not specified take the whole dataset
                batch_X, batch_y = X, y
                batch_X, batch_y = batch_X.astype(int), batch_y.astype(int)

            # Forward pass
            self.forward(batch_X, training=True)

            # Backward pass
            self.backward(batch_y)

            # Update parameters
            self.optimizer.pre_update_parameters()
            self.optimizer.update_parameters(self)
            self.optimizer.post_update_parameters()

            # Store current loss
            B, T, C = self.output.shape
            loss = self.loss_function.calculate(self.output.reshape(B*T, C), batch_y.reshape(B*T))
            self.loss_vals.append(loss)

            if print_every is not None and not i % print_every:
                print(f'===== EPOCH : {i} ===== LOSS : {loss} =====')
                
    def generate(self, idx: np.ndarray, context_window: int, max_new_tokens: int = 100):
        """
        Generate sequences autoregressively using the trained Transformer Decoder model.

        Parameters
        ----------
        idx : np.ndarray
            The initial input sequence of tokens, shape `(batch_size, seq_len)`. This sequence will be used as the starting context for generation.

        context_window : int
            The length of the context window to consider for each step in the autoregressive generation. 
            Typically, this is the number of previous tokens the model uses to generate the next token.
            
        max_new_tokens : int, default=1000
            The maximum length of the sequence to be generated. Generation will stop either when
            this length is reached or when an end token (`end_token`) is generated.

        Returns
        -------
        np.ndarray
            The generated sequence of tokens for each input in the batch of shape `(batch_size, max_len)`.
        """
            
        for _ in range(max_new_tokens):

            # Get last context_window tokens from idx
            current_context = idx[:, -context_window:]

            # Pass through decoder
            self.forward(current_context, training=False)
            # Compute probabilities from raw logits
            self.softm.forward(self.output, training=False)
    
            probs = self.softm.output
            probs = probs[:, -1, :].reshape(-1)

            # Sample next token
            idx_next = np.argmax(np.random.multinomial(n=1, pvals=probs, size=1), axis=1).reshape(1, 1)

            # Concatenate the new token and previous tokens
            idx = np.concatenate((idx, idx_next), axis=1)

        return idx