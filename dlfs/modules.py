import numpy as np
from .base import Module, Activation, Layer
from .layers import *
from .activation import Softmax, ReLU

class SequentialWrapper(Module):

    def __init__(self, layers: list[Layer | Module | Activation] = []) -> None:
        """
        SequentialWrapper is module which sequentially processes input data, forwarding it through every layer.

        Parameters
        ----------
        layers: list[Layer | Module | Activation], default=[]
            List of network components.
        """
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = False) -> None:
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

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the SequentialWrapper. Creates dinputs gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Upstream gradient array.

        Returns
        -------
        None
        """
        self.layers[-1].backward(delta)
        
        for idx, layer in reversed(list(enumerate(self.layers[:-1]))):
            layer.backward(self.layers[idx + 1].dinputs)

        self.dinputs = self.layers[0].dinputs

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

    def __init__(self, input_size: int, head_size: int, dropout=0.1, use_mask: bool = False) -> None:
        """
        A single head of attention for computing the attention mechanism in models like Transformers.

        Parameters
        ----------
        input_size : int
            The size of the input features. This is the dimension of the input to the attention mechanism (token embedding size).
            
        head_size : int
            The size of each attention head, dimensionality of the query, key, and value vectors.
            
        dropout : float, optional, default=0.1
            The dropout rate applied to the attention weights during training.
            
        use_mask : bool, optional, default=False
            If True, applies a mask to ensure that the attention mechanism cannot attend to future tokens.
        """
        self.key = DenseLayer(input_size, head_size)
        self.query = DenseLayer(input_size, head_size)
        self.value = DenseLayer(input_size, head_size)
        self.softmax = Softmax()
        self.dropout = DropoutLayer(dropout)

        self.normalize_factor = head_size**0.5
        self.use_mask = use_mask

    def forward(self, query_input: np.ndarray, context_input: np.ndarray, training: bool = False) -> None:
        """
        Forward pass for the Single Attention Head. Creates output attribute.

        Parameters
        ----------
        query_input : np.ndarray
            Input array of shape `(batch_size, seq_len, input_size)` used in computing query matrix.

        context_input : np.ndarray
            Input array of shape `(batch_size, seq_len, input_size)` used in computing key and value matrices.
            In self-attention, this array will be the same as `query_input`, but in encoder-decoder cross-attention, 
            the `context_input` comes from the encoder.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to the attention
            weights.

        Returns
        -------
        None
        """

        # Compute Q, K and V matrices
        self.query.forward(query_input)
        self.key.forward(context_input)
        self.value.forward(context_input)

        self.q = self.query.output
        self.k = self.key.output
        self.v = self.value.output
        
        # Compute similarity scores (unnormalized attention scores) between all vector pairs of Q and K 
        self.w = np.matmul(self.q, self.k.swapaxes(-2, -1)) / self.normalize_factor

        if self.use_mask:
            B, T, _ = self.q.shape
            # Create lower triangular mask matrix
            mask = np.tril(np.ones((T, T), dtype=bool))
            # Apply mask to future tokens
            self.w = np.where(mask[None, :, :], self.w, -np.inf)

        # Compute attention scores
        self.softmax.forward(self.w)
        self.w = self.softmax.output
        self.attn_weights = self.softmax.output.copy()

        # Add dropout
        self.dropout.forward(self.w, training)
        self.w = self.dropout.output

        # Compute final output
        self.output = np.matmul(self.w, self.v)

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the Single Attention Head. Creates dinputs gradient attributes.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, head_size)`.

        Returns
        -------
        None
        """
        # Calculate dW
        dW = np.matmul(delta, self.v.swapaxes(-2, -1))

        # Backprop dW through Dropout
        self.dropout.backward(dW)
        dW = self.dropout.dinputs

        # Backprop dW through Softmax
        self.softmax.backward(dW)
        # dA we set to Softmax grad
        dA = self.softmax.dinputs

        if self.use_mask:
            B, T, _ = dA.shape
            # Create lower triangular mask matrix
            mask = np.tril(np.ones((T, T), dtype=bool))
            # Apply mask to future tokens
            dA = dA * mask[None, :, :]

        # Grad with respect to query
        dQ = np.matmul(dA, self.k) / self.normalize_factor
        # Grad with respect to key
        dK = np.matmul(dA.swapaxes(-2, -1), self.q) / self.normalize_factor
        # Grad with respect to value
        dV = np.matmul(self.attn_weights.transpose(0, 2, 1), delta)

        # Backprop grads to their respective layers
        self.key.backward(dK)
        self.query.backward(dQ)
        self.value.backward(dV)

        # Grad with respect to decoder inputs
        self.dinputs_query = self.query.dinputs
        # Grad with respect to encoder inputs
        self.dinputs_context = self.key.dinputs + self.value.dinputs

class MultiHeadAttention(Module):

    def __init__(self, n_embed: int, n_heads: int, dropout: float = 0.1, use_mask: bool = False) -> None:
        """
        Multi Head Attention module which consists of multiple `SingleAttentionHead` objects.

        Parameters
        ----------
        n_embed : int
            Dimensionality of input embeddings.

        n_heads : int
            Number of single attention heads to create in the module.
            `n_embed` must be divisible by this number.

        dropout : float, default=0.1
            The dropout rate applied to the attention weights during training.
            
        use_mask : bool, optional, default=False
            If True, applies a mask to ensure that the attention mechanism cannot attend to future tokens.
        """
        # Assertion to ensure n_embed is divisible by n_heads
        assert n_embed % n_heads == 0, f"n_embed ({n_embed}) must be divisible by n_heads ({n_heads})."

        self.n_heads = n_heads
        self.head_size = n_embed // n_heads

        self.attention_heads = [
            SingleAttentionHead(n_embed, self.head_size, dropout, use_mask)
            for _ in range(n_heads)
        ]

        self.output_dense = DenseLayer(n_embed, n_embed)

        self.dropout = DropoutLayer(dropout)

    def forward(self, query_input: np.ndarray, context_input: np.ndarray = None, training: bool = False) -> None:
        """
        Forward pass for the MultiHeadAttention. Creates output attribute.

        Parameters
        ----------
        query_input : np.ndarray
            Input array of shape `(batch_size, seq_len, input_size)` used in computing query matrix.

        context_input : np.ndarray, default=None
            Input array of shape `(batch_size, seq_len, input_size)` used in computing key and value matrices.
            In self-attention, this array will be the same as `query_input`, but in encoder-decoder cross-attention, 
            the `context_input` comes from the encoder.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to the attention
            weights.

        Returns
        -------
        None
        """
        # Check if context_input is given
        self.is_cross_attention = context_input is not None
        if context_input is None:
            # If there is no context input, query input gets passed to heads twice
            context_input = query_input

        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head.forward(query_input, context_input, training)
            head_outputs.append(head.output)

        # Concatenate head results
        concatenated_output = np.concatenate(head_outputs, axis=-1)
        self.output_dense.forward(concatenated_output)
        self.dropout.forward(self.output_dense.output, training)
        self.output = self.dropout.output

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the MultiHeadAttention. Creates dinputs gradient attributes.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, input_size)`.

        Returns
        -------
        None
        """

        self.dropout.backward(delta)

        self.output_dense.backward(self.dropout.dinputs)

        d_concatenated_output = self.output_dense.dinputs

        d_head_outputs = np.split(d_concatenated_output, self.n_heads, axis=-1)

        self.attention_heads[0].backward(d_head_outputs[0])
        self.dinputs_query = np.zeros_like(self.attention_heads[0].dinputs_query)
        self.dinputs_context = np.zeros_like(self.attention_heads[0].dinputs_context)

        self.dinputs_query += self.attention_heads[0].dinputs_query
        self.dinputs_context += self.attention_heads[0].dinputs_context

        for i, head in enumerate(self.attention_heads[1:], start=1):
            head.backward(d_head_outputs[i])
            self.dinputs_query += self.attention_heads[i].dinputs_query
            self.dinputs_context += self.attention_heads[i].dinputs_context

        if self.is_cross_attention:
            self.dinputs = None
        else:
            self.dinputs = self.dinputs_query + self.dinputs_context

class FeedForward(Module):

    def __init__(self, d_model: int, hidden_dim: int = 2048, dropout: float = 0.1, activation: Activation = ReLU) -> None:
        """
        FeedForward network used in Transformer architectures.

        Parameters
        ----------
        d_model : int
            The dimensionality of the input and output embeddings. It should match dimension of MultiHeadAttention
            or other modules preceeding FeedForward.

        hidden_dim : int, default=2048
            The dimensionality of the hidden layer in the FeedForward network.

        dropout : float, default=0.1
            The dropout rate applied to the outputs during training.

        activation : Activation, default=ReLU
            The activation function applied after the first fully connected layer.
        """
        self.fc1 = DenseLayer(d_model, hidden_dim)
        self.activation = activation()
        self.dropout1 = DropoutLayer(dropout)
        self.fc2 = DenseLayer(hidden_dim, d_model)
        self.dropout2 = DropoutLayer(dropout)

    def forward(self, inputs: np.ndarray, training: bool = False) -> None:
        """
        Forward pass for the FeedForward network. Creates output attribute.

        Parameters
        ----------
        inputs : np.ndarray
            Input array of shape `(batch_size, seq_len, d_model)` used in computing query matrix.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to outputs.

        Returns
        -------
        None
        """
        self.fc1.forward(inputs)
        self.activation.forward(self.fc1.output)
        self.dropout1.forward(self.activation.output, training)
        self.fc2.forward(self.dropout1.output)
        self.dropout2.forward(self.fc2.output, training)
        self.output = self.dropout2.output

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the FeedForward. Creates dinputs gradient attributes.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        None
        """
        self.dropout2.backward(delta)
        self.fc2.backward(self.dropout2.dinputs)
        self.dropout1.backward(self.fc2.dinputs)
        self.activation.backward(self.dropout1.dinputs)
        self.fc1.backward(self.activation.dinputs)
        self.dinputs = self.fc1.dinputs
        
class TransformerDecoderBlock(Module):
    
    def __init__(self, d_model: int, n_head: int, dim_ff: int = 2048, dropout: float = 0.1, 
                 activation: Activation = ReLU, layer_norm_eps: float = 1e-5) -> None:
        """
        Transformer Decoder block as presented in the paper 'Attention is all you need' by Vaswani et al. (2017).
        Consists of Masked Multi-Head Attention, Cross Attention, Feed Forward and LayerNorm modules.
        This implementation uses pre-ln (LayerNorm applied first).

        Parameters
        ----------
        d_model : int
            Dimensionality of input embeddings.

        n_heads : int
            Number of heads to create in the MultiHeadAttention modules.
            `d_model` must be divisible by this number.

        dim_ff : int, default=2048
            The dimensionality of the hidden layer in the FeedForward module.

        dropout : float, default=0.1
            The dropout rate applied to module outputs during training.

        activation : Activation, default=ReLU
            The activation function applied in the FeedForward module.

        layer_norm_eps : float, default=1e-5
            A small value added to the denominator for numerical stability when performing layer normalization. 
        """
        self.mask_mha = MultiHeadAttention(d_model, n_head, dropout=dropout, use_mask=True)
        self.ln1 = LayerNorm(d_model, epsilon=layer_norm_eps)

        self.cross_mha = MultiHeadAttention(d_model, n_head, dropout=dropout, use_mask=False)
        self.ln2 = LayerNorm(d_model, epsilon=layer_norm_eps)

        self.ffwd = FeedForward(d_model, hidden_dim=dim_ff, dropout=dropout, activation=activation)
        self.ln3 = LayerNorm(d_model, epsilon=layer_norm_eps)

    def forward(self, x: np.ndarray, enc_output: np.ndarray = None, training: bool = False) -> None:
        """
        Forward pass for the TransformerDecoderBlock. Creates output attribute.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape `(batch_size, seq_len, d_model)`.

        enc_output : np.ndarray, default=None
            Encoder output array used in cross attention of shape `(batch_size, seq_len, d_model)`. Used only in encoder-decoder scenario.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to outputs.

        Returns
        -------
        None
        """
        self.ln1.forward(x, training=training)
        self.mask_mha.forward(self.ln1.output, training=training)
        x = x + self.mask_mha.output

        if enc_output is not None:
            self.has_cross_attn = True
            self.ln2.forward(x, training=training)
            self.cross_mha.forward(self.ln2.output, enc_output, training=training)
            x = x + self.cross_mha.output  # Residual connection
        else:
            self.has_cross_attn = False
            self.cross_mha = None
            self.ln2 = None

        self.ln3.forward(x, training=training)
        self.ffwd.forward(self.ln3.output, training=training)
        x = x + self.ffwd.output

        self.output = x

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the TransformerDecoderBlock. Creates dinputs gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        None
        """
        self.ln3.backward(delta)
        self.ffwd.backward(self.ln3.dinputs)
        d_after_ffwd = delta + self.ffwd.dinputs

        if self.has_cross_attn:
            self.ln2.backward(d_after_ffwd)
            self.cross_mha.backward(self.ln2.dinputs)
            d_after_cross = self.cross_mha.dinputs_query + d_after_ffwd  # Residual gradient
            self.grad_wrt_encoder_output = self.cross_mha.dinputs_context
        else:
            # No cross attention — gradients just pass through
            d_after_cross = d_after_ffwd
            self.grad_wrt_encoder_output = None

        self.ln1.backward(d_after_cross)
        self.mask_mha.backward(self.ln1.dinputs)
        self.dinputs = self.mask_mha.dinputs

class TransformerEncoderBlock(Module):

    def __init__(self, d_model: int, n_head: int, dim_ff: int = 2048, dropout: float = 0.1, 
                 activation: Activation = ReLU, layer_norm_eps: float = 1e-5) -> None:
        """
        Transformer Encoder block as presented in the paper 'Attention is all you need' by Vaswani et al. (2017).
        Consists of Multi-Head Attention, Feed Forward and LayerNorm modules.
        This implementation uses pre-ln (LayerNorm applied first).

        Parameters
        ----------
        d_model : int
            Dimensionality of input embeddings.

        n_head : int
            Number of heads to create in the MultiHeadAttention modules.
            `d_model` must be divisible by this number.

        dim_ff : int, default=2048
            The dimensionality of the hidden layer in the FeedForward module.

        dropout : float, default=0.1
            The dropout rate applied to module outputs during training.

        activation : Activation, default=ReLU
            The activation function applied in the FeedForward module.

        layer_norm_eps : float, default=1e-5
            A small value added to the denominator for numerical stability when performing layer normalization. 
        """
        self.mha = MultiHeadAttention(d_model, n_head, dropout=dropout, use_mask=False)
        self.ln1 = LayerNorm(d_model, epsilon=layer_norm_eps)

        self.ffwd = FeedForward(d_model, hidden_dim=dim_ff, dropout=dropout, activation=activation)
        self.ln2 = LayerNorm(d_model, epsilon=layer_norm_eps)

    def forward(self, x: np.ndarray, training: bool = False) -> None:
        """
        Forward pass for the TransformerEncoderBlock. Creates output attribute.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape `(batch_size, seq_len, d_model)`.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to outputs.

        Returns
        -------
        None
        """
        self.ln1.forward(x, training=training)
        self.mha.forward(self.ln1.output, training=training)
        x = x + self.mha.output

        self.ln2.forward(x, training=training)
        self.ffwd.forward(self.ln2.output, training=training)
        x = x + self.ffwd.output

        self.output = x

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the TransformerEncoderBlock. Creates dinputs gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        None
        """
        d_ffn_residual = delta
        self.ffwd.backward(d_ffn_residual)
        self.ln2.backward(self.ffwd.dinputs)
        d_mha_residual = self.ln2.dinputs + d_ffn_residual 

        self.mha.backward(d_mha_residual)
        self.ln1.backward(self.mha.dinputs)
        self.dinputs = self.ln1.dinputs

class TransformerEncoder(Module):

    def __init__(self, d_model: int, n_head: int, n_enc_layers: int = 2, dim_ff: int = 2048, 
                    dropout: float = 0.1, activation: Activation = ReLU, layer_norm_eps: float = 1e-5) -> None:
        """
        Stack of Transformer Encoder blocks. 

        Parameters
        ----------
        d_model : int
            Dimensionality of input embeddings.

        n_head : int
            Number of heads to create in the MultiHeadAttention modules.
            `d_model` must be divisible by this number.

        n_enc_layers : int, default=2
            The number of layers in the Transformer encoder.

        dim_ff : int, default=2048
            The dimensionality of the feedforward layers in the TransformerEncoderBlock.

        dropout : float, default=0.1
            The dropout rate applied to module outputs during training.

        activation : Activation, default=ReLU
            The activation function used in the feedforward layers.

        layer_norm_eps : float, default=1e-5
            A small value added to the denominator for numerical stability when performing layer normalization.

        Returns
        -------
        None
        """
        self.encoder_layers = [TransformerEncoderBlock(d_model=d_model, 
                                                       n_head=n_head, 
                                                       dim_ff=dim_ff, 
                                                       dropout=dropout, 
                                                       activation=activation,
                                                       layer_norm_eps=layer_norm_eps) for _ in range(n_enc_layers)]
        
    def forward(self, x: np.ndarray, training: bool = False) -> None:
        """
        Forward pass for the TransformerEncoder. Creates output attribute.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape `(batch_size, seq_len, d_model)`.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to outputs.

        Returns
        -------
        None
        """
        self.encoder_layers[0].forward(x, training)
        for idx, enc in enumerate(self.encoder_layers[1:], start=1):
            enc.forward(self.encoder_layers[idx-1].output, training)
        self.output = self.encoder_layers[-1].output

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the TransformerEncoder. Creates dinputs gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        None
        """
        self.encoder_layers[-1].backward(delta)
        for idx, enc in reversed(list(enumerate(self.encoder_layers[:-1]))):
            enc.backward(self.encoder_layers[idx + 1].dinputs)
        self.dinputs = self.encoder_layers[0].dinputs

class TransformerDecoder(Module):

    def __init__(self, d_model: int, n_head: int, n_dec_layers: int = 2, dim_ff: int = 2048, 
                    dropout: float = 0.1, activation: Activation = ReLU, layer_norm_eps: float = 1e-5) -> None:
        """
        Stack of Transformer Decoder blocks. 

        Parameters
        ----------
        d_model : int
            Dimensionality of input embeddings.

        n_head : int
            Number of heads to create in the MultiHeadAttention modules.
            `d_model` must be divisible by this number.

        n_dec_layers : int, default=2
            The number of layers in the Transformer decoder.

        dim_ff : int, default=2048
            The dimensionality of the feedforward layers in the TransformerDecoderBlock.

        dropout : float, default=0.1
            The dropout rate applied to module outputs during training.

        activation : Activation, default=ReLU
            The activation function used in the feedforward layers.

        layer_norm_eps : float, default=1e-5
            A small value added to the denominator for numerical stability when performing layer normalization.

        Returns
        -------
        None
        """

        self.decoder_layers = [TransformerDecoderBlock(d_model=d_model, 
                                                       n_head=n_head, 
                                                       dim_ff=dim_ff, 
                                                       dropout=dropout,
                                                       activation=activation, 
                                                       layer_norm_eps=layer_norm_eps) for _ in range(n_dec_layers)]
        
    def forward(self, x: np.ndarray, enc_output: np.ndarray = None, training: bool = False) -> None:
        """
        Forward pass for the TransformerDecoder. Creates output attribute.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape `(batch_size, seq_len, d_model)`.

        enc_output : np.ndarray, default=None
            Encoder output array used in cross attention of shape `(batch_size, seq_len, d_model)`. Used only in encoder-decoder scenario.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to outputs.

        Returns
        -------
        None
        """
        self.decoder_layers[0].forward(x, enc_output=enc_output, training=training)
        for idx, dec in enumerate(self.decoder_layers[1:], start=1):
            dec.forward(self.decoder_layers[idx-1].output, enc_output=enc_output, training=training)
        self.output = self.decoder_layers[-1].output

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the TransformerDecoder. Creates dinputs gradient attribute.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        None
        """
        self.decoder_layers[-1].backward(delta)
        for idx, dec in reversed(list(enumerate(self.decoder_layers[:-1]))):
            dec.backward(self.decoder_layers[idx + 1].dinputs)
        self.dinputs = self.decoder_layers[0].dinputs

class Transformer(Module):

    def __init__(self, d_model: int, n_head: int, n_enc_layers: int = 2, n_dec_layers: int = 2, dim_ff: int = 2048, 
                    dropout: float = 0.1, activation: Activation = ReLU, layer_norm_eps: float = 1e-5) -> None:
        """
        Original Transformer architecture from the paper 'Attention is All You Need' by Vaswani et al. (2017).
        This architecture unites Transformer Encoder and Transformer Decoder.

        Parameters
        ----------
        d_model : int
            Dimensionality of input embeddings.

        n_head : int
            The number of attention heads in each attention layer.

        n_enc_layers : int, default=2
            The number of layers in the Transformer encoder.

        n_dec_layers : int, default=2
            The number of layers in the Transformer decoder.

        dim_ff : int, default=2048
            The dimensionality of the feedforward layers in the encoder and decoder.

        dropout : float, default=0.1
            The dropout rate applied to layers during training to help prevent overfitting.

        activation : Activation, default=ReLU
            The activation function used in the feedforward layers.

        layer_norm_eps : float, default=1e-5
            A small value added to the denominator for numerical stability when performing layer normalization.

        Returns
        -------
        None
        """
        self.encoder = TransformerEncoder(d_model=d_model,
                                            n_head=n_head,
                                            n_enc_layers=n_enc_layers,
                                            dim_ff=dim_ff,
                                            dropout=dropout,
                                            activation=activation,
                                            layer_norm_eps=layer_norm_eps)

        self.decoder = TransformerDecoder(d_model=d_model,
                                            n_head=n_head,
                                            n_dec_layers=n_dec_layers,
                                            dim_ff=dim_ff,
                                            dropout=dropout,
                                            activation=activation,
                                            layer_norm_eps=layer_norm_eps)

    def _decoder_backward(self, delta: np.ndarray) -> None:
        """
        Helper method for executing TransformerDecoder backward pass. Computes gradient with respect to encoder output.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        None
        """
        self.decoder.backward(delta)

        # Gradient for encoder comes from decoder    
        grad_wrt_encoder_output_total = np.zeros_like(self.encoder.output)
        for idx, dec in enumerate(self.decoder.decoder_layers):
            grad_wrt_encoder_output_total += dec.grad_wrt_encoder_output

        self.grad_wrt_encoder_output_total = grad_wrt_encoder_output_total
        
    def forward(self, inputs_enc: np.ndarray, inputs_dec: np.ndarray, training: bool = False) -> None:
        """
        Forward pass for the Transformer. Creates output attribute.

        Parameters
        ----------
        inputs_enc : np.ndarray
            Encoder input array of shape `(batch_size, seq_len, d_model)`.

        inputs_dec : np.ndarray, default=None
            Decoder input array of shape `(batch_size, seq_len, d_model)`.

        training : bool, default=False
            A flag indicating whether the model is in training mode. If True, dropout is applied to outputs.

        Returns
        -------
        None
        """
        self.encoder.forward(inputs_enc, training)
        self.decoder.forward(inputs_dec, self.encoder.output, training)
        self.output = self.decoder.output

    def backward(self, delta: np.ndarray) -> None:
        """
        Backward pass for the Transformer.

        Parameters
        ----------
        delta : np.ndarray
            Gradient array of shape `(batch_size, seq_len, d_model)`.

        Returns
        -------
        None
        """
        self._decoder_backward(delta)
        self.encoder.backward(self.grad_wrt_encoder_output_total)

    def _print_grad_norms(self):
        print(f'-------------------GRADS------------------')
        for i in range(len(self.decoder_layers)):
            print(f'Decoder ffwd {i}: {np.linalg.norm(self.decoder_layers[i].ffwd.dinputs)}')
            print(f'Decoder cross MHA {i}: {np.linalg.norm(self.decoder_layers[i].cross_mha.dinputs_query)} | {np.linalg.norm(self.decoder_layers[i].cross_mha.dinputs_context)}')
            print(f'Decoder mask MHA {i}: {np.linalg.norm(self.decoder_layers[i].mask_mha.dinputs)}')
            print(f'Decoder grad for encoder {i}: {np.linalg.norm(self.decoder_layers[i].grad_wrt_encoder_output)}')
        for i in range(len(self.encoder)):
            print(f'Encoder ffwd {i}: {np.linalg.norm(self.encoder_layers[i].ffwd.dinputs)}')
            print(f'Encoder MHA {i}: {np.linalg.norm(self.encoder_layers[i].mha.dinputs_query)} | {np.linalg.norm(self.encoder_layers[i].mha.dinputs_context)}')
            print(f'Encoder dipnuts {i}: {np.linalg.norm(self.encoder_layers[i].dinputs)}')
        print(f'------------------------------------------')

class VAE(Module):

    def __init__(self, encoder: Module, decoder: Module, 
                       mu: Module, logvar: Module, 
                       loss_function = None, optimizer = None):
        """
        Variational Autoencoder generative model.

        Parameters
        ----------
        encoder: Module
            Encoder part of VAE.

        decoder: Module
            Decoder part of VAE.

        mu: Module
            Feed-forward network for predicting mean of distribution.

        logvar: Module
            Feed-forward network for predicting log variance of distribution.
        
        loss_function : Loss, default=None
            Loss function.
        
        optimizer : Optimizer, default=None
            Optimizer algorithm.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.mu = mu
        self.logvar = logvar
        self.loss_function = loss_function
        self.optimizer = optimizer

    def _reparametrize(self, mu: np.ndarray, logvar: np.ndarray) -> None:
        """
        Helper method for the reparametrization trick. Creates latent variable attribute.

        Parameters
        ----------
        mu : np.ndarray
            Mean used to calculate latent variable.

        logvar : np.ndarray
            Log variance used to calculate latent variable.

        Returns
        -------
        None
        """
        self.std = np.exp(0.5 * logvar)
        self.eps = np.random.randn(*self.std.shape)  # random noise
        self.z = mu + self.eps * self.std

    def _reparametrize_backward(self, decoder_dinputs: np.ndarray, grad_kl_mu: np.ndarray, grad_kl_logvar: np.ndarray) -> None:
        """
        Helper method for the backpropagation of reparametrization trick. Creates mu and logvar gradient attributes.

        Parameters
        ----------
        decoder_dinputs : np.ndarray
            Decoder upstream gradient.

        grad_kl_mu : np.ndarray
            Mean upstream gradient from KL loss.

        grad_kl_logvar : np.ndarray
            Log variance upstream gradient from KL loss.

        Returns
        -------
        None
        """
        self.dmu = decoder_dinputs * 1 + grad_kl_mu
        self.dlogvar = decoder_dinputs * (0.5 * self.eps * self.std) + grad_kl_logvar

    def forward(self, x: np.ndarray, training: bool = False) -> None:
        """
        Forward pass for the Variational Autoencoder model. Creates output attribute.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        training : bool, default=False
            A flag indicating whether the model is in training mode (`True`) or inference mode (`False`).
            This controls dropout, which is only applied during training.

        Returns
        -------
        None
        """

        # Pass data through encoder
        self.encoder.forward(x, training=training)

        # Compute distribution parameters from encoder output
        self.mu.forward(self.encoder.output, training=training)
        self.logvar.forward(self.encoder.output, training=training)

        # Using distribution parameters apply the reparametrization
        self._reparametrize(self.mu.output, self.logvar.output)

        # Pass sampled latent variable from reparametrization through decoder 
        self.decoder.forward(self.z, training=training)

        # Network output is the decoder output
        self.output = self.decoder.output

    def backward(self, y_true: np.ndarray, kl_beta: float) -> None:
        """
        Backward pass for the Variational Autoencoder model.

        Parameters
        ----------
        y_true : np.ndarray
            Original inputs used to compute the loss function gradient.

        kl_beta : float
            Coefficient used to scale the KL divergence loss and gradients during the KL warmup phase.

        Returns
        -------
        None
        """

        # Compute loss function grads
        self.loss_function.backward(self.output, y_true, self.mu.output, self.logvar.output, kl_beta)
        recon_grad, mu_grad, logvar_grad = self.loss_function.recon_loss.dinputs, self.loss_function.dmu, self.loss_function.dlogvar

        # Backprop reconstruction loss grad through decoder
        self.decoder.backward(recon_grad)

        # Backprop decoder grad, loss function grads with respect to mu and logvar through reparametrize
        self._reparametrize_backward(self.decoder.dinputs, mu_grad, logvar_grad)

        # Backprop distribution grads through their respective ffwd nets
        self.mu.backward(self.dmu)
        self.logvar.backward(self.dlogvar)

        # Add up the distribution parameter grads for encoder
        total_enc_grad = self.mu.dinputs + self.logvar.dinputs

        # Backprop through encoder
        self.encoder.backward(total_enc_grad)

    def train(self, X: np.ndarray, epochs: int = 1000, batch_size: int = None, print_every: int = None, kl_warmup_steps: int = 0) -> None:
        """
        Train the VAE model for a specified number of epochs, updating parameters using backpropagation.

        Parameters
        ----------
        X : np.ndarray
            Input and ground truth output data for the model.

        epochs : int, default=1000
            The number of epochs to train the model.

        batch_size : int, default=None
            The size of the batch to use for each training step. If `None`, the entire dataset is used
            for each forward/backward pass.

        print_every : int, default=None
            If provided, will print the current loss value every 
            `print_every` epochs.

        kl_warmup_steps : int, default=0
            Number of training epochs to apply the KL divergence downscaling. 
            This is done in order for the model to learn to reconstruct data before regularizing the latent space.

        Returns
        -------
        None
        """

        # Store recon and kl losses separately
        self.rec_loss = []
        self.kl_loss = []

        for i in range(epochs + 1):

            if kl_warmup_steps and kl_warmup_steps > 0:
                # Linear kl_beta during warmup
                self.kl_beta = min(1.0, (i + 1) / float(kl_warmup_steps))
            else:
                self.kl_beta = 1.0

            if batch_size is None:

                # Forward pass
                self.forward(X, training=True)

                # Backward pass
                self.backward(X, self.kl_beta)

                # Update parameters
                self.optimizer.pre_update_parameters()
                self.optimizer.update_parameters(self)
                self.optimizer.post_update_parameters()

                # Caluculate and record losses
                rec_l, kl_l, loss = self.loss_function.calculate(self.output, X, self.mu.output, self.logvar.output, self.kl_beta)
                self.rec_loss.append(rec_l)
                self.kl_loss.append(kl_l)

                if print_every is not None:
                    if not i % print_every:
                        print(f'===== EPOCH : {i} ===== LOSS : {loss:.5f} (Recon: {rec_l:.5f} | KL: {kl_l:.5f})')

            else:

                batch_loss = 0
                rec_loss = 0
                kl_loss = 0
                num_batches = 0

                for j in range(0, len(X), batch_size):

                    # Subset data into a batch
                    batch_X = X[j:j+batch_size, :]

                    # Forward pass
                    self.forward(batch_X)

                    # Calculate current batch loss
                    rec, kl, loss = self.loss_function.calculate(self.output, batch_X, self.mu.output, self.logvar.output)

                    batch_loss += loss
                    rec_loss += rec
                    kl_loss += kl
                    num_batches += 1

                    # Backward pass
                    self.backward(batch_X, self.kl_beta)

                    # Update parameters
                    self.optimizer.pre_update_parameters()
                    self.optimizer.update_parameters(self)
                    self.optimizer.post_update_parameters()

                # Record average batch losses
                avg_batch_loss = batch_loss / num_batches
                avg_rec_loss = rec_loss / num_batches
                avg_kl_loss = kl_loss / num_batches

                self.rec_loss.append(avg_rec_loss)
                self.kl_loss.append(avg_kl_loss)

                if print_every is not None:
                    if not i % print_every:
                        print(f'===== EPOCH : {i} ===== LOSS : {avg_batch_loss:.5f} (Recon: {avg_rec_loss:.5f} | KL: {avg_kl_loss:.5f}) =====')