import numpy as np
from .base import Module
from .layers import *
from .activation import Softmax, ReLU, GELU

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


class SingleAttentionHead(Module):

    def __init__(self, input_size, head_size, dropout=0.1, use_mask=False):
        self.key = DenseLayer(input_size, head_size)
        self.query = DenseLayer(input_size, head_size)
        self.value = DenseLayer(input_size, head_size)
        self.softmax = Softmax()
        self.dropout = DropoutLayer(dropout)

        self.normalize_factor = head_size**0.5
        self.use_mask = use_mask

    def forward(self, query_input, context_input, training):

        self.query.forward(query_input)
        self.key.forward(context_input)
        self.value.forward(context_input)

        self.q = self.query.output
        self.k = self.key.output
        self.v = self.value.output
        
        self.w = np.matmul(self.q, self.k.swapaxes(-2, -1)) / self.normalize_factor

        if self.use_mask:
            B, T, _ = self.q.shape
            mask = np.tril(np.ones((T, T), dtype=bool))  # causal mask
            self.w = np.where(mask[None, :, :], self.w, -np.inf)


        self.softmax.forward(self.w)
        self.w = self.softmax.output
        self.attn_weights = self.softmax.output.copy()
        self.dropout.forward(self.w, training)
        self.w = self.dropout.output
        self.output = np.matmul(self.w, self.v)

    def backward(self, delta):
        d_w = np.matmul(delta, self.v.swapaxes(-2, -1))

        self.dropout.backward(d_w)
        d_w = self.dropout.dinputs

        self.softmax.backward(d_w)
        d_w = self.softmax.dinputs

        if self.use_mask:
            B, T, _ = d_w.shape
            mask = np.tril(np.ones((T, T), dtype=bool))
            d_w = d_w * mask[None, :, :]

        d_q = np.matmul(d_w, self.k) / self.normalize_factor
        d_k = np.matmul(d_w.swapaxes(-2, -1), self.q) / self.normalize_factor

        d_v_input = np.matmul(self.attn_weights.transpose(0, 2, 1), delta)

        self.key.backward(d_k)
        self.query.backward(d_q)
        self.value.backward(d_v_input)

        self.dinputs_query = self.query.dinputs
        self.dinputs_context = self.key.dinputs + self.value.dinputs

class MultiHeadAttention(Module):

    def __init__(self, n_embed, n_heads, dropout=0.1, use_mask=False):
        self.n_heads = n_heads
        self.head_size = n_embed // n_heads

        self.attention_heads = [
            SingleAttentionHead(n_embed, self.head_size, dropout, use_mask)
            for _ in range(n_heads)
        ]

        self.output_dense = DenseLayer(n_embed, n_embed)

        self.dropout = DropoutLayer(dropout)

    def forward(self, x, training, context=None):
        self.is_cross_attention = context is not None
        if context is None:
            context = x

        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head.forward(x, context, training)
            head_outputs.append(head.output)

        concatenated_output = np.concatenate(head_outputs, axis=-1)
        self.output_dense.forward(concatenated_output)
        self.dropout.forward(self.output_dense.output, training)
        self.output = self.dropout.output

    def backward(self, delta):

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

        #self.dinputs_query /= self.n_heads
        #self.dinputs_context /= self.n_heads

        if self.is_cross_attention:
            self.dinputs = None
        else:
            self.dinputs = self.dinputs_query + self.dinputs_context

class FeedForward(Module):
    def __init__(self, d_model=512, hidden_dim=2048, dropout=0.1, activation=ReLU()):
        self.fc1 = DenseLayer(d_model, hidden_dim)
        self.activation = activation
        self.dropout1 = DropoutLayer(dropout)
        self.fc2 = DenseLayer(hidden_dim, d_model)
        self.dropout2 = DropoutLayer(dropout)

    def forward(self, inputs, training):
        self.fc1.forward(inputs)
        self.activation.forward(self.fc1.output)
        self.dropout1.forward(self.activation.output, training)
        self.fc2.forward(self.dropout1.output)
        self.dropout2.forward(self.fc2.output, training)
        self.output = self.dropout2.output

    def backward(self, delta):
        self.dropout2.backward(delta)
        self.fc2.backward(self.dropout2.dinputs)
        self.dropout1.backward(self.fc2.dinputs)
        self.activation.backward(self.dropout1.dinputs)
        self.fc1.backward(self.activation.dinputs)
        self.dinputs = self.fc1.dinputs
        
class TransformerDecoderBlock(Module):
    
    def __init__(self, d_model=512, n_head=4, dim_ff=2048, dropout=0.1, activation=ReLU(), layer_norm_eps=1e-5):
        self.mask_mha = MultiHeadAttention(d_model, n_head, dropout=dropout, use_mask=True)
        self.ln1 = LayerNorm(d_model, epsilon=layer_norm_eps)

        self.cross_mha = MultiHeadAttention(d_model, n_head, dropout=dropout, use_mask=False)
        self.ln2 = LayerNorm(d_model, epsilon=layer_norm_eps)

        self.ffwd = FeedForward(d_model, hidden_dim=dim_ff, dropout=dropout, activation=activation)
        self.ln3 = LayerNorm(d_model, epsilon=layer_norm_eps)

    def forward(self, x, enc_output=None, training=False):
        self.ln1.forward(x, training)
        self.mask_mha.forward(self.ln1.output, training)
        x = x + self.mask_mha.output

        if enc_output is not None:
            self.has_cross_attn = True
            self.ln2.forward(x, training)
            self.cross_mha.forward(self.ln2.output, training, context=enc_output)
            x = x + self.cross_mha.output  # Residual connection
        else:
            self.has_cross_attn = False
            self.cross_mha = None
            self.ln2 = None

        self.ln3.forward(x, training)
        self.ffwd.forward(self.ln3.output, training)
        x = x + self.ffwd.output

        self.output = x

    def backward(self, delta):
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
    def __init__(self, d_model=512, n_head=4, dim_ff=2048, dropout=0.1, activation=ReLU(), layer_norm_eps=1e-5):
        self.mha = MultiHeadAttention(d_model, n_head, dropout=dropout, use_mask=False)
        self.ln1 = LayerNorm(d_model, epsilon=layer_norm_eps)

        self.ffwd = FeedForward(d_model, hidden_dim=dim_ff, dropout=dropout, activation=activation)
        self.ln2 = LayerNorm(d_model, epsilon=layer_norm_eps)

    def forward(self, x, training):
        self.ln1.forward(x, training)
        self.mha.forward(self.ln1.output, training)
        x = x + self.mha.output

        self.ln2.forward(x, training)
        self.ffwd.forward(self.ln2.output, training)
        x = x + self.ffwd.output

        self.output = x

    def backward(self, delta):
        d_ffn_residual = delta
        self.ffwd.backward(d_ffn_residual)
        self.ln2.backward(self.ffwd.dinputs)
        d_mha_residual = self.ln2.dinputs + d_ffn_residual 

        self.mha.backward(d_mha_residual)
        self.ln1.backward(self.mha.dinputs)
        self.dinputs = self.ln1.dinputs

class Transformer(Module):

    def __init__(self, d_model=512, n_head = 4, n_enc_layers=2, n_dec_layers=2, dim_ff=2048, 
                       dropout=0.1, activation=ReLU(), layer_norm_eps=1e-5):
    
        self.encoder_layers = [TransformerEncoderBlock(d_model=d_model, 
                                                n_head=n_head, 
                                                dim_ff=dim_ff, 
                                                dropout=dropout, 
                                                layer_norm_eps=layer_norm_eps) for _ in range(n_enc_layers)]

        self.decoder_layers = [TransformerDecoderBlock(d_model=d_model, 
                                                n_head=n_head, 
                                                dim_ff=dim_ff, 
                                                dropout=dropout, 
                                                layer_norm_eps=layer_norm_eps) for _ in range(n_dec_layers)]

    def _encoder_forward(self, inputs_enc, training=False):
        self.encoder_layers[0].forward(inputs_enc, training)
        for idx, enc in enumerate(self.encoder_layers[1:], start=1):
            enc.forward(self.encoder_layers[idx-1].output, training)
        self.encoder_output = self.encoder_layers[-1].output

    def _decoder_forward(self, inputs_dec, outputs_enc, training=False):
        self.decoder_layers[0].forward(inputs_dec, outputs_enc, training)
        for idx, dec in enumerate(self.decoder_layers[1:], start=1):
            dec.forward(self.decoder_layers[idx-1].output, self.encoder_layers[-1].output, training)
        self.decoder_output = self.decoder_layers[-1].output

    def _encoder_backward(self, delta):
        self.encoder_layers[-1].backward(delta)
        for idx, enc in reversed(list(enumerate(self.encoder_layers[:-1]))):
            enc.backward(self.encoder_layers[idx + 1].dinputs)
        self.encoder_dinputs = self.encoder_layers[0].dinputs

    def _decoder_backward(self, delta):
        self.decoder_layers[-1].backward(delta)
        for idx, dec in reversed(list(enumerate(self.decoder_layers[:-1]))):
            dec.backward(self.decoder_layers[idx + 1].dinputs)
        self.decoder_dinputs = self.decoder_layers[0].dinputs
        
        grad_wrt_encoder_output_total = np.zeros_like(self.decoder_layers[0].output)
        for idx, dec in enumerate(self.decoder_layers):
            grad_wrt_encoder_output_total += dec.grad_wrt_encoder_output

        self.grad_wrt_encoder_output_total = grad_wrt_encoder_output_total
        
    def forward(self, inputs_enc, inputs_dec, training=False):
        self._encoder_forward(inputs_enc, training)
        self._decoder_forward(inputs_dec, self.encoder_output, training)
        self.output = self.decoder_output

    def backward(self, delta):
        self._decoder_backward(delta)
        self._encoder_backward(self.grad_wrt_encoder_output_total)

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

class TransformerEncoder(Module):

    def __init__(self, d_model=512, n_head = 4, n_enc_layers=2, dim_ff=2048, 
                    dropout=0.1, activation=ReLU(), layer_norm_eps=1e-5):

        self.encoder_layers = [TransformerEncoderBlock(d_model=d_model, 
                                                    n_head=n_head, 
                                                    dim_ff=dim_ff, 
                                                    dropout=dropout, 
                                                    layer_norm_eps=layer_norm_eps) for _ in range(n_enc_layers)]
        
    def forward(self, inputs_enc, training=False):
        self.encoder_layers[0].forward(inputs_enc, training)
        for idx, enc in enumerate(self.encoder_layers[1:], start=1):
            enc.forward(self.encoder_layers[idx-1].output, training)
        self.output = self.encoder_layers[-1].output

    def backward(self, delta):
        self.encoder_layers[-1].backward(delta)
        for idx, enc in reversed(list(enumerate(self.encoder_layers[:-1]))):
            enc.backward(self.encoder_layers[idx + 1].dinputs)
        self.dinputs = self.encoder_layers[0].dinputs

class TransformerDecoder(Module):

    def __init__(self, d_model=512, n_head = 4, n_dec_layers=2, dim_ff=2048, 
                    dropout=0.1, activation=ReLU(), layer_norm_eps=1e-5):

        self.decoder_layers = [TransformerDecoderBlock(d_model=d_model, 
                                                       n_head=n_head, 
                                                       dim_ff=dim_ff, 
                                                       dropout=dropout, 
                                                       layer_norm_eps=layer_norm_eps) for _ in range(n_dec_layers)]
        
    def forward(self, inputs_dec, training=False):
        self.decoder_layers[0].forward(inputs_dec, training=training)
        for idx, dec in enumerate(self.decoder_layers[1:], start=1):
            dec.forward(self.decoder_layers[idx-1].output, training=training)
        self.output = self.decoder_layers[-1].output

    def backward(self, delta):
        self.decoder_layers[-1].backward(delta)
        for idx, dec in reversed(list(enumerate(self.decoder_layers[:-1]))):
            dec.backward(self.decoder_layers[idx + 1].dinputs)
        self.dinputs = self.decoder_layers[0].dinputs