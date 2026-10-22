import numpy as np
from .base import Loss
from .tensor import Tensor

class BCE_Loss(Loss):

    def calculate(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)

        predictions = y_pred.data
        targets = y_true.data

        pred_clipped = np.clip(predictions, 1e-7, 1-1e-7)
        loss = -(targets * np.log(pred_clipped) + (1 - targets) * np.log(1 - pred_clipped))

        result = Tensor(
            np.mean(np.sum(loss, axis=1)),
            _children=(y_pred,),
            _op="bce_loss"
        )

        def _backward():
            if y_pred.grad is None:
                y_pred.init_grad()
            bce_grad = (pred_clipped - targets) / (pred_clipped * (1 - pred_clipped))
            y_pred.grad += bce_grad * result.grad

        # Store tensor result
        result._backward = _backward
        return result

class MSE_Loss(Loss):

    def calculate(self, y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)

        predictions = y_pred.data
        targets = y_true.data

        loss = 0.5 * (predictions - targets)**2

        result = Tensor(
            np.mean(loss),
            _children=(y_pred,),
            _op="mse_loss"
        )

        def _backward():
            if y_pred.grad is None:
                y_pred.init_grad()
            mse_grad = predictions - targets
            y_pred.grad += mse_grad * result.grad

        # Store tensor result
        result._backward = _backward
        self.output = result

        return result

class CCE_Loss(Loss):

    def __init__(self, from_logits: bool = False) -> None:
        """
        Categorical Cross Entropy loss function.
        """
        self.from_logits = from_logits

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Categorical Cross Entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values of shape `(batch_size, num_classes)`

        y_true : np.ndarray
            True labels, either as class indices of shape `(batch_size, )` or one-hot encoded labels of shape `(batch_size, num_classes)`.

        Returns
        -------
        loss : float

        Notes
        -----
        - If `from_logits=True`, softmax is applied to the `y_pred` values before calculating the loss.
        """

        #if self.from_logits:
            #y_pred = Softmax.calculate(y_pred)

        y_pred = y_pred if isinstance(y_pred, Tensor) else Tensor(y_pred)
        y_true = y_true if isinstance(y_true, Tensor) else Tensor(y_true)

        samples = np.arange(len(y_pred.data))
        y_pred_clipped = np.clip(y_pred.data, 1e-7, 1 - 1e-7)

        if len(y_true.data.shape) == 1:
            correct_confidences = y_pred_clipped[samples, y_true.data]
        elif len(y_true.data.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true.data, axis=-1)

        result = Tensor(
            np.mean(-np.log(correct_confidences)),
            _children=(y_pred, ),
            _op="cce_loss"
        )

        def _backward():
            if y_pred.grad is None:
                y_pred.init_grad()
                
            samples = len(y_pred.data)

            if len(y_true.data.shape) == 1:
                y_true.data = np.eye(y_pred.data.shape[1])[y_true.data]

            #if self.from_logits:
            #    self.dinputs = (y_pred.data - y_true.data) / samples
            #else:
            y_pred.grad = (-y_true.data / y_pred.data) / samples

        result._backward = _backward
        return result