import numpy as np
from .base import Loss
from .tensor.base import Tensor, ensure_tensor

class BCE_Loss(Loss):

    def calculate(self, y_pred, y_true):
        y_pred = ensure_tensor(y_pred)
        y_true = ensure_tensor(y_true)

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
            bce_grad = (pred_clipped - targets) / (pred_clipped * (1 - pred_clipped))
            up_grad = result.grad * bce_grad
            y_pred.grad = up_grad if y_pred.grad is None else y_pred.grad + up_grad

        # Store tensor result
        result._backward = _backward
        return result

class MSE_Loss(Loss):

    def calculate(self, y_pred, y_true):
        y_pred = ensure_tensor(y_pred)
        y_true = ensure_tensor(y_true)

        predictions = y_pred.data
        targets = y_true.data

        loss = 0.5 * (predictions - targets)**2

        result = Tensor(
            np.mean(loss),
            _children=(y_pred,),
            _op="mse_loss"
        )

        def _backward():
            mse_grad = predictions - targets
            up_grad = mse_grad * result.grad
            y_pred.grad = up_grad if y_pred.grad is None else y_pred.grad + up_grad

        # Store tensor result
        result._backward = _backward
        self.output = result

        return result

class CCE_Loss(Loss):

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        #if self.from_logits:
            #y_pred = Softmax.calculate(y_pred)

        y_pred = ensure_tensor(y_pred)
        y_true = ensure_tensor(y_true)

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
            samples = len(y_pred.data)

            if len(y_true.data.shape) == 1:
                y_true.data = np.eye(y_pred.data.shape[1])[y_true.data]

            #if self.from_logits:
            #    self.dinputs = (y_pred.data - y_true.data) / samples
            #else:
            cce_grad = (-y_true.data / y_pred.data) / samples
            up_grad = cce_grad * result.grad
            y_pred.grad = up_grad if y_pred.grad is None else y_pred.grad + up_grad

        result._backward = _backward
        return result