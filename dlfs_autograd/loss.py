import numpy as np
from .tensor import Tensor

class BCE_Loss:

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
            bce_grad = (pred_clipped - targets) / (pred_clipped * (1 - pred_clipped))
            y_pred.grad += bce_grad * result.grad

        # Store tensor result
        result._backward = _backward
        self.output = result

        # For display purposes
        #self.value = result.data
        return result

    def backward(self):
        self.output.backward()

class MSE_Loss:

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
            mse_grad = predictions - targets
            y_pred.grad += mse_grad * result.grad

        # Store tensor result
        result._backward = _backward
        self.output = result

        # For display purposes
        self.value = result.data

    def backward(self):
        self.output.backward()