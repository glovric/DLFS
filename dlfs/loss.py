import numpy as np
from .base import Loss

class BCE_Loss(Loss):

    def __init__(self) -> None:
        """
        Binary Cross Entropy loss function.
        """
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Binary Cross Entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        loss : float
        """
        # Clip y_pred so logarithm doesn't become unstable
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return np.mean(loss)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backward pass using Binary Cross Entropy loss. Creates gradient attribute with respect to predicted values.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        None
        """
        # Clip y_pred so the denominator doesn't become unstable
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # Derivative of BCE with respect to y_pred
        self.dinputs = (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped))

class MSE_Loss(Loss):

    def __init__(self) -> None:
        """
        Mean Squared Error loss function.
        """
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate Mean Squared Error loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        loss : float
        """
        loss = 0.5 * (y_pred - y_true)**2
        return np.mean(loss)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backward pass using Mean Squared Error loss. Creates gradient attribute with respect to predicted values.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        Returns
        -------
        None
        """
        # Derivative of MSE with respect to y_pred
        self.dinputs = (y_pred - y_true)

class CCE_Loss(Loss):

    def calculate(self, y_pred, y_true):
        samples = range(len(y_pred))
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[samples, y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        return np.mean(-np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        samples = len(y_pred)

        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        self.dinputs = (y_pred - y_true) / samples
