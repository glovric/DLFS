import numpy as np
from .base import Loss
from .activation import Softmax, Sigmoid

class BCE_Loss(Loss):

    def __init__(self, from_logits=False) -> None:
        """
        Binary Cross Entropy loss function.
        """
        self.from_logits = from_logits

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
        if self.from_logits:
            x = y_pred
            loss = np.maximum(x, 0) - x * y_true + np.log1p(np.exp(-np.abs(x)))
        else:
            # Clip y_pred so logarithm doesn't become unstable
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
            loss = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        return np.mean(np.sum(loss, axis=1))
    
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
        if self.from_logits:
            sig = Sigmoid.calculate(y_pred)
            self.dinputs = (sig - y_true) / y_true.shape[0]
        else:
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

        if self.from_logits:
            y_pred = Softmax.calculate(y_pred)

        samples = np.arange(len(y_pred))
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[samples, y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=-1)

        return np.mean(-np.log(correct_confidences))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Backward pass using Categorical Cross Entropy loss. Creates gradient attribute with respect to predicted values.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values of shape `(batch_size, num_classes)`
        
        y_true : np.ndarray
            True labels, either as class indices of shape `(batch_size, )` or one-hot encoded labels of shape `(batch_size, num_classes)`.

        Returns
        -------
        None

        Notes
        -----
        - If `y_true` is provided as class indices of shape `(batch_size, )`, the method will one-hot encode it.
        - If `from_logits=True`, softmax is applied to the `y_pred` values before calculating the gradients.
        """

        if self.from_logits:
            y_pred = Softmax.calculate(y_pred)

        samples = len(y_pred)

        if len(y_true.shape) == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        if self.from_logits:
            self.dinputs = (y_pred - y_true) / samples
        else:
            self.dinputs = -y_true / y_pred
            self.dinputs = self.dinputs / samples

class VAE_Loss(Loss):

    def __init__(self, recon_loss: Loss) -> None:
        """
        Variational Autoencoder loss function consisting of reconstruction loss and KL divergence loss.

        Parameters
        ----------
        recon_loss : Loss
            Loss function used to learn reconstruction of data.
        """
        self.recon_loss = recon_loss()

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray, mu: np.ndarray, logvar: np.ndarray, kl_beta: float = 1.0) -> tuple[float, float, float]:
        """
        Calculate Variational Autoencoder loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        mu : np.ndarray
            Mean of the latent space distribution.

        logvar : np.ndarray
            Log variance of the latent space distribution.

        kl_beta : float, default=1.0
            Scaling term of the KL divergence loss. Used during KL warmup in training.

        Returns
        -------
        rec_loss, kl_loss, total_loss : tuple[float, float, float]
            Separate recon, KL, and cumulative loss.
        """

        # Calculate recon loss
        recon_loss = self.recon_loss.calculate(y_pred, y_true)

        # Calculate KL divergence loss
        kl_loss = -0.5* np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))

        # Add up two losses
        total_loss = recon_loss + kl_beta * kl_loss

        return recon_loss, kl_beta * kl_loss, total_loss

    def backward(self, y_pred, y_true, mu, logvar, kl_beta=1.0) -> None:
        """
        Backward pass using Variational Autoencoder loss. Creates gradient attributes with respect to mean and log variance.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.

        y_true : np.ndarray
            True values.

        mu : np.ndarray
            Mean of the latent space distribution.

        logvar : np.ndarray
            Log variance of the latent space distribution.

        kl_beta : float, default=1.0
            Scaling term of the KL divergence loss. Used during KL warmup in training.

        Returns
        -------
        None
        """

        # Calculate recon loss gradient
        self.recon_loss.backward(y_pred, y_true)

        # Calculate KL divergence gradients
        dmu = mu
        dlogvar = -0.5 * (1 - np.exp(logvar))

        # Scale KL divergence gradients by kl_beta
        self.dmu = kl_beta * dmu
        self.dlogvar = kl_beta * dlogvar