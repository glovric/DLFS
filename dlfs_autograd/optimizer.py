import numpy as np

from .tensor import Tensor
from .base import Optimizer, Layer, Module

class Optimizer_SGD(Optimizer):

    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0., decay: float = 0.) -> None:
        """
        Stochastic gradient descent optimizing algorithm with momentum and decay.

        Parameters
        ----------
        learning_rate : float, default=0.001
            Step size used in gradient descent.

        momentum : float, default=0.
            Factor used to scale past gradients.

        decay : float, default=0.
            Factor used to reduce learning rate over time.

        Attributes
        ----------
        iterations : int, default=0
            Number of training iterations used to calculate new learning rate with decay.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iterations = 0

    def _update_sgd_parameters(self, params: Tensor, momentum: Tensor = None) -> tuple[Tensor, Tensor | None]:
        """
        Helper method for calculating new parameters of a layer. Uses momentum if provided.

        Parameters
        ----------
        params : np.ndarray
            Layer parameter to be updated.

        gradient : np.ndarray
            Layer gradient used to update the parameter.

        momentum : np.ndarray, default=None
            Momentum used for better performance of the algorithm.

        Returns
        -------
        update_params, updated_momentum : tuple[np.ndarray, np.ndarray]
        """
        gradient = params.grad
        param_data = params.data

        if momentum is not None:
            # Calculate new parameters using momentum
            new_momentum = self.momentum * momentum.data - self.current_learning_rate * gradient
            return param_data + new_momentum, new_momentum
        else:
            # Calculate new parameters using vanilla SGD
            update = -self.current_learning_rate * gradient
            return param_data + update, None
        
    def _update_layer_parameters(self, layer: Layer) -> None:
        """
        Helper method for updating parameters of a layer.

        Parameters
        ----------
        layer: Layer
            Layer to be updated.

        Returns
        -------
        None
        """
        params = layer.get_parameters()

        # Loop through parameters of a layer
        for param_name, param_value in params.items():

            momentum = getattr(layer, param_name + "_momentum", None)

            # Calculate new parameters and new momentum
            new_params, new_momentum = self._update_sgd_parameters(params=param_value, momentum=momentum)

            # Set new parameters attribute to layer
            #setattr(layer, param_name, new_params)
            param_value.data = new_params
            if new_momentum is not None:
                # Set new momentum attribute to layer
                new_momentum = Tensor(new_momentum, use_grad=False)
                setattr(layer, param_name + "_momentum", new_momentum)
        
    def _init_layer_parameters(self, layer: Layer) -> None:
        """
        Helper method for initializing momentum of a layer.

        Parameters
        ----------
        layer : Layer
            Layer to be initialized.

        Returns
        -------
        None
        """
        params = layer.get_parameters()
        for p in params:
            momentum = Tensor(
                np.zeros_like(params[p].data),
                use_grad=False
            )
            # Create empty momentum array for every parameter 
            setattr(layer, f"{p}_momentum", momentum)
    
    def pre_update_parameters(self) -> None:
        """
        Method for updating current learning rate.

        Returns
        -------
        None
        """
        if self.decay:
            # Inverse decay method
            self.current_learning_rate = self.learning_rate  / (1 + self.iterations * self.decay)

    def update_parameters(self, layer: Layer) -> None:
        """
        Method for updating layer parameters recursively.

        Parameters
        ----------
        layer : Layer
            Layer that is being updated.

        Returns
        -------
        None
        """

        # Base case: if layer object is a Layer instance it can be updated
        if isinstance(layer, Layer):
            params = layer.get_parameters()
            if params is None:
                return
            param_name = list(params.keys())[0] 

            # Check if momentum are initialized and should they be initialized
            if not hasattr(layer, param_name + "_momentum") and self.momentum:
                self._init_layer_parameters(layer)

            # Update layer parameters
            self._update_layer_parameters(layer)

        # Case 2: if layer object is a list perform recursive update for every element of the list
        elif isinstance(layer, list):
            for l in layer:
                self.update_parameters(l)

        # Case 3: if layer object has its own class attributes perform recursive update for every class attribute
        elif isinstance(layer, Module):
            for _, attr in vars(layer).items():
                self.update_parameters(attr)

    def post_update_parameters(self) -> None:
        """
        Method for updating number of iterations.

        Returns
        -------
        None
        """
        self.iterations += 1

    def zero_grad(self, layer):


        if isinstance(layer, Layer):
            for _, attr in vars(layer).items():
                if isinstance(attr, Tensor) and attr.use_grad:
                    attr.zero_grad()

        elif isinstance(layer, list):
            for l in layer:
                self.zero_grad(l)

        elif isinstance(layer, Module):
            for _, attr in vars(layer).items():
                self.zero_grad(attr)