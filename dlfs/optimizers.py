import numpy as np
from .base import Optimizer, Layer

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

    def _update_sgd_parameters(self, params: np.ndarray, gradients: np.ndarray, momentums: np.ndarray = None) -> tuple:
        if momentums is not None:
            new_momentums = self.momentum * momentums - self.current_learning_rate * gradients
            return params + new_momentums, new_momentums
        else:
            update = -self.current_learning_rate * gradients
            return params + update, None
        
    def _update_parameters(self, layer: Layer) -> None:
        params = layer.get_parameters()

        for param_name, param_value in params.items():
            gradient = getattr(layer, "d" + param_name)
            momentums = getattr(layer, param_name + "_momentums", None)
            new_params, new_momentums = self._update_sgd_parameters(params=param_value, gradients=gradient, momentums=momentums)
            setattr(layer, param_name, new_params)
            if new_momentums is not None:
                setattr(layer, param_name + "_momentums", new_momentums)
        
    def _init_parameters(self, layer: Layer) -> None:
        """
        Helper method for initializing momentums of a layer.

        Parameters
        ----------
        params : tuple
            Tuple of layer parameters.

        Returns
        -------
        adam_params : list
            List of zero-valued momentums or cache arrays.
        """
        params = layer.get_parameters()
        for p in params:
            setattr(layer, f"{p}_momentums", np.zeros_like(params[p]))
    
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

    def update_layer_parameters(self, layer: Layer) -> None:
        """
        Method for updating layer parameters.

        Parameters
        ----------
        layer : Layer
            Layer that is being updated.

        Returns
        -------
        None
        """

        if isinstance(layer, Layer):
            params = layer.get_parameters()
            if params is None:
                return
            param_name = list(params.keys())[0] 

            if not hasattr(layer, param_name + "_momentums") and self.momentum:
                self._init_parameters(layer)

            self._update_parameters(layer)

        elif isinstance(layer, list):
            for l in layer:
                self.update_layer_parameters(l)

        elif hasattr(layer, "__dict__"):
            for attr_name, attr in vars(layer).items():
                self.update_layer_parameters(attr)

    def post_update_parameters(self) -> None:
        self.iterations += 1

class Optimizer_Adam(Optimizer):

    def __init__(self, learning_rate=1e-3, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999) -> None:
        """
        Adam optimizing algorithm.

        Parameters
        ----------
        learning_rate : float, default=0.001
            Step size used in gradient descent.

        decay : float, default=0
            Factor used to reduce learning rate over time.

        epsilon : float, default=1e-7
            Factor used to avoid divison by zero while updating parameters.

        beta_1 : float, default=0.9
            Exponential decay rate for momentum.
        
        beta_2 : float, default=0.999
            Exponential decay rate for cache.

        Attributes
        ----------
        iterations : int, default=0
            Number of training iterations used to calculate new learning rate with decay.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0 

    def _init_parameters(self, layer: Layer) -> None:
        """
        Helper method for initializing Adam parameters of a layer (momentums or cache).

        Parameters
        ----------
        params : tuple
            Tuple of layer parameters.

        Returns
        -------
        adam_params : list
            List of zero-valued momentums or cache arrays.
        """
        params = layer.get_parameters()
        for p in params:
            setattr(layer, f"{p}_cache", np.zeros_like(params[p]))
            setattr(layer, f"{p}_momentums", np.zeros_like(params[p]))

    def _update_parameters(self, layer: Layer) -> None:
        params = layer.get_parameters()

        for param_name, param_value in params.items():
            gradient = getattr(layer, "d" + param_name)
            cache = getattr(layer, param_name + "_cache")
            momentums = getattr(layer, param_name + "_momentums")
            new_params, new_momentums, new_cache = self._update_adam_parameters(params=param_value, gradients=gradient, momentums=momentums, cache=cache)
            setattr(layer, param_name, new_params)
            setattr(layer, param_name + "_momentums", new_momentums)
            setattr(layer, param_name + "_cache", new_cache)
    
    def _update_adam_parameters(self, params: np.ndarray, gradients: np.ndarray, momentums: np.ndarray, cache: np.ndarray) -> tuple:
        """
        Helper method for updating Adam parameters of a layer (momentums and caches).

        Parameters
        ----------
        params : np.ndarray
            Layer parameter to be updated.

        gradient : np.ndarray
            Layer gradient used for update.

        momentums : np.ndarray
            Layer momentums used for update.

        cache : np.ndarray
            Layer cache used for update.

        Returns
        -------
        updated_param, new_momentums, new_cache : tuple
            Updated parameter, momentums and cache.
        """
        new_momentums = self.beta_1 * momentums + (1 - self.beta_1) * gradients
        momentums_corrected = new_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        new_cache = self.beta_2 * cache + (1 - self.beta_2) * gradients**2
        cache_corrected = new_cache / (1 - self.beta_2 ** (self.iterations + 1))

        parameter_update = -self.current_learning_rate * momentums_corrected / (np.sqrt(cache_corrected) + self.epsilon)
        return params + parameter_update, new_momentums, new_cache

    def pre_update_parameters(self) -> None:
        """
        Method for updating learning rate based on current number of iterations and decay.

        Returns
        -------
        None
        """
        if self.decay:
            # Inverse decay method
            self.current_learning_rate = self.learning_rate  / (1 + self.iterations * self.decay)

    def update_layer_parameters(self, layer: Layer) -> None:
        """
        Method for updating layer parameters.

        Parameters
        ----------
        layer : Layer
            Layer to update.

        Returns
        -------
        None
        """

        if isinstance(layer, Layer):
            params = layer.get_parameters()
            if params is None:
                return
            param_name = list(params.keys())[0] 

            if not hasattr(layer, param_name + "_cache"):
                self._init_parameters(layer)

            self._update_parameters(layer)

        elif isinstance(layer, list):
            for l in layer:
                self.update_layer_parameters(l)

        elif hasattr(layer, "__dict__"):
            for attr_name, attr in vars(layer).items():
                self.update_layer_parameters(attr)

    def post_update_parameters(self) -> None:
        """
        Method for updating current number of iterations.

        Returns
        -------
        None
        """
        self.iterations += 1