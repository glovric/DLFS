import numpy as np
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

    def _update_sgd_parameters(self, params: np.ndarray, gradient: np.ndarray, momentum: np.ndarray = None) -> tuple:
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
        if momentum is not None:
            # Calculate new parameters using momentum
            new_momentum = self.momentum * momentum - self.current_learning_rate * gradient
            return params + new_momentum, new_momentum
        else:
            # Calculate new parameters using vanilla SGD
            update = -self.current_learning_rate * gradient
            return params + update, None
        
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

            # Get the gradient attribute and momentum attributes from the layer
            gradient = getattr(layer, "d" + param_name)
            momentum = getattr(layer, param_name + "_momentum", None)

            # Calculate new parameters and new momentum
            new_params, new_momentum = self._update_sgd_parameters(params=param_value, gradient=gradient, momentum=momentum)

            # Set new parameters attribute to layer
            setattr(layer, param_name, new_params)
            if new_momentum is not None:
                # Set new momentum attribute to layer
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
            # Create empty momentum array for every parameter 
            setattr(layer, f"{p}_momentum", np.zeros_like(params[p]))
    
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

class Optimizer_Adam(Optimizer):

    def __init__(self, learning_rate: float = 1e-3, decay: float = 0, epsilon: float = 1e-7, 
                 beta_1: float = 0.9, beta_2: float = 0.999, clip_grad: bool = False):
        """
        Adam optimizing algorithm. It uses first and second momentum estimates to update model parameters.
        First momentum estimate is referred to as momentum and second momentum is referred to as variance.
        Momentum can be interpreted as mean (central tendency) of the gradients and variance as how much gradients
        are dispersed.

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
            Exponential decay rate for variance.

        clip_grad : bool, default=False
            Flag indicating whether adaptive gradient clipping will be applied.

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
        self.clip_grad = clip_grad
        self.iterations = 0

    def _init_layer_parameters(self, layer: Layer) -> None:
        """
        Helper method for initializing Adam parameters of a layer (momentum and variance).

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
            # Create empty momentum and variance arrays for every parameter 
            setattr(layer, f"{p}_variance", np.zeros_like(params[p]))
            setattr(layer, f"{p}_momentum", np.zeros_like(params[p]))

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

            # Get gradient, variance and momentum attributes from the layer
            gradient = getattr(layer, "d" + param_name)
            variance = getattr(layer, param_name + "_variance")
            momentum = getattr(layer, param_name + "_momentum")

            # Calculate new parameters, new variance and  new momentum
            new_params, new_momentum, new_variance = self._update_adam_parameters(params=param_value, gradient=gradient, momentum=momentum, variance=variance)

            # Set new parameters, new variance and new momentum attributes to layer
            setattr(layer, param_name, new_params)
            setattr(layer, param_name + "_momentum", new_momentum)
            setattr(layer, param_name + "_variance", new_variance)
    
    def _update_adam_parameters(self, params: np.ndarray, gradient: np.ndarray, momentum: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper method for calculating new Adam parameters of a layer (momentum and variance).

        Parameters
        ----------
        params : np.ndarray
            Parameter to be updated.

        gradient : np.ndarray
            Parameter gradient used for update.

        momentum : np.ndarray
            Parameter gradient momentum used for update.

        variance : np.ndarray
            Parameter gradient variance used for update.

        Returns
        -------
        updated_param, new_momentum, new_variance : tuple
            Updated parameter, momentum and variance.
        """
        new_momentum = self.beta_1 * momentum + (1 - self.beta_1) * gradient
        momentum_corrected = new_momentum / (1 - self.beta_1 ** (self.iterations + 1))

        new_variance = self.beta_2 * variance + (1 - self.beta_2) * gradient**2
        variance_corrected = new_variance / (1 - self.beta_2 ** (self.iterations + 1))

        parameter_update = -self.current_learning_rate * (momentum_corrected / (np.sqrt(variance_corrected) + self.epsilon))
        return params + parameter_update, new_momentum, new_variance

    def _clip_gradients(self, layer: Layer, clip_factor: float = 0.2, eps: float = 1e-3) -> None:
        """
        Adaptive Gradient Clipping (AGC) of gradients based on parameter norms.

        Parameters
        ----------
        layer : Layer
            Layer whose gradients will be clipped.
        clip_factor : float
            Ratio of gradient norm to parameter norm to clip at.
        eps : float
            Small value to avoid division by zero.

        Returns
        -------
        None
        """
        params = layer.get_parameters()

        for param_name in params:
            param = getattr(layer, param_name)
            grad = getattr(layer, "d" + param_name, None)
            if grad is not None:
                param_norm = np.linalg.norm(param)
                grad_norm = np.linalg.norm(grad)

                max_grad_norm = clip_factor * max(param_norm, eps)

                if grad_norm > max_grad_norm:
                    scale = max_grad_norm / (grad_norm + eps)
                    clipped_grad = grad * scale
                    setattr(layer, "d" + param_name, clipped_grad)

    def pre_update_parameters(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.iterations * self.decay)

    def update_parameters(self, layer: Layer) -> None:
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

            if not hasattr(layer, param_name + "_variance"):
                self._init_layer_parameters(layer)

            if self.clip_grad:
                self._clip_gradients(layer)

            self._update_layer_parameters(layer)

        elif isinstance(layer, list):
            for l in layer:
                self.update_parameters(l)

        elif isinstance(layer, Module):
            for _, attr in vars(layer).items():
                self.update_parameters(attr)

    def post_update_parameters(self) -> None:
        """
        Method for updating current number of iterations.

        Returns
        -------
        None
        """
        self.iterations += 1

class Optimizer_AdamW(Optimizer_Adam):

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 0.01, lr_decay: float = 0, 
                 epsilon: float = 1e-7, beta_1: float = 0.9, beta_2: float = 0.999, amsgrad: bool = False, clip_grad: bool = False):
        """
        AdamW optimizing algorithm. 
        Compared to vanilla Adam, AdamW applies weight decay (effectively L2 regularization) during optimizer step.

        Parameters
        ----------
        learning_rate : float, default=0.001
            Step size used in gradient descent.

        weight_decay : float, default=0.01
            Factor used to regularize gradient.

        lr_decay : float, default=0
            Factor used to reduce learning rate over time.

        epsilon : float, default=1e-7
            Factor used to avoid divison by zero while updating parameters.

        beta_1 : float, default=0.9
            Exponential decay rate for momentum.
        
        beta_2 : float, default=0.999
            Exponential decay rate for variance.

        amsgrad : bool, default=False
            Flag indicating whether AMSGrad second moment calculation is applied.

        clip_grad : bool, default=False
            Flag indicating whether adaptive gradient clipping will be applied.

        Attributes
        ----------
        iterations : int, default=0
            Number of training iterations used to calculate new learning rate with decay.
        """
        super().__init__(learning_rate=learning_rate, decay=lr_decay, epsilon=epsilon, 
                         beta_1=beta_1, beta_2=beta_2, clip_grad=clip_grad)
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def _update_adamw_parameters(self, params: np.ndarray, gradient: np.ndarray, momentum: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper method for calculating new AdamW parameters of a layer (momentum and variance).

        Parameters
        ----------
        params : np.ndarray
            Parameter to be updated.

        gradient : np.ndarray
            Parameter gradient used for update.

        momentum : np.ndarray
            Parameter gradient momentum used for update.

        variance : np.ndarray
            Parameter gradient variance used for update.

        Returns
        -------
        updated_param, new_momentum, new_variance : tuple
            Updated parameter, momentum and variance.
        """
        new_momentum = self.beta_1 * momentum + (1 - self.beta_1) * gradient
        momentum_corrected = new_momentum / (1 - self.beta_1 ** (self.iterations + 1))

        new_variance = self.beta_2 * variance + (1 - self.beta_2) * gradient**2
        if self.amsgrad:
            new_variance = np.maximum(variance, new_variance)
        variance_corrected = new_variance / (1 - self.beta_2 ** (self.iterations + 1))

        parameter_update = -self.current_learning_rate * ((momentum_corrected / (np.sqrt(variance_corrected) + self.epsilon)) + self.weight_decay * gradient)
    
        return params + parameter_update, new_momentum, new_variance

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

            # Get gradient, variance and momentum attributes from the layer
            gradient = getattr(layer, "d" + param_name)
            variance = getattr(layer, param_name + "_variance")
            momentum = getattr(layer, param_name + "_momentum")

            # Calculate new parameters, new variance and  new momentum
            new_params, new_momentum, new_variance = self._update_adamw_parameters(params=param_value, gradient=gradient, momentum=momentum, variance=variance)

            # Set new parameters, new variance and new momentum attributes to layer
            setattr(layer, param_name, new_params)
            setattr(layer, param_name + "_momentum", new_momentum)
            setattr(layer, param_name + "_variance", new_variance)