import numpy as np
from .base import Optimizer
from .tensor import Parameter

class Optimizer_SGD(Optimizer):

    def __init__(self, parameters: list[Parameter] = [], learning_rate: float = 1e-3, momentum: float = 0., decay: float = 0.) -> None:
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
        self.model_params = parameters
        self.state = self._init_state(parameters)

    def _update_sgd_parameters(self, params: Parameter, momentum: np.ndarray = None) -> tuple[np.ndarray, np.ndarray | None]:
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
            new_momentum = self.momentum * momentum - self.current_learning_rate * gradient
            return param_data + new_momentum, new_momentum
        else:
            # Calculate new parameters using vanilla SGD
            update = -self.current_learning_rate * gradient
            return param_data + update, None
        
    def _update_layer_parameters(self, param: Parameter) -> None:
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
        momentum = self.state.get(id(param), {}).get("momentum", None)

        # Calculate new parameters and new momentum
        new_params, new_momentum = self._update_sgd_parameters(params=param, momentum=momentum)

        param.data = new_params

        if new_momentum is not None:
            self.state[id(param)]["momentum"] = new_momentum
        
    def _init_state(self, parameters: list[Parameter]) -> dict[int, dict[str, np.ndarray]]:
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
        state = {}
        if self.momentum:
            for p in parameters:
                state[id(p)] = { "momentum": np.zeros_like(p.data) }
        return state

    def set_parameters(self, parameters: list[Parameter]):
        self.model_params = parameters
        self.state = self._init_state(parameters)
    
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

    def update_parameters(self) -> None:
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
        for param in self.model_params:
            self._update_layer_parameters(param)

    def post_update_parameters(self) -> None:
        """
        Method for updating number of iterations.

        Returns
        -------
        None
        """
        self.iterations += 1

    def zero_grad(self):
        for param in self.model_params:
            if param.use_grad:
                param.zero_grad()

class Optimizer_Adam(Optimizer):

    def __init__(self, parameters: list[Parameter] = [], learning_rate: float = 1e-3, decay: float = 0, epsilon: float = 1e-7, 
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
        self.model_params = parameters
        self.state = self._init_state(parameters)

    def _update_layer_parameters(self, param: Parameter) -> None:
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
        # Get gradient, variance and momentum attributes from the layer
        variance = self.state.get(id(param), {}).get("variance", None)
        momentum = self.state.get(id(param), {}).get("momentum", None)

        # Calculate new parameters, new variance and  new momentum
        new_params, new_momentum, new_variance = self._update_adam_parameters(params=param.data, gradient=param.grad, momentum=momentum, variance=variance)

        param.data = new_params

        self.state[id(param)]["momentum"] = new_momentum
        self.state[id(param)]["variance"] = new_variance
    
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

    def pre_update_parameters(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.iterations * self.decay)

    def update_parameters(self) -> None:
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
        for param in self.model_params:
            self._update_layer_parameters(param)

    def post_update_parameters(self) -> None:
        """
        Method for updating current number of iterations.

        Returns
        -------
        None
        """
        self.iterations += 1

    def set_parameters(self, parameters: list[Parameter]):
        self.model_params = parameters
        self.state = self._init_state(parameters)

    def zero_grad(self):
        for param in self.model_params:
            if param.use_grad:
                param.zero_grad()

    def _init_state(self, parameters: list[Parameter]) -> dict[int, dict[str, np.ndarray]]:
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
        state = {}
        for p in parameters:
            state[id(p)] = { 
                "momentum": np.zeros_like(p.data), 
                "variance": np.zeros_like(p.data) 
            }
        return state