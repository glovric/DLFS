import numpy as np

class Layer:
    """
    Layer abstract base class.
    """
    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass

    def get_parameters(self) -> dict:
        pass

    def _filter_parameters(self, param_names: list[str]) -> dict:
        parameters = {}
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray):
                if attr_name in param_names:
                    parameters[attr_name] = attr_value
        return parameters

class Activation:
    """
    Activation function abstract base class.
    """
    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass

class Loss:
    """
    Loss function abstract base class.
    """
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        pass

class Optimizer:
    """
    Optimizer abstract base class.
    """
    def pre_update_parameters(self) -> None:
        pass

    def update_layer_parameters(self, layer: Layer) -> None:
        pass

    def post_update_parameters(self) -> None:
        pass

class Module:
    """
    Module abstract base class.
    """

    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass

class Model:
    """
    Model abstract base class.
    """

    def forward(self, inputs: np.ndarray) -> None:
        pass

    def backward(self, delta: np.ndarray) -> None:
        pass