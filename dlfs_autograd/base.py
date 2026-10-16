import numpy as np
from .tensor import Tensor

class Layer:
    """
    Layer abstract base class.
    """
    def forward(self, inputs: Tensor) -> None:
        pass

    def get_parameters(self) -> dict[str, Tensor]:
        pass

    def _filter_parameters(self, param_names: list[str]) -> dict[str, Tensor]:
        """
        Helper method for filtering layer attributes. Method selects only the layer attributes which are of `np.ndarray` type
        and are contained within `param_names` list.

        Layer instances can contain multiple `Tensor` attributes such as output, dinputs, cell_states etc. and we wouldn't like
        for wrong `np.ndarray` objects to be recognized as trainable layer parameters.

        Parameters
        ----------
        param_names : list[str]
            List of Layer trainable parameter names.

        Returns
        -------
        param_dict : dict[str, np.ndarray]
            Dictionary mapping each parameter name with its respective `np.ndarray` object.
        """
        param_dict = {}
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, Tensor) and attr_name in param_names:
                    param_dict[attr_name] = attr_value
        return param_dict

class Module:
    """
    Module abstract base class.
    """
    def forward(self, inputs: Tensor) -> None:
        pass

class Activation:
    """
    Activation function abstract base class.
    """
    def forward(self, inputs: Tensor) -> None:
        pass

class Loss:
    """
    Loss function abstract base class.
    """
    def calculate(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass
    

class Optimizer:
    """
    Optimizer abstract base class.
    """
    def pre_update_parameters(self) -> None:
        pass

    def update_parameters(self, module: Layer | Module) -> None:
        pass

    def post_update_parameters(self) -> None:
        pass