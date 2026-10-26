import numpy as np
from .tensor.base import Tensor, Parameter

class Module:
    """
    Module abstract base class.
    """
    def forward(self, inputs: Tensor) -> None:
        pass

    def parameters(self) -> list[Parameter]:
        params = []

        for value in vars(self).values():
            if isinstance(value, Parameter):
                params.append(value)

            elif isinstance(value, Module):
                params.extend(value.parameters())

            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())

        return params

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

    def update_parameters(self) -> None:
        pass

    def post_update_parameters(self) -> None:
        pass