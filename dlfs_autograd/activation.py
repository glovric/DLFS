import numpy as np
from .tensor.base import Tensor
from .tensor.math import sigmoid, relu, softmax
from .base import Activation

class ReLU(Activation):

    def forward(self, inputs, training=False):
        self.output = relu(inputs)

class Sigmoid(Activation):

    def forward(self, inputs, training=False):
        self.output = sigmoid(inputs)

class Softmax(Activation):

    def forward(self, inputs: Tensor, training=False) -> None:
        self.output = softmax(inputs)