import numpy as np
from .base import Module, Activation, Loss, Optimizer
from .modules import SequentialWrapper
from .helpers import get_random_batch
from .tensor import Tensor


"""import gc
import os
import psutil

process = psutil.Process(os.getpid())

def memory_report(epoch):
    gc.collect()

    rss = process.memory_info().rss / 1024**3

    print(
        f"Epoch {epoch}: "
        f"RSS={rss:.2f} GiB, "
        f"GC objects={len(gc.get_objects()):,}"
    )"""



class SequentialModel(Module):

    def __init__(self, layers: list[Module | Activation] = None, loss_function: Loss = None, optimizer: Optimizer = None) -> None:
        """
        Neural network model.

        Parameters
        ----------
        layers : list[Layer | Module | Activation], default=None
            List of layers and activation functions.

        loss_function : Loss, default=None
            Loss function.
        
        optimizer : Optimizer, default=None
            Optimizer algorithm.
        """
        self.wrapper = SequentialWrapper(layers)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.optimizer.set_parameters(self.parameters())

    def forward(self, X: Tensor, training: bool = False) -> None:
        """
        Forward pass.

        Parameters
        ----------
        X : Tensor
            Input values.

        Returns
        -------
        None
        """

        # Pass data through SequentialWrapper
        self.wrapper.forward(X, training)

        # Output of the model is the output of SequentialWrapper
        self.output = self.wrapper.output

    def print_mb(self, objekat: Module):

        for value in vars(objekat).values():

            if isinstance(value, Optimizer):
                if value.state:
                    for key, sub_dict in value.state.items():
                        for k, v in sub_dict.items():
                            print(f"Optimizer {k} state: {v.nbytes / 1e6:.2f} MB")

            if isinstance(value, Tensor):
                print(f"Data: {value.data.nbytes / 1e6:.2f} MB")
                if value.grad is not None:
                    print(f"Grad: {value.grad.nbytes / 1e6:.2f} MB")

            elif isinstance(value, Module):
                self.print_mb(value)

            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Tensor):
                        print(f"Data: {value.data.nbytes / 1e6:.2f} MB")
                        if value.grad is not None:
                            print(f"Grad: {value.grad.nbytes / 1e6:.2f} MB")
                    elif isinstance(item, Module):
                        self.print_mb(item)


    def train(self, X: Tensor, y: Tensor, epochs: int = 1000, batch_size: int = None, print_every: int = None) -> None:
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        y : np.ndarray
            Output values.

        epochs : int, default=1000
            Number of training epochs.

        batch_size : int, default=None
            Number of samples in a data subset. Batch size of None results in using all samples.

        print_every : int, default=None
            If given an integer, prints loss value.

        Returns
        -------
        None
        """

        for i in range(epochs + 1):

            #self.print_mb(self)

            self.optimizer.zero_grad()

            if batch_size is not None:
                batch_X, batch_y = get_random_batch(X, y, batch_size=batch_size)
            else:
                # If batch size is not specified take the whole dataset
                batch_X, batch_y = X, y

            # Forward pass
            self.forward(batch_X, training=True)

            loss = self.loss_function.calculate(self.output, batch_y)

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.pre_update_parameters()
            self.optimizer.update_parameters()
            self.optimizer.post_update_parameters()

            if print_every is not None:
                if not i % print_every:
                    print(f'===== EPOCH : {i} ===== LOSS : {loss.data} =====')

            #self.print_mb(self)
            log_all_arrays(self)
            log_process_memory(i)
            #memory_report(i)

    def predict(self, X: Tensor) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Input values.

        Returns
        -------
        prediction : np.ndarray
        """
        self.forward(X, training=False)
        return self.output.data

import numpy as np
import inspect

def log_all_arrays(self=None):
    print("\n========== NUMPY MEMORY REPORT ==========")

    seen_arrays = set()
    seen_objects = set()
    total = 0

    def scan(obj, path, depth=0, max_depth=5):
        nonlocal total

        if id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))

        if isinstance(obj, np.ndarray):
            if id(obj) not in seen_arrays:
                seen_arrays.add(id(obj))
                total += obj.nbytes
                print(
                    f"{obj.nbytes / 1024**2:9.2f} MiB | "
                    f"{str(obj.shape):20s} | "
                    f"{str(obj.dtype):8s} | {path}"
                )
            return

        if depth >= max_depth:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                scan(value, f"{path}[{key!r}]", depth + 1, max_depth)

        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                scan(value, f"{path}[{i}]", depth + 1, max_depth)

        elif hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                scan(value, f"{path}.{name}", depth + 1, max_depth)

    # Scan every active stack frame.
    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            for name, obj in frame.f_locals.items():
                scan(obj, f"{frame.f_code.co_name}.{name}")
            frame = frame.f_back
    finally:
        del frame

    # Also scan the Module explicitly.
    if self is not None:
        scan(self, "self")

    print("-----------------------------------------")
    print(f"Total unique array storage: {total / 1024**2:.2f} MiB")
    print("=========================================\n")

"""import os
import psutil

process = psutil.Process(os.getpid())
def log_process_memory(epoch):
    rss_gib = process.memory_info().rss / (1024 ** 3)
    print(f"Epoch {epoch}: process RSS = {rss_gib:.2f} GiB")"""

import os
import gc
import psutil
import tracemalloc

process = psutil.Process(os.getpid())

tracemalloc.start()

def log_process_memory(epoch):
    gc.collect()

    rss = process.memory_info().rss / 1024**3
    current, peak = tracemalloc.get_traced_memory()

    print(
        f"Epoch {epoch}: "
        f"RSS={rss:.2f} GiB | "
        f"tracemalloc current={current / 1024**2:.1f} MiB | "
        f"peak={peak / 1024**2:.1f} MiB | "
        f"GC objects={len(gc.get_objects()):,}"
    )