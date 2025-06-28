from abc import ABC, abstractmethod
from torch.nn import Parameter
from typing import Iterable, Optional

import torch

class Optimizer(ABC):

    def __init__(self, lr: float, params: Iterable[Parameter]) -> None:
        self.lr = lr
        self.params = list(params)
    
    def zero_grad(self) -> None:
        """
        Zeros out gradients
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

class SGD(Optimizer):

    def __init__(self, lr: float, params: Iterable[Parameter], momentum: Optional[float]=None) -> None:
        super().__init__(lr, params)
        self.momentum = momentum
        if momentum is not None:
            assert momentum > 0, ValueError(f"momentum must be positive not {momentum}")
            self.v = [torch.ones_like(p) for p in params]
        else:
            self.v = None

    def step(self):
        if self.v is None:
            super().step()
            return
        
        for (p, v) in zip(self.params, self.v):
            if p.requires_grad and p.grad is not None:
                v.data = self.momentum * v.data - self.lr * p.grad
                p.data += v.data