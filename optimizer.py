from abc import ABC
from torch.nn import Parameter
from typing import Iterable, Optional

import torch

class Optimizer(ABC):

    def __init__(self, lr: float, params: Iterable[Parameter]) -> None:
        self.lr = lr
        self.params = list(params)
        self.state = {}
    
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
        self.momentum = None
        if momentum is not None:
            assert isinstance(momentum, float) and momentum > 0, ValueError(f"momentum must be positive not {momentum}")
            self.momentum = momentum
            for p in self.params:
                self.state[p] = {
                    "v": torch.zeros_like(p)
                }

    def step(self):
        if self.momentum is None:
            super().step()
            return
        
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                v: torch.Tensor = self.state[p]["v"]
                v.data = self.momentum * v.data - self.lr * p.grad
                p.data += v.data


class AdaGrad(Optimizer):
    
    def __init__(self, lr: float, params: Iterable[Parameter], eps: float=1e-9) -> None:
        super().__init__(lr, params)
        self.eps = eps
        for p in self.params:
            self.state[p] = {
                "grad_squared": torch.zeros_like(p)
            }
    
    def step(self):
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                grad_squared: torch.Tensor = self.state[p]["grad_squared"]
                grad_squared.add_(p.grad ** 2) # updates tensor inplace so that we don't have to update the dict again.
                p.data -= (self.lr / torch.sqrt(grad_squared + self.eps)) * p.grad


class AdaDelta(Optimizer):
    
    def __init__(self, lr: float, params: Iterable[Parameter], alpha: float=0.5, eps: float=1e-9) -> None:
        super().__init__(lr, params)
        self.eps = eps
        self.alpha = alpha
        for p in self.params:
            self.state[p] = {
                "grad_squared": torch.zeros_like(p),
                "param_diff_squared": torch.zeros_like(p)
            }
    
    def step(self):
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                grad_squared: torch.Tensor = self.state[p]["grad_squared"]
                grad_squared.mul_(self.alpha).add_((1-self.alpha) * p.grad ** 2)
                param_diff_squared: torch.Tensor = self.state[p]["param_diff_squared"]
                # step
                delta = torch.sqrt(param_diff_squared + self.eps) / torch.sqrt(grad_squared + self.eps) * p.grad
                p.data -= delta
                # update ema for delta
                param_diff_squared.mul_(self.alpha).add_((1 - self.alpha) * delta ** 2)


class Adam(Optimizer):

    def __init__(self, params: Iterable[Parameter], lr: float=1e-3, beta_1: float=0.9, beta_2: float=0.999, eps: float=1e-8) -> None:
        super().__init__(lr, params)
        self.beta_1 = beta_1
        self.beta_2 = beta_2    
        self.eps = eps
        self.t = 0

        for p in self.params:
            self.state[p] = {
                "m": torch.zeros_like(p),
                "v": torch.zeros_like(p)
            }
    
    def step(self) -> None:
        self.t += 1
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                m: torch.Tensor = self.state[p]['m']
                v: torch.Tensor = self.state[p]["v"]
                # adam update
                m.mul_(self.beta_1).add_((1 - self.beta_1) * p.grad)
                v.mul_(self.beta_2).add_((1 - self.beta_2) * p.grad ** 2)
                # remove bias
                m_hat = m / (1 - self.beta_1 ** self.t)
                v_hat = v / (1 - self.beta_2 ** self.t)

                p.data -= self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps))