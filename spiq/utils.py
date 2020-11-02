r"""Miscellaneous tools such as modules, functionals and more.
"""

###########
# Imports #
###########

import torch
import torch.nn as nn

from typing import List, Tuple


#############
# Functions #
#############

def gaussian_kernel(kernel_size: int, sigma: float = 1., n: int = 2) -> torch.Tensor:
    r"""Returns the `n`-dimensional Gaussian kernel of size `kernel_size`.

    The distribution is centered around the kernel's center and the standard deviation is `sigma`.

    Args:
        kernel_size: size of the kernel
        sigma: standard deviation of the distribution
        n: number of dimensions of the kernel

    Wikipedia:
        https://en.wikipedia.org/wiki/Normal_distribution
    """

    distrib = torch.arange(kernel_size).float()
    distrib -= (kernel_size - 1) / 2
    distrib = distrib ** 2

    kernel = distrib.clone()

    for i in range(1, n):
        distrib = distrib.unsqueeze(0)
        kernel = kernel.unsqueeze(i)
        kernel = kernel + distrib

    kernel = torch.exp(-kernel / (2 * sigma ** 2))
    kernel /= kernel.sum()

    return kernel


def tensor_norm(x: torch.Tensor, dim: Tuple[int, ...] = (), keepdim: bool = False, norm: str = 'L2') -> torch.Tensor:
    r"""Returns the norm of `x`.

    Args:
        x: input tensor
        dim: dimension(s) along which to calculate the norm
        keepdim: whether the output tensor has `dim` retained or not
        norm: norm function name (`'L1'`, `'L2'` or `'L2_squared'`)

    Wikipedia:
        https://en.wikipedia.org/wiki/Norm_(mathematics)
    """

    if norm in ['L2', 'L2_squared']:
        x = x ** 2
    else: # norm == 'L1'
        x = x.abs()

    x = x.sum(dim=dim, keepdim=keepdim)

    if norm == 'L2':
        x = x.sqrt()

    return x


def normalize_tensor(x: torch.Tensor, dim: Tuple[int, ...] = (), norm: str = 'L2', epsilon: float = 1e-8) -> torch.Tensor:
    r"""Returns `x` normalized.

    Args:
        x: input tensor
        dim: dimension(s) along which to normalize
        norm: norm function name (`'L1'`, `'L2'` or `'L2_squared'`)
        epsilon: numerical stability term
    """

    norm = tensor_norm(x, dim=dim, keepdim=True, norm=norm)

    return x / (norm + epsilon)


###########
# Classes #
###########

class Intermediary(nn.Module):
    r"""Module that catches and returns the outputs of indermediate target layers of a sequential module during its forward pass.

    Args:
        layers: sequential module
        targets: target layer indexes
    """

    def __init__(self, layers: nn.Sequential, targets: List[int]):
        super().__init__()

        self.layers = layers
        self.targets = set(targets)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        output = []

        for i, layer in enumerate(self.layers):
            input = layer(input)

            if i in self.targets:
                output.append(input.clone())

            if len(output) == len(self.targets):
                break

        return output
