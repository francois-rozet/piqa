r"""Miscellaneous tools such as modules, functionals and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple


def build_reduce(
    reduction: str = 'mean',
    dim: Tuple[int, ...] = (),
    keepdim: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Returns a reduce function.

    Args:
        reduction: A reduction type (`'mean'`, `'sum'` or `'none'`).
        dim: The dimension(s) along which to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.
    """

    if reduction == 'mean':
        return lambda x: x.mean(dim=dim, keepdim=keepdim)
    elif reduction == 'sum':
        return lambda x: x.sum(dim=dim, keepdim=keepdim)

    return lambda x: x


def gaussian_kernel(
    kernel_size: int,
    sigma: float = 1.,
    n: int = 2,
) -> torch.Tensor:
    r"""Returns the `n`-dimensional Gaussian kernel of size `kernel_size`.

    The distribution is centered around the kernel's center
    and the standard deviation is `sigma`.

    Args:
        kernel_size: The size of the kernel.
        sigma: The standard deviation of the distribution.
        n: The number of dimensions of the kernel.

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


def gradient2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Returns the 2D gradient of `x` with respect to `kernel`.

    Args:
        x: An input tensor, (N, 1, H, W).
        kernel: A 2D derivative kernel, (2, K, K).
    """

    return F.conv2d(x, kernel, padding=kernel.size(-1) // 2)


def prewitt_kernel() -> torch.Tensor:
    r"""Returns the (horizontal) 3x3 Prewitt kernel.

    Wikipedia:
        https://en.wikipedia.org/wiki/Prewitt_operator
    """

    return torch.Tensor([
        [1., 0., -1.],
        [1., 0., -1.],
        [1., 0., -1.],
    ]) / 3


def sobel_kernel() -> torch.Tensor:
    r"""Returns the (horizontal) 3x3 Sobel kernel.

    Wikipedia:
        https://en.wikipedia.org/wiki/Sobel_operator
    """

    return torch.Tensor([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.],
    ]) / 4


def scharr_kernel() -> torch.Tensor:
    r"""Returns the (horizontal) 3x3 Scharr kernel.

    Wikipedia:
        https://en.wikipedia.org/wiki/Scharr_operator
    """

    return torch.Tensor([
        [3., 0., -3.],
        [10., 0., -10.],
        [3., 0., -3.],
    ]) / 16


def tensor_norm(
    x: torch.Tensor,
    dim: Tuple[int, ...] = (),
    keepdim: bool = False,
    norm: str = 'L2',
) -> torch.Tensor:
    r"""Returns the norm of `x`.

    Args:
        x: An input tensor.
        dim: The dimension(s) along which to calculate the norm.
        keepdim: Whether the output tensor has `dim` retained or not.
        norm: A norm function name (`'L1'`, `'L2'` or `'L2_squared'`).

    Wikipedia:
        https://en.wikipedia.org/wiki/Norm_(mathematics)
    """

    if norm in ['L2', 'L2_squared']:
        x = x ** 2
    else:  # norm == 'L1'
        x = x.abs()

    x = x.sum(dim=dim, keepdim=keepdim)

    if norm == 'L2':
        x = x.sqrt()

    return x


def normalize_tensor(
    x: torch.Tensor,
    epsilon: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    r"""Returns `x` normalized.

    Args:
        x: An input tensor.
        epsilon: A numerical stability term.

        `**kwargs` are transmitted to `tensor_norm`.
    """

    norm = tensor_norm(x, **kwargs)

    return x / (norm + epsilon)


class Intermediary(nn.Module):
    r"""Module that catches and returns the outputs of indermediate
    target layers of a sequential module during its forward pass.

    Args:
        layers: A sequential module.
        targets: A list of target layer indexes.
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
