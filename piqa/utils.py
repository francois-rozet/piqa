r"""Miscellaneous tools such as modules, functionals and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Tuple, Union


def build_reduce(
    reduction: str = 'mean',
    dim: Union[int, Tuple[int, ...]] = (),
    keepdim: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Returns a reduce function.

    Args:
        reduction: Specifies the reduce function:
            `'none'` | `'mean'` | `'sum'`.
        dim: The dimension(s) along which to reduce.
        keepdim: Whether the output tensor has `dim` retained or not.

    Example:
        >>> red = build_reduce(reduction='sum')
        >>> type(red)
        <class 'function'>
        >>> red(torch.arange(5))
        tensor(10)
    """

    if reduction == 'mean':
        return lambda x: x.mean(dim=dim, keepdim=keepdim)
    elif reduction == 'sum':
        return lambda x: x.sum(dim=dim, keepdim=keepdim)

    return lambda x: x


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        coord: The unraveled coordinates, (*, N, D).

    Example:
        >>> unravel_index(torch.arange(9), (3, 3))
        tensor([[0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
                [2, 2]])
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


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

    Example:
        >>> gaussian_kernel(5, sigma=1.5, n=2)
        tensor([[0.0144, 0.0281, 0.0351, 0.0281, 0.0144],
                [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                [0.0351, 0.0683, 0.0853, 0.0683, 0.0351],
                [0.0281, 0.0547, 0.0683, 0.0547, 0.0281],
                [0.0144, 0.0281, 0.0351, 0.0281, 0.0144]])
    """

    shape = (kernel_size,) * n

    kernel = unravel_index(
        torch.arange(kernel_size ** n),
        shape,
    ).float()

    kernel -= (kernel_size - 1) / 2
    kernel = (kernel ** 2).sum(1) / (2. * sigma ** 2)
    kernel = torch.exp(-kernel)
    kernel /= kernel.sum()

    return kernel.reshape(shape)


def filter2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    padding: Union[int, Tuple[int, int]] = 0,
) -> torch.Tensor:
    r"""Returns the 2D (channel-wise) filter of `x` with respect to `kernel`.

    Args:
        x: An input tensor, (N, C, H, W).
        kernel: A 2D filter kernel, (C', 1, K, L).
        padding: The implicit paddings on both sides of the input.

    Example:
        >>> x = torch.arange(25).float().view(1, 1, 5, 5)
        >>> x
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
        >>> kernel = gaussian_kernel(3, sigma=1.5)
        >>> kernel
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> filter2d(x, kernel.view(1, 1, 3, 3))
        tensor([[[[ 6.0000,  7.0000,  8.0000],
                  [11.0000, 12.0000, 13.0000],
                  [16.0000, 17.0000, 18.0000]]]])
    """

    return F.conv2d(x, kernel, padding=padding, groups=x.size(1))


def prewitt_kernel() -> torch.Tensor:
    r"""Returns the (horizontal) 3x3 Prewitt kernel.

    Wikipedia:
        https://en.wikipedia.org/wiki/Prewitt_operator

    Example:
        >>> prewitt_kernel()
        tensor([[ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333],
                [ 0.3333,  0.0000, -0.3333]])
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

    Example:
        >>> sobel_kernel()
        tensor([[ 0.2500,  0.0000, -0.2500],
                [ 0.5000,  0.0000, -0.5000],
                [ 0.2500,  0.0000, -0.2500]])
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

    Example:
        >>> scharr_kernel()
        tensor([[ 0.1875,  0.0000, -0.1875],
                [ 0.6250,  0.0000, -0.6250],
                [ 0.1875,  0.0000, -0.1875]])
    """

    return torch.Tensor([
        [3., 0., -3.],
        [10., 0., -10.],
        [3., 0., -3.],
    ]) / 16


def tensor_norm(
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = (),
    keepdim: bool = False,
    norm: str = 'L2',
) -> torch.Tensor:
    r"""Returns the norm of `x`.

    Args:
        x: An input tensor.
        dim: The dimension(s) along which to calculate the norm.
        keepdim: Whether the output tensor has `dim` retained or not.
        norm: Specifies the norm funcion to apply:
            `'L1'` | `'L2'` | `'L2_squared'`.

    Wikipedia:
        https://en.wikipedia.org/wiki/Norm_(mathematics)

    Example:
        >>> x = torch.arange(9).float().view(3, 3)
        >>> x
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])
        >>> tensor_norm(x, dim=0)
        tensor([6.7082, 8.1240, 9.6437])
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
    dim: Tuple[int, ...] = (),
    norm: str = 'L2',
    epsilon: float = 1e-8,
) -> torch.Tensor:
    r"""Returns `x` normalized.

    Args:
        x: An input tensor.
        dim: The dimension(s) along which to normalize.
        norm: Specifies the norm funcion to use:
            `'L1'` | `'L2'` | `'L2_squared'`.
        epsilon: A numerical stability term.

    Example:
        >>> x = torch.arange(9).float().view(3, 3)
        >>> x
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]])
        >>> normalize_tensor(x, dim=0)
        tensor([[0.0000, 0.1231, 0.2074],
                [0.4472, 0.4924, 0.5185],
                [0.8944, 0.8616, 0.8296]])
    """

    norm = tensor_norm(x, dim=dim, keepdim=True, norm=norm)

    return x / (norm + epsilon)


class Intermediary(nn.Module):
    r"""Module that catches and returns the outputs of indermediate
    target layers of a sequential module during its forward pass.

    Args:
        layers: A sequential module.
        targets: A list of target layer indexes.
    """

    def __init__(self, layers: nn.Sequential, targets: List[int]):
        r""""""
        super().__init__()

        self.layers = layers
        self.targets = set(targets)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        r"""Defines the computation performed at every call.
        """

        output = []

        for i, layer in enumerate(self.layers):
            input = layer(input)

            if i in self.targets:
                output.append(input.clone())

            if len(output) == len(self.targets):
                break

        return output
