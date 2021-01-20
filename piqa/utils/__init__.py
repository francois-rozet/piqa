r"""Miscellaneous tools and general purpose components
"""

import os
import torch

from typing import List, Tuple


if os.getenv('PIQA_JIT') == '0':
    _jit = lambda f: f
else:
    _jit = torch.jit.script


def _debug(mode: bool = __debug__) -> bool:
    r"""Returns whether debugging is enabled or not.
    """

    return mode


def _assert_type(
    tensors: List[torch.Tensor],
    device: torch.device,
    dim_range: Tuple[int, int] = (0, -1),
    n_channels: int = 0,
    value_range: Tuple[float, float] = (0., -1.),
) -> None:
    r"""Asserts that the types, devices, shapes and values of `tensors` are
    valid with respect to some requirements.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> _assert_type([x, y], device=x.device, dim_range=(4, 4), n_channels=3)
    """

    if not _debug():
        return

    ref = tensors[0]

    for t in tensors:
        assert t.device == device, (
            f'Expected tensors to be on {device}, got {t.device}'
        )

        assert t.shape == ref.shape, (
            'Expected tensors to be of the same shape, got'
            f' {ref.shape} and {t.shape}'
        )

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], (
                'Expected number of dimensions to be'
                f' {dim_range[0]}, got {t.dim()}'
            )
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], (
                'Expected number of dimensions to be between'
                f' {dim_range[0]} and {dim_range[1]}, got {t.dim()}'
            )
        elif dim_range[0] > 0:
            assert dim_range[0] <= t.dim(), (
                'Expected number of dimensions to be greater or equal to'
                f' {dim_range[0]}, got {t.dim()}'
            )

        if n_channels > 0:
            assert t.size(1) == n_channels, (
                'Expected number of channels to be'
                f' {n_channels}, got {t.size(1)}'
            )

        if value_range[0] < value_range[1]:
            assert value_range[0] <= t.min(), (
                'Expected values to be greater or equal to'
                f' {value_range[0]}, got {t.min()}'
            )

            assert t.max() <= value_range[1], (
                'Expected values to be lower or equal to'
                f' {value_range[1]}, got {t.max()}'
            )


def _reduce(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    r"""Returns a reducing module.

    Args:
        reduction: Specifies the reduction type:
            `'none'` | `'mean'` | `'sum'`.

    Example:
        >>> x = torch.arange(5)
        >>> _reduce(x, reduction='sum')
        tensor(10)
    """

    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()

    return x
