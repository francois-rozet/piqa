r"""Miscellaneous tools and general purpose components
"""

import os
import torch

from typing import List, Tuple


if os.getenv('PIQA_JIT') == '1':
    _jit = torch.jit.script
else:
    _jit = lambda f: f


__piqa_debug__ = __debug__

def set_debug(mode: bool = False) -> bool:
    r"""Sets and returns whether debugging is enabled or not.
    If `__debug__` is `False`, this function has not effect.

    Example:
        >>> set_debug(False)
        False
    """

    global __piqa_debug__

    __piqa_debug__ = __debug__ and mode

    return __piqa_debug__


def assert_type(
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
        >>> assert_type([x, y], device=x.device, dim_range=(4, 4), n_channels=3)
    """

    if not __piqa_debug__:
        return

    ref = tensors[0]

    for t in tensors:
        assert t.device == device, (
            f'Tensors expected to be on {device}, got {t.device}'
        )

        assert t.shape == ref.shape, (
            'Tensors expected to be of the same shape, got'
            f' {ref.shape} and {t.shape}'
        )

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], (
                'Number of dimensions expected to be'
                f' {dim_range[0]}, got {t.dim()}'
            )
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], (
                'Number of dimensions expected to be between'
                f' {dim_range[0]} and {dim_range[1]}, got {t.dim()}'
            )
        elif dim_range[0] > 0:
            assert dim_range[0] <= t.dim(), (
                'Number of dimensions expected to be greater or equal to'
                f' {dim_range[0]}, got {t.dim()}'
            )

        if n_channels > 0:
            assert t.size(1) == n_channels, (
                'Number of channels expected to be'
                f' {n_channels}, got {t.size(1)}'
            )

        if value_range[0] < value_range[1]:
            assert value_range[0] <= t.min(), (
                'Values expected to be greater or equal to'
                f' {value_range[0]}, got {t.min()}'
            )

            assert t.max() <= value_range[1], (
                'Values expected to be lower or equal to'
                f' {value_range[1]}, got {t.max()}'
            )


@_jit
def reduce_tensor(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    r"""Returns the reduction of \(x\).

    Args:
        x: A tensor, \((*,)\).
        reduction: Specifies the reduction type:
            `'none'` | `'mean'` | `'sum'`.

    Example:
        >>> x = torch.arange(5)
        >>> reduce_tensor(x, reduction='sum')
        tensor(10)
    """

    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()

    return x
