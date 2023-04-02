r"""Miscellaneous tools and general purpose components"""

import torch

from torch import Tensor, Size
from typing import *


__piqa_debug__ = __debug__

def set_debug(mode: bool = False) -> bool:
    r"""Sets and returns whether debugging is enabled or not.
    If `__debug__` is :py:`False`, this function has not effect.

    Example:
        >>> set_debug(False)
        False
    """

    global __piqa_debug__

    __piqa_debug__ = __debug__ and mode

    return __piqa_debug__


def broadcastable(*shapes: Size) -> bool:
    r"""Returns whether `shapes` are broadcastable.

    Example:
        >>> x = torch.rand(3, 2, 1)
        >>> y = torch.rand(1, 2, 3)
        >>> z = torch.rand(2, 2, 2)
        >>> broadcastable(x.shape, y.shape)
        True
        >>> broadcastable(y.shape, z.shape)
        False
    """

    try:
        torch.broadcast_shapes(*shapes)
    except RuntimeError as e:
        return False
    else:
        return True


def assert_type(
    *tensors: Tensor,
    device: torch.device = None,
    dim_range: Tuple[int, int] = None,
    n_channels: int = None,
    value_range: Tuple[float, float] = None,
) -> None:
    r"""Asserts that the types, devices, shapes and values of `tensors` are
    valid with respect to some requirements.

    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> assert_type(x, y, dim_range=(4, 4), n_channels=3)
    """

    if torch.jit.is_tracing() or not __piqa_debug__:
        return

    if device is None:
        device = tensors[0].device

    shapes = [tuple(t.shape) for t in tensors]

    assert broadcastable(*shapes), f"Expected all tensors to have broadcastable shapes, but got {shapes}."

    for t in tensors:
        assert t.device == device, f"Expected all tensors to be on the same device, but got {str(t.device)} and {str(device)}."

        if dim_range is None:
            pass
        elif dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f"Expected number of dimensions to be {dim_range[0]}, but got {t.dim()}."
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim(), f"Expected number of dimensions to be greater or equal to {dim_range[0]}, but got {t.dim()}."
            assert t.dim() <= dim_range[1], f"Expected number of dimensions to be lower or equal to {dim_range[1]}, but got {t.dim()}."
        else:
            assert dim_range[0] <= t.dim(), f"Expected number of dimensions to be greater or equal to {dim_range[0]}, but got {t.dim()}."

        if n_channels is not None:
            assert t.shape[1] == n_channels, f"Expected number of channels to be {n_channels}, but got {t.shape[1]}."

        if value_range is not None:
            assert value_range[0] <= t.min(), f"Expected all values to be greater or equal to {value_range[0]}, but got {t.min().item()}."
            assert t.max() <= value_range[1], f"Expected all values to be lower or equal to {value_range[1]}, but got {t.max().item()}."
