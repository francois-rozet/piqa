r"""Differentiable and JITable complex tensor API"""

import torch

from torch import Tensor


def complx(real: Tensor, imag: Tensor) -> Tensor:
    r"""Returns a complex tensor with its real part equal to :math:`\Re` and
    its imaginary part equal to :math:`\Im`.

    .. math:: c = \Re + i \Im

    Args:
        real: A tensor :math:`\Re`, :math:`(*,)`.
        imag: A tensor :math:`\Im`, :math:`(*,)`.

    Returns:
        The complex tensor, :math:`(*, 2)`.

    Example:
        >>> x = torch.tensor([2., 0.7071])
        >>> y = torch.tensor([0., 0.7071])
        >>> complx(x, y)
        tensor([[2.0000, 0.0000],
                [0.7071, 0.7071]])
    """

    return torch.stack([real, imag], dim=-1)


def real(x: Tensor) -> Tensor:
    r"""Returns the real part of :math:`x`.

    Args:
        x: A complex tensor, :math:`(*, 2)`.

    Returns:
        The real tensor, :math:`(*,)`.

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> real(x)
        tensor([2.0000, 0.7071])
    """

    return x[..., 0]


def imag(x: Tensor) -> Tensor:
    r"""Returns the imaginary part of :math:`x`.

    Args:
        x: A complex tensor, :math:`(*, 2)`.

    Returns:
        The imaginary tensor, :math:`(*,)`.

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> imag(x)
        tensor([0.0000, 0.7071])
    """

    return x[..., 1]


def conj(x: Tensor) -> Tensor:
    r"""Returns the element-wise conjugate of :math:`x`.

    .. math:: \bar{x} = \Re(x) - i \Im(x)

    Args:
        x: A complex tensor, :math:`(*, 2)`.

    Returns:
        The conjugate tensor, :math:`(*, 2)`.

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> conj(x)
        tensor([[ 2.0000, -0.0000],
                [ 0.7071, -0.7071]])
    """

    return x * torch.tensor([1., -1.]).to(x)


def turn(x: Tensor) -> Tensor:
    r"""Returns the element-wise product of :math:`x` with :math:`i`.

    .. math:: i x = -\Im(x) + i \Re(x)

    Args:
        x: A complex tensor, :math:`(*, 2)`.

    Returns:
        The turned tensor, :math:`(*, 2)`.

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> turn(x)
        tensor([[-0.0000,  2.0000],
                [-0.7071,  0.7071]])
    """

    return complx(-imag(x), real(x))


def polar(r: Tensor, phi: Tensor) -> Tensor:
    r"""Returns a complex tensor with its modulus equal to :math:`r`
    and its phase equal to :math:`\phi`.

    .. math:: c = r \exp(i \phi)

    Args:
        r: A tensor :math:`r`, :math:`(*,)`.
        phi: A tensor :math:`\phi`, :math:`(*,)`.

    Returns:
        The complex tensor, :math:`(*, 2)`.

    Example:
        >>> x = torch.tensor([2., 1.])
        >>> y = torch.tensor([0., 0.7854])
        >>> polar(x, y)
        tensor([[2.0000, 0.0000],
                [0.7071, 0.7071]])
    """

    return complx(r * torch.cos(phi), r * torch.sin(phi))


def mod(x: Tensor, squared: bool = False) -> Tensor:
    r"""Returns the modulus (absolute value) of :math:`x`.

    .. math:: \left| x \right| = \sqrt{ \Re(x)^2 + \Im(x)^2 }

    Args:
        x: A complex tensor, :math:`(*, 2)`.
        squared: Whether the output is squared or not.

    Returns:
        The modulus tensor, :math:`(*,)`.

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> mod(x)
        tensor([2.0000, 1.0000])
    """

    x = x.square().sum(dim=-1)

    if not squared:
        x = torch.sqrt(x)

    return x


def angle(x: Tensor) -> Tensor:
    r"""Returns the angle (phase) of :math:`x`.

    .. math:: \phi(x) = \operatorname{atan2}(\Im(x), \Re(x))

    Args:
        x: A complex tensor, :math:`(*, 2)`.

    Returns:
        The angle tensor, :math:`(*,)`.

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> angle(x)
        tensor([0.0000, 0.7854])
    """

    return torch.atan2(imag(x), real(x))


def prod(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the element-wise product of :math:`x` and :math:`y`.

    .. math::
        x y = \Re(x) \Re(y) - \Im(x) \Im(y)
            + i \left( \Re(x) \Im(y) - \Im(x) \Re(y) \right)

    Args:
        x: A complex tensor, :math:`(*, 2)`.
        y: A complex tensor, :math:`(*, 2)`.

    Returns:
        The product tensor, :math:`(*, 2)`.

    Example:
        >>> x = torch.tensor([[2.,  0.], [0.7071,  0.7071]])
        >>> y = torch.tensor([[2., -0.], [0.7071, -0.7071]])
        >>> prod(x, y)
        tensor([[4.0000, 0.0000],
                [1.0000, 0.0000]])
    """

    x_r, x_i = real(x), imag(x)
    y_r, y_i = real(y), imag(y)

    return complx(x_r * y_r - x_i * y_i, x_i * y_r + x_r * y_i)


def dot(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the element-wise dot-product of :math:`x` and :math:`y`.

    .. math:: x \odot y = \Re(x) \Re(y) + \Im(x) \Im(y)

    Args:
        x: A complex tensor, :math:`(*, 2)`.
        y: A complex tensor, :math:`(*, 2)`.

    Returns:
        The dot-product tensor, :math:`(*,)`.

    Example:
        >>> x = torch.tensor([[2.,  0.], [0.7071,  0.7071]])
        >>> y = torch.tensor([[2., -0.], [0.7071, -0.7071]])
        >>> dot(x, y)
        tensor([4., 0.])
    """

    return (x * y).sum(dim=-1)


def pow(x: Tensor, exponent: float) -> Tensor:
    r"""Returns the power of :math:`x` with `exponent`.

    .. math:: x^p = \left| x \right|^p \exp(i \phi(x))^p

    Args:
        x: A complex tensor, :math:`(*, 2)`.
        exponent: The exponent :math:`p`.

    Returns:
        The power tensor, :math:`(*, 2)`.

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> pow(x, 2.)
        tensor([[ 4.0000e+00,  0.0000e+00],
                [-4.3711e-08,  9.9998e-01]])
    """

    r = mod(x, squared=True) ** (exponent / 2)
    phi = angle(x) * exponent

    return polar(r, phi)
