r"""Differentiable and JITable complex tensor API
"""

import torch


def complx(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    r"""Returns a complex tensor with its real part equal to \(\Re\) and
    its imaginary part equal to \(\Im\).

    $$ c = \Re + i \Im $$

    Args:
        real: A tensor \(\Re\), \((*,)\).
        imag: A tensor \(\Im\), \((*,)\).

    Returns:
        The complex tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([2., 0.7071])
        >>> y = torch.tensor([0., 0.7071])
        >>> complx(x, y)
        tensor([[2.0000, 0.0000],
                [0.7071, 0.7071]])
    """

    return torch.stack([real, imag], dim=-1)


def real(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the real part of \(x\).

    Args:
        x: A complex tensor, \((*, 2)\).

    Returns:
        The real tensor, \((*,)\).

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> real(x)
        tensor([2.0000, 0.7071])
    """

    return x[..., 0]


def imag(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the imaginary part of \(x\).

    Args:
        x: A complex tensor, \((*, 2)\).

    Returns:
        The imaginary tensor, \((*,)\).

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> imag(x)
        tensor([0.0000, 0.7071])
    """

    return x[..., 1]


def conj(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the element-wise conjugate of \(x\).

    $$ \bar{x} = \Re(x) - i \Im(x) $$

    Args:
        x: A complex tensor, \((*, 2)\).

    Returns:
        The conjugate tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> conj(x)
        tensor([[ 2.0000, -0.0000],
                [ 0.7071, -0.7071]])
    """

    return x * torch.tensor([1., -1.]).to(x)


def turn(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the element-wise product of \(x\) with \(i\).

    $$ i x = -\Im(x) + i \Re(x) $$

    Args:
        x: A complex tensor, \((*, 2)\).

    Returns:
        The turned tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> turn(x)
        tensor([[-0.0000,  2.0000],
                [-0.7071,  0.7071]])
    """

    return complx(-imag(x), real(x))


def polar(r: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    r"""Returns a complex tensor with its modulus equal to \(r\)
    and its phase equal to \(\phi\).

    $$ c = r \exp(i \phi) $$

    Args:
        r: A tensor \(r\), \((*,)\).
        phi: A tensor \(\phi\), \((*,)\).

    Returns:
        The complex tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([2., 1.])
        >>> y = torch.tensor([0., 0.7854])
        >>> polar(x, y)
        tensor([[2.0000, 0.0000],
                [0.7071, 0.7071]])
    """

    return complx(r * torch.cos(phi), r * torch.sin(phi))


def mod(x: torch.Tensor, squared: bool = False) -> torch.Tensor:
    r"""Returns the modulus (absolute value) of \(x\).

    $$ \left| x \right| = \sqrt{ \Re(x)^2 + \Im(x)^2 } $$

    Args:
        x: A complex tensor, \((*, 2)\).
        squared: Whether the output is squared or not.

    Returns:
        The modulus tensor, \((*,)\).

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> mod(x)
        tensor([2.0000, 1.0000])
    """

    x = x.square().sum(dim=-1)

    if not squared:
        x = torch.sqrt(x)

    return x


def angle(x: torch.Tensor) -> torch.Tensor:
    r"""Returns the angle (phase) of \(x\).

    $$ \phi(x) = \operatorname{atan2}(\Im(x), \Re(x)) $$

    Args:
        x: A complex tensor, \((*, 2)\).

    Returns:
        The angle tensor, \((*,)\).

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> angle(x)
        tensor([0.0000, 0.7854])
    """

    return torch.atan2(x[..., 1], x[..., 0])


def prod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Returns the element-wise product of \(x\) and \(y\).

    $$ x y = \Re(x) \Re(y) - \Im(x) \Im(y)
        + i \left( \Re(x) \Im(y) - \Im(x) \Re(y) \right) $$

    Args:
        x: A complex tensor, \((*, 2)\).
        y: A complex tensor, \((*, 2)\).

    Returns:
        The product tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([[2.,  0.], [0.7071,  0.7071]])
        >>> y = torch.tensor([[2., -0.], [0.7071, -0.7071]])
        >>> prod(x, y)
        tensor([[4.0000, 0.0000],
                [1.0000, 0.0000]])
    """

    x_r, x_i = x[..., 0], x[..., 1]
    y_r, y_i = y[..., 0], y[..., 1]

    return complx(x_r * y_r - x_i * y_i, x_i * y_r + x_r * y_i)


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Returns the element-wise dot-product of \(x\) and \(y\).

    $$ x \odot y = \Re(x) \Re(y) + \Im(x) \Im(y) $$

    Args:
        x: A complex tensor, \((*, 2)\).
        y: A complex tensor, \((*, 2)\).

    Returns:
        The dot-product tensor, \((*,)\).

    Example:
        >>> x = torch.tensor([[2.,  0.], [0.7071,  0.7071]])
        >>> y = torch.tensor([[2., -0.], [0.7071, -0.7071]])
        >>> dot(x, y)
        tensor([4., 0.])
    """

    return (x * y).sum(dim=-1)


def pow(x: torch.Tensor, exponent: float) -> torch.Tensor:
    r"""Returns the power of \(x\) with `exponent`.

    $$ x^p = \left| x \right|^p \exp(i \phi(x))^p $$

    Args:
        x: A complex tensor, \((*, 2)\).
        exponent: The exponent \(p\).

    Returns:
        The power tensor, \((*, 2)\).

    Example:
        >>> x = torch.tensor([[2., 0.], [0.7071, 0.7071]])
        >>> pow(x, 2.)
        tensor([[ 4.0000e+00,  0.0000e+00],
                [-4.3711e-08,  9.9998e-01]])
    """

    r = mod(x, squared=True) ** (exponent / 2)
    phi = angle(x) * exponent

    return polar(r, phi)
