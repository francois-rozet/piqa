r"""Learned Perceptual Image Patch Similarity (LPIPS)

This module implements the LPIPS in PyTorch.

Original:
    https://github.com/richzhang/PerceptualSimilarity

References:
    | The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Zhang et al., 2018)
    | https://arxiv.org/abs/1801.03924
"""

import torch
import torch.nn as nn
import torchvision

from torch import Tensor
from typing import *

from .utils import assert_type
from .utils.color import ImageNetNorm
from .utils.functional import l2_norm, reduce_tensor


ORIGIN: str = 'https://github.com/richzhang/PerceptualSimilarity'


def get_weights(network: str = 'alex', version: str = 'v0.1') -> List[Tensor]:
    r"""Returns the official LPIPS weights for `network`.

    Args:
        network: Specifies the perception network that is used:
            `'alex'`, `'squeeze'` or `'vgg'`.
        version: Specifies the official version release:
            `'v0.0'` or `'v0.1'`.
    """

    weights = torch.hub.load_state_dict_from_url(
        f'{ORIGIN}/raw/master/lpips/weights/{version}/{network}.pth',
        map_location='cpu',
    )

    return [w.flatten() for w in weights.values()]


class Perceptual(nn.Module):
    r"""Perceptual network that intercepts and returns the output of target layers
    within its foward pass.

    Args:
        layers: A list of layers.
        targets: A list of target layer indices.
    """

    def __init__(self, layers: List[nn.Module], targets: List[int]):
        super().__init__()

        self.blocks = nn.ModuleList()

        i = 0
        for j in targets:
            self.blocks.append(nn.Sequential(*layers[i : j + 1]))
            i = j + 1

    def forward(self, x: Tensor) -> List[Tensor]:
        y = []

        for block in self.blocks:
            x = block(x)
            y.append(x)

        return y


class LPIPS(nn.Module):
    r"""Measures the LPIPS between an input and a target.

    .. math::
        \text{LPIPS}(x, y) = \sum_{l \, \in \, \mathcal{F}}
            w_l \cdot \text{MSE}(\phi_l(x), \phi_l(y))

    where :math:`\phi_l` represents the normalized output of an intermediate layer
    :math:`l` in a perceptual network :math:`\mathcal{F}` and :math:`w_l` are the
    official weights of Zhang et al. (2018).

    Args:
        network: Specifies the perceptual network :math:`\mathcal{F}` to use:
            `'alex'`, `'squeeze'` or `'vgg'`.
        epsilon: A numerical stability term.
        reduction: Specifies the reduction to apply to the output:
            `'none'`, `'mean'` or `'sum'`.

    Example:
        >>> criterion = LPIPS()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> l = criterion(x, y)
        >>> l.shape
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        network: str = 'alex',
        epsilon: float = 1e-10,
        reduction: str = 'mean',
    ):
        super().__init__()

        # ImageNet normalization
        self.normalize = ImageNetNorm()

        # Perception layers
        if network == 'alex':  # AlexNet
            layers = torchvision.models.alexnet(weights='DEFAULT').features
            targets = [1, 4, 7, 9, 11]
        elif network == 'squeeze':  # SqueezeNet
            layers = torchvision.models.squeezenet1_1(weights='DEFAULT').features
            targets = [1, 4, 7, 9, 10, 11, 12]
        elif network == 'vgg':  # VGG16
            layers = torchvision.models.vgg16(weights='DEFAULT').features
            targets = [3, 8, 15, 22, 29]
        else:
            raise ValueError(f"Unknown network architecture {network}")

        self.net = Perceptual(list(layers), targets)
        self.net.eval()

        # Weights
        self.weights = nn.ParameterList(get_weights(network=network))

        # Disable gradients
        for p in self.parameters():
            p.requires_grad = False

        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Args:
            x: An input tensor, :math:`(N, 3, H, W)`.
            y: A target tensor, :math:`(N, 3, H, W)`.

        Returns:
            The LPIPS vector, :math:`(N,)` or :math:`()` depending on `reduction`.
        """

        assert_type(
            x, y,
            device=self.normalize.shift.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0.0, 1.0),
        )

        # ImageNet normalization
        x = self.normalize(x)
        y = self.normalize(y)

        # LPIPS
        lpips = 0

        for w, fx, fy in zip(self.weights, self.net(x), self.net(y)):
            fx = fx / (l2_norm(fx, dim=1, keepdim=True) + self.epsilon)
            fy = fy / (l2_norm(fy, dim=1, keepdim=True) + self.epsilon)

            mse = (fx - fy).square().mean(dim=(-1, -2))
            lpips = lpips + mse @ w

        return reduce_tensor(lpips, self.reduction)
