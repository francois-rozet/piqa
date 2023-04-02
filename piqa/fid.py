r"""Fréchet Inception Distance (FID)

This module implements the FID in PyTorch.

Original:
    https://github.com/bioinf-jku/TTUR

Wikipedia:
    https://wikipedia.org/wiki/Frechet_inception_distance

References:
    | GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (Heusel et al., 2017)
    | https://arxiv.org/abs/1706.08500
"""

import torch
import torch.nn as nn
import torchvision

from torch import Tensor
from typing import *

from .utils import assert_type
from .utils.color import ImageNetNorm


@torch.jit.script_if_tracing
def frechet_distance(
    mu_x: Tensor,
    sigma_x: Tensor,
    mu_y: Tensor,
    sigma_y: Tensor,
) -> Tensor:
    r"""Returns the Fréchet distance between two multivariate Gaussian distributions.

    .. math:: d^2 = \left\| \mu_x - \mu_y \right\|_2^2 +
        \operatorname{tr} \left( \Sigma_x + \Sigma_y - 2 \sqrt{\Sigma_x \Sigma_y} \right)

    Wikipedia:
        https://wikipedia.org/wiki/Frechet_distance

    Args:
        mu_x: The mean :math:`\mu_x` of the first distribution, :math:`(*, D)`.
        sigma_x: The covariance :math:`\Sigma_x` of the first distribution, :math:`(*, D, D)`.
        mu_y: The mean :math:`\mu_y` of the second distribution, :math:`(*, D)`.
        sigma_y: The covariance :math:`\Sigma_y` of the second distribution, :math:`(*, D, D)`.

    Example:
        >>> mu_x = torch.arange(3).float()
        >>> sigma_x = torch.eye(3)
        >>> mu_y = 2 * mu_x + 1
        >>> sigma_y = 2 * sigma_x + 1
        >>> frechet_distance(mu_x, sigma_x, mu_y, sigma_y)
        tensor(15.8710)
    """

    a = (mu_x - mu_y).square().sum(dim=-1)
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class InceptionV3(nn.Sequential):
    r"""Pretrained Inception-v3 network.

    References:
        | Rethinking the Inception Architecture for Computer Vision (Szegedy et al., 2015)
        | https://arxiv.org/abs/1512.00567

    Args:
        logits: Whether to return the class logits or the last pooling features.

    Example:
        >>> x = torch.randn(5, 3, 256, 256)
        >>> inception = InceptionV3()
        >>> logits = inception(x)
        >>> logits.shape
        torch.Size([5, 1000])
    """

    def __init__(self, logits: bool = True):
        net = torchvision.models.inception_v3(weights='DEFAULT')

        layers = [
            net.Conv2d_1a_3x3,
            net.Conv2d_2a_3x3,
            net.Conv2d_2b_3x3,
            net.maxpool1,
            net.Conv2d_3b_1x1,
            net.Conv2d_4a_3x3,
            net.maxpool2,
            net.Mixed_5b,
            net.Mixed_5c,
            net.Mixed_5d,
            net.Mixed_6a,
            net.Mixed_6b,
            net.Mixed_6c,
            net.Mixed_6d,
            net.Mixed_6e,
            net.Mixed_7a,
            net.Mixed_7b,
            net.Mixed_7c,
            net.avgpool,
            nn.Flatten(-3),
        ]

        if logits:
            layers.append(net.fc)

        super().__init__(*layers)


class FID(nn.Module):
    r"""Measures the FID between two set of inception features.

    Note:
        See :meth:`FID.features` for how to get inception features.

    Example:
        >>> criterion = FID()
        >>> x = torch.randn(1024, 256)
        >>> y = torch.randn(2048, 256)
        >>> l = criterion(x, y)
        >>> l.shape
        torch.Size([])
    """

    def __init__(self):
        super().__init__()

        # ImageNet normalization
        self.normalize = ImageNetNorm()

        # Inception-v3
        self.inception = InceptionV3(logits=False)
        self.inception.eval()

        # Disable gradients
        for p in self.parameters():
            p.requires_grad = False

    def features(self, x: Tensor, no_grad: bool = True) -> Tensor:
        r"""Returns the inception features of an input.

        Tip:
            If you cannot get the inception features of your input at once, for instance
            because of memory limitations, you can split it in smaller batches and
            concatenate the outputs afterwards.

        Args:
            x: An input tensor, :math:`(N, 3, H, W)`.
            no_grad: Whether to disable gradients or not.

        Returns:
            The features, :math:`(N, 2048)`.
        """

        assert_type(
            x,
            device=self.normalize.shift.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0.0, 1.0),
        )

        # ImageNet normalization
        x = self.normalize(x)

        # Features
        if no_grad:
            with torch.no_grad():
                return self.inception(x)
        else:
            return self.inception(x)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""
        Args:
            x: An input tensor, :math:`(M, D)`.
            y: A target tensor, :math:`(N, D)`.

        Returns:
            The FID, :math:`()`.
        """

        # Mean & covariance
        mu_x, sigma_x = torch.mean(x, dim=0), torch.cov(x.T)
        mu_y, sigma_y = torch.mean(y, dim=0), torch.cov(y.T)

        # Fréchet distance
        return frechet_distance(mu_x, sigma_x, mu_y, sigma_y)
