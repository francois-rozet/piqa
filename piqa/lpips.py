r"""Learned Perceptual Image Patch Similarity (LPIPS)

This module implements the LPIPS in PyTorch.

Original:
    https://github.com/richzhang/PerceptualSimilarity

References:
    .. [Zhang2018] The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Zhang et al., 2018)

    .. [Deng2009] ImageNet: A large-scale hierarchical image database (Deng et al, 2009)
"""

import inspect
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.hub as hub

from torch import Tensor
from typing import Dict, List

from .utils import _jit, assert_type, reduce_tensor


ORIGIN: str = 'https://github.com/richzhang/PerceptualSimilarity'
SHIFT: Tensor = torch.tensor([0.485, 0.456, 0.406])
SCALE: Tensor = torch.tensor([0.229, 0.224, 0.225])


def get_weights(
    network: str = 'alex',
    version: str = 'v0.1',
) -> Dict[str, Tensor]:
    r"""Returns the official LPIPS weights for `network`.

    Args:
        network: Specifies the perception network that is used:
            `'alex'` | `'squeeze'` | `'vgg'`.
        version: Specifies the official version release:
            `'v0.0'` | `'v0.1'`.

    Example:
        >>> w = get_weights(network='alex')
        >>> w.keys()
        dict_keys(['0.1.weight', '1.1.weight', '2.1.weight', '3.1.weight', '4.1.weight'])
    """

    # Load from URL
    weights = hub.load_state_dict_from_url(
        f'{ORIGIN}/raw/master/lpips/weights/{version}/{network}.pth',
        map_location='cpu',
    )

    # Format keys
    weights = {
        k.replace('lin', '').replace('.model', ''): v
        for (k, v) in weights.items()
    }

    return weights


class Intermediary(nn.Module):
    r"""Module that catches and returns the outputs of indermediate
    target layers of a sequential module during its forward pass.

    Args:
        layers: A sequential module.
        targets: A list of target layer indexes.
    """

    def __init__(self, layers: nn.Sequential, targets: List[int]):
        super().__init__()

        self.layers = nn.ModuleList()
        j = 0

        seq: List[nn.Module] = []

        for i, layer in enumerate(layers):
            seq.append(layer)

            if i == targets[j]:
                self.layers.append(nn.Sequential(*seq))
                seq.clear()

                j += 1
                if j == len(targets):
                    break

    def forward(self, input: Tensor) -> List[Tensor]:
        output = []

        for layer in self.layers:
            input = layer(input)
            output.append(input)

        return output


class LPIPS(nn.Module):
    r"""Creates a criterion that measures the LPIPS
    between an input :math:`x` and a target :math:`y`.

    .. math::
        \text{LPIPS}(x, y) = \sum_{l \, \in \, \mathcal{F}}
            w_l \cdot \text{MSE}(\phi_l(x), \phi_l(y))

    where :math:`\phi_l` represents the normalized output of an intermediate
    layer :math:`l` in a perceptual network :math:`\mathcal{F}`.

    Note:
        :class:`LPIPS` is a trainable metric. For more details, refer to [Zhang2018]_.

    Args:
        network: Specifies the perceptual network :math:`\mathcal{F}` to use:
            `'alex'` | `'squeeze'` | `'vgg'`.
        scaling: Whether the input and target need to be scaled w.r.t. [Deng2009]_.
        dropout: Whether dropout is used or not.
        pretrained: Whether the official weights :math:`w_l` are used or not.
        eval: Whether to initialize the object in evaluation mode or not.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

    Shapes:
        input: :math:`(N, 3, H, W)`
        target: :math:`(N, 3, H, W)`
        output: :math:`(N,)` or :math:`()` depending on `reduction`

    Example:
        >>> criterion = LPIPS().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        network: str = 'alex',
        scaling: bool = True,
        dropout: bool = False,
        pretrained: bool = True,
        eval: bool = True,
        reduction: str = 'mean',
    ):
        super().__init__()

        # ImageNet scaling
        self.scaling = scaling
        self.register_buffer('shift', SHIFT.view(1, -1, 1, 1))
        self.register_buffer('scale', SCALE.view(1, -1, 1, 1))

        # Perception layers
        if network == 'alex':  # AlexNet
            layers = models.alexnet(pretrained=True).features
            targets = [1, 4, 7, 9, 11]
            channels = [64, 192, 384, 256, 256]
        elif network == 'squeeze':  # SqueezeNet
            layers = models.squeezenet1_1(pretrained=True).features
            targets = [1, 4, 7, 9, 10, 11, 12]
            channels = [64, 128, 256, 384, 384, 512, 512]
        elif network == 'vgg':  # VGG16
            layers = models.vgg16(pretrained=True).features
            targets = [3, 8, 15, 22, 29]
            channels = [64, 128, 256, 512, 512]
        else:
            raise ValueError(f'Unknown network architecture {network}')

        self.net = Intermediary(layers, targets)
        for p in self.net.parameters():
            p.requires_grad = False

        # Linear comparators
        self.lins = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(inplace=True) if dropout else nn.Identity(),
                nn.Conv2d(c, 1, kernel_size=1, bias=False),
            ) for c in channels
        ])

        if pretrained:
            self.lins.load_state_dict(get_weights(network=network))

        if eval:
            self.eval()

        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert_type(
            input, target,
            device=self.shift.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0., 1.) if self.scaling else (0., -1.),
        )

        # ImageNet scaling
        if self.scaling:
            input = (input - self.shift) / self.scale
            target = (target - self.shift) / self.scale

        # LPIPS
        residuals = []

        for lin, fx, fy in zip(self.lins, self.net(input), self.net(target)):
            fx = fx / torch.linalg.norm(fx, dim=1, keepdim=True)
            fy = fy / torch.linalg.norm(fy, dim=1, keepdim=True)

            mse = ((fx - fy) ** 2).mean(dim=(-1, -2), keepdim=True)
            residuals.append(lin(mse).flatten())

        l = torch.stack(residuals, dim=-1).sum(dim=-1)

        return reduce_tensor(l, self.reduction)
