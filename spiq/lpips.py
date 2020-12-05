r"""Learned Perceptual Image Patch Similarity (LPIPS)

This module implements the LPIPS in PyTorch.

Credits:
    Inspired by [lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch)

References:
    [1] The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
    (Zhang et al., 2018)
    https://arxiv.org/abs/1801.03924
"""

import inspect
import os
import torch
import torch.nn as nn
import torchvision.models as models

from spiq.utils import build_reduce, normalize_tensor, Intermediary

_SHIFT = torch.Tensor([0.485, 0.456, 0.406])
_SCALE = torch.Tensor([0.229, 0.224, 0.225])


class LPIPS(nn.Module):
    r"""Creates a criterion that measures the LPIPS
    between an input and a target.

    Args:
        network: Specifies the perception network to use:
            `'AlexNet'` | `'SqueezeNet'` | `'VGG16'`.
        scaling: Whether the input and target need to
            be scaled w.r.t. ImageNet.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

    Shape:
        * Input: (N, 3, H, W)
        * Target: (N, 3, H, W)
        * Output: (N,) or (1,) depending on `reduction`
    """

    def __init__(
        self,
        network: str = 'AlexNet',
        scaling: bool = True,
        reduction: str = 'mean',
        trainable: bool = False,
    ):
        r""""""
        super().__init__()

        # ImageNet scaling
        self.scaling = scaling
        self.register_buffer('shift', _SHIFT.view(1, -1, 1, 1))
        self.register_buffer('scale', _SCALE.view(1, -1, 1, 1))

        # Perception layers
        if network == 'AlexNet':
            layers = models.alexnet(pretrained=True).features
            targets = [1, 4, 7, 9, 11]
            channels = [64, 192, 384, 256, 256]
        elif network == 'SqueezeNet':
            layers = models.squeezenet1_1(pretrained=True).features
            targets = [1, 4, 7, 9, 10, 11, 12]
            channels = [64, 128, 256, 384, 384, 512, 512]
        elif network == 'VGG16':
            layers = models.vgg16(pretrained=True).features
            targets = [3, 8, 15, 22, 29]
            channels = [64, 128, 256, 512, 512]
        else:
            raise ValueError('Unknown network architecture ' + network)

        self.net = Intermediary(layers, targets)

        # Linear comparators
        state_path = os.path.join(
            os.path.dirname(inspect.getsourcefile(self.__init__)),
            f'weights/lpips_{network}.pth'
        )

        self.lin = nn.ModuleList([
            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0, bias=False)
            for c in channels
        ])
        self.lin.load_state_dict(torch.load(state_path))

        if not trainable:
            # Prevent gradients
            for p in self.parameters():
                p.requires_grad = False

        self.reduce = build_reduce(reduction)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Defines the computation performed at every call.
        """

        if self.scaling:
            input = (input - self.shift) / self.scale
            target = (target - self.shift) / self.scale

        residuals = []

        for loss, fx, fy in zip(self.lin, self.net(input), self.net(target)):
            fx = normalize_tensor(fx, dim=1, norm='L2')
            fy = normalize_tensor(fy, dim=1, norm='L2')

            residuals.append(loss((fx - fy) ** 2).mean(dim=(-1, -2)))

        l = torch.cat(residuals, dim=-1).sum(dim=-1)

        return self.reduce(l)
