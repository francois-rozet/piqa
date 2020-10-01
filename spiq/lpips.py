r"""Learned Perceptual Image Patch Similarity (LPIPS)

This module implements the LPIPS in PyTorch.

Credits:
    Inspired by lpips-pytorch
    https://github.com/S-aiueo32/lpips-pytorch

References:
    [1] The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
    (Zhang et al., 2018)
    https://arxiv.org/abs/1801.03924
"""

###########
# Imports #
###########

import inspect
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


#############
# Constants #
#############

_SHIFT = torch.Tensor([-.030, -.088, -.188])
_SCALE = torch.Tensor([.458, .448, .450])


#############
# Functions #
#############

def normalize(x: torch.Tensor, dim=(), norm='L2', epsilon: float=1e-8) -> torch.Tensor:
    r"""Returns `x` normalized.

    Args:
        x: input tensor
        dim: dimension(s) to normalize
        norm: norm function name ('L1' or 'L2')
        epsilon: numerical stability

    Wikipedia:
        https://en.wikipedia.org/wiki/Norm_(mathematics)
    """

    if norm == 'L1':
        norm = x.abs().sum(dim=dim, keepdim=True)
    else: # norm == 'L2'
        norm = torch.sqrt((x ** 2).sum(dim=dim, keepdim=True))

    return x / (norm + epsilon)


###########
# Classes #
###########

class Intermediate(nn.Module):
    r"""Module that returns the outputs of target indermediate layers of a sequential module during its forward pass.

    Args:
        layers: sequential module
        targets: target layer indexes
    """

    def __init__(self, layers: nn.Sequential, targets: list):
        super().__init__()

        self.layers = layers
        self.targets = set(targets)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input: input tensor
        """

        output = []

        for i, layer in enumerate(self.layers):
            input = layer(input)

            if i in self.targets:
                output.append(input.clone())

            if len(output) == len(self.targets):
                break

        return output


class LPIPS(nn.Module):
    r"""Creates a criterion that measures the LPIPS between an input and a target.
    """

    def __init__(self, network='AlexNet', normalize=False, reduction='mean'):
        super().__init__()

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

        self.net = Intermediate(layers, targets)

        state_path = os.path.join(
            os.path.dirname(inspect.getsourcefile(self.__init__)),
            f'weights/lpips_{network}.pth'
        )

        self.lin = nn.ModuleList([
            nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0, bias=False)
            for c in channels
        ])
        self.lin.load_state_dict(torch.load(state_path))

        self.register_buffer('shift', _SHIFT.view(1, -1, 1, 1))
        self.register_buffer('scale', _SCALE.view(1, -1, 1, 1))

        for x in [self.parameters(), self.buffers()]:
            for y in x:
                y.requires_grad = False

        self.normalize = normalize
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input: input tensor, (N, C, H, W)
            target: target tensor, (N, C, H, W)
        """

        if self.normalize:
            input = input * 2 - 1
            target = target * 2 - 1

        input_features = self.net((input - self.shift) / self.scale)
        target_features = self.net((target - self.shift) / self.scale)

        residuals = []

        for loss, (fx, fy) in zip(self.lin, zip(input_features, target_features)):
            fx = normalize(fx, dim=1, norm='L2')
            fy = normalize(fy, dim=1, norm='L2')

            residuals.append(loss((fx - fy) ** 2).mean(dim=(-1, -2)))

        l = torch.cat(residuals, dim=-1).sum(dim=-1)

        if self.reduction == 'mean':
            return l.mean()
        elif self.reduction == 'sum':
            return l.sum()

        return l
