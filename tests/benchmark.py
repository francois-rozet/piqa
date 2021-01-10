#!/usr/bin/env python

r"""Benchmark against other IQA packages

Packages:
    kornia: https://pypi.org/project/kornia/
    piq: https://pypi.org/project/piq/
    IQA-pytorch: https://pypi.org/project/IQA-pytorch/
"""

import os
import pandas as pd
import sys
import timeit
import torch
import urllib.request as request

from torchvision import transforms
from PIL import Image, ImageFilter

import kornia.losses as kornia
import piq
import IQA_pytorch as IQA

sys.path.append(os.path.abspath('..'))

from piqa import (
    tv,
    psnr,
    ssim,
    lpips,
    mdsi,
    gmsd,
    haarpsi,
)


if __name__ == '__main__':
    lenna = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

    x = Image.open(request.urlopen(lenna))
    y = x.filter(ImageFilter.BLUR)

    x = transforms.ToTensor()(x).unsqueeze(0).cuda()
    y = transforms.ToTensor()(y).unsqueeze(0).cuda()

    x.requires_grad_()
    y.requires_grad_()

    metrics = {
        ('TV', 1): {
            'piq.tv': piq.total_variation,
            'tv': tv.tv,
            'piq.TV': piq.TVLoss(),
            'TV': tv.TV(),
        },
        ('PSNR', 2): {
            'piq.psnr': piq.psnr,
            'psnr': psnr.psnr,
            'kornia.PSNR': kornia.PSNRLoss(max_val=1.),
            'PSNR': psnr.PSNR(),
        },
        ('SSIM', 2): {
            'piq.ssim': piq.ssim,
            'ssim': ssim.ssim,
            'kornia.SSIM-halfloss': kornia.SSIM(window_size=11, reduction='mean'),
            'piq.SSIM-loss': piq.SSIMLoss(),
            'IQA.SSIM-loss': IQA.SSIM(),
            'SSIM': ssim.SSIM(),
        },
        ('MS-SSIM', 2): {
            'piq.msssim': piq.multi_scale_ssim,
            'msssim': ssim.msssim,
            'piq.MSSSIM-loss': piq.MultiScaleSSIMLoss(),
            'IQA.MSSSIM-loss': IQA.MS_SSIM(),
            'MSSSIM': ssim.MSSSIM(),
        },
        ('LPIPS', 2): {
            'piq.LPIPS': piq.LPIPS(),
            'IQA.LPIPS': IQA.LPIPSvgg(),
            'LPIPS': lpips.LPIPS(network='vgg')
        },
        ('GMSD', 2): {
            'piq.gmsd': piq.gmsd,
            'gmsd': gmsd.gmsd,
            'piq.GMSD': piq.GMSDLoss(),
            'IQA.GMSD': IQA.GMSD(),
            'GMSD': gmsd.GMSD(),
        },
        ('MS-GMSD', 2): {
            'piq.msgmsd': piq.multi_scale_gmsd,
            'msgmsd': gmsd.msgmsd,
            'piq.MSGMSD': piq.MultiScaleGMSDLoss(),
            'MSGMSD': gmsd.MSGMSD(),
        },
        ('MDSI', 2): {
            'piq.mdsi': piq.mdsi,
            'mdsi': mdsi.mdsi,
            'piq.MDSI-loss': piq.MDSILoss(),
            'MDSI': mdsi.MDSI(),
        },
        ('HaarPSI', 2): {
            'piq.haarpsi': piq.haarpsi,
            'haarpsi': haarpsi.haarpsi,
            'piq.HaarPSI-loss': piq.HaarPSILoss(),
            'HaarPSI': haarpsi.HaarPSI(),
        },
    }

    for (name, nargs), methods in metrics.items():
        print(name)
        print('-' * len(name))

        data = {
            'method': [],
            'value': [],
            'time': []
        }

        for key, method in methods.items():
            if hasattr(method, 'to'):
                method.to(x.device)

            if nargs == 1:
                f = lambda: method(x).squeeze()
            else:
                f = lambda: method(x, y).squeeze()

            if key.endswith('-loss'):
                key = key.replace('-loss', '')
                g = lambda: 1. - f()
            elif key.endswith('-halfloss'):
                key = key.replace('-halfloss', '')
                g = lambda: 1. - 2. * f()
            else:
                g = f

            data['method'].append(key)
            data['value'].append(g().item())
            data['time'].append(timeit.timeit(g, number=42))

        print(pd.DataFrame(data).sort_values(by='time', ignore_index=True))
