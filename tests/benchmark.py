#!/usr/bin/env python

r"""Benchmark against other packages

Packages:
    scikit-image: https://pypi.org/project/scikit-image/
    kornia: https://pypi.org/project/kornia/
    piq: https://pypi.org/project/piq/
    IQA-pytorch: https://pypi.org/project/IQA-pytorch/
    pytorch-msssim: https://pypi.org/project/pytorch-msssim/
"""

import numpy as np
import os
import pandas as pd
import sys
import time
import torch
import urllib.request as request

from torchvision import transforms
from PIL import Image, ImageFilter

import skimage.metrics as sk
import kornia.losses as kornia
import piq
import IQA_pytorch as IQA
import pytorch_msssim as vainf

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


def timeit(f, n: int = 420) -> (float, float):
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)

    start = time.perf_counter()
    cuda_start.record()

    for _ in range(n):
        f()

    cuda_end.record()
    end = time.perf_counter()

    torch.cuda.synchronize()

    return cuda_start.elapsed_time(cuda_end) / 1000, end - start


if __name__ == '__main__':
    lenna = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

    truth = Image.open(request.urlopen(lenna))
    noisy = truth.filter(ImageFilter.BLUR)

    noisy, truth = np.array(noisy), np.array(truth)

    x = transforms.ToTensor()(noisy).unsqueeze(0).cuda()
    y = transforms.ToTensor()(truth).unsqueeze(0).cuda()

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
            'sk.psnr': sk.peak_signal_noise_ratio,
            'piq.psnr': piq.psnr,
            'psnr': psnr.psnr,
            'kornia.PSNR': kornia.PSNRLoss(max_val=1.),
            'PSNR': psnr.PSNR(),
        },
        ('SSIM', 2): {
            'sk.ssim': lambda x, y: sk.structural_similarity(
                x, y,
                win_size=11,
                multichannel=True,
                gaussian_weights=True,
            ),
            'piq.ssim': piq.ssim,
            'ssim': ssim.ssim,
            'kornia.SSIM-halfloss': kornia.SSIM(
                window_size=11,
                reduction='mean',
            ),
            'piq.SSIM-loss': piq.SSIMLoss(),
            'IQA.SSIM-loss': IQA.SSIM(),
            'vainf.SSIM': vainf.SSIM(data_range=1.),
            'SSIM': ssim.SSIM(),
        },
        ('MS-SSIM', 2): {
            'piq.msssim': piq.multi_scale_ssim,
            'msssim': ssim.msssim,
            'piq.MSSSIM-loss': piq.MultiScaleSSIMLoss(),
            'IQA.MSSSIM-loss': IQA.MS_SSIM(),
            'vainf.MSSSIM': vainf.MS_SSIM(data_range=1.),
            'MSSSIM': ssim.MSSSIM(),
        },
        # ('LPIPS', 2): {
        #     'piq.LPIPS': piq.LPIPS(),
        #     'IQA.LPIPS': IQA.LPIPSvgg(),
        #     'LPIPS': lpips.LPIPS(network='vgg')
        # },
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
            'time-sync': [],
            'time-async': []
        }

        for key, method in methods.items():
            if hasattr(method, 'to'):
                method.to(x.device)

            if key.startswith('sk.'):
                a, b = truth, noisy
            else:
                a, b = x, y

            if nargs == 1:
                f = lambda: method(a).mean()
            else:
                f = lambda: method(a, b).mean()

            if key.endswith('-loss'):
                key = key.replace('-loss', '')
                g = lambda: 1. - f()
            elif key.endswith('-halfloss'):
                key = key.replace('-halfloss', '')
                g = lambda: 1. - 2. * f()
            else:
                g = f

            data['method'].append(key)
            data['value'].append(float(g()))

            if key.startswith('sk.'):
                data['time-sync'].append(0.)
                data['time-async'].append(0.)
                continue

            _ = timeit(g, n=42)  # activate JIT

            t_sync, t_async = timeit(g)

            data['time-sync'].append(t_sync)
            data['time-async'].append(t_async)

        print(pd.DataFrame(data).sort_values(by='time-sync', ignore_index=True))
