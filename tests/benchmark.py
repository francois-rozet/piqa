#!/usr/bin/env python

r"""Benchmark against other packages

Packages:
    scikit-image: https://pypi.org/project/scikit-image/
    kornia: https://pypi.org/project/kornia/
    piq: https://pypi.org/project/piq/
    IQA-pytorch: https://pypi.org/project/IQA-pytorch/
    pytorch-msssim: https://pypi.org/project/pytorch-msssim/
"""

import contextlib
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


LENNA = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

METRICS = {
    'TV': lambda: (1, {
        'kornia.tv': kornia.total_variation,
        'piq.tv': lambda x: piq.total_variation(x, norm_type='l1'),
        'piqa.tv': tv.tv,
        'piq.TV': piq.TVLoss(norm_type='l1'),
        'piqa.TV': tv.TV(),
    }),
    'PSNR': lambda: (2, {
        'sk.psnr': sk.peak_signal_noise_ratio,
        'piq.psnr': piq.psnr,
        'piqa.psnr': psnr.psnr,
        'kornia.PSNR': kornia.PSNRLoss(max_val=1.),
        'piqa.PSNR': psnr.PSNR(),
    }),
    'SSIM': lambda: (2, {
        'sk.ssim': lambda x, y: sk.structural_similarity(
            x, y,
            win_size=11,
            multichannel=True,
            gaussian_weights=True,
        ),
        'piq.ssim': piq.ssim,
        'kornia.SSIM-halfloss': kornia.SSIM(
            window_size=11,
            reduction='mean',
        ),
        'piq.SSIM-loss': piq.SSIMLoss(),
        'IQA.SSIM-loss': IQA.SSIM(),
        'vainf.SSIM': vainf.SSIM(data_range=1.),
        'piqa.SSIM': ssim.SSIM(),
    }),
    'MS-SSIM': lambda: (2, {
        'piq.ms_ssim': piq.multi_scale_ssim,
        'piq.MS_SSIM-loss': piq.MultiScaleSSIMLoss(),
        'IQA.MS_SSIM-loss': IQA.MS_SSIM(),
        'vainf.MS_SSIM': vainf.MS_SSIM(data_range=1.),
        'piqa.MS_SSIM': ssim.MS_SSIM(),
    }),
    'LPIPS': lambda: (2, {
        'piq.LPIPS': piq.LPIPS(),
        'IQA.LPIPS': IQA.LPIPSvgg(),
        'piqa.LPIPS': lpips.LPIPS(network='vgg')
    }),
    'GMSD': lambda: (2, {
        'piq.gmsd': piq.gmsd,
        'piq.GMSD': piq.GMSDLoss(),
        'IQA.GMSD': IQA.GMSD(),
        'piqa.GMSD': gmsd.GMSD(),
    }),
    'MS-GMSD': lambda: (2, {
        'piq.ms_gmsd': piq.multi_scale_gmsd,
        'piq.MS_GMSD': piq.MultiScaleGMSDLoss(),
        'piqa.MS_GMSD': gmsd.MS_GMSD(),
    }),
    'MDSI': lambda: (2, {
        'piq.mdsi': piq.mdsi,
        'piq.MDSI-loss': piq.MDSILoss(),
        'piqa.MDSI': mdsi.MDSI(),
    }),
    'HaarPSI': lambda: (2, {
        'piq.haarpsi': piq.haarpsi,
        'piq.HaarPSI-loss': piq.HaarPSILoss(),
        'piqa.HaarPSI': haarpsi.HaarPSI(),
    }),
}


def timeit(f, n: int) -> (float, float):
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


def main(
    url: str,
    metrics: list = [],
    batch: int = 1,
    warm: int = 69,
    loops: int = 420,
    device: str = 'cuda',
    grad: bool = True,
    backend: bool = True,
):
    # Backend
    torch.backends.cudnn.enabled = backend
    torch.backends.cudnn.benchmark = backend

    # Images
    truth = Image.open(request.urlopen(url))
    noisy = truth.filter(ImageFilter.BLUR)

    noisy, truth = np.array(noisy), np.array(truth)

    totensor = transforms.ToTensor()

    x = totensor(noisy).repeat(batch, 1, 1, 1).to(device)
    y = totensor(truth).repeat(batch, 1, 1, 1).to(device)

    x.requires_grad_()
    y.requires_grad_()

    # Metrics
    if metrics:
        metrics = {
            k: v() for (k, v) in METRICS.items()
            if k in metrics
        }
    else:
        metrics = {k: v() for (k, v) in METRICS.items()}

    # Benchmark
    for name, (nargs, methods) in metrics.items():
        print(name)
        print('-' * len(name))

        data = {
            'method': [],
            'value': [],
            'time-sync': [],
            'time-async': []
        }

        with contextlib.nullcontext() if grad else torch.no_grad():
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
                    data['time-sync'].append(-1.)
                    data['time-async'].append(-1.)
                    continue

                _ = timeit(g, n=warm)  # activate JIT

                t_sync, t_async = timeit(g, n=loops)

                data['time-sync'].append(t_sync)
                data['time-async'].append(t_async)

        df = pd.DataFrame(data)

        print(df.sort_values(by='time-sync', ignore_index=True))
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark')

    parser.add_argument('-u', '--url', default=LENNA, help='image URL')
    parser.add_argument('-m', '--metrics', nargs='+', default=[], choices=list(METRICS.keys()), help='metrics to benchmark')
    parser.add_argument('-b', '--batch', type=int, default=1, help='batch size')
    parser.add_argument('-w', '--warm', type=int, default=69, help='number of warming loops')
    parser.add_argument('-l', '--loops', type=int, default=420, help='number of loops')
    parser.add_argument('-d', '--device', default='cuda', choices=['cpu', 'cuda'], help='computation device')
    parser.add_argument('-grad', default=True, action='store_false', help='enable gradients')
    parser.add_argument('-backend', default=True, action='store_false', help='enable backends')

    args = parser.parse_args()

    main(
        metrics=args.metrics,
        url=args.url,
        batch=args.batch,
        warm=args.warm,
        loops=args.loops,
        device=args.device,
        grad=args.grad,
        backend=args.backend,
    )
