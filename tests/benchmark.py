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

import piqa


LENNA = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

METRICS = {
    'TV': (1, {
        'kornia.tv': kornia.total_variation,
        'piq.tv': lambda x: piq.total_variation(x, norm_type='l1'),
        'piqa.TV': piqa.TV(),
    }),
    'PSNR': (2, {
        'sk.psnr-np': sk.peak_signal_noise_ratio,
        'piq.psnr': piq.psnr,
        'kornia.PSNR': kornia.PSNRLoss(1.),
        'piqa.PSNR': piqa.PSNR(),
    }),
    'SSIM': (2, {
        'sk.ssim-np': lambda x, y: sk.structural_similarity(
            x, y,
            win_size=11,
            multichannel=True,
            gaussian_weights=True,
        ),
        'piq.ssim': lambda x, y: piq.ssim(x, y, downsample=False),
        'kornia.SSIM-halfloss': kornia.SSIMLoss(11),
        'IQA.SSIM-loss': IQA.SSIM(),
        'vainf.SSIM': vainf.SSIM(data_range=1.),
        'piqa.SSIM': piqa.SSIM(),
    }),
    'MS-SSIM': (2, {
        'piq.ms_ssim': piq.multi_scale_ssim,
        'IQA.MS_SSIM-loss': IQA.MS_SSIM(),
        'vainf.MS_SSIM': vainf.MS_SSIM(data_range=1.),
        'piqa.MS_SSIM': piqa.MS_SSIM(),
    }),
    'LPIPS': (2, {
        # 'piq.LPIPS': piq.LPIPS(),
        # 'IQA.LPIPS': IQA.LPIPSvgg(),
        'piqa.LPIPS': piqa.LPIPS(network='vgg')
    }),
    'GMSD': (2, {
        'piq.gmsd': piq.gmsd,
        'piqa.GMSD': piqa.GMSD(),
    }),
    'MS-GMSD': (2, {
        'piq.ms_gmsd': piq.multi_scale_gmsd,
        'piqa.MS_GMSD': piqa.MS_GMSD(),
    }),
    'MDSI': (2, {
        'piq.mdsi': piq.mdsi,
        'piqa.MDSI': piqa.MDSI(),
    }),
    'HaarPSI': (2, {
        'piq.haarpsi': piq.haarpsi,
        'piqa.HaarPSI': piqa.HaarPSI(),
    }),
    'VSI': (2, {
        'piq.vsi': piq.vsi,
        'piqa.VSI': piqa.VSI(),
    }),
    'FSIM': (2, {
        'piq.fsim': piq.fsim,
        'piqa.FSIM': piqa.FSIM(),
    }),
}


def timeit(f, n: int) -> float:
    start = time.perf_counter()

    for _ in range(n):
        f()

    end = time.perf_counter()

    return (end - start) * 1000


def cuda_timeit(f, n: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(n):
        f()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end)


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
    # Device
    if not torch.cuda.is_available():
        device = 'cpu'

    # Backend
    if device == 'cuda':
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

    # Metrics
    if metrics:
        metrics = {
            k: v for (k, v) in METRICS.items()
            if k in metrics
        }
    else:
        metrics = {k: v for (k, v) in METRICS.items()}

        del metrics['LPIPS']

    # Benchmark
    for name, (nargs, methods) in metrics.items():
        print(name)
        print('-' * len(name))

        data = {
            'method': [],
            'value': [],
            'time': []
        }

        with contextlib.nullcontext() if grad else torch.no_grad():
            for key, method in methods.items():
                if hasattr(method, 'to'):
                    method.to(x.device)

                if '-np' in key:
                    a, b = truth, noisy
                else:
                    a, b = x, y

                if nargs == 1:
                    f = lambda: method(a).mean()
                else:
                    f = lambda: method(a, b).mean()

                if '-loss' in key:
                    g = lambda: 1. - f()
                elif '-halfloss' in key:
                    g = lambda: 1. - 2. * f()
                else:
                    g = f

                if '-' in key:
                    base = key[:key.find('-')]
                else:
                    base = key

                data['method'].append(base)
                data['value'].append(float(g()))

                time = timeit(g, n=warm) / warm

                if device == 'cuda' and '-np' not in key:
                    time = cuda_timeit(g, n=loops) / loops

                data['time'].append(time)

        df = pd.DataFrame(data)

        print(df.sort_values(by='time', ignore_index=True))
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
