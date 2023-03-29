#!/usr/bin/env python

r"""Benchmark against other packages

Packages:
    piq: https://pypi.org/project/piq/
    pytorch-msssim: https://pypi.org/project/pytorch-msssim/
"""

import argparse
import pandas as pd
import time
import torch

from PIL import Image, ImageFilter
from torch import Tensor
from torchvision.transforms.functional import to_tensor
from typing import *
from urllib.request import urlopen

import piq
import piqa
import pytorch_msssim as vainf


LENNA = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_(test_image).png'

NO_REF = {
    'TV': {
        'piq.tv': piq.total_variation,
        'piqa.TV': piqa.TV(norm='L2'),
    },
}

FULL_REF = {
    'PSNR': {
        'piq.psnr': piq.psnr,
        'piqa.PSNR': piqa.PSNR(),
    },
    'SSIM': {
        'piq.ssim': lambda x, y: piq.ssim(x, y, downsample=False),
        'vainf.SSIM': vainf.SSIM(data_range=1),
        'piqa.SSIM': piqa.SSIM(),
    },
    'MS-SSIM': {
        'piq.ms_ssim': piq.multi_scale_ssim,
        'vainf.MS_SSIM': vainf.MS_SSIM(data_range=1),
        'piqa.MS_SSIM': piqa.MS_SSIM(),
    },
    'LPIPS': {
        # 'piq.LPIPS': piq.LPIPS(),
        'piqa.LPIPS': piqa.LPIPS(network='vgg'),
    },
    'GMSD': {
        'piq.gmsd': piq.gmsd,
        'piqa.GMSD': piqa.GMSD(),
    },
    'MS-GMSD': {
        'piq.ms_gmsd': piq.multi_scale_gmsd,
        'piqa.MS_GMSD': piqa.MS_GMSD(),
    },
    'MDSI': {
        'piq.mdsi': piq.mdsi,
        'piqa.MDSI': piqa.MDSI(),
    },
    'HaarPSI': {
        'piq.haarpsi': piq.haarpsi,
        'piqa.HaarPSI': piqa.HaarPSI(),
    },
    'VSI': {
        'piq.vsi': piq.vsi,
        'piqa.VSI': piqa.VSI(),
    },
    'FSIM': {
        'piq.fsim': piq.fsim,
        'piqa.FSIM': piqa.FSIM(),
    },
}

MANY_REF = {
    'FID': {
        'piq.FID': piq.FID(),
        'piqa.FID': piqa.FID(),
    },
}

METRICS = [*NO_REF, *FULL_REF, *MANY_REF]


def timeit(f: Callable, n: int) -> float:
    start = time.perf_counter()

    for _ in range(n):
        f()

    end = time.perf_counter()

    return (end - start) * 1000 / n


def cuda_timeit(f: Callable, n: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(n):
        f()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / n


def benchmark(
    metrics: List[str] = METRICS,
    batch: int = 1,
    warmup: int = 16,
    repeat: int = 64,
    device: str = 'cpu',
    tracing: bool = False,
):
    # Image
    lenna = Image.open(urlopen(LENNA))
    noisy = lenna.filter(ImageFilter.GaussianBlur)

    lenna = to_tensor(lenna).repeat(batch, 1, 1, 1)
    noisy = to_tensor(noisy).repeat(batch, 1, 1, 1)

    # Features
    A = torch.randn(256, 256) / 256 ** 0.5
    fx = torch.randn(4096, 256)
    fy = torch.randn(4096, 256) @ A + torch.randn(256)

    # Benchmark
    if metrics is None:
        metrics = METRICS.copy()
        metrics.remove('LPIPS')
        metrics.remove('FID')

    for name in metrics:
        if name in NO_REF:
            versions = NO_REF[name]
            x, y = lenna.to(device), None
        elif name in FULL_REF:
            versions = FULL_REF[name]
            x, y = noisy.to(device), lenna.to(device)
        elif name in MANY_REF:
            versions = MANY_REF[name]
            x, y = fx.to(device), fy.to(device)
        else:
            continue

        print(name)
        print('-' * len(name))

        rows = []

        for key, metric in versions.items():
            if hasattr(metric, 'to'):
                metric.to(device)

            if y is None:
                f = lambda x: metric(x)
            else:
                f = lambda x: metric(x, y)

            if tracing and 'piqa' in key:
                f = torch.jit.trace(f, x)

            g = lambda: torch.autograd.functional.jacobian(f, x)

            for _ in range(warmup):
                g()

            if device == 'cuda':
                time = cuda_timeit(g, n=repeat)
            else:
                time = timeit(g, n=repeat)

            rows.append({
                'version': key,
                'value': f(x).item(),
                'gradient': g().norm(2).item(),
                'time': time,
            })

        df = pd.DataFrame(rows)

        print(df.sort_values(by='time', ignore_index=True))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark')

    parser.add_argument('-m', '--metrics', nargs='+', default=None, choices=METRICS, help='metrics to benchmark')
    parser.add_argument('-b', '--batch', type=int, default=1, help='batch size')
    parser.add_argument('-w', '--warmup', type=int, default=16, help='number of warmups')
    parser.add_argument('-r', '--repeat', type=int, default=64, help='number of repeats')
    parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda'], help='computation device')
    parser.add_argument('-t', '--tracing', default=False, action='store_true', help='enable tracing')

    args = parser.parse_args()

    benchmark(**vars(args))
