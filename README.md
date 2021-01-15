<p align="center"><img src="https://raw.githubusercontent.com/francois-rozet/piqa/master/banner.svg" width="80%"></p>

> PIQA is not endorsed by Facebook, Inc.; PyTorch, the PyTorch logo and any related marks are trademarks of Facebook, Inc.

# PyTorch Image Quality Assessment

This package is a collection of measures and metrics for image quality assessment in various image processing tasks such as denoising, super-resolution, image interpolation, etc. It relies heavily on [PyTorch](https://github.com/pytorch/pytorch) and takes advantage of its efficiency and automatic differentiation.

It should be noted that `piqa` is directly inspired from the [`piq`](https://github.com/photosynthesis-team/piq) project, while focusing on the conciseness, readability and understandability of its (sub-)modules, such that anyone can freely and easily reuse and/or adapt them to its needs.

However, conciseness should never be at the expense of efficency; `piqa`'s implementations are up to 2 times faster than those of other IQA PyTorch packages like [`kornia`](https://github.com/kornia/kornia), [`piq`](https://github.com/photosynthesis-team/piq) and [`IQA-pytorch`](https://github.com/dingkeyan93/IQA-optimization).

> `piqa` should be pronounced *pika* (like Pikachu ⚡️)

## Installation

The `piqa` package is available on [PyPI](https://pypi.org/project/piqa/), which means it is installable with `pip`:

```bash
pip install piqa
```

Alternatively, if you need the lastest features, you can install it using

```bash
git clone https://github.com/francois-rozet/piqa
cd piqa
python setup.py install
```

or copy the package directly to your project, with

```bash
git clone https://github.com/francois-rozet/piqa
cd piqa
cp -R piqa <path/to/project>/piqa
```

## Getting started

The `piqa` package is divided in several submodules, each of which implements the functions and/or classes related to a specific image quality assessement metric.

```python
import torch
from piqa import psnr, ssim

x = torch.rand(3, 3, 256, 256, requires_grad=True).cuda()
y = torch.rand(3, 3, 256, 256, requires_grad=True).cuda()

# PSNR function
l = psnr.psnr(x, y)

# SSIM instantiable object
criterion = ssim.SSIM().cuda()
l = criterion(x, y)
l.backward()
```

### Metrics

| Acronym | Module         | Year | Metric                                                                                               |
|:-------:|----------------|:----:|------------------------------------------------------------------------------------------------------|
|    TV   | [piqa.tv]      | 1937 | [Total Variation](https://en.wikipedia.org/wiki/Total_variation)                                     |
|   PSNR  | [piqa.psnr]    |   /  | [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)               |
|   SSIM  | [piqa.ssim]    | 2004 | [Structural Similarity](https://en.wikipedia.org/wiki/Structural_similarity)                         |
| MS-SSIM | [piqa.ssim]    | 2004 | [Multi-Scale Structural Similarity](https://ieeexplore.ieee.org/abstract/document/1292216/)          |
|  LPIPS  | [piqa.lpips]   | 2018 | [Learned Perceptual Image Patch Similarity](https://arxiv.org/abs/1801.03924)                        |
|   GMSD  | [piqa.gmsd]    | 2013 | [Gradient Magnitude Similarity Deviation](https://arxiv.org/abs/1308.3052)                           |
| MS-GMSD | [piqa.gmsd]    | 2017 | [Multi-Scale Gradient Magnitude Similiarity Deviation](https://ieeexplore.ieee.org/document/7952357) |
|   MDSI  | [piqa.mdsi]    | 2016 | [Mean Deviation Similarity Index](https://arxiv.org/abs/1608.07433)                                  |
| HaarPSI | [piqa.haarpsi] | 2018 | [Haar Perceptual Similarity Index](https://arxiv.org/abs/1607.06140)                                 |

[piqa.tv]: piqa/tv.py
[piqa.psnr]: piqa/psnr.py
[piqa.ssim]: piqa/ssim.py
[piqa.lpips]: piqa/lpips.py
[piqa.gmsd]: piqa/gmsd.py
[piqa.mdsi]: piqa/mdsi.py
[piqa.haarpsi]: piqa/haarpsi.py

## Documentation

The [documentation](https://francois-rozet.github.io/piqa/) of this package is generated automatically using [`pdoc`](https://github.com/pdoc3/pdoc).

### Code style

The code follows the [Google Python style](https://google.github.io/styleguide/pyguide.html) and is compliant with [YAPF](https://github.com/google/yapf).
