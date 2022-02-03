<p align="center"><img src="https://raw.githubusercontent.com/francois-rozet/piqa/master/banner.svg" width="80%"></p>

> PIQA is not endorsed by Facebook, Inc.; PyTorch, the PyTorch logo and any related marks are trademarks of Facebook, Inc.

# PyTorch Image Quality Assessment

The `piqa` package is a collection of measures and metrics for image quality assessment in various image processing tasks such as denoising, super-resolution, image interpolation, etc. It relies only on [PyTorch](https://github.com/pytorch/pytorch) and takes advantage of its efficiency and automatic differentiation.

PIQA is directly inspired from the [`piq`](https://github.com/photosynthesis-team/piq) project, but focuses on the conciseness, readability and understandability of its (sub-)modules, such that anyone can easily reuse and/or adapt them to its needs.

However, conciseness should never be at the expense of efficiency; PIQA's implementations are up to 3 times faster than those of other IQA PyTorch packages like [`kornia`](https://github.com/kornia/kornia), [`piq`](https://github.com/photosynthesis-team/piq) and [`IQA-pytorch`](https://github.com/dingkeyan93/IQA-optimization).

> PIQA should be pronounced *pika* (like Pikachu ⚡️)

## Installation

The `piqa` package is available on [PyPI](https://pypi.org/project/piqa/), which means it is installable with `pip`:

```bash
pip install piqa
```

Alternatively, if you need the latest features, you can install it using

```bash
pip install git+https://github.com/francois-rozet/piqa
```

or copy the package directly to your project, with

```bash
git clone https://github.com/francois-rozet/piqa
cp -R piqa/piqa <path/to/project>/piqa
```

## Documentation

The [documentation](https://francois-rozet.github.io/piqa/) of this package is generated automatically by [`Sphinx`](https://pypi.org/project/Sphinx/).

## Getting started

In `piqa`, each metric is associated to a class, child of `torch.nn.Module`, which has to be instantiated to evaluate the metric.

```python
import torch

# PSNR
from piqa import PSNR

x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)

psnr = PSNR()
l = psnr(x, y)

# SSIM
from piqa import SSIM

x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
y = torch.rand(5, 3, 256, 256).cuda()

ssim = SSIM().cuda()
l = 1 - ssim(x, y)
l.backward()
```

Like `torch.nn` built-in components, these classes are based on functional definitions of the metrics, which are less user-friendly, but more versatile.

```python
import torch

from piqa.ssim import ssim
from piqa.utils.functional import gaussian_kernel

x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)

kernel = gaussian_kernel(11, sigma=1.5).repeat(3, 1, 1)

l = ssim(x, y, kernel=kernel, channel_avg=False)
```

### Metrics

| Acronym | Class     | Range    | Objective | Year | Metric                                                                                               |
|:-------:|:---------:|:--------:|:---------:|:----:|------------------------------------------------------------------------------------------------------|
| TV      | `TV`      | `[0, ∞]` | /         | 1937 | [Total Variation](https://en.wikipedia.org/wiki/Total_variation)                                     |
| PSNR    | `PSNR`    | `[0, ∞]` | max       | /    | [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)               |
| SSIM    | `SSIM`    | `[0, 1]` | max       | 2004 | [Structural Similarity](https://en.wikipedia.org/wiki/Structural_similarity)                         |
| MS-SSIM | `MS_SSIM` | `[0, 1]` | max       | 2004 | [Multi-Scale Structural Similarity](https://ieeexplore.ieee.org/document/1292216/)                   |
| LPIPS   | `LPIPS`   | `[0, ∞]` | min       | 2018 | [Learned Perceptual Image Patch Similarity](https://arxiv.org/abs/1801.03924)                        |
| GMSD    | `GMSD`    | `[0, ∞]` | min       | 2013 | [Gradient Magnitude Similarity Deviation](https://arxiv.org/abs/1308.3052)                           |
| MS-GMSD | `MS_GMSD` | `[0, ∞]` | min       | 2017 | [Multi-Scale Gradient Magnitude Similarity Deviation](https://ieeexplore.ieee.org/document/7952357)  |
| MDSI    | `MDSI`    | `[0, ∞]` | min       | 2016 | [Mean Deviation Similarity Index](https://arxiv.org/abs/1608.07433)                                  |
| HaarPSI | `HaarPSI` | `[0, 1]` | max       | 2018 | [Haar Perceptual Similarity Index](https://arxiv.org/abs/1607.06140)                                 |
| VSI     | `VSI`     | `[0, 1]` | max       | 2014 | [Visual Saliency-based Index](https://ieeexplore.ieee.org/document/6873260)                          |
| FSIM    | `FSIM`    | `[0, 1]` | max       | 2011 | [Feature Similarity](https://ieeexplore.ieee.org/document/5705575)                                   |

### JIT

Most functional components of `piqa` support  PyTorch's JIT, *i.e.* [TorchScript](https://pytorch.org/docs/stable/jit.html), which is a way to create serializable and optimizable functions from PyTorch code.

By default, jitting is disabled for those components. To enable it, the `PIQA_JIT` environment variable has to be set to `1`. To do so temporarily,

* UNIX-like `bash`

```bash
export PIQA_JIT=1
```

* Windows `cmd`

```cmd
set PIQA_JIT=1
```

* Microsoft `PowerShell`

```powershell
$env:PIQA_JIT=1
```

### Assert

PIQA uses type assertions to raise meaningful messages when an object-oriented component doesn't receive an input of the expected type. This feature eases a lot early prototyping and debugging, but it might hurt a little the performances.

If you need the absolute best performances, the assertions can be disabled with the Python flag [`-O`](https://docs.python.org/3/using/cmdline.html#cmdoption-o). For example,

```bash
python -O your_awesome_code_using_piqa.py
```

Alternatively, you can disable PIQA's type assertions within your code with

```python
from piqa.utils import set_debug
set_debug(False)
```
