<p align="center"><img src="https://raw.githubusercontent.com/francois-rozet/piqa/master/docs/images/banner.svg" width="80%"></p>

# PyTorch Image Quality Assessment

PIQA is a collection of PyTorch metrics for image quality assessment in various image processing tasks such as generation, denoising, super-resolution, interpolation, etc. It focuses on the efficiency, conciseness and understandability of its (sub-)modules, such that anyone can easily reuse and/or adapt them to its needs.

> PIQA should be pronounced *pika* (like Pikachu ⚡️)

## Installation

The `piqa` package is available on [PyPI](https://pypi.org/project/piqa), which means it is installable via `pip`.

```
pip install piqa
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/francois-rozet/piqa
```

## Getting started

In `piqa`, each metric is associated to a class, child of `torch.nn.Module`, which has to be instantiated to evaluate the metric. All metrics are differentiable and support CPU and GPU (CUDA).

```python
import torch
import piqa

# PSNR
x = torch.rand(5, 3, 256, 256)
y = torch.rand(5, 3, 256, 256)

psnr = piqa.PSNR()
l = psnr(x, y)

# SSIM
x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
y = torch.rand(5, 3, 256, 256).cuda()

ssim = piqa.SSIM().cuda()
l = 1 - ssim(x, y)
l.backward()
```

Like `torch.nn` built-in components, these classes are based on functional definitions of the metrics, which are less user-friendly, but more versatile.

```python
from piqa.ssim import ssim
from piqa.utils.functional import gaussian_kernel

kernel = gaussian_kernel(11, sigma=1.5).expand(3, 11, 11)

l = 1 - ssim(x, y, kernel=kernel)
```

For more information, check out the documentation at [piqa.readthedocs.io](https://piqa.readthedocs.io).

### Available metrics

| Class     | Range  | Objective | Year | Metric                                                                                               |
|:---------:|:------:|:---------:|:----:|------------------------------------------------------------------------------------------------------|
| `TV`      | [0, ∞] | /         | 1937 | [Total Variation](https://en.wikipedia.org/wiki/Total_variation)                                     |
| `PSNR`    | [0, ∞] | max       | /    | [Peak Signal-to-Noise Ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)               |
| `SSIM`    | [0, 1] | max       | 2004 | [Structural Similarity](https://en.wikipedia.org/wiki/Structural_similarity)                         |
| `MS_SSIM` | [0, 1] | max       | 2004 | [Multi-Scale Structural Similarity](https://ieeexplore.ieee.org/document/1292216/)                   |
| `LPIPS`   | [0, ∞] | min       | 2018 | [Learned Perceptual Image Patch Similarity](https://arxiv.org/abs/1801.03924)                        |
| `GMSD`    | [0, ∞] | min       | 2013 | [Gradient Magnitude Similarity Deviation](https://arxiv.org/abs/1308.3052)                           |
| `MS_GMSD` | [0, ∞] | min       | 2017 | [Multi-Scale Gradient Magnitude Similarity Deviation](https://ieeexplore.ieee.org/document/7952357)  |
| `MDSI`    | [0, ∞] | min       | 2016 | [Mean Deviation Similarity Index](https://arxiv.org/abs/1608.07433)                                  |
| `HaarPSI` | [0, 1] | max       | 2018 | [Haar Perceptual Similarity Index](https://arxiv.org/abs/1607.06140)                                 |
| `VSI`     | [0, 1] | max       | 2014 | [Visual Saliency-based Index](https://ieeexplore.ieee.org/document/6873260)                          |
| `FSIM`    | [0, 1] | max       | 2011 | [Feature Similarity](https://ieeexplore.ieee.org/document/5705575)                                   |
| `FID`     | [0, ∞] | min       | 2017 | [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500)                                       |

### Tracing

All metrics of `piqa` support [PyTorch's tracing](https://pytorch.org/docs/stable/generated/torch.jit.trace.html), which optimizes their execution, especially on GPU.

```python
ssim = piqa.SSIM().cuda()
ssim_traced = torch.jit.trace(ssim, (x, y))

l = 1 - ssim_traced(x, y)  # should be faster ¯\_(ツ)_/¯
```

### Assert

PIQA uses type assertions to raise meaningful messages when a metric doesn't receive an input of the expected type. This feature eases a lot early prototyping and debugging, but it might hurt a little the performances. If you need the absolute best performances, the assertions can be disabled with the Python flag [`-O`](https://docs.python.org/3/using/cmdline.html#cmdoption-o). For example,

```
python -O your_awesome_code_using_piqa.py
```

Alternatively, you can disable PIQA's type assertions within your code with

```python
piqa.utils.set_debug(False)
```

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](https://github.com/francois-rozet/piqa/blob/master/CONTRIBUTING.md).
