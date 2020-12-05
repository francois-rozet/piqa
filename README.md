# PyTorch Image Quality Assessment

This package is a collection of measures and metrics for image quality assessment in various image processing tasks such as denoising, super-resolution, image interpolation, etc. It relies heavily on [PyTorch](https://github.com/pytorch/pytorch) and takes advantage of its efficiency and automatic differentiation.

It should noted that `piqa` is directly inspired from the [`piq`](https://github.com/photosynthesis-team/piq) project. However, it focuses on the conciseness, readability and understandability of its (sub-)modules, such that anyone can freely and easily reuse and/or adapt them to its needs.

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

```python
import torch
import piqa.psnr as psnr
import piqa.ssim as ssim

x = torch.rand(3, 3, 256, 256)
y = torch.rand(3, 3, 256, 256)

# PSNR function
l = psnr.psnr(x, y)

# SSIM instantiable object
criterion = ssim.SSIM().cuda()
l = criterion(x, y)
```

## Documentation

The [documentation](https://francois-rozet.github.io/piqa/) of this package is generated automatically using [`pdoc`](https://github.com/pdoc3/pdoc).

> The code follows the [Google Python style](https://google.github.io/styleguide/pyguide.html) and is compliant with [YAPF](https://github.com/google/yapf).
