# Simple PyTorch Image Quality

This package is a collection of measures and metrics for image quality assessment in various image processing tasks such as denoising, super-resolution, image interpolation, etc. It relies heavily on [PyTorch](https://github.com/pytorch/pytorch) and takes advantage of its efficiency and automatic differentiation.

It should noted that `spiq` is directly inspired from the [`piq`](https://github.com/photosynthesis-team/piq) project. However, it focuses on the conciseness, readability and understandability of its (sub-)modules, such that anyone can freely and easily reuse and/or adapt them to its needs.

## Installation

To install the current version of `spiq`,

```bash
git clone https://github.com/francois-rozet/spiq
cd spiq
python setup.py install
```

You can also copy the package directly to your project.

```bash
git clone https://github.com/francois-rozet/spiq
cd spiq
cp -R spiq <path/to/project>/spiq
```

## Getting started

```python
import torch
import spiq.psnr as psnr
import spiq.ssim as ssim

x = torch.rand(3, 3, 256, 256)
y = torch.rand(3, 3, 256, 256)

# PSNR function
l = psnr.psnr(x, y)

# SSIM instantiable object
criterion = ssim.SSIM().cuda()
l = criterion(x, y)
```

## Documentation

The [documentation](https://francois-rozet.github.io/spiq/) of this package is generated automatically using [`pdoc`](https://github.com/pdoc3/pdoc).

```bash
pdoc spiq --html --config "git_link_template='https://github.com/francois-rozet/spiq/blob/{commit}/{path}#L{start_line}-L{end_line}'"
```

> The code follows the [Google Python style](https://google.github.io/styleguide/pyguide.html) and is compliant with [YAPF](https://github.com/google/yapf).
