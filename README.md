# Simple PyTorch Image Quality

Collection of measures and metrics for automatic image quality assessment in various image-to-image tasks such as denoising, super-resolution, image interpolation, etc.

This library is directly inspired from [piq](https://github.com/photosynthesis-team/piq). However, it focuses on the simplicity, readability and understandability of its modules, such that anyone can freely and easily reuse and/or adapt them to its needs.

## Installation

To install the current version of `spiq`,

```bash
git clone https://github.com/francois-rozet/spiq
cd spiq
python setup.py install
```

You can also copy the library directly to your project.

```bash
git clone https://github.com/francois-rozet/spiq
cd spiq
cp -R spiq <path/to/project>/spiq
```

## Getting started

```python
import torch
import spiq

x = torch.rand(3, 3, 256, 256)
y = torch.rand(3, 3, 256, 256)

a = spiq.psnr(x, y)
b = spiq.ssim(x, y)
```
