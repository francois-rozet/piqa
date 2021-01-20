#!/usr/bin/env python

import doctest
import os
import sys
import unittest

sys.path.append(os.path.abspath('..'))

from piqa import utils
from piqa.utils import color, complex, functional
from piqa import (
    tv,
    psnr,
    ssim,
    lpips,
    mdsi,
    gmsd,
    haarpsi,
)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    modules = [
        utils,
        color,
        complex,
        functional,
        tv,
        psnr,
        ssim,
        lpips,
        mdsi,
        gmsd,
        haarpsi,
    ]

    for m in modules:
        suite.addTests(doctest.DocTestSuite(m))

    runner = unittest.TextTestRunner()
    runner.run(suite)
