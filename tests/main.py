#!/usr/bin/env python

import doctest
import os
import sys
import unittest

sys.path.append(os.path.abspath('..'))

from piqa import utils

utils._jit = lambda f: f

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
