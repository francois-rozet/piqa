#!/usr/bin/env python

import doctest
import os
import sys
import unittest

sys.path.append(os.path.abspath('..'))

import piqa.utils
import piqa.tv
import piqa.psnr
import piqa.ssim
import piqa.lpips
import piqa.mdsi
import piqa.gmsd


def add_doctests(suite, modules):
    for m in modules:
        suite.addTests(doctest.DocTestSuite(m))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    add_doctests(
        suite,
        [
            piqa.utils,
            piqa.tv,
            piqa.psnr,
            piqa.ssim,
            piqa.lpips,
            piqa.mdsi,
            piqa.gmsd,
        ],
    )

    runner = unittest.TextTestRunner()
    runner.run(suite)
