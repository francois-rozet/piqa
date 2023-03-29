#!/usr/bin/env python

import doctest
import piqa
import unittest


def tests():
    suite = unittest.TestSuite()
    modules = [
        piqa.utils,
        piqa.utils.color,
        piqa.utils.functional,
        piqa.tv,
        piqa.psnr,
        piqa.ssim,
        piqa.lpips,
        piqa.mdsi,
        piqa.gmsd,
        piqa.haarpsi,
        piqa.vsi,
        piqa.fsim,
        piqa.fid,
    ]

    for m in modules:
        suite.addTests(doctest.DocTestSuite(m))

    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    tests()
