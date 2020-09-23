#!/usr/bin/env python

import os
import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='spiq',
    version='0.0.1',
    description='Image quality metrics in PyTorch.',
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='pytorch image processing metrics',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    url='https://github.com/francois-rozet/spiq',
    install_requires=required,
    packages=setuptools.find_packages()
)
