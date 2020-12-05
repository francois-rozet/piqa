#!/usr/bin/env python

import os
import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='piqa',
    version='1.0.0',
    description='PyTorch Image Quality Assessment',
    long_description=readme,
    long_description_content_type='text/markdown',
    keywords='pytorch image processing metrics',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    url='https://github.com/francois-rozet/piqa',
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
