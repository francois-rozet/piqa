#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='piqa',
    version='1.2.2',
    packages=setuptools.find_packages(),
    description='PyTorch Image Quality Assessment',
    keywords='image quality processing metrics torch vision',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    license='MIT license',
    url='https://github.com/francois-rozet/piqa',
    project_urls={
        'Documentation': 'https://francois-rozet.github.io/piqa/',
        'Source': 'https://github.com/francois-rozet/piqa',
        'Tracker': 'https://github.com/francois-rozet/piqa/issues',
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    install_requires=required,
    python_requires='>=3.6',
)
