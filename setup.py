# -*- coding: utf-8 -*-
# @Author: schlowm0 (Moritz Milde)
# @Date:   2019-09-16

from setuptools import setup
import os

setup(
    name="speed",
    version="0.1",
    author="Moritz Milde",
    author_email="m.milde@westernsydney.edu.au",
    description=("This compiler converts networks"
                 "described using brian2 oder teili"
                 "such that these networks can be simulated"
                 "on the FPGA-based Neuromorphic Signal Processsor: ORCA."),
    license="MIT",
    keywords="Neural algorithms, Large-scale simulation, Spiking Neural Networks",
    url="https://github.com/neuromorphicsystems/speed",
    packages=[
        'speed',
    ],

    install_requires=[
        'setuptools>=39.2.0',
        'numpy>=1.14.5',
        'pyqtgraph>=0.10.0',
        'h5py>=2.8.0',
        'pyqt5>=5.10.1'
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Computational neuroscientists",
        "Intended Audience :: Neuromorphic engineers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python3",
    ],
)

