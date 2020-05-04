#!/usr/bin/env python

import imp
from setuptools import setup, find_packages

VERSION = imp.load_source("", "morph_validator/version.py").__version__

setup(
    name="morph-validator",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="tool for validating morphologies against existing morphologies of the same type",
    url="https://bbpteam.epfl.ch/documentation/projects/morph-validator",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
        "Source": "ssh://bbpcode.epfl.ch/nse/morph-validator",
    },
    license="BBP-internal-confidential",
    python_requires='>=3.6',
    install_requires=[
        'pandas>=0.25',
        'numpy>=1.14',
        'scipy>=1.3',
        'lxml>=4.3.4',
        'neurom>=1.4.15',
        'bluepy>=0.14',
    ],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
