#!/usr/bin/env python

import imp
from setuptools import setup, find_packages

VERSION = imp.load_source("", "morph_validator/version.py").__version__

with open('README.rst', encoding='utf-8') as f:
    README = f.read()

setup(
    name="morph-validator",
    author="bbp-ou-nse",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    long_description=README,
    long_description_content_type="text/x-rst",
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
        'joblib>=0.14',
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
