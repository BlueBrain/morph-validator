#!/usr/bin/env python
import importlib.util

from setuptools import setup, find_packages

# read the contents of the README file
with open("README.rst", "r", encoding="utf-8") as f:
    README = f.read()

spec = importlib.util.spec_from_file_location(
    "morph_validator.version",
    "morph_validator/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

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
        "Source": "git@bbpgitlab.epfl.ch:nse/morph-validator.git",
    },
    license="BBP-internal-confidential",
    python_requires='>=3.7',
    install_requires=[
        'pandas>=0.25,<1.3',  # version 1.3 does not sum list items of groupby
        'joblib>=0.14',
        'numpy>=1.14',
        'scipy>=1.3',
        'lxml>=4.3.4',
        'morph-tool>=2.9.0,<3.0',
        'neurom>=3.0,<4.0',
        'bluepy>=2.3.0,<3.0',
        'seaborn>=0.10.1',
        'tqdm>=4.46.0',
        'matplotlib>=2.2.0',
    ],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
