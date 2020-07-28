Morph Validator
===============
This project is a collection of various modules for morphology validation.

- `features` module allows to extract morphology features.
- `zscores` module allows to validate the extracted features.
- `spatial` module validates morphologies spatially in a given volume of space (voxel data).


Installation
------------
In a fresh virtualenv:

.. code:: bash

    pip install --index-url https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ morph-validator

Usage
-----
For usage of `features` and `zscores` see `examples/zscores.ipynb`. For usage of `spatial` module
see `examples/spatial.py`.


