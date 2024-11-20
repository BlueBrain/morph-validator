Morph Validator
===============
This project is a collection of various modules for morphology validation.

- `features` module allows to extract morphology features.
- `zscores` module allows to validate the extracted features.
- `spatial` module validates morphologies spatially in a given volume of space (voxel data).


Installation
------------

Note, this originally used bluepy, but was not updated to use bluepysnap.
Updating it is left as an exercise to the reader.

In a fresh virtualenv:

.. code:: bash

    pip install morph-validator


Usage
-----
For usage of `features` and `zscores` see `examples/zscores.ipynb`. For usage of `spatial` module
see `examples/spatial.py`.


Acknowledgements
================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license see LICENSE.txt.

Copyright © 2022-2024 Blue Brain Project/EPFL
