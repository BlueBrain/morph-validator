Morph Validator
===============
Project that allows to validate morphologies.

Installation
------------
In a fresh virtualenv:

.. code:: bash

    pip install --index-url https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/ morph-validator

Usage
-----
**Python**

.. code:: python

    failed_features_per_file = validate(
        Path('../tests/data/morphologies/valid/mini'),
        Path('../tests/data/morphologies/test'))
    for file_failed_features in failed_features_per_file:
        print(file_failed_features)
        print('--------------------------------------')

Important
---------
Current implementation has been chosen among other alternatives as the most short and neat. To see
what other alternatives were, please look at the directory ``validator_alternatives`` in the commit
**@a98beb18**, Change-Id: **I2af778cb9c8d97a8f7fa4f26769f21a5268f5511**.


