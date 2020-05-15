"""test `plotting` module"""
import tempfile
from subprocess import call
import numpy as np
import pandas as pd
import neurom as nm
from pandas import testing

from morph_validator.spatial import _sample_morph_points, count_points_distribution

from tests.utils import TEST_DATA_DIR

SPATIAL_DATA_DIR = TEST_DATA_DIR / 'spatial'


def test_sample_morph_points():
    morph_path = SPATIAL_DATA_DIR / 'sample_morph_points.asc'
    morph = nm.load_neuron(morph_path)
    sampled_points = _sample_morph_points(morph, 10)
    expected_basal = np.array([
        [0., 11., 0., ],
        [0., 21., 0., ],
        [-5.65685445, 28.65685445, 0., ],
        [-13.8053271, 34.31732977, 0., ],
        [0., 31., 0., ],
        [0., 41., 0., ],
        [1.38485692, 30.30600158, 0., ],
        [11., 0., 0., ]])
    assert np.allclose(expected_basal, sampled_points[nm.NeuriteType.basal_dendrite])
    expected_axon = np.array([[0., -11., 0., ]])
    assert np.allclose(expected_axon, sampled_points[nm.NeuriteType.axon])


def test_count_points_distribution():
    expected_df = pd.DataFrame([
        ['L2_X', 'r1', nm.NeuriteType.axon, -1, 0],
        ['L2_X', 'r1', nm.NeuriteType.axon, 2, 1],
        ['L2_X', 'r1', nm.NeuriteType.basal_dendrite, -1, 0],
        ['L2_X', 'r1', nm.NeuriteType.basal_dendrite, 2, 8],
        ['L6_Y', 'r2', nm.NeuriteType.axon, -1, 1],
        ['L6_Y', 'r2', nm.NeuriteType.axon, 3, 1],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, -1, 8],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, 2, 6],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, 3, 1],
    ], columns=['mtype', 'soma_region', 'neurite', 'region', 'count'])
    with tempfile.TemporaryDirectory() as tmpdirname:
        call(['brainbuilder', 'atlases',
              '-n', '1,2,3,4,5,6',
              '-t', '200,100,100,100,100,200',
              '-d', '100',
              '-o', tmpdirname,
              'column',
              '-a', '1000', ])
        circuit_config = {
            "cells": str(SPATIAL_DATA_DIR / 'circuit.mvd3'),
            "morphologies": str(SPATIAL_DATA_DIR),
            "atlas": tmpdirname}
        points_df = count_points_distribution(circuit_config)
        expected_df.set_index(['mtype', 'soma_region', 'neurite', 'region'], inplace=True)
        testing.assert_frame_equal(expected_df, points_df)
