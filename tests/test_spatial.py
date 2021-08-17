"""test `plotting` module"""
import tempfile
from mock import Mock
from collections import Counter
from subprocess import call
import numpy as np
import pandas as pd
import neurom as nm
from pandas import testing

from morph_validator.spatial import\
    iter_positions, count_circuit_points_distribution,\
    sample_morph_voxel_values, _count_values_in_bins,\
    count_cells_points_distribution, relative_depth_volume
from voxcell import VoxelData
from bluepy import Circuit
from tests.utils import TEST_DATA_DIR

SPATIAL_DATA_DIR = TEST_DATA_DIR / 'spatial'


def test_iter_positions():
    morph_path = SPATIAL_DATA_DIR / 'iter_positions.asc'
    morph = nm.load_neuron(morph_path)
    expected_basal = np.array([
        [0., 11., 0., ],
        [0., 21., 0., ],
        [-5.65685445, 28.65685445, 0., ],
        [-13.8053271, 34.31732977, 0., ],
        [0., 31., 0., ],
        [0., 41., 0., ],
        [1.38485692, 30.30600158, 0., ],
        [11., 0., 0., ]])
    basal_filter = lambda s: s.type == nm.NeuriteType.basal_dendrite
    assert np.allclose(
        expected_basal,
        np.array(list(
            iter_positions(morph, 10, basal_filter))))
    expected_axon = np.array([[0., -11., 0., ]])
    axon_filter = lambda s: s.type == nm.NeuriteType.axon
    assert np.allclose(
        expected_axon,
        np.array(list(
            iter_positions(morph, 10, axon_filter))))


def test_iter_positions_no_skips_branch_points():
    morph = nm.load_neuron(SPATIAL_DATA_DIR / 'simple.swc')
    points = iter_positions(morph, 1, None)
    exp_points = np.array([
        [0, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 4, 0],
        [0, 5, 0],
        [-1, 5, 0],
        [-2, 5, 0],
        [-3, 5, 0],
        [-4, 5, 0],
        [-5, 5, 0],
        [1, 5, 0],
        [2, 5, 0],
        [3, 5, 0],
        [4, 5, 0],
        [5, 5, 0],
        [6, 5, 0],
        [0, -1, 0],
        [0, -2, 0],
        [0, -3, 0],
        [0, -4, 0],
        [1, -4, 0],
        [2, -4, 0],
        [3, -4, 0],
        [4, -4, 0],
        [5, -4, 0],
        [6, -4, 0],
        [-1, -4, 0],
        [-2, -4, 0],
        [-3, -4, 0],
        [-4, -4, 0],
        [-5, -4, 0]
    ])

    np.testing.assert_almost_equal(exp_points, np.array(list(points)))


def test_sample_morph_voxel_values():
    voxeldata = VoxelData(
        np.array([[[0], [1]], [[2], [3]]]),
        (25, 25, 25))
    morph_path = SPATIAL_DATA_DIR / 'sample_morph_points.asc'
    morph = nm.load_neuron(morph_path)
    single_cell_values = sample_morph_voxel_values(
        morph, 10, voxeldata, out_of_bounds_value=-1)
    np.testing.assert_array_equal(
        single_cell_values[nm.NeuriteType.axon], np.array([-1]))
    np.testing.assert_array_equal(
        single_cell_values[nm.NeuriteType.basal_dendrite],
        np.array([0, 0, -1, -1, 1, 1, 1, 0]))


def test_count_values_in_bins_default_uses_unique():
    values = np.array([1, 2, 2.3, 1, 2.3])
    counts = _count_values_in_bins(values)
    assert counts == Counter({1.0: 2, 2.0: 1, 2.3: 2})

def test_count_values_in_bins_counter_with_bin_centers():
    values = np.array([1, 1.9, 3, 4, 5, 6])
    counts = _count_values_in_bins(
        values, bin_edges=[0, 2, 2.5, 4, 5, 6])
    assert counts == Counter(
        {1.0: 2, 2.25: 0, 3.25: 1, 4.5: 1, 5.5:2})

def test_cells_neurites_distribution():
    arr = np.zeros((4, 10, 4))
    for i in range(arr.shape[1]):
        arr[:, i, :] = i
    test_voxeldata = VoxelData(arr, (100, 100, 100))
    circuit_config = {
            "cells": str(SPATIAL_DATA_DIR / 'circuit.mvd3'),
            "morphologies": str(SPATIAL_DATA_DIR)}
    circuit = Circuit(circuit_config)
    counts = count_cells_points_distribution(
        circuit, [1, 2], test_voxeldata)
    assert counts[nm.NeuriteType.axon] == Counter(
        {0: 1, 2: 1})
    assert counts[nm.NeuriteType.basal_dendrite] == Counter(
        {1: 14, 2: 1})


def test_count_circuit_points_distribution():
    expected_df = pd.DataFrame([
        ['L2_X', 'r1', nm.NeuriteType.axon, 2, 1],
        ['L2_X', 'r1', nm.NeuriteType.basal_dendrite, 2, 8],
        ['L6_Y', 'r2', nm.NeuriteType.axon, -1, 1],
        ['L6_Y', 'r2', nm.NeuriteType.axon, 3, 1],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, -1, 8],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, 2, 6],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, 3, 1],
    ], columns=['mtype', 'soma_region', 'neurite', 'voxel_value', 'count'])
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
        points_df = count_circuit_points_distribution(circuit_config)
        expected_df.set_index(['mtype', 'soma_region', 'neurite', 'voxel_value'], inplace=True)
        testing.assert_frame_equal(expected_df, points_df)


def test_count_circuit_points_distribution_other_voxeldata():
    expected_df = pd.DataFrame([
        ['L2_X', 'r1', nm.NeuriteType.axon, 0.0, 1],
        ['L2_X', 'r1', nm.NeuriteType.basal_dendrite, 1.0, 8],
        ['L6_Y', 'r2', nm.NeuriteType.axon, 2.0, 1],
        ['L6_Y', 'r2', nm.NeuriteType.axon, 9.0, 1],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, 1.0, 6],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, 2.0, 1],
        ['L6_Y', 'r2', nm.NeuriteType.basal_dendrite, 9.0, 8],
    ], columns=['mtype', 'soma_region', 'neurite', 'voxel_value', 'count'])

    arr = np.zeros((4, 10, 4))
    for i in range(arr.shape[1]):
        arr[:, i, :] = i
    test_voxeldata = VoxelData(arr, (100, 100, 100))
    circuit_config = {
        "cells": str(SPATIAL_DATA_DIR / 'circuit.mvd3'),
        "morphologies": str(SPATIAL_DATA_DIR)}
    points_df = count_circuit_points_distribution(
        circuit_config, voxeldata=test_voxeldata)
    expected_df.set_index(['mtype', 'soma_region', 'neurite', 'voxel_value'], inplace=True)
    print(points_df)
    testing.assert_frame_equal(expected_df, points_df)


def test_relative_depth_volume():
    atlas_region = np.array(
        [[[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1],
          [1, 1, 1],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]], dtype=np.bool)

    atlas_phy = np.array(
        [[[np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan],
          [6, 6, 6],
          [5, 5, 5],
          [4, 4, 4],
          [3, 3, 3],
          [2, 2, 2],
          [1, 1, 1],
          [np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan],
          [np.nan, np.nan, np.nan]]])

    atlas_ph1 = np.zeros(atlas_phy.shape + (2, ))
    atlas_ph1[..., 1] = 6
    atlas_ph1[atlas_region == 0] = np.nan

    expected_atlas_reldepth = np.array(
        [[[-3/6, -3/6, -3/6],
          [-2/6, -2/6, -2/6],
          [-1/6, -1/6, -1/6],
          [0, 0, 0],
          [1/6, 1/6, 1/6],
          [2/6, 2/6, 2/6],
          [3/6, 3/6, 3/6],
          [4/6, 4/6, 4/6],
          [5/6, 5/6, 5/6],
          [7/6, 7/6, 7/6],
          [8/6, 8/6, 8/6],
          [9/6, 9/6, 9/6]]])

    mockatlas = Mock()

    def mockdata(thing):
        return {
            '[PH]y': VoxelData(atlas_phy, (1, 1, 1)),
            '[PH]1': VoxelData(atlas_ph1, (1, 1, 1))}[
                thing]

    mockatlas.load_data = mockdata
    mockatlas.get_region_mask = lambda acro:\
        VoxelData(atlas_region, (1, 1, 1))

    np.testing.assert_array_almost_equal(
        relative_depth_volume(
            mockatlas, top_layer=1, bottom_layer=1, in_region='blabla').raw,
        expected_atlas_reldepth)
