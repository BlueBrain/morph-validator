"""Test `validator` module"""
import neurom as nm
import numpy as np
from neurom import NeuriteType

from morph_validator import features

from tests.utils import MORPHOLOGIES_DIR


def test_get_soma_feature():
    neuron = nm.load_morphology(MORPHOLOGIES_DIR / 'test' / 'Unknown' / 'ca3b-N2.CNG.swc')
    for neurite in features.NeuriteType:
        area = features._get_soma_feature('soma_surface_area', neuron, neurite)
        if neurite == NeuriteType.soma:
            assert np.allclose(area, np.array([370.9672678]), 1e-10, 1e-10)
        else:
            assert area.size == 0


def test_collect_features():
    valid_dir = MORPHOLOGIES_DIR / 'valid' / 'mini'
    filenames_per_mtype = {
        'L5_MC': ['C040426', 'C040601'],
        'L23_BTC': ['rat_20160906_E1_LH5_cell2'],
    }
    files_per_mtype = {mtype: [valid_dir.joinpath(filename + '.h5') for filename in filenames]
                       for mtype, filenames in filenames_per_mtype.items()}

    discrete_features, continuous_features = features.collect(files_per_mtype)
    assert set(discrete_features.columns) - set(features.DISCRETE_FEATURES) == set()
    assert set(continuous_features.columns) - set(features.CONTINUOUS_FEATURES) == set()
    expected_index = {(mtype, filename + '.h5', neurite.name)
                      for mtype, filenames in filenames_per_mtype.items()
                      for filename in filenames
                      for neurite in NeuriteType}
    assert set(discrete_features.index) == expected_index
    assert set(continuous_features.index) == expected_index
    for col in continuous_features.to_numpy():
        for cell in col:
            assert isinstance(cell, list)
    assert np.issubdtype(discrete_features.to_numpy().dtype, np.number)
