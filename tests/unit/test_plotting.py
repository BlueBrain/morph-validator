"""test `plotting` module"""
import numpy as np
import neurom as nm

from morph_validator.spatial import _sample_morph_points

from tests.utils import TEST_DATA_DIR


def test_sample_morph_points():
    morph_path = TEST_DATA_DIR / 'sample_morph_points.asc'
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
