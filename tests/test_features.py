"""Test `validator` module"""
import neurom as nm
import numpy as np
import pandas as pd
from neurom import NeuriteType

from morph_validator import features

from tests.utils import MORPHOLOGIES_DIR


def test_get_soma_feature():
    neuron = nm.load_neuron(MORPHOLOGIES_DIR / 'test' / 'Unknown' / 'ca3b-N2.CNG.swc')
    for neurite in features.NeuriteType:
        area = features._get_soma_feature('soma_surface_areas', neuron, neurite)
        if neurite == NeuriteType.soma:
            assert np.allclose(area, np.array([370.96746105]), 1e-10, 1e-10)
        else:
            assert area.size == 0


def test_get_test_files_per_mtype():
    test_dir = MORPHOLOGIES_DIR / 'test'
    test_files_dict = features.get_test_files_per_mtype(test_dir)

    expected_filenames_dict = {
        'L23_BTC': ['C210401C'],
        'L5_MC': ['mtC171001A_idC', 'rp101020_2_INT_idA', 'vd101020A_idA'],
        'Unknown': ['ca3b-N2.CNG'],
    }
    assert set(test_files_dict.keys()) - (expected_filenames_dict.keys()) == set()
    for mtype, file_list in test_files_dict.items():
        expected_filename_list = expected_filenames_dict[mtype]
        assert len(file_list) == len(expected_filename_list)
        for file in file_list:
            assert file.stem in expected_filename_list


def test_collect_features():
    valid_dir = MORPHOLOGIES_DIR / 'valid' / 'mini'
    filenames_per_mtype = {
        'L5_MC': ['C040426', 'C040601'],
        'L23_BTC': ['rat_20160906_E1_LH5_cell2'],
    }
    files_per_mtype = {mtype: [valid_dir.joinpath(filename + '.h5') for filename in filenames]
                       for mtype, filenames in filenames_per_mtype.items()}

    discrete_features, continuous_features = features.collect_features(files_per_mtype)
    assert set(discrete_features.columns) - set(features.DISCRETE_FEATURES) == set()
    assert set(continuous_features.columns) - set(features.CONTINUOUS_FEATURES) == set()
    expected_index = {(mtype, filename, neurite.name)
                      for mtype, filenames in filenames_per_mtype.items()
                      for filename in filenames
                      for neurite in NeuriteType}
    assert set(discrete_features.index) - expected_index == set()
    assert set(continuous_features.index) - expected_index == set()
    for col in continuous_features.to_numpy():
        for cell in col:
            assert isinstance(cell, list)
    assert np.issubdtype(discrete_features.to_numpy().dtype, np.number)


def test_ks_2samp():
    assert features._ks_2samp([1], [1, 2]) == (0.5, 1, 1)
    assert features._ks_2samp([], [1, 2]) == (np.nan, np.nan, np.nan)


def test_get_ks_among_features():
    index = pd.MultiIndex.from_product(
        [['type1', 'type2'], ['filename1', 'filename2', 'filename3'], ['neurite1']],
        names=features.FEATURES_INDEX)
    columns = ['a']
    data = [
        [[1, 1]],
        [[2]],
        [[1, 1, 1]],
        [[1]],
        [[2, 2]],
        [[2, 2, 2]]]
    features_df = pd.DataFrame(data, columns=columns, index=index)

    ks_df = features._get_ks_among_features(features_df)
    expected_columns = pd.MultiIndex.from_product([columns, features.KS_INDEX])
    expected_ks_values = [
        [0.25, 1, 2],
        [1, 0.33333333, 1],
        [0.33333333, 1, 3],
        [1, 0.33333333, 1],
        [0.25, 1, 2],
        [0.33333333, 1, 3]]
    assert ks_df.columns.equals(expected_columns)
    assert np.allclose(ks_df, expected_ks_values)


def test_get_ks_features_to_distr():
    columns = ['feature1']
    features_index = pd.MultiIndex.from_product(
        [['type1', 'type2'], ['filename1', 'filename2'], ['neurite1']],
        names=features.FEATURES_INDEX)
    features_data = [
        [[1, 1]],
        [[2]],
        [[1]],
        [[2, 2]]]
    features_df = pd.DataFrame(features_data, columns=columns, index=features_index)
    distr_index = pd.MultiIndex.from_product(
        [['type1', 'type2'], ['neurite1']],
        names=['mtype', 'neurite'])
    distr_df = pd.DataFrame([[[1, 1, 1]], [[2, 2, 2]]], columns=columns, index=distr_index)
    ks_df = features._get_ks_features_to_distr(features_df, distr_df)

    expected_columns = pd.MultiIndex.from_product([columns, features.KS_INDEX])
    expected_ks_values = [
        [0., 1., 2],
        [1., 0.5, 1],
        [1., 0.5, 1],
        [0., 1., 2]]
    assert ks_df.columns.equals(expected_columns)
    assert np.allclose(ks_df, expected_ks_values)


def test_get_discrete_distr():
    columns = ['feature1']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1', 'type2'], ['filename1', 'filename2'], ['neurite1']],
        names=features.FEATURES_INDEX)
    data = [
        [1],
        [2],
        [1],
        [2]]
    morph_features = pd.DataFrame(data, columns=columns, index=morph_features_index)
    distr = features.get_discrete_distr(morph_features)

    expected_index = pd.MultiIndex.from_product([['type1', 'type2'], ['neurite1']])
    assert expected_index.equal_levels(distr.index)

    assert np.array_equal([1, 2], distr.loc['type1', 'neurite1'][0])
    assert distr.loc['type1', 'neurite1'].size == 1
    assert np.array_equal([1, 2], distr.loc['type2', 'neurite1'][0])
    assert distr.loc['type2', 'neurite1'].size == 1


def test_get_continuous_distr():
    columns = ['feature1']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1', 'type2'], ['filename1', 'filename2'], ['neurite1']],
        names=features.FEATURES_INDEX)
    data = [
        [[1, 1]],
        [[2]],
        [[1]],
        [[2, 2]]]
    morph_features = pd.DataFrame(data, columns=columns, index=morph_features_index)
    distr = features.get_continuous_distr(morph_features)

    expected_index = pd.MultiIndex.from_product([['type1', 'type2'], ['neurite1']])
    assert expected_index.equal_levels(distr.distr.index)
    assert expected_index.equal_levels(distr.ks_distr.index)

    assert np.array_equal([1, 1, 2], distr.distr.loc['type1', 'neurite1'][0])
    assert distr.distr.loc['type1', 'neurite1'].size == 1
    assert np.array_equal([1, 2, 2], distr.distr.loc['type2', 'neurite1'][0])
    assert distr.distr.loc['type2', 'neurite1'].size == 1


def test_stats():
    columns = ['feature1']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1', 'type2'], ['neurite1']],
        names=['mtype', 'neurite'])
    data = [[[1, 2, 3, 4, 5]], [[5, 6, 7, 8, 9]]]
    distr = pd.DataFrame(data, columns=columns, index=morph_features_index)
    stats = features.Stats(distr, 50)
    assert np.allclose(stats.median.to_numpy().flatten(), [3., 7.])
    assert np.allclose(stats.iqr.to_numpy().flatten(), [2., 2.])


def test_get_discrete_scores():
    columns = ['feature1', 'feature2']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1'], ['filename1', 'filename2'], ['neurite1']],
        names=features.FEATURES_INDEX)
    morph_features = pd.DataFrame(
        [[1, 3],
         [2, 4]], columns=columns, index=morph_features_index)
    distr_index = morph_features_index.droplevel('filename').drop_duplicates()
    distr = pd.DataFrame([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], columns=columns, index=distr_index)
    stats = features.Stats(distr, 50)
    scores = features.get_discrete_scores(stats, morph_features)
    assert np.allclose(scores, [[-1., 0.],
                                [-.5, .5]])


def test_get_continuous_scores():
    columns = ['feature3', 'feature4']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1'], ['filename1', 'filename2'], ['neurite1']],
        names=features.FEATURES_INDEX)
    morph_features = pd.DataFrame(
        [[[1], [1]],
         [[2, 2], [2, 2]]], columns=columns, index=morph_features_index)
    distr_index = morph_features_index.droplevel('filename').drop_duplicates()
    distr = pd.DataFrame([[[1, 2, 3], [3, 4, 5]]], columns=columns, index=distr_index)
    stats = features.Stats(pd.DataFrame(
        [[[1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1]]],
        columns=pd.MultiIndex.from_product([columns, features.KS_INDEX]),
        index=distr_index), 95)
    scores = features.get_continuous_scores(distr, stats, morph_features)
    assert np.allclose(scores,
                       [[-.35088, 0., 0., 0., -.52632, 0.],
                        [-.70175, 0., 1.05263, 0., -.84211, 1.05263]])


def test_failed_scores():
    discrete_columns = ['feature1', 'feature2']
    continuous_columns = ['feature3', 'feature4']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1'], ['filename1', 'filename2'], ['neurite1']],
        names=features.FEATURES_INDEX)
    continuous_data = [
        [0.5, 2.],
        [-.34, .16]]
    discrete_data = [
        [-1.96, 0.3],
        [.5, .6]]
    discrete_scores = pd.DataFrame(
        discrete_data, columns=discrete_columns, index=morph_features_index)
    continuous_scores = pd.DataFrame(
        continuous_data, columns=continuous_columns, index=morph_features_index)
    scores = pd.concat([discrete_scores, continuous_scores], axis=1, sort=True)
    failed_scores = features.failed_scores(scores, 1.95)
    assert len(failed_scores) == 2
    assert np.array_equal(failed_scores[0].columns, ['feature1', 'feature4'])
    assert np.allclose(failed_scores[0], [-1.96, 2.0])
    assert failed_scores[1].empty
