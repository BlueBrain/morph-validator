"""Test `validator` module"""
import numpy as np
import pandas as pd

from morph_validator import zscores


def test_ks_2samp():
    assert zscores._ks_2samp([1], [1, 2]) == (0.5, 1, 1)
    assert zscores._ks_2samp([], [1, 2]) == (np.nan, np.nan, np.nan)


def test_get_ks_among_features():
    index = pd.MultiIndex.from_product(
        [['type1', 'type2'], ['filename1', 'filename2', 'filename3'], ['neurite1']],
        names=zscores.FEATURES_INDEX)
    columns = ['a']
    data = [
        [[1, 1]],
        [[2]],
        [[1, 1, 1]],
        [[1]],
        [[2, 2]],
        [[2, 2, 2]]]
    features_df = pd.DataFrame(data, columns=columns, index=index)

    ks_df = zscores._get_ks_among_features(features_df)
    expected_columns = pd.MultiIndex.from_product([columns, zscores.KS_INDEX])
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
        names=zscores.FEATURES_INDEX)
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
    ks_df = zscores._get_ks_features_to_distr(features_df, distr_df)

    expected_columns = pd.MultiIndex.from_product([columns, zscores.KS_INDEX])
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
        names=zscores.FEATURES_INDEX)
    data = [
        [1],
        [2],
        [1],
        [2]]
    morph_features = pd.DataFrame(data, columns=columns, index=morph_features_index)
    distr = zscores.get_discrete_distr(morph_features)

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
        names=zscores.FEATURES_INDEX)
    data = [
        [[1, 1]],
        [[2]],
        [[1]],
        [[2, 2]]]
    morph_features = pd.DataFrame(data, columns=columns, index=morph_features_index)
    distr = zscores.get_continuous_distr(morph_features)

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
    stats = zscores.Stats(distr, 50)
    assert np.allclose(stats.median.to_numpy().flatten(), [3., 7.])
    assert np.allclose(stats.iqr.to_numpy().flatten(), [2., 2.])


def test_get_discrete_scores():
    columns = ['feature1', 'feature2']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1'], ['filename1', 'filename2'], ['neurite1']],
        names=zscores.FEATURES_INDEX)
    morph_features = pd.DataFrame(
        [[1, 3],
         [2, 4]], columns=columns, index=morph_features_index)
    distr_index = morph_features_index.droplevel('filename').drop_duplicates()
    distr = pd.DataFrame([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]], columns=columns, index=distr_index)
    stats = zscores.Stats(distr, 50)
    scores = zscores.get_discrete_scores(stats, morph_features)
    assert np.allclose(scores, [[-1., 0.],
                                [-.5, .5]])


def test_get_continuous_scores():
    columns = ['feature3', 'feature4']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1'], ['filename1', 'filename2'], ['neurite1']],
        names=zscores.FEATURES_INDEX)
    morph_features = pd.DataFrame(
        [[[1], [1]],
         [[2, 2], [2, 2]]], columns=columns, index=morph_features_index)
    distr_index = morph_features_index.droplevel('filename').drop_duplicates()
    distr = pd.DataFrame([[[1, 2, 3], [3, 4, 5]]], columns=columns, index=distr_index)
    stats = zscores.Stats(pd.DataFrame(
        [[[1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1]]],
        columns=pd.MultiIndex.from_product([columns, zscores.KS_INDEX]),
        index=distr_index), 95)
    scores = zscores.get_continuous_scores(distr, stats, morph_features)
    assert np.allclose(scores,
                       [[-.35088, 0., 0., 0., -.52632, 0.],
                        [-.70175, 0., 1.05263, 0., -.84211, 1.05263]])


def test_failed_scores():
    discrete_columns = ['feature1', 'feature2']
    continuous_columns = ['feature3', 'feature4']
    morph_features_index = pd.MultiIndex.from_product(
        [['type1'], ['filename1', 'filename2'], ['neurite1']],
        names=zscores.FEATURES_INDEX)
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
    failed_scores = zscores.failed_scores(scores, 1.95)
    assert len(failed_scores) == 2
    assert np.array_equal(failed_scores[0].columns, ['feature1', 'feature4'])
    assert np.allclose(failed_scores[0], [-1.96, 2.0])
    assert failed_scores[1].empty
