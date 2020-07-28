"""
Validation by Z scores of test versus valid morphologies. Z score is `standard score
<https://en.wikipedia.org/wiki/Standard_score>`__.
"""

import itertools
import logging
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats

from morph_validator.features import collect, FEATURES_INDEX
from morph_validator.utils import get_mtype_files_db, get_mtype_files_dir

L = logging.getLogger(__name__)
KS_INDEX = ['distance', 'p', 'sample_size']


class Stats:
    """Statistics holder of distribution"""

    def __init__(self, distr: DataFrame, ci: float):
        """Builds stats from distribution according confidence interval.

        Args:
            distr: dataframe of values distribution to take stats from
            ci: confidence interval range in percents
        """
        assert 0 < ci < 100
        self.median = distr.applymap(lambda x: np.median(x) if len(x) > 0 else np.nan)
        self.mean = distr.applymap(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        self.std = distr.applymap(lambda x: np.std(x) if len(x) > 0 else np.nan)
        percentile_min = distr.applymap(
            lambda x: np.percentile(x, (100 - ci) / 2.) if len(x) > 0 else np.nan)
        percentile_max = distr.applymap(
            lambda x: np.percentile(x, (100 + ci) / 2.) if len(x) > 0 else np.nan)
        # Interquartile Range
        self.iqr = (percentile_max - percentile_min).abs()


ContinuousDistr = namedtuple('ContinuousDistr', 'distr ks_distr')


def _ks_2samp(l1: List, l2: List) -> Tuple:
    """scipy.stats.ks_2samp of two lists + the size of the first list.

    Args:
        l1: first list
        l2: second list

    Returns:
        tuple of `(ks.distance, ks.pvalue, l1 size)`. (NaN, NaN, NaN) is returned if any of
        arguments is empty.
    """
    if l1 and l2:
        return stats.ks_2samp(l1, l2) + (len(l1),)
    return np.nan, np.nan, np.nan


def _get_ks_among_features(features: DataFrame) -> DataFrame:
    """Calculates `scipy.stats.ks_2samp` among each Index row and the rest of Index for each
    feature.

    Args:
        features: dataframe of all features of all files

    Returns:
        dataframe of `ks_2samp` stats.
    """

    def get_ks_per_feature(feature: Series) -> DataFrame:
        def flat_concat(l1, l2):
            return list(itertools.chain(*l1, *l2))

        fs_list = feature.to_list()
        data = [_ks_2samp(fs_list[i], flat_concat(fs_list[:i], fs_list[i + 1:]))
                for i in range(0, len(fs_list))]
        return DataFrame(data, index=feature.index, columns=KS_INDEX)

    def get_ks_per_neurite(mtype_df):
        ks_features = []
        for feature in mtype_df:
            ks_features.append(get_ks_per_feature(mtype_df[feature]))
        return pd.concat(ks_features, axis=1, keys=mtype_df.columns)

    return features.groupby(['mtype', 'neurite']).apply(get_ks_per_neurite)


def _get_ks_features_to_distr(features: DataFrame, distr: DataFrame) -> DataFrame:
    """Calculates `scipy.stats.ks_2samp` between feature values and its expected distribution
    of values.

    Args:
        features: dataframe of all features values of all files
        distr: dataframe of all features distributions of all files

    Returns:
        dataframe of `ks_2samp` stats.
    """

    def get_ks_per_feature(feature: Series) -> DataFrame:
        mtype = feature.index.get_level_values('mtype').unique()[0]
        mtype_series = distr.loc[mtype][feature.name]
        data = [_ks_2samp(file_, mtype) for file_, mtype in zip(feature, mtype_series)]
        return DataFrame(data, index=feature.index, columns=KS_INDEX)

    def get_ks_per_file(file_df: DataFrame) -> DataFrame:
        ks_features = []
        for feature in file_df:
            ks_features.append(get_ks_per_feature(file_df[feature]))
        return pd.concat(ks_features, axis=1, keys=file_df.columns)

    distr_mtypes = distr.index.get_level_values('mtype').unique()
    distr_features = features.groupby('mtype').filter(
        lambda x: x.index.get_level_values('mtype').unique()[0] in distr_mtypes)
    return distr_features.groupby(['mtype', 'filename']).apply(get_ks_per_file)


def get_features_distrs(features: DataFrame) -> DataFrame:
    """Distribution of features grouped by `mtype` and `neurite`.

    Args:
        features: features dataframe

    Returns:
        Distribution of features grouped by `mtype` and `neurite`
    """
    return features.groupby(['mtype', 'neurite']).sum()


def get_discrete_distr(discrete_features: DataFrame) -> DataFrame:
    """Distribution of discrete features.

    Args:
        discrete_features: discrete features

    Returns:
        Distribution of discrete features.
    """
    return get_features_distrs(discrete_features.applymap(lambda x: [x]))


def get_continuous_distr(continuous_features: DataFrame) -> ContinuousDistr:
    """Distribution of continuous features.

    Args:
        continuous_features: continuous features

    Returns:
        Distribution of continuous features.
    """
    continuous_distr = get_features_distrs(continuous_features)
    continuous_ks = _get_ks_among_features(continuous_features)
    continuous_ks_distr = (continuous_ks
                           .applymap(lambda x: [x] if not np.isnan(x) else [])
                           .groupby(['mtype', 'neurite'])
                           .agg('sum'))
    return ContinuousDistr(continuous_distr, continuous_ks_distr)


def _reorder_scores(scores):
    return (scores
            .reorder_levels(FEATURES_INDEX)
            .sort_index()
            .dropna(how='all'))


def get_discrete_scores(discrete_stats: Stats, discrete_features: DataFrame) -> DataFrame:
    """Scores of discrete features against discrete statistics. Scores can be negative.

    Args:
        discrete_stats: discrete statistics
        discrete_features: discrete features

    Returns:
        Dataframe of scores.
    """
    return _reorder_scores((discrete_features - discrete_stats.median) / discrete_stats.iqr)


def get_continuous_scores(continuous_distr: DataFrame, continuous_stats: Stats,
                          continuous_features: DataFrame) -> DataFrame:
    """Scores of continuous features against continuous distributions and statistics.
    Scores can be negative.

    Args:
        continuous_distr: continuous distributions
        continuous_stats: continuous statistics
        continuous_features: continuous features

    Returns:
        Dataframe of scores.
    """
    continuous_ks = _get_ks_features_to_distr(continuous_features, continuous_distr)
    return _reorder_scores((continuous_ks - continuous_stats.median) / continuous_stats.iqr)


def validate(valid_neurondb_file: Path, test_files: Path, ci: float = 95) -> DataFrame:
    """Validates directory of test morphologies against directory of valid morphologies.

    Args:
        valid_neurondb_file: file that contains mapping of valid morphology names to their mtype.
        Valid morphology files must be located in the same directory as this file.
        test_files: Either a neurondb file of test morphologies or a directory of test morphologies
            files as in ``utils.get_mtype_files_dir``
        ci: confidence interval range in percents

    Returns:
        Dataframe: dataframe of features scores for each test file. Each score is:
        (feature value - feature median) / (feature interquartile range for ci).
    """
    valid_files_per_mtype = get_mtype_files_db(valid_neurondb_file)
    valid_discrete_features, valid_continuous_features = collect(valid_files_per_mtype)
    valid_discrete_distr = get_discrete_distr(valid_discrete_features)
    valid_discrete_stats = Stats(valid_discrete_distr, ci)
    valid_continuous_distr = get_continuous_distr(valid_continuous_features)
    valid_continuous_stats = Stats(valid_continuous_distr.ks_distr, ci)
    if test_files.is_file():
        test_files_per_mtype = get_mtype_files_db(test_files)
    else:
        test_files_per_mtype = get_mtype_files_dir(test_files)
    test_discrete_features, test_continuous_features = collect(test_files_per_mtype)
    discrete_scores = get_discrete_scores(valid_discrete_stats, test_discrete_features)
    continuous_scores = get_continuous_scores(
        valid_continuous_distr.distr, valid_continuous_stats, test_continuous_features)
    return pd.concat([discrete_scores, continuous_scores], axis=1, sort=True)


def failed_scores(scores: DataFrame, threshold: float = 0.5) -> List[DataFrame]:
    """Checks what features failed.

    Args:
        scores: dataframe of features scores for each test morphology
        threshold: scores higher than this are considered as failed.

    Returns:
        List of dataframes with failed features per test morphology.
    """
    assert threshold > 0
    # some cells in z_score are NaN so we use `> threshold` + `any`
    # instead of `<= threshold` + `all`.
    return [df.loc[:, (df.abs() > threshold).any(axis=0)]
            for _, df in scores.groupby(['mtype', 'filename'])]
