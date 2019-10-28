"""
API for single morphology validation.
"""

import itertools
import logging
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

import neurom as nm
import numpy as np
import pandas as pd
from lxml import etree
from neurom import NeuriteType
from neurom.fst import FstNeuron
from scipy import stats

L = logging.getLogger(__name__)
pd.options.display.width = 0
MORPH_FILETYPES = ['.h5', '.swc', '.asc']
KS_INDEX = ['distance', 'p', 'sample_size']
FEATURES_INDEX = ['mtype', 'filename', 'neurite']
DISCRETE_FEATURES = [
    'total_length',
    'total_area_per_neurite',
    'soma_surface_areas',
    'neurite_volumes',
    'number_of_sections',
    'number_of_bifurcations',
    'number_of_terminations',
]
CONTINUOUS_FEATURES = [
    'section_lengths',
    'section_radial_distances',
    'section_path_distances',
    'partition_asymmetry',
    'segment_radii',
]


def _get_soma_area(neuron: FstNeuron, neurite: NeuriteType) -> np.array:
    """Gets soma area

    Args:
        neuron: neuron object from neurom
        neurite: neurite of neuron
    Returns:
        Soma area value of neuron
    """
    if neurite == NeuriteType.soma:
        return nm.get('soma_surface_areas', neuron)
    return np.empty(0)


_FEATURE_CUSTOM_GETTERS = {
    'soma_surface_areas': _get_soma_area,
}


class Stats:
    """Statistics holder of distribution"""

    def __init__(self, distr: pd.DataFrame, ci: float):
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


def _get_neuron_features(neuron: FstNeuron, feature_names: List[str]) -> pd.DataFrame:
    """Get features of neuron as a dataframe.

    Args:
        neuron: neuron object from neurom
        feature_names: list of feature names

    Returns:
        features of neuron with INDEX as index and feature names as columns
    """
    index = [neurite.name for neurite in NeuriteType]
    df = pd.DataFrame(index=index, columns=feature_names)
    for neurite, feature_name in itertools.product(NeuriteType, feature_names):
        if feature_name in _FEATURE_CUSTOM_GETTERS:
            val = _FEATURE_CUSTOM_GETTERS[feature_name](neuron, neurite)
        else:
            val = nm.get(feature_name, neuron, neurite_type=neurite)
        df.loc[neurite.name, feature_name] = val.tolist()
    return df


def get_valid_files_per_mtype(valid_dir: Path) -> Dict[str, List[Path]]:
    """Gets valid morphologies files from a target directory.

    Args:
        valid_dir: directory with valid morphologies files

    Returns:
        dictionary of files per mtype
    """
    db_file = valid_dir.joinpath('neuronDB.xml')
    if not valid_dir.is_dir() or not db_file.exists():
        raise ValueError(
            '"{}" must be a directory with morphology files and "neuronDB.xml"'.format(valid_dir))
    root = etree.parse(str(db_file)).getroot()
    files_dict = defaultdict(list)
    for morphology in root.iterfind('.//morphology'):
        name = morphology.findtext('name')
        if not name:
            L.warning('Empty morphology name in %s', db_file)
        mtype = morphology.findtext('mtype')
        if not mtype:
            L.warning('Empty morphology mtype in %s', db_file)
        file = valid_dir.joinpath(name + '.h5')
        if file.exists() and file not in files_dict[mtype]:
            files_dict[mtype].append(file)
    return dict(files_dict)


def get_test_files_per_mtype(test_dir: Path) -> Dict[str, List[Path]]:
    """Gets test morphologies files from a target directory.

    Args:
        test_dir: directory with directories of test morphologies per mtype

    Returns:
        dictionary of files per mtype
    """
    if not test_dir.is_dir():
        raise ValueError('"{}" must be a directory'.format(test_dir))
    files_dict = defaultdict(list)
    for mtype_dir in test_dir.iterdir():
        mtype = mtype_dir.name
        for file in mtype_dir.iterdir():
            if file.suffix.lower() in MORPH_FILETYPES:
                files_dict[mtype].append(file)
    return dict(files_dict)


def collect_features(files_per_mtype: Dict[str, List[Path]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collects features of all files. Returns them as two dataframes. One for discrete,
    another for continuous features.

    Args:
        files_per_mtype: collection of files per mtype

    Returns:
        a dataframe of all discrete features for all files,
        a dataframe of all continuous features for all files.
    """
    index, discrete, continuous = [], [], []
    for mtype, files in files_per_mtype.items():
        for file in files:
            neuron = nm.load_neuron(str(file))
            index.append((mtype, neuron.name))
            discrete.append(_get_neuron_features(neuron, DISCRETE_FEATURES))
            continuous.append(_get_neuron_features(neuron, CONTINUOUS_FEATURES))
    discrete = pd.concat(discrete, keys=index, names=FEATURES_INDEX).applymap(np.sum)
    continuous = pd.concat(continuous, keys=index, names=FEATURES_INDEX)
    return discrete, continuous


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


def _get_ks_among_features(features: pd.DataFrame) -> pd.DataFrame:
    """Calculates `scipy.stats.ks_2samp` among each Index row and the rest of Index for each
    feature.

    Args:
        features: dataframe of all features of all files

    Returns:
        dataframe of `ks_2samp` stats.
    """

    def get_ks_per_feature(feature: pd.Series) -> pd.DataFrame:
        def flat_concat(l1, l2):
            return list(itertools.chain(*l1, *l2))

        fs_list = feature.to_list()
        data = [_ks_2samp(fs_list[i], flat_concat(fs_list[:i], fs_list[i + 1:]))
                for i in range(0, len(fs_list))]
        return pd.DataFrame(data, index=feature.index, columns=KS_INDEX)

    def get_ks_per_neurite(mtype_df):
        ks_features = []
        for feature in mtype_df:
            ks_features.append(get_ks_per_feature(mtype_df[feature]))
        return pd.concat(ks_features, axis=1, keys=mtype_df.columns)

    return features.groupby(['mtype', 'neurite']).apply(get_ks_per_neurite)


def _get_ks_features_to_distr(features: pd.DataFrame, distr: pd.DataFrame) -> pd.DataFrame:
    """Calculates `scipy.stats.ks_2samp` between feature values and its expected distribution
    of values.

    Args:
        features: dataframe of all features values of all files
        distr: dataframe of all features distributions of all files

    Returns:
        dataframe of `ks_2samp` stats.
    """

    def get_ks_per_feature(feature: pd.Series) -> pd.DataFrame:
        mtype = feature.index.get_level_values('mtype').unique()[0]
        mtype_series = distr.loc[mtype][feature.name]
        data = [_ks_2samp(file_, mtype) for file_, mtype in zip(feature, mtype_series)]
        return pd.DataFrame(data, index=feature.index, columns=KS_INDEX)

    def get_ks_per_file(file_df: pd.DataFrame) -> pd.DataFrame:
        ks_features = []
        for feature in file_df:
            ks_features.append(get_ks_per_feature(file_df[feature]))
        return pd.concat(ks_features, axis=1, keys=file_df.columns)

    distr_mtypes = distr.index.get_level_values('mtype').unique()
    distr_features = features.groupby('mtype').filter(
        lambda x: x.index.get_level_values('mtype').unique()[0] in distr_mtypes)
    return distr_features.groupby(['mtype', 'filename']).apply(get_ks_per_file)


def get_features_distrs(features: pd.DataFrame) -> pd.DataFrame:
    """Distribution of features grouped by `mtype` and `neurite`.

    Args:
        features: features dataframe

    Returns:
        Distribution of features grouped by `mtype` and `neurite`
    """
    return features.groupby(['mtype', 'neurite']).agg('sum')


def get_discrete_distr(discrete_features: pd.DataFrame) -> pd.DataFrame:
    """Distribution of discrete features.

    Args:
        discrete_features: discrete features

    Returns:
        Distribution of discrete features.
    """
    return get_features_distrs(discrete_features.applymap(lambda x: [x]))


def get_continuous_distr(continuous_features: pd.DataFrame) -> ContinuousDistr:
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


def get_discrete_scores(discrete_stats: Stats,
                        discrete_features: pd.DataFrame) -> pd.DataFrame:
    """Scores of discrete features against discrete statistics. Scores can be negative.

    Args:
        discrete_stats: discrete statistics
        discrete_features: discrete features

    Returns:
        Dataframe of scores.
    """
    return _reorder_scores((discrete_features - discrete_stats.median) / discrete_stats.iqr)


def get_continuous_scores(continuous_distr: pd.DataFrame, continuous_stats: Stats,
                          continuous_features: pd.DataFrame) -> pd.DataFrame:
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


def validate(valid_dir: Path, test_dir: Path, ci: float = 95) -> pd.DataFrame:
    """Validates directory of test morphologies against directory of valid morphologies.

    Args:
        valid_dir: directory of valid morphologies files
        test_dir: directory of test morphologies files
        ci: confidence interval range in percents

    Returns:
        Dataframe of features scores for each test file. Each score is:
        (feature value - feature median) / (feature interquartile range for ci).
    """
    valid_files_per_mtype = get_valid_files_per_mtype(valid_dir)
    valid_discrete_features, valid_continuous_features = collect_features(valid_files_per_mtype)
    valid_discrete_distr = get_discrete_distr(valid_discrete_features)
    valid_discrete_stats = Stats(valid_discrete_distr, ci)
    valid_continuous_distr = get_continuous_distr(valid_continuous_features)
    valid_continuous_stats = Stats(valid_continuous_distr.ks_distr, ci)
    test_files_per_mtype = get_test_files_per_mtype(test_dir)
    test_discrete_features, test_continuous_features = collect_features(test_files_per_mtype)
    discrete_scores = get_discrete_scores(valid_discrete_stats, test_discrete_features)
    continuous_scores = get_continuous_scores(
        valid_continuous_distr.distr, valid_continuous_stats, test_continuous_features)
    return pd.concat([discrete_scores, continuous_scores], axis=1, sort=True)


def failed_scores(scores: pd.DataFrame, threshold: float = 0.5) -> List[pd.DataFrame]:
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
    return [grp[1].loc[:, (grp[1].abs() > threshold).any(axis=0)]
            for grp in scores.groupby(['mtype', 'filename'])]
