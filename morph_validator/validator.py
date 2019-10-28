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

    def __init__(self, distr: pd.DataFrame, ci: float = 95):
        """Builds stats from distribution according confidence interval.

        Args:
            distr: dataframe of values distribution to take stats from
            ci: confidence interval in percents
        """
        assert 0. < ci < 100.
        self.median = distr.applymap(lambda x: np.median(x) if len(x) > 0 else np.nan)
        self.mean = distr.applymap(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        self.std = distr.applymap(lambda x: np.std(x) if len(x) > 0 else np.nan)
        # self.percentile_low = distr.applymap(lambda x: np.percentile(x, (100 - ci) / 2.))
        # self.percentile_up = distr.applymap(lambda x: np.percentile(x, (100 - ci) / 2 + ci))


DistrAndStats = namedtuple('DistrAndStats', 'distr stats')


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
        val = val.tolist()
        if len(val) == 1 and val[0] == 0:
            val = []
        df.loc[neurite.name, feature_name] = val
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


def get_discrete_distr_stats(discrete_features: pd.DataFrame) -> DistrAndStats:
    """Distribution and statistics of discrete features.

    Args:
        discrete_features: discrete features

    Returns:
        Distribution and statistics of discrete features.
    """
    discrete_features = discrete_features.applymap(lambda x: [x])
    discrete_distr = get_features_distrs(discrete_features)
    return DistrAndStats(discrete_distr, Stats(discrete_distr))


def get_continuous_distr_stats(continuous_features: pd.DataFrame) -> DistrAndStats:
    """Distribution and statistics of continuous features.

    Args:
        continuous_features: continuous features

    Returns:
        Distribution and statistics of continuous features.
    """
    continuous_distr = get_features_distrs(continuous_features)
    continuous_ks = _get_ks_among_features(continuous_features)
    continuous_ks_distr = (continuous_ks
                           .applymap(lambda x: [x] if not np.isnan(x) else [])
                           .groupby(['mtype', 'neurite'])
                           .agg('sum'))
    return DistrAndStats(continuous_distr, Stats(continuous_ks_distr))


def _reorder_zscores(zscores):
    return (zscores
            .reorder_levels(FEATURES_INDEX)
            .sort_index()
            .dropna(how='all'))


def get_discrete_zscores(discrete_distr_stats: DistrAndStats,
                         discrete_features: pd.DataFrame) -> pd.DataFrame:
    """Z scores of discrete features against discrete distributions and statistics.

    Args:
        discrete_distr_stats: discrete distributions and its statistics
        discrete_features: discrete features

    Returns:
        Z scores of discrete features against discrete distributions and statistics
    """
    _, _stats = discrete_distr_stats
    return _reorder_zscores((discrete_features - _stats.mean) / _stats.std)


def get_continuous_zscores(continuous_distr_stats: DistrAndStats,
                           continuous_features: pd.DataFrame) -> pd.DataFrame:
    """Z scores of continuous features against continuous distributions and statistics.

    Args:
        continuous_distr_stats: continuous distributions and its statistics
        continuous_features: continuous features

    Returns:
        Z scores of continuous features against continuous distributions and statistics
    """
    distr, _stats = continuous_distr_stats
    continuous_ks = _get_ks_features_to_distr(continuous_features, distr)
    return _reorder_zscores((continuous_ks - _stats.mean) / _stats.std)


def failed_features(zscores: pd.DataFrame, p_value=0.05) -> List[pd.DataFrame]:
    """dataframe of failed features for each tested morphology.

    Args:
        zscores: features Z scores
        p_value:

    Returns:
        list of dataframes of failed features
    """
    assert 0. <= p_value <= 1.
    threshold = np.abs(stats.norm.ppf(p_value / 2.))
    # some cells in z_score are NaN so we use `> threshold` + `any`
    # instead of `<= threshold` + `all`.
    return [grp[1].loc[:, (grp[1].abs() > threshold).any(axis=0)]
            for grp in zscores.groupby(['mtype', 'filename'])]


def validate(valid_dir: Path, test_dir: Path) -> List[pd.DataFrame]:
    """Validates directory of test morphologies against directory of valid morphologies.

    Args:
        valid_dir: directory of valid morphologies files
        test_dir: directory of test morphologies files

    Returns:
        list of failed features for each test file
    """
    valid_files_per_mtype = get_valid_files_per_mtype(valid_dir)
    valid_discrete_features, valid_continuous_features = collect_features(valid_files_per_mtype)
    valid_discrete_distr_stats = get_discrete_distr_stats(valid_discrete_features)
    valid_continuous_distr_stats = get_continuous_distr_stats(valid_continuous_features)
    test_files_per_mtype = get_test_files_per_mtype(test_dir)
    test_discrete_features, test_continuous_features = collect_features(test_files_per_mtype)
    discrete_zscores = get_discrete_zscores(valid_discrete_distr_stats, test_discrete_features)
    continuous_zscores = get_continuous_zscores(
        valid_continuous_distr_stats, test_continuous_features)
    zscores = pd.concat([discrete_zscores, continuous_zscores], axis=1, sort=True)
    return failed_features(zscores)
