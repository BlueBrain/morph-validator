"""
use pandas.DataFrame
"""
import itertools
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path

import neurom as nm
import numpy as np
import pandas as pd
from lxml import etree
from neurom import NeuriteType
from scipy import stats

L = logging.getLogger(__name__)
pd.options.display.width = 0

MORPH_FILETYPES = ['.h5', '.swc', '.asc']
DISCRETE_FEATURES_NAMES = [
    'total_length',
    'total_area_per_neurite',
    'soma_surface_areas',
    'neurite_volumes',
    'number_of_sections',
    'number_of_bifurcations',
    'number_of_terminations',
]
CONTINUOUS_FEATURES_NAMES = [
    'section_lengths',
    'section_radial_distances',
    'section_path_distances',
    'partition_asymmetry',
    'segment_radii',
]
NEURITES = [neurite for neurite in NeuriteType]


class INDEX(Enum):
    MTYPE = 'mtype'
    FILENAME = 'filename'
    NEURITE = 'neurite'

    @classmethod
    def values(cls):
        return [name.value for name in cls]


class Container(object):
    def __init__(self, discrete=None, continuous=None):
        self.discrete = discrete
        self.continuous = continuous


class Stats(object):
    def __init__(self, distr, ci=95):
        assert 0. < ci < 100.
        self.median = distr.applymap(lambda x: np.median(x))
        self.mean = distr.applymap(lambda x: np.mean(x))
        self.std = distr.applymap(lambda x: np.std(x))
        # self.percentile_low = distr.applymap(lambda x: np.percentile(x, (100 - ci) / 2.))
        # self.percentile_up = distr.applymap(lambda x: np.percentile(x, (100 - ci) / 2 + ci))


def get_neuron_features(neuron, feature_names) -> pd.DataFrame:
    index = [neurite.name for neurite in NEURITES]
    df = pd.DataFrame(index=index, columns=feature_names)
    for neurite, feature_name in itertools.product(NEURITES, feature_names):
        try:
            val = nm.get(feature_name, neuron, neurite_type=neurite).tolist()
        except AssertionError as err:
            # None values for features without support of `neurite`
            if 'Neurite type' in err.args[0]:
                val = []
            else:
                raise
        if len(val) == 1 and val[0] == 0:
            val = []
        df.loc[neurite.name, feature_name] = val.tolist()
    return df


def get_valid_files_per_mtype(valid_dirpath: Path) -> dict:
    db_file = valid_dirpath.joinpath('neuronDB.xml')
    if not valid_dirpath.is_dir() or not db_file.exists():
        raise ValueError(
            '"{}" must be a directory with morphology files and "neuronDB.xml"'
            .format(valid_dirpath))
    root = etree.parse(str(db_file)).getroot()
    files_dict = defaultdict(list)
    for morphology in root.iterfind('.//morphology'):
        name = morphology.findtext('name')
        if not name:
            L.warning('Empty morphology name in %s', db_file)
        mtype = morphology.findtext('mtype')
        if not mtype:
            L.warning('Empty morphology mtype in %s', db_file)
        file = valid_dirpath.joinpath(name + '.h5')
        if file.exists():
            files_dict[mtype].append(file)
    return files_dict


def get_test_files_per_mtype(test_dirpath: Path) -> dict:
    if not test_dirpath.is_dir():
        raise ValueError(
            '"{}" must be a directory'.format(test_dirpath))
    files_dict = defaultdict(list)
    for mtype_dir in test_dirpath.iterdir():
        mtype = mtype_dir.name
        for file in mtype_dir.iterdir():
            if file.suffix.lower() in MORPH_FILETYPES:
                files_dict[mtype].append(file)
    return files_dict


def collect_features(files_per_mtype):
    index, discrete, continuous = [], [], []
    for mtype, files in files_per_mtype.items():
        for file in files:
            neuron = nm.load_neuron(str(file))
            index.append((mtype, neuron.name))
            discrete.append(get_neuron_features(neuron, DISCRETE_FEATURES_NAMES))
            continuous.append(get_neuron_features(neuron, CONTINUOUS_FEATURES_NAMES))
    return Container(
        pd.concat(discrete, keys=index, names=INDEX.values()),
        pd.concat(continuous, keys=index, names=INDEX.values()))


def ks_2samp(a, b):
    return stats.ks_2samp(a, b) + (len(a),)


def expand_ks_tuples(ks_tuples, ks_columns):
    """transform tuple values to their separate columns"""
    tmp_list = []
    for col in ks_columns:
        expanded_splt = ks_tuples.apply(lambda x: pd.Series(x[col]), axis=1)
        columns = pd.MultiIndex.from_product([[col], ['distance', 'p', 'sample_size']])
        expanded_splt.columns = columns
        tmp_list.append(expanded_splt)
    return pd.concat(tmp_list, axis=1)


def get_ks_among_features(features):
    def ks_valid(feature_series):
        def ks(a, b):
            b = np.concatenate(b)
            if a and b.size:
                return ks_2samp(a, b)

        fs_list = feature_series.to_list()
        return [ks(fs_list[i], fs_list[:i] + fs_list[i + 1:]) for i in range(0, len(fs_list))]

    ks_as_tuples = features \
        .groupby([INDEX.MTYPE.value, INDEX.NEURITE.value]) \
        .transform(ks_valid)
    return expand_ks_tuples(ks_as_tuples, features.columns)


def get_ks_features_to_distr(features, distr):
    def ks_test(file_series):
        mtype = file_series.index.get_level_values('mtype').unique()[0]
        if not mtype in distr.index.levels[0]:
            return None
        mtype_series = distr.loc[mtype][file_series.name]
        return [ks_2samp(fm[0], fm[1])
                if fm[0] and fm[1] else None for fm in zip(file_series, mtype_series)]

    ks_as_tuples = features \
        .groupby([INDEX.MTYPE.value, INDEX.FILENAME.value]).transform(ks_test)
    return expand_ks_tuples(ks_as_tuples, features.columns)


def get_valid_distrs(valid_features):
    def build_distrs(features):
        return features.groupby([INDEX.MTYPE.value, INDEX.NEURITE.value]).agg('sum')

    return Container(
        build_distrs(valid_features.discrete),
        build_distrs(valid_features.continuous)
    )


def get_valid_stats(valid_features, valid_distrs):
    valid_stats = Container()
    valid_stats.discrete = Stats(valid_distrs.discrete)

    continuous_ks = get_ks_among_features(valid_features.continuous)
    continuous_ks_distr = continuous_ks \
        .applymap(lambda x: [x] if not np.isnan(x) else []) \
        .groupby([INDEX.MTYPE.value, INDEX.NEURITE.value]).agg('sum')
    valid_stats.continuous = Stats(continuous_ks_distr)
    return valid_stats


def get_zscores(test_features, valid_distrs, valid_stats):
    # TODO fix mix of indices
    def reorder(df):
        return df.reorder_levels([INDEX.MTYPE.value, INDEX.FILENAME.value, INDEX.NEURITE.value])

    zscores = Container()
    discrete_stats = valid_stats.discrete
    zscores.discrete = reorder((test_features.discrete - discrete_stats.mean) / discrete_stats.std)

    continuous_stats = valid_stats.continuous
    continuous_ks = get_ks_features_to_distr(test_features.continuous, valid_distrs.continuous)
    zscores.continuous = reorder((continuous_ks - continuous_stats.mean) / continuous_stats.std)
    return zscores


def report(zscores: Container, p_value=0.05):
    assert 0. <= p_value <= 1.
    threshold = np.abs(stats.norm.ppf(p_value / 2.))
    # some cells in z_score are NaN so we use `failed_neurites` + `any`
    # instead of `valid_neurites` + `all`.
    zscore = pd.concat([zscores.discrete, zscores.continuous], axis=1, sort=True)
    failed_zscore = zscore.abs() > threshold
    failed_features = zscore.loc[:, failed_zscore.any(axis=0)]
    return (~failed_features).groupby(['filename', 'mtype']).all()


if __name__ == '__main__':
    valid_files_per_mtype = get_valid_files_per_mtype(Path('../tests/data/morphologies/valid/mini'))
    valid_features = collect_features(valid_files_per_mtype)
    valid_distrs = get_valid_distrs(valid_features)
    valid_stats = get_valid_stats(valid_features, valid_distrs)

    test_files_per_mtype = get_test_files_per_mtype(Path('../tests/data/morphologies/valid/mini'))
    test_features = collect_features(test_files_per_mtype)
    test_features.discrete = test_features.discrete.applymap(np.sum)
    zscores = get_zscores(test_features, valid_distrs, valid_stats)
