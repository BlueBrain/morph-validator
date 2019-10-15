"""
use pandas.DataFrame
"""
import itertools
import logging
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


class DISCRETE_FEATURE_NAMES(Enum):
    LEN = 'total_length'
    SURFACE_AREA = 'total_area_per_neurite'
    VOLUMES = 'neurite_volumes'
    NUMBER_OF_SECTIONS = 'number_of_sections'
    NUMBER_OF_BIFURCATIONS = 'number_of_bifurcations'
    NUMBER_OF_TERMINATIONS = 'number_of_terminations'


class CONTINUOUS_FEATURE_NAMES(Enum):
    SECTION_LEN = 'section_lengths'
    SECTION_RADIAL_DISTANCES = 'section_radial_distances'
    SECTION_PATH_DISTANCES = 'section_path_distances'
    PARTITION_ASYMMETRY = 'partition_asymmetry'
    SEGMENT_RADII = 'segment_radii'


NEURITES = (NeuriteType.soma,
            NeuriteType.axon,
            NeuriteType.basal_dendrite,
            NeuriteType.apical_dendrite,
            NeuriteType.undefined,)
NEURITE_NAMES = [type.name for type in NEURITES]

CUSTOM_FEATURE_LOAD = {
    DISCRETE_FEATURE_NAMES.SURFACE_AREA.value: {
        NeuriteType.soma.name: lambda neuron: nm.get('soma_surface_areas', neuron),
    }
}

MORPH_FILETYPES = ['.h5', '.swc', '.asc']


class Features(object):
    class INDEX(Enum):
        MTYPE = 'mtype'
        FILENAME = 'filename'
        NEURITE = 'neurite'

    _INDEX_NAMES = [index.value for index in INDEX]

    def __init__(self, index, discrete, continuous):
        self.discrete = pd.concat(discrete, keys=index, names=self._INDEX_NAMES)
        self.continuous = pd.concat(continuous, keys=index, names=self._INDEX_NAMES)


def get_discrete_features(neuron) -> pd.DataFrame:
    feature_names = [name.value for name in DISCRETE_FEATURE_NAMES]
    df = get_features(neuron, feature_names)
    df = df.applymap(np.sum)
    df.loc[NeuriteType.all.name] = df.sum()
    return df


def get_continuous_features(neuron) -> pd.DataFrame:
    feature_names = [name.value for name in CONTINUOUS_FEATURE_NAMES]
    df = get_features(neuron, feature_names)
    # `np.concatenate(x).tolist()` is used instead `np.concatenate(x)` to treat the return object
    # as a single value. Without it pandas treat it as a Series and tries to broadcast it.
    # When `np.concatenate(x).tolist()` returns a list of length equal to len(df) => may be an error
    df.loc[NeuriteType.all.name] = df.aggregate(lambda x: np.concatenate(x).tolist())
    return df


def get_features(neuron, feature_names) -> pd.DataFrame:
    df = pd.DataFrame(index=NEURITE_NAMES, columns=feature_names)
    for neurite, feature_name in itertools.product(NEURITES, feature_names):
        val = None
        if feature_name in CUSTOM_FEATURE_LOAD:
            if neurite.name in CUSTOM_FEATURE_LOAD[feature_name]:
                val = CUSTOM_FEATURE_LOAD[feature_name][neurite.name](neuron)
        if val is None:
            val = nm.get(feature_name, neuron, neurite_type=neurite)
        df.loc[neurite.name, feature_name] = val.tolist()
    return df


def get_mtype_dict(db_file: Path) -> dict:
    root = etree.parse(str(db_file)).getroot()
    mtype_dict = {}
    for morphology in root.iterfind('.//morphology'):
        name = morphology.findtext('name')
        if not name:
            L.warning('Empty morphology name in %s', db_file)
        mtype = morphology.findtext('mtype')
        if not mtype:
            L.warning('Empty morphology mtype in %s', db_file)
        if name in mtype_dict and mtype_dict[name] != mtype:
            L.warning('Multiple mtypes %s %s for %s', mtype, mtype_dict[name], name)
        mtype_dict[name] = mtype
    return mtype_dict


def get_valid_morph_features(morph_dirpath: Path) -> Features:
    if not morph_dirpath.is_dir():
        raise ValueError(
            '"{}" must be a directory with morphology files'.format(morph_dirpath))
    mtype_dict = get_mtype_dict(morph_dirpath.joinpath('neuronDB.xml'))
    index, discrete, continuous = [], [], []
    for file in morph_dirpath.iterdir():
        if file.suffix in MORPH_FILETYPES:
            neuron = nm.load_neuron(str(file))
            mtype = mtype_dict[neuron.name]
            index.append((mtype, neuron.name))
            discrete.append(get_discrete_features(neuron))
            continuous.append(get_continuous_features(neuron))
    return Features(index, discrete, continuous)


def get_test_morph_features(morph_dirpath: Path) -> Features:
    if not morph_dirpath.is_dir():
        raise ValueError(
            '"{}" must be a directory'.format(morph_dirpath))
    index, discrete, continuous = [], [], []
    for mtype_dir in morph_dirpath.iterdir():
        mtype = mtype_dir.name
        for file in mtype_dir.iterdir():
            if file.suffix in MORPH_FILETYPES:
                neuron = nm.load_neuron(str(file))
                index.append((mtype, neuron.name))
                discrete.append(get_discrete_features(neuron))
                continuous.append(get_continuous_features(neuron))
    return Features(index, discrete, continuous)


def ks_2samp(a, b):
    return stats.ks_2samp(a, b) + (len(a),)


def get_ks_of_valid_features(valid_features):
    def ks_valid(feature_series):
        def ks(a, b):
            b = np.concatenate(b)
            if a and b.size:
                return ks_2samp(a, b)

        fs_list = feature_series.to_list()
        return [ks(fs_list[i], fs_list[:i] + fs_list[i + 1:]) for i in range(0, len(fs_list))]

    return valid_features.continuous \
        .groupby([Features.INDEX.MTYPE.value, Features.INDEX.NEURITE.value]) \
        .transform(ks_valid)


def get_ks_of_test_features(test_features, valid_features):
    def ks_test(file_series):
        mtype = file_series.index.get_level_values('mtype').unique()[0]
        if not mtype in neurite_feature_distr.index.levels[0]:
            return None
        mtype_series = neurite_feature_distr.loc[mtype][file_series.name]
        return [ks_2samp(fm[0], fm[1])
                if fm[0] and fm[1] else None for fm in zip(file_series, mtype_series)]

    neurite_feature_distr = valid_features.continuous \
        .groupby([Features.INDEX.MTYPE.value, Features.INDEX.NEURITE.value]) \
        .agg(lambda feature: np.concatenate(feature).tolist())

    return test_features.continuous \
        .groupby([Features.INDEX.MTYPE.value, Features.INDEX.FILENAME.value]).transform(ks_test)


def discrete_z_score(valid_features: Features, test_features: Features):
    valid_mean = valid_features.discrete.groupby(
        [Features.INDEX.MTYPE.value, Features.INDEX.NEURITE.value]).mean()
    valid_std = valid_features.discrete.groupby(
        [Features.INDEX.MTYPE.value, Features.INDEX.NEURITE.value]).std()
    return ((test_features.discrete - valid_mean) / valid_std).dropna(how='all')


def discrete_report(z_score: pd.DataFrame, p_value=0.05):
    assert 0. <= p_value <= 1.
    threshold = np.abs(stats.norm.ppf(p_value / 2.))
    # some cells in z_score are NaN so we use `failed_neurites` + `any`
    # instead of `valid_neurites` + `all`.
    failed_neurites = (z_score.abs() > threshold).any(axis=1)
    return (~failed_neurites).groupby(['filename', 'mtype']).all()


def valid_ks_neurite_distr(valid_ks):
    def tt(mtype_df):
        print(mtype_df)
        return mtype_df

    return valid_ks.groupby([Features.INDEX.MTYPE.value, Features.INDEX.NEURITE.value]).apply(tt)


if __name__ == '__main__':
    valid_features = get_valid_morph_features(Path('../tests/data/morphologies/valid/mini'))
    test_features = get_test_morph_features(Path('../tests/data/morphologies/test'))
    # z_score = discrete_z_score(valid_features, test_features)
    # discrete_report(z_score)
    valid_ks = get_ks_of_valid_features(valid_features)
    valid_ks_neurite_distr(valid_ks)
