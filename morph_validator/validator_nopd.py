"""
Don't use pandas DataFrame
"""
import itertools
import logging
from collections import defaultdict, OrderedDict
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


class DISCRETE_FEATURE(Enum):
    TOTAL_LEN = 'total_length'
    NEURITE_AREA = 'total_area_per_neurite'
    # SOMA_AREA = 'soma_surface_areas'
    NEURITE_VOLUME = 'neurite_volumes'
    # SOMA_VOLUME = 'soma_volumes'
    NUMBER_OF_SECTIONS = 'number_of_sections'
    NUMBER_OF_BIFURCATIONS = 'number_of_bifurcations'
    NUMBER_OF_TERMINATIONS = 'number_of_terminations'


class CONTINUOUS_FEATURE(Enum):
    SECTION_LEN = 'section_lengths'
    SECTION_RADIAL_DISTANCES = 'section_radial_distances'
    SECTION_PATH_DISTANCES = 'section_path_distances'
    PARTITION_ASYMMETRY = 'partition_asymmetry'
    SEGMENT_RADII = 'segment_radii'


NEURITE_NAMES = [type.name for type in NeuriteType]
DISCRETE_FEATURE_NAMES = [feature.value for feature in DISCRETE_FEATURE]
CONTINUOUS_FEATURE_NAMES = [feature.value for feature in CONTINUOUS_FEATURE]


def is_discrete_feature(name):
    return name in DISCRETE_FEATURE_NAMES


class Feature:
    def __init__(self, name: Enum, neuron):
        self.name = name
        self.per_neurite_values = OrderedDict()
        for neurite in NEURITE_NAMES:
            val = nm.get(name.value, neuron, neurite_type=getattr(NeuriteType, neurite))
            if is_discrete_feature(name):
                val = val[0]
            self.per_neurite_values[neurite] = val


class Mfile:
    def __init__(self, path: Path, mtype: str):
        self.name = path.stem
        self.mtype = mtype
        self.features = OrderedDict()
        neuron = nm.load_neuron(str(path))
        for name in DISCRETE_FEATURE:
            self.features[name] = Feature(name, neuron)
        for name in CONTINUOUS_FEATURE:
            self.features[name] = Feature(name, neuron)


class Mtype:
    def __init__(self, name: str):
        self.name = name
        self.mfiles = []

    def add_file(self, mfile: Mfile):
        self.mfiles.append(mfile)


class FeatureDistr:
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.per_neurite_values = OrderedDict()
        for neurite in NEURITE_NAMES:
            self.per_neurite_values[neurite] = []

    def add_feature(self, feature: Feature):
        for name, value in feature.per_neurite_values.items():
            if value:
                self.per_neurite_values[name].append(value)


class Stats:
    def __init__(self, distr):
        self.median = np.median(distr)
        self.mean = np.mean(distr)
        self.std = np.std(distr)
        self.percentile5 = np.percentile(distr, 5)
        self.percentile95 = np.percentile(distr, 95)


class ContinuousStats:
    def __init__(self, distr):
        ks_list = []
        for i in range(0, len(distr)):
            if distr[i]:
                others = distr[:i] + distr[i + 1:]
                others = np.concatenate(others)
                if others.size:
                    ks_tuple = stats.ks_2samp(distr[i], others.tolist()) + (len(distr[i]),)
                    ks_list.append(ks_tuple)
        self.distance = Stats([ks_tuple[0] for ks_tuple in ks_list])
        self.pvalue = Stats([ks_tuple[1] for ks_tuple in ks_list])
        self.sample_size = Stats([ks_tuple[2] for ks_tuple in ks_list])


class DiscreteFeatureStats:
    def __init__(self, feature_distr: FeatureDistr):
        self.feature_name = feature_distr.feature_name
        self.per_neurite_values = OrderedDict()
        for neurite, feature_distr in feature_distr.per_neurite_values.items():
            self.per_neurite_values[neurite] = Stats(feature_distr)

    def zscore(self, feature):
        for value in feature.per_neurite_values.values():




class ContinuousFeatureStats:
    def __init__(self, feature_distr: FeatureDistr):
        self.feature_name = feature_distr.feature_name
        self.per_neurite_values = OrderedDict()
        for neurite, feature_distr in feature_distr.per_neurite_values.items():
            self.per_neurite_values[neurite] = ContinuousStats(feature_distr)

    def zscore(self, feature, feature_distr):
        # TODO not `feature` but its ks_2samp with `feature_distr`
        pass


class MtypeDistr:
    def __init__(self, mtype: Mtype):
        self.feature_distrs = OrderedDict()
        self.feature_stats = OrderedDict()

        for mfile in mtype.mfiles:
            for feature in mfile.features.values():
                if feature.name not in self.feature_distrs:
                    self.feature_distrs[feature.name] = FeatureDistr(feature.name)
                self.feature_distrs[feature.name].add_feature(feature)

        for distr in self.feature_distrs.values():
            if is_discrete_feature(distr.feature_name):
                stats = DiscreteFeatureStats(distr)
            else:
                stats = ContinuousFeatureStats(distr)
            self.feature_stats[distr.feature_name] = stats

    def zscore(self, test_mfile: Mfile) -> (pd.DataFrame, pd.DataFrame):
        discrete_data = []
        continuous_data = []
        for feature in test_mfile.features.values():
            feature_stats = self.feature_stats[feature.name]
            if is_discrete_feature(feature.name):
                discrete_data.append(feature_stats.zscore(feature))
            else:
                feature_distr = self.feature_distrs[feature.name]
                continuous_data.append(feature_stats.zscore(feature, feature_distr))
        discrete_zscore = pd.DataFrame(
            discrete_data, columns=NEURITE_NAMES, index=DISCRETE_FEATURE_NAMES)
        # TODO index include `distance`, `pvalue`, `sample_size`
        continuous_zscore = pd.DataFrame(
            continuous_data, columns=NEURITE_NAMES, index=CONTINUOUS_FEATURE_NAMES)
        return discrete_zscore, continuous_zscore


class Validator:
    def __init__(self, mtype_dict):
        self._mtype_distr_dict = {}
        for mtype in mtype_dict.values():
            self._mtype_distr_dict[mtype.name] = MtypeDistr(mtype)

    def validate(self, test_mfile: Mfile, pvalue):
        assert 0. <= pvalue <= 1.
        mtype_distr = self._mtype_distr_dict[test_mfile.mtype]
        discrete_zscore, continuous_zscore = mtype_distr.zscore(test_mfile)
        threshold = np.abs(stats.norm.ppf(pvalue / 2.))
        # some cells in z_score are NaN so we use `failed_neurites` + `any`
        # instead of `valid_neurites` + `all`.
        failed_neurites = (discrete_zscore.abs() > threshold).any(axis=1)
        return (~failed_neurites).all()


def get_mtype_dict(valid_path: Path) -> dict:
    db_file = valid_path.joinpath('neuronDB.xml')
    root = etree.parse(str(db_file)).getroot()
    mtype_dict = {}
    for morphology in root.iterfind('.//morphology'):
        filename = morphology.findtext('name')
        if not filename:
            L.warning('Empty morphology name in %s', db_file)
        mtype = morphology.findtext('mtype')
        if not mtype:
            L.warning('Empty morphology mtype in %s', db_file)
        mfile_path = valid_path.joinpath(filename + '.h5')
        if mfile_path.exists():
            if mtype not in mtype_dict:
                mtype_dict[mtype] = Mtype(mtype)
            mtype_dict[mtype].add_file(Mfile(mfile_path, mtype))
    return mtype_dict


if __name__ == '__main__':
    valid_path = Path('../tests/data/morphologies/valid/mini')
    test_path = Path('../tests/data/morphologies/test')
    mtype_dict = get_mtype_dict(valid_path)
    validator = Validator(mtype_dict)

    for mtype_dir in test_path.iterdir():
        mtype = mtype_dir.name
        for file in mtype_dir.iterdir():
            if file.suffix in MORPH_FILETYPES:
                validator.validate(Mfile(file, mtype), 0.05)
