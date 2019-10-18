"""
OOP instead of pandas.DataFrame
"""
import logging
from collections import OrderedDict
from enum import Enum
from pathlib import Path

import neurom as nm
import numpy as np
import pandas as pd
from lxml import etree
from neurom import NeuriteType
from scipy.stats import ks_2samp, norm

L = logging.getLogger(__name__)
pd.options.display.width = 0

MORPH_FILETYPES = ['.h5', '.swc', '.asc']


class DISCRETE_FEATURE(Enum):
    TOTAL_LEN = 'total_length'
    NEURITE_AREA = 'total_area_per_neurite'
    SOMA_AREA = 'soma_surface_areas'
    NEURITE_VOLUME = 'neurite_volumes'
    NUMBER_OF_SECTIONS = 'number_of_sections'
    NUMBER_OF_BIFURCATIONS = 'number_of_bifurcations'
    NUMBER_OF_TERMINATIONS = 'number_of_terminations'

    @classmethod
    def has_value(cls, value):
        return value in [feature.value for feature in cls]


class CONTINUOUS_FEATURE(Enum):
    SECTION_LEN = 'section_lengths'
    SECTION_RADIAL_DISTANCES = 'section_radial_distances'
    SECTION_PATH_DISTANCES = 'section_path_distances'
    PARTITION_ASYMMETRY = 'partition_asymmetry'
    SEGMENT_RADII = 'segment_radii'


NEURITE_NAMES = [type.name for type in NeuriteType]


class Feature:
    def __init__(self, feature_name: Enum, neuron):
        self.name = feature_name.value
        self.neurite_values = OrderedDict()
        for neurite in NEURITE_NAMES:
            try:
                val = nm.get(self.name, neuron, neurite_type=getattr(NeuriteType, neurite))
                if DISCRETE_FEATURE.has_value(self.name):
                    val = np.sum(val) if val.size else None
                else:
                    val = val.tolist()
            except AssertionError as err:
                # None values for features without support of `neurite`
                if 'Neurite type' in err.args[0]:
                    val = None
                else:
                    raise
            self.neurite_values[neurite] = val


class Mfile:
    def __init__(self, path: Path, mtype: str):
        self.name = path.stem
        self.mtype = mtype
        self.features = OrderedDict()
        neuron = nm.load_neuron(str(path))
        for name in DISCRETE_FEATURE:
            self.features[name.value] = Feature(name, neuron)
        for name in CONTINUOUS_FEATURE:
            self.features[name.value] = Feature(name, neuron)


class Mtype:
    def __init__(self, name: str):
        self.name = name
        self.mfiles = []

    def add_file(self, mfile: Mfile):
        self.mfiles.append(mfile)


class FeatureDistr:
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.neurite_distrs = OrderedDict()
        for neurite in NEURITE_NAMES:
            self.neurite_distrs[neurite] = []

    def add_feature(self, feature: Feature):
        for name, value in feature.neurite_values.items():
            if value:
                self.neurite_distrs[name].append(value)


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
                    ks = ks_2samp(distr[i], others) + (len(distr[i]),)
                    ks_list.append(ks)
        self.distance = Stats([ks[0] for ks in ks_list])
        self.pvalue = Stats([ks[1] for ks in ks_list])
        self.size = Stats([ks[2] for ks in ks_list])


def calc_zscore(val, stats):
    """TODO confirm from Lida that it is OK to calc like that"""
    if val - stats.mean < 1e-10:
        return 0.0
    else:
        return (val - stats.mean) / stats.std


class DiscreteFeatureStats:
    def __init__(self, feature_distr: FeatureDistr):
        self.feature_name = feature_distr.feature_name
        self.neurite_stats = OrderedDict()
        for neurite, neurite_distr in feature_distr.neurite_distrs.items():
            if neurite_distr:
                self.neurite_stats[neurite] = Stats(neurite_distr)

    def zscore(self, feature):
        neurite_scores = OrderedDict()
        for neurite, value in feature.neurite_values.items():
            stats = self.neurite_stats.get(neurite)
            if stats:
                neurite_scores[neurite] = calc_zscore(value, stats)
        return pd.Series(data=neurite_scores, name=feature.name)


class ContinuousFeatureStats:
    def __init__(self, feature_distr: FeatureDistr):
        # if feature_distr.feature_name == CONTINUOUS_FEATURE.SEGMENT_RADII:
        #     print('g')
        self.feature_name = feature_distr.feature_name
        self.neurite_stats = OrderedDict()
        for neurite, neurite_distr in feature_distr.neurite_distrs.items():
            if neurite_distr:
                self.neurite_stats[neurite] = ContinuousStats(neurite_distr)

    def zscore(self, feature, feature_distr):
        columns = ['distance', 'pvalue', 'size']
        df = pd.DataFrame(index=feature.neurite_values.keys(), columns=columns)
        for neurite, value in feature.neurite_values.items():
            neurite_distr = feature_distr.neurite_distrs[neurite]
            if neurite_distr and value:
                ks = ks_2samp(np.concatenate(neurite_distr), value) + (len(value),)
                stats = self.neurite_stats.get(neurite)
                if stats:
                    df.loc[neurite, 'distance'] = calc_zscore(ks[0], stats.distance)
                    df.loc[neurite, 'pvalue'] = calc_zscore(ks[1], stats.pvalue)
                    df.loc[neurite, 'size'] = calc_zscore(ks[2], stats.size)
        df.columns = pd.MultiIndex.from_product([[feature.name], df.columns])
        return df


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
            if DISCRETE_FEATURE.has_value(distr.feature_name):
                stats = DiscreteFeatureStats(distr)
            else:
                stats = ContinuousFeatureStats(distr)
            self.feature_stats[distr.feature_name] = stats

    def zscore(self, test_mfile: Mfile) -> (pd.DataFrame, pd.DataFrame):
        discrete_data = []
        continuous_data = []
        for feature in test_mfile.features.values():
            feature_stats = self.feature_stats[feature.name]
            if DISCRETE_FEATURE.has_value(feature.name):
                discrete_data.append(feature_stats.zscore(feature))
            else:
                feature_distr = self.feature_distrs[feature.name]
                continuous_data.append(feature_stats.zscore(feature, feature_distr))
        discrete_zscore = pd.concat(discrete_data, axis=1, sort=True)
        continuous_zscore = pd.concat(continuous_data, axis=1)
        return pd.concat([discrete_zscore, continuous_zscore], axis=1, sort=True)


class Validator:
    def __init__(self, mtype_dict):
        self._mtype_distr_dict = {}
        for mtype in mtype_dict.values():
            self._mtype_distr_dict[mtype.name] = MtypeDistr(mtype)

    def zscore(self, test_mfile: Mfile):
        mtype_distr = self._mtype_distr_dict.get(test_mfile.mtype)
        if mtype_distr:
            return mtype_distr.zscore(test_mfile)

    def get_failed_zscore(self, test_mfile: Mfile, pvalue):
        assert 0. <= pvalue <= 1.
        zscore = self.zscore(test_mfile)
        if zscore is not None:
            threshold = np.abs(norm.ppf(pvalue / 2.))
            # some cells in z_score are NaN so we use `failed_neurites` + `any`
            # instead of `valid_neurites` + `all`.
            failed_zscore = zscore.abs() > threshold
            failed_features = zscore.loc[:, failed_zscore.any(axis=0)]
            return failed_features


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
                failed_zscore = validator.get_failed_zscore(Mfile(file, mtype), 0.05)
                if failed_zscore is not None:
                    if failed_zscore.empty:
                        print(file.name, ' is Valid')
                    else:
                        print(file.name)
                        print(failed_zscore)
                    print('--------------------------------------------')
