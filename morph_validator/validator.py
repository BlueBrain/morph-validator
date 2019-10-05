""""""
import itertools
import logging
from enum import Enum
from pathlib import Path

import neurom as nm
import numpy as np
import pandas as pd
from lxml import etree
from neurom import NeuriteType

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


def get_discrete_features(neuron) -> pd.DataFrame:
    feature_names = [name.value for name in DISCRETE_FEATURE_NAMES]
    df = get_features(neuron, feature_names)
    df = df.applymap(np.sum)
    df.loc[NeuriteType.all.name] = df.sum()
    return df


def get_continuous_features(neuron) -> pd.DataFrame:
    feature_names = [name.value for name in CONTINUOUS_FEATURE_NAMES]
    df = get_features(neuron, feature_names)
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
        df.loc[neurite.name, feature_name] = val
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


def build_valid_morphologies(morph_dirpath: Path):
    if not morph_dirpath.is_dir():
        raise ValueError(
            '"{}" must be a directory with morphology files'.format(morph_dirpath))
    mtype_dict = get_mtype_dict(morph_dirpath.joinpath('neuronDB.xml'))
    discrete_features = []
    continuous_features = []
    features_index = []
    features_index_names = ['mtype', 'filename', 'neurite']
    for file in morph_dirpath.iterdir():
        if file.suffix in ['.h5', '.swc', '.asc']:
            neuron = nm.load_neuron(str(file))
            mtype = mtype_dict[neuron.name]
            features_index.append((mtype, neuron.name))
            discrete_features.append(get_discrete_features(neuron))
            continuous_features.append(get_continuous_features(neuron))
    discrete_features = pd.concat(
        discrete_features, keys=features_index, names=features_index_names)
    continuous_features = pd.concat(
        continuous_features, keys=features_index, names=features_index_names)
    return discrete_features, continuous_features


if __name__ == '__main__':
    build_valid_morphologies(Path('../tests/data/valid_morphologies'))
