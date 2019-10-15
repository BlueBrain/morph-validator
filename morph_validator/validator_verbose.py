"""
use separate function for each feature extraction
"""
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


def set_total_len(df, neuron):
    data = [np.sum(nm.get(DISCRETE_FEATURE_NAMES.LEN.value, neuron, neurite_type=neurite))
            for neurite in NEURITES]
    df.loc[DISCRETE_FEATURE_NAMES.LEN.value] = data


def set_total_area(df, neuron):
    data = [np.sum(nm.get(DISCRETE_FEATURE_NAMES.SURFACE_AREA.value, neuron, neurite_type=neurite))
            for neurite in NEURITES]
    idx = NEURITES.index(NeuriteType.soma)
    data[idx] = nm.get('soma_surface_areas', neuron)[0]
    df.loc[DISCRETE_FEATURE_NAMES.SURFACE_AREA.value] = data


def set_volumes(df, neuron):
    data = [np.sum(nm.get(DISCRETE_FEATURE_NAMES.VOLUMES.value, neuron, neurite_type=neurite))
            for neurite in NEURITES]
    df.loc[DISCRETE_FEATURE_NAMES.VOLUMES.value] = data


def set_number_of_sections(df, neuron):
    data = [np.sum(nm.get(
        DISCRETE_FEATURE_NAMES.NUMBER_OF_SECTIONS.value, neuron, neurite_type=neurite))
        for neurite in NEURITES]
    df.loc[DISCRETE_FEATURE_NAMES.NUMBER_OF_SECTIONS.value] = data


def set_number_of_bifurcations(df, neuron):
    data = [
        np.sum(nm.get(DISCRETE_FEATURE_NAMES.NUMBER_OF_BIFURCATIONS.value, neuron,
            neurite_type=neurite))
        for neurite in NEURITES]
    df.loc[DISCRETE_FEATURE_NAMES.NUMBER_OF_BIFURCATIONS.value] = data


def set_number_of_terminations(df, neuron):
    data = [np.sum(nm.get(
        DISCRETE_FEATURE_NAMES.NUMBER_OF_TERMINATIONS.value, neuron, neurite_type=neurite))
        for neurite in NEURITES]
    df.loc[DISCRETE_FEATURE_NAMES.NUMBER_OF_TERMINATIONS.value] = data


def set_section_len(df, neuron):
    data = [nm.get(
        CONTINUOUS_FEATURE_NAMES.SECTION_LEN.value, neuron, neurite_type=neurite)
        for neurite in NEURITES]
    df.loc[CONTINUOUS_FEATURE_NAMES.SECTION_LEN.value] = data


def set_section_radial_distances(df, neuron):
    data = [nm.get(
        CONTINUOUS_FEATURE_NAMES.SECTION_RADIAL_DISTANCES.value, neuron, neurite_type=neurite)
        for neurite in NEURITES]
    df.loc[CONTINUOUS_FEATURE_NAMES.SECTION_RADIAL_DISTANCES.value] = data


def set_section_path_distances(df, neuron):
    data = [nm.get(
        CONTINUOUS_FEATURE_NAMES.SECTION_PATH_DISTANCES.value, neuron, neurite_type=neurite)
        for neurite in NEURITES]
    df.loc[CONTINUOUS_FEATURE_NAMES.SECTION_PATH_DISTANCES.value] = data


def set_partition_asymmetry(df, neuron):
    data = [nm.get(
        CONTINUOUS_FEATURE_NAMES.PARTITION_ASYMMETRY.value, neuron, neurite_type=neurite)
        for neurite in NEURITES]
    df.loc[CONTINUOUS_FEATURE_NAMES.PARTITION_ASYMMETRY.value] = data


def set_segment_radii(df, neuron):
    data = [nm.get(
        CONTINUOUS_FEATURE_NAMES.SEGMENT_RADII.value, neuron, neurite_type=neurite)
        for neurite in NEURITES]
    df.loc[CONTINUOUS_FEATURE_NAMES.SEGMENT_RADII.value] = data


def get_features(neuron) -> pd.DataFrame:
    df = pd.DataFrame(columns=[neurite.name for neurite in NEURITES])
    set_total_len(df, neuron)
    set_total_area(df, neuron)
    set_volumes(df, neuron)
    set_number_of_sections(df, neuron)
    set_number_of_bifurcations(df, neuron)
    set_number_of_terminations(df, neuron)

    set_section_len(df, neuron)
    set_section_radial_distances(df, neuron)
    set_section_path_distances(df, neuron)
    set_partition_asymmetry(df, neuron)
    set_segment_radii(df, neuron)
    return df.T


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
    features = []
    features_index = []
    features_index_names = ['mtype', 'filename', 'neurite']
    for file in morph_dirpath.iterdir():
        if file.suffix in ['.h5', '.swc', '.asc']:
            neuron = nm.load_neuron(str(file))
            mtype = mtype_dict[neuron.name]
            features_index.append((mtype, neuron.name))
            features.append(get_features(neuron))
    features = pd.concat(
        features, keys=features_index, names=features_index_names)
    return features


if __name__ == '__main__':
    print(build_valid_morphologies(Path('../tests/data/valid_morphologies')))
