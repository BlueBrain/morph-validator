"""
Don't use pandas DataFrame
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


def get_valid_files_per_mtype(valid_path: Path) -> dict:
    db_file = valid_path.joinpath('neuronDB.xml')
    root = etree.parse(str(db_file)).getroot()
    mtype_dict = defaultdict(list)
    for morphology in root.iterfind('.//morphology'):
        name = morphology.findtext('name')
        if not name:
            L.warning('Empty morphology name in %s', db_file)
        mtype = morphology.findtext('mtype')
        if not mtype:
            L.warning('Empty morphology mtype in %s', db_file)
        if name in mtype_dict and mtype_dict[name] != mtype:
            L.warning('Multiple mtypes %s %s for %s', mtype, mtype_dict[name], name)
        file = valid_path.joinpath(name + '.h5')
        if file.exists():
            mtype_dict[mtype].append(str(file))
    return mtype_dict


if __name__ == '__main__':
    valid_path = Path('../tests/data/morphologies/valid/mini')
    test_path = Path('../tests/data/morphologies/test')
    valid_files_per_mtype = get_valid_files_per_mtype(valid_path)
    valid_neurons_per_mtype = {
        mtype: nm.load_neurons(files) for mtype, files in valid_files_per_mtype.items()}

    for mtype_dir in test_path.iterdir():
        mtype = mtype_dir.name
        for test_file in mtype_dir.iterdir():



# valid_features = get_valid_morph_features(Path('../tests/data/morphologies/valid/mini'))
    # test_features = get_test_morph_features(Path('../tests/data/morphologies/test'))
