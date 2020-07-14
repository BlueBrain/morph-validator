"""
API for collecting neurom features from morphologies.
"""

import itertools
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import neurom as nm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from neurom import NeuriteType
from pandas import DataFrame
from tqdm import tqdm

L = logging.getLogger(__name__)
FEATURES_INDEX = ['mtype', 'filename', 'neurite']
DISCRETE_FEATURES = [
    'total_length',
    'total_area_per_neurite',
    'soma_surface_areas',
    'neurite_volumes',
    'soma_volumes',
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


def _get_soma_feature(feature: str, neuron, neurite: NeuriteType) -> np.array:
    """Gets soma area

    Args:
        feature: feature of neurom
        neuron: neuron object from neurom
        neurite: neurite of neuron
    Returns:
        Soma area value of neuron
    """
    if neurite == NeuriteType.soma:
        return nm.get(feature, neuron)
    return np.empty(0)


_FEATURE_CUSTOM_GETTERS = {
    'soma_surface_areas': partial(_get_soma_feature, 'soma_surface_areas'),
    'soma_volumes': partial(_get_soma_feature, 'soma_volumes'),
}


def _get_neuron_features(neuron, feature_names: List[str]) -> DataFrame:
    """Get features of neuron as a dataframe.

    Args:
        neuron: neuron object from neurom
        feature_names: list of feature names

    Returns:
        features of neuron with INDEX as index and feature names as columns
    """
    index = [neurite.name for neurite in NeuriteType]
    df = DataFrame(index=index, columns=feature_names)
    for neurite, feature_name in itertools.product(NeuriteType, feature_names):
        if feature_name in _FEATURE_CUSTOM_GETTERS:
            val = _FEATURE_CUSTOM_GETTERS[feature_name](neuron, neurite)
        else:
            val = nm.get(feature_name, neuron, neurite_type=neurite)
        df.loc[neurite.name, feature_name] = val.tolist()
    return df


def _collect(file):
    """Inner function to collect in parallel."""
    neuron = nm.load_neuron(str(file))
    name = neuron.name
    discrete = _get_neuron_features(neuron, DISCRETE_FEATURES)
    continuous = _get_neuron_features(neuron, CONTINUOUS_FEATURES)
    return name, discrete, continuous


def collect(files_per_mtype: Dict[str, List[Path]]) -> Tuple[DataFrame, DataFrame]:
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
        L.info('Extracting features for %s', mtype)
        features_per_file = Parallel(-1)(delayed(_collect)(file) for file in tqdm(files))
        for name_, discrete_, continuous_ in features_per_file:
            index.append((mtype, name_))
            discrete.append(discrete_)
            continuous.append(continuous_)
    discrete = pd.concat(discrete, keys=index, names=FEATURES_INDEX).applymap(np.sum)
    continuous = pd.concat(continuous, keys=index, names=FEATURES_INDEX)
    return discrete, continuous
