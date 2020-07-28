"""Plotting functions."""
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from neurom.apps.morph_stats import extract_dataframe

from morph_validator.utils import get_mtype_files_db
from morph_validator.feature_configs import get_feature_configs

matplotlib.use("Agg")
L = logging.getLogger(__name__)


def get_features_df(morphologies_mtypes: Dict, features_config: Dict, n_workers: int = 1):
    """Create a feature dataframe from a dictionary of morphology_folders per mtypes.

    Args:
        morphologies_mtypes (dict): dict of morphology_folders files per mtype
        features_config (dict): configuration dict for features extraction
            (see ``neurom.apps.morph_stats.extract_dataframe``)
        n_workers (int) : number of workers for feature extractions
        """
    features_df = pd.DataFrame()
    for mtype, morphology_folders in tqdm(morphologies_mtypes.items()):
        features_df_tmp = extract_dataframe(morphology_folders, features_config,
                                            n_workers=n_workers)
        features_df_tmp['mtype'] = mtype
        features_df = features_df.append(features_df_tmp.replace(0, np.nan))
    return features_df


def _morphologies_to_features(neurondb: Path,
                              morphology_folders: Dict[str, Path],
                              features_config: Dict,
                              ext='.h5', n_workers=1):
    """Creates a single dataframe from ``neurondb`` and ``morphology_folders``.

    Args:
        neurondb: path to neurondb
        morphology_folders: a dict of labels, morphology folders
    """
    L.info('Extracting features.')
    neurondb_mtypes = get_mtype_files_db(neurondb, verify_path=False, ext=ext)
    features = []
    for morphologies_label, morphologies_path in morphology_folders.items():
        morphologies_mtypes = {mtype: [Path(morphologies_path, file.name) for file in mtype_files]
                               for mtype, mtype_files in neurondb_mtypes.items()}
        df = get_features_df(morphologies_mtypes, features_config, n_workers=n_workers)
        df['label'] = morphologies_label
        features.append(df)
    return pd.concat(features)


def _expand_lists(data):
    """Convert list element of dataframe to duplicated rows with float values."""
    data_expanded = pd.DataFrame()
    for row_id in data.index:
        if isinstance(data.loc[row_id, 'value'], list):
            for value in data.loc[row_id, 'value']:
                new_row = data.loc[row_id].copy()
                new_row['value'] = value
                data_expanded = data_expanded.append(new_row)
    return data_expanded


def _normalize(data):
    """Normalize data witht mean and std."""
    data_tmp = data.set_index(['feature', 'neurite_type', 'mtype'])
    groups = data_tmp.groupby(['feature', 'neurite_type', 'mtype'])
    means = groups.mean().reset_index()
    stds = groups.std().reset_index()
    for feat_id in means.index:
        mask = (data.feature == means.loc[feat_id, 'feature']) \
            & (data.neurite_type == means.loc[feat_id, 'neurite_type']) \
            & (data.mtype == means.loc[feat_id, 'mtype'])
        data.loc[mask, 'value'] = (
            data.loc[mask, 'value'] - means.loc[feat_id, 'value']
        ) / stds.loc[feat_id, 'value']
    return data


def plot_violin_features(features: pd.DataFrame, neurite_types: List, output_dir: Path,
                         bw: float, normalize=True):
    """Create violin plots from features dataframe.

    Args:
        features (pandas.DataFrame): features dataframe to plot
        neurite_types (list): list of neurite types to plot (one plot per neurite_type)
        output_dir (Path): path to folder for saving plots
        bw (float): resolution of violins
        normalize (bool): normalize feature values with mean/std
    """
    L.info('Plotting features.')
    output_dir.mkdir(parents=True, exist_ok=True)
    features = features.melt(var_name='feature', id_vars=['name', 'mtype', 'neurite_type', 'label'])
    for neurite_type in neurite_types:
        with PdfPages(output_dir / f'morphometrics_{neurite_type}.pdf') as pdf:
            for mtype in tqdm(features.mtype.unique()):
                data = (
                    features[
                        (features.mtype == mtype)
                        & (features.neurite_type == neurite_type)
                    ]
                    .drop('name', axis=1)
                    .dropna()
                )
                data = _expand_lists(data)
                if normalize:
                    data = _normalize(data)

                if len(data.index) > 0:
                    plt.figure()
                    ax = plt.gca()
                    sns.violinplot(
                        x='feature',
                        y='value',
                        hue='label',
                        data=data,
                        split=True,
                        bw=bw,
                        ax=ax,
                        inner='quartile',
                    )

                    ax.tick_params(axis='x', rotation=90)
                    plt.suptitle(f'mtype: {mtype}')
                    pdf.savefig(bbox_inches='tight')
                    plt.close()


def plot_violin_comparison(
    neurondb: Path,
    morphology_folders: Dict[str, Path],
    output_dir: Path,
    features_config: Dict = None,
    neurite_types: List = None,
    bw: float = 0.2,
    normalize: bool = True,
    n_workers=1,
    ext='.h5',
):
    """Does comparative plots of features from two collections of morphology_folders.

    For an usage example see `examples/plotting.py`.

    Args:
        neurondb: path to neurondb file. All morphology_folders files from ``morphology_folders``
            argument must be valid for this ``neurondb``.
        morphology_folders: dictionary with two items of format:
            {<morphology_folders label>: <morphology_folders path>,
             <morphology_folders label>: <morphology_folders path>}
            where `morphology_folders label` is a legend label of morphology_folders on plots
            `morphology_folders path` is a path to a folder with morphology_folders files
        features_config: dict with configuration for feature extractions
            (see ``neurom.apps.morph_stats.extract_dataframe``)
        output_dir: path to folder for saving plots.
        neurite_types: list of neurite types to plot (one pdf per neurite_type)
        bw: resolution of violins
        normalize: normalize feature values with mean/std
        n_workers : number of workers for feature extractions
        ext: morphology file extension
    """
    assert len(morphology_folders.keys()) == 2, 'two sets of morphology_folders required'
    if neurite_types is None:
        neurite_types = ['basal_dendrite', 'apical_dendrite', 'axon']
    if features_config is None:
        features_config = get_feature_configs()

    features = _morphologies_to_features(neurondb, morphology_folders, features_config,
                                         ext=ext, n_workers=n_workers)
    plot_violin_features(features, neurite_types, output_dir, bw=bw, normalize=normalize)
