"""Validation tools for morphology repair."""
import logging
from itertools import starmap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import neurom as nm
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from neurom.apps.morph_stats import extract_dataframe
from tqdm import tqdm

from morph_tool.utils import neurondb_dataframe

matplotlib.use('Agg')
L = logging.getLogger(__name__)

CONFIG_TOTAL_LENGTH = {
    'neurite': {'total_length_per_neurite': ['total']},
}


def create_pdf(masses: pd.DataFrame,
               output_pdf: Path):
    '''Create pdfs containing the mass plots.

    Args:
        masses: the dataframe returned by the compare_masse function
        output_pdf: the path to the output

    '''
    with PdfPages(output_pdf) as pdf:
        for neurite_type, df in masses.groupby('neurite_type'):
            fig = plt.figure()
            df.plot.bar(x='mtype', ax=plt.gca())
            fig.suptitle(neurite_type)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def _get_masses(df, morph_stats_config):
    """Get the dendrite masses of cells."""
    mass_df = pd.DataFrame()
    for _, row in tqdm(df[['mtype', 'path']].iterrows()):
        mass_df_tmp = extract_dataframe(
            nm.load_neurons(row.path), morph_stats_config
        )
        mass_df_tmp['mtype'] = row.mtype
        mass_df = mass_df.append(mass_df_tmp)
    return mass_df


def compare_morphometrics(data: List[Tuple[Path, Path]],
                          morph_stats_config: Optional[Dict] = None):
    """Create plot to compare morphometrics of neurons.

    For axonal, basal and apical dendrites, and all dendrites,
    a pdf will be generated, with a page per mtype, comparing the masses
    between different morphology_paths.

    Args:
        data: list of 2-tuples (path to neurondb, path to morphology folder)
        morph_stats_config: a config of morphometrics in the NeuroM morph-stat format
            See: https://neurom.readthedocs.io/en/latest/morph_stats.html
    """

    L.info('Get masses from morphologies')
    fat = None
    all_mtype = None
    morph_stats_config = morph_stats_config or CONFIG_TOTAL_LENGTH

    for i, df in enumerate(starmap(neurondb_dataframe, data)):
        index = ['mtype', 'neurite_type']
        masses = _get_masses(df, morph_stats_config)
        by_mtype_neurite = masses.groupby(index).mean()
        by_neurite = masses.groupby('neurite_type').mean()
        suffix = str(i)
        if i == 0:
            fat = by_mtype_neurite
            all_mtype = by_neurite
        else:
            fat = fat.join(by_mtype_neurite, rsuffix=suffix)
            all_mtype = all_mtype.join(by_neurite, rsuffix=suffix)

    all_mtype = all_mtype.reset_index()
    all_mtype['mtype'] = 'ALL'

    return pd.concat([all_mtype, fat.reset_index()]).reset_index()
