"""Validation tools for morphology repair."""
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.morphdb import MorphDB

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


def pdf_all_metrics_by_metric(metrics: pd.DataFrame,  # pylint: disable=too-many-locals
                              output_pdf: Path):
    '''Create pdfs containing the mass plots.

    Args:
        masses: the dataframe returned by the compare_masse function
        output_pdf: the path to the output

    '''
    idx = pd.IndexSlice
    df = metrics.loc[:, idx[['properties', 'all'], :]]

    metrics = metrics.loc[:, idx[['all'], :]].columns.droplevel()
    metrics = metrics[metrics.str.startswith('mean_')]
    df.columns = df.columns.droplevel()

    groups = df.groupby(['mtype', 'label'])
    labels = df.label.unique()
    mtypes = {mtype: i for i, mtype in enumerate(df.mtype.unique())}
    with PdfPages(output_pdf) as pdf:
        for metric in metrics:
            fig = plt.figure()

            x = {label: [] for label in labels}
            y = {label: [] for label in labels}
            yerr = {label: [] for label in labels}
            fig.suptitle(f'metric: {metric[5:]}')

            for (mtype, label), df in groups:
                mean = df[metric].mean()
                std = df[metric].std()
                if np.isfinite(mean) and np.isfinite(std):
                    y[label].append(mean)
                    yerr[label].append(std)
                    x[label].append(mtypes[mtype])

            options = dict(
                capsize=2,
                fmt='o',
                elinewidth=1,
                markersize=2,
            )

            for offset, label in enumerate(labels):
                plt.errorbar([item + 0.3 * offset for item in x[label]],
                             y[label], yerr=yerr[label], label=label, **options)

            plt.xticks(list(mtypes.values()), list(mtypes.keys()), rotation='vertical', fontsize=5)
            plt.legend()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def compare_morphometrics(db: MorphDB,
                          morph_stats_config,
                          n_workers=1,
                          output_pdf: str = 'morphometrics.pdf'):
    """Create plot to compare morphometrics of neurons.

    For axonal, basal and apical dendrites, and all dendrites,
    a pdf will be generated, with a page per mtype, comparing the masses
    between different morphology_paths.

    Args:
        db: a morphology db
        morph_stats_config: a config of morphometrics in the NeuroM morph-stat format
            See: https://neurom.readthedocs.io/en/latest/morph_stats.html
        n_workers: the number of workers
        output_pdf: the path to the output PDF
    """
    df = db.features(morph_stats_config, n_workers=n_workers)
    pdf_all_metrics_by_metric(df, output_pdf)
