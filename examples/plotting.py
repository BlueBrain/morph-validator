import os
import logging
from pathlib import Path
import warnings
from morph_validator.plotting import plot_violin_comparison

warnings.simplefilter("ignore")
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))


def plot_violin_comparison_example():
    example_dir = Path('/gpfs/bbp.cscs.ch/data/project_no_backup/proj82_no_backup/mgevaert')
    unrepaired_path = example_dir / '04_ZeroDiameterFix-asc'
    repaired_path = example_dir / '06_RepairUnravel-asc'
    neurondb = repaired_path / 'neuronDB.xml'
    plot_violin_comparison(
        neurondb,
        {'unrepaired': unrepaired_path, 'repaired': repaired_path},
        Path('./plot_violin_comparison'),
        ext='.asc',
        normalize=False,
        n_workers=10,
    )


if __name__ == '__main__':
    plot_violin_comparison_example()
