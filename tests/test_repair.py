from pathlib import Path
from tempfile import TemporaryDirectory

from morph_tool.morphdb import MorphDB
from morph_validator import repair

DATA = Path(__file__).parent / 'data'


def test_compare_morphometrics():
    release = DATA / 'repair'
    config = {'neurite': {'total_length': ['min', 'max', 'median', 'mean', 'std'],
                          'total_length_per_neurite': ['min', 'max', 'median', 'mean', 'std']}}

    db = MorphDB.from_neurondb(release / 'neuronDB.xml',
                               morphology_folder=release / '04_ZeroDiameterFix')
    db += MorphDB.from_neurondb(release / 'neuronDB.xml',
                                morphology_folder=release / '06_RepairUnravel')

    with TemporaryDirectory() as folder:
        out = Path(folder, 'masses.pdf')
        repair.compare_morphometrics(db, morph_stats_config=config,
                                     output_pdf=out)
        assert out.exists()
