from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from morph_validator import repair


from neurom.apps.morph_stats import full_config


DATA = Path(__file__).parent / 'data'


def test_compare_morphometrics():
    release = DATA / 'repair'
    df = repair.compare_morphometrics([(release / 'neuronDB.xml', release / '04_ZeroDiameterFix'),
                                (release / 'neuronDB.xml', release / '06_RepairUnravel')])
    assert_frame_equal(df, pd.read_csv(release / 'expected-compare-masses.csv'))

    with TemporaryDirectory() as folder:
        out = Path(folder, 'masses.pdf')
        repair.create_pdf(df, out)
        assert out.exists()


def test_compare_morphometrics2():
    release = DATA / 'repair'
    df = repair.compare_morphometrics([(release / 'neuronDB.xml', release / '04_ZeroDiameterFix'),
                                (release / 'neuronDB.xml', release / '06_RepairUnravel')],
                               morph_stats_config=full_config())

    with TemporaryDirectory() as folder:
        out = Path(folder, 'masses.pdf')
        repair.create_pdf(df, out)
        assert out.exists()
