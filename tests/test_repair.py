from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from morph_validator import repair

DATA = Path(__file__).parent / 'data'


def test_compare_masses():
    release = DATA / 'repair'
    df = repair.compare_masses([(release / 'neuronDB.xml', release / '04_ZeroDiameterFix'),
                                (release / 'neuronDB.xml', release / '06_RepairUnravel')])
    assert_frame_equal(df, pd.read_csv(release / 'expected-compare-masses.csv'))
