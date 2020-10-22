import os
import logging
from pathlib import Path

from morph_validator.repair import compare_morphometrics

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == "__main__":
    unrepaired_path = Path(
        "/gpfs/bbp.cscs.ch/data/project_no_backup/proj82_no_backup/mgevaert/04_ZeroDiameterFix"
    )
    repaired_path = Path(
        "/gpfs/bbp.cscs.ch/data/project_no_backup/proj82_no_backup/mgevaert/06_RepairUnravel"
    )
    neurondb = Path(
        "/gpfs/bbp.cscs.ch/data/project_no_backup/proj82_no_backup/mgevaert/06_RepairUnravel/neuronDB.xml"
    )

    compare_morphometrics(neurondb, [unrepaired_path, repaired_path], ['_unrep', '_rep'], relative_to = '_unrep')
