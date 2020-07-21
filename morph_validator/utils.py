"""Utils functions."""
from pathlib import Path
from typing import Dict, List

from morph_tool.utils import neurondb_dataframe


def get_valid_mtype_files(valid_mtype_db_file: Path,
                          verify_path=True, ext='.h5') -> Dict[str, List[Path]]:
    """Gets valid morphologies files.

    Args:
        valid_mtype_db_file: file of mappings between morphology name and mtype. Morphology files
        must be located in the same directory as this file.

    Returns:
        dictionary of files per full mtype (mtype:msubtype)
    """
    df = neurondb_dataframe(valid_mtype_db_file)
    df['path'] = df.apply(lambda x: Path(valid_mtype_db_file.parent, x['name'] + ext), axis=1)
    if verify_path:
        df = df[df.apply(lambda x: x['path'].exists(), axis=1)]
    mtype_dict = {mtype: list(set(df.path)) for mtype, df in df.groupby('mtype')}

    if not mtype_dict.keys():
        raise ValueError('No mtypes in {}'.format(valid_mtype_db_file))

    return mtype_dict
