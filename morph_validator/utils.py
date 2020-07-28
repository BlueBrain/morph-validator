"""Utils functions."""
from pathlib import Path
from typing import Dict, List

from morph_tool.utils import neurondb_dataframe, iter_morphology_files


def get_mtype_files_db(
        neurondb_file: Path, verify_path: bool = True, ext: str = '.h5') -> Dict[str, List[Path]]:
    """Gets morphologies files defined by a neurondb file.

    Args:
        neurondb_file: file of mappings between morphology name and mtype. Morphology files
        must be located in the same directory as this file.
        verify_path: return only existing morphologies
        ext: return morphologies with this extension only

    Returns:
        dict: dictionary of files per full mtype (mtype:msubtype)
    """
    df = neurondb_dataframe(neurondb_file)
    df['path'] = df.apply(lambda x: Path(neurondb_file.parent, x['name'] + ext), axis=1)
    if verify_path:
        df = df[df.apply(lambda x: x['path'].exists(), axis=1)]
    mtype_dict = {mtype: list(set(df.path)) for mtype, df in df.groupby('mtype')}

    if not mtype_dict.keys():
        raise ValueError('No mtypes in {}'.format(neurondb_file))

    return mtype_dict


def get_mtype_files_dir(dir_: Path) -> Dict[str, List[Path]]:
    """Gets morphologies files defined by scanning a directory.

    Args:
        dir_: directory with morphologies. It must contain directories with morphologies files.
            Those directories must be named after morphology type which files they contain.

    Returns:
        dict: dictionary of files per mtype
    """
    if not dir_.is_dir():
        raise ValueError('"{}" must be a directory'.format(dir_))
    return {mtype_dir.name: list(iter_morphology_files(mtype_dir))
            for mtype_dir in dir_.iterdir()}
