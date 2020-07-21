"""Test `validator` module"""
from morph_validator import utils

from tests.utils import MORPHOLOGIES_DIR


def _assert_valid_file_dict(valid_files_dict, expected_filenames_dict, valid_dir):
    assert set(valid_files_dict.keys()) == (expected_filenames_dict.keys())
    for mtype, file_list in valid_files_dict.items():
        expected_filename_list = expected_filenames_dict[mtype]
        assert len(file_list) == len(expected_filename_list)
        for file in file_list:
            assert file.parent == valid_dir
            assert file.stem in expected_filename_list


def test_get_valid_files_per_mtype_xml():
    valid_mtype_db_file = MORPHOLOGIES_DIR / 'valid' / 'mini' / 'neuronDB.xml'
    valid_files_dict = utils.get_valid_mtype_files(valid_mtype_db_file)

    expected_filenames_dict = {
        'L23_BTC': [
            'mtC020502A_idA', 'mtC031100A_idB', 'mtC061100A_idC', 'mtC121100B_idJ', 'mtC240300A_idB'
        ],
        'L5_MC': ['C040426', 'C040601', 'C050896A-I', 'C180298B-I3', 'C290500C-I4'],
        'L5_TPC:A': ['rat_20160906_E1_LH5_cell2', 'rat_20160914_E1_LH4_cell1', 'rat_P16_S1_RH3_20140129'],
        'L5_TPC:B': ['rat_20170523_E1_LH2_cell1'],
    }
    _assert_valid_file_dict(valid_files_dict, expected_filenames_dict, valid_mtype_db_file.parent)


def test_get_valid_files_per_mtype_dat():
    valid_mtype_db_file = MORPHOLOGIES_DIR / 'valid' / 'mini' / 'neuronDB.dat'
    valid_files_dict = utils.get_valid_mtype_files(valid_mtype_db_file)

    expected_filenames_dict = {
        'L4_MC': ['C040426', 'C040601', 'C180298B-I3', 'C290500C-I4'],
        'L5_MC': ['C040426', 'C040601', 'C050896A-I', 'C180298B-I3', 'C290500C-I4'],
        'L6_MC': ['C040426', 'C040601', 'C180298B-I3', 'C290500C-I4'],
        'L23_BTC': ['mtC031100A_idB'],
    }
    _assert_valid_file_dict(valid_files_dict, expected_filenames_dict, valid_mtype_db_file.parent)
