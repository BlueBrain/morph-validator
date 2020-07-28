"""Test `validator` module"""
from morph_validator import utils

from tests.utils import MORPHOLOGIES_DIR


def _assert_mtype_files(actual_mtype_files, expected_mtype_files):
    assert set(actual_mtype_files.keys()) == (expected_mtype_files.keys())
    for mtype, file_list in actual_mtype_files.items():
        expected_filename_list = expected_mtype_files[mtype]
        assert len(file_list) == len(expected_filename_list)
        for file in file_list:
            assert file.stem in expected_filename_list


def test_get_mtype_files_db_xml():
    valid_mtype_db_file = MORPHOLOGIES_DIR / 'valid' / 'mini' / 'neuronDB.xml'
    valid_files_dict = utils.get_mtype_files_db(valid_mtype_db_file)

    expected_filenames_dict = {
        'L23_BTC': [
            'mtC020502A_idA', 'mtC031100A_idB', 'mtC061100A_idC', 'mtC121100B_idJ', 'mtC240300A_idB'
        ],
        'L5_MC': ['C040426', 'C040601', 'C050896A-I', 'C180298B-I3', 'C290500C-I4'],
        'L5_TPC:A': ['rat_20160906_E1_LH5_cell2', 'rat_20160914_E1_LH4_cell1', 'rat_P16_S1_RH3_20140129'],
        'L5_TPC:B': ['rat_20170523_E1_LH2_cell1'],
    }
    _assert_mtype_files(valid_files_dict, expected_filenames_dict)


def test_get_mtype_files_db_dat():
    valid_mtype_db_file = MORPHOLOGIES_DIR / 'valid' / 'mini' / 'neuronDB.dat'
    valid_files_dict = utils.get_mtype_files_db(valid_mtype_db_file)

    expected_filenames_dict = {
        'L4_MC:A': ['C040426', 'C040601', 'C180298B-I3', 'C290500C-I4'],
        'L5_MC:subtype': ['C040426', 'C040601', 'C050896A-I', 'C180298B-I3', 'C290500C-I4'],
        'L6_MC': ['C040426', 'C040601', 'C180298B-I3', 'C290500C-I4'],
        'L23_BTC': ['mtC031100A_idB'],
    }
    _assert_mtype_files(valid_files_dict, expected_filenames_dict)


def test_get_mtype_files_dir():
    test_dir = MORPHOLOGIES_DIR / 'test'
    test_files_dict = utils.get_mtype_files_dir(test_dir)

    expected_filenames_dict = {
        'L23_BTC': ['C210401C'],
        'L5_MC': ['mtC171001A_idC', 'rp101020_2_INT_idA', 'vd101020A_idA'],
        'Unknown': ['ca3b-N2.CNG'],
    }
    _assert_mtype_files(test_files_dict, expected_filenames_dict)
