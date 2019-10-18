"""
Plain dictionaries instead of Pandas.DataFrame
"""

test_file = {
                'filename': '',
                'mtype': '',
                'feature_1': {
                    'axon': [],
                    'soma': [],
                    # ...
                },
                'feature_2': {
                    'axon': [],
                    'soma': [],
                    # ...
                },
                # ...
            },

valid_mtype_dict = {
    'mtype1': {
        'file1': {
            'filename': '',
            'mtype': '',
            'feature_1': {
                'axon': [],
                'soma': [],
                # ...
            },
            'feature_2': {
                'axon': [],
                'soma': [],
                # ...
            },
            # ...
        }
    },
    'mtype2': {
        'file1': {
            'filename': '',
            'mtype': '',
            'feature_1': {
                'axon': [],
                'soma': [],
                # ...
            },
            'feature_2': {
                'axon': [],
                'soma': [],
                # ...
            },
            # ...
        }
    },
    # ...
}
