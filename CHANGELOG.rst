Changelog
=========

Version 0.2.4
-------------

- Use BluePy >= 2.3.0

Version 0.2.3
-------------

- Use NeuroM v2

Version 0.2.0
-------------

- Split `features` among `features`, `zscores` and `utils` modules
- Rename
  `features.collect_features` to `features.collect`
  `utils.get_test_files_per_mtype` to `utils.get_mtype_files_dir`
  `utils.get_valid_mtype_files` to `utils.get_mtype_files_db`
- Add `plotting` module for various plots to validate morphologies visually
- Add `plot_violin_comparison` to `plotting`
- Add `repair` module to validate morphology repair

Version 0.1.1
-------------

- Changed Z score validation to be parallel

Version 0.1.0
-------------

- Added spatial validations

Version 0.0.1
-------------

- Added Z score validation of features
