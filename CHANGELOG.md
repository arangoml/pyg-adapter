## 1.1.2 (2023-10-25)

### Other

* Code Cleanup (#23) [Anthony Mahanna]

  * code cleanup

  copy of https://github.com/arangoml/dgl-adapter/pull/30/commits/6b36a5056939efa56a857f709857c3b364c06ccb

  * attempt fix: `pip install torch`

  * attempt fix: remove `-e`

  * attempt fix: remove cache

  * fix: bad comment

  * attempt fix: `pip install torch-sparse`

  * attempt fix: torch installations

  * attempt fix: upgrade pip

  hitting a wall...

  * cleanup: workflows

  * cleanup: `adapter.py`

  * cleanup: workflows

  * fix: install publishing dependencies

  * extracting more logic into separate functions

  * fix: mypy

  * rename kwargs

  * bump `adbpyg-adapter` to `1.1.2` in notebooks

  * fix param order

  * bump

  * Delete build_self_hosted.yml

  * cleanup workflows

  * Update build.yml

  * Update build.yml

  * Update build.yml

  * parameter renaming, new `use_async` param

  * fix abc

  * fix mypy

  * add `strict` param to other adb -> pyg methods

  * Update setup.py

* Changelog: release 1.1.1 (#22) [github-actions[bot]]

  !gitchangelog


## 1.1.1 (2023-08-08)

### New

* Batch_size (pyg to adb) [Anthony Mahanna]

  + some miscellaneous cleanup

* Get_aql_return_value. [Anthony Mahanna]

* Batch_size test param. [Anthony Mahanna]

  (adb to pyg)

* `cache` field in build workflow. [Anthony Mahanna]

* Video link. [aMahanna]

### Fix

* Release.yml. [Anthony Mahanna]

* Mypy. [Anthony Mahanna]

* Udf behaviour when batch_size is set. [Anthony Mahanna]

* Aql kwargs. [Anthony Mahanna]

* Flake8. [Anthony Mahanna]

* Mypy. [Anthony Mahanna]

* `PyGMetagraph` typing. [Anthony Mahanna]

* Mypy. [Anthony Mahanna]

* Type ignore. [Anthony Mahanna]

### Other

* Merge pull request #20 from arangoml/housekeeping. [Chris Woodward]

  More housekeeping

* Lock python-arango version. [Anthony Mahanna]

* Merge branch 'master' into housekeeping. [Anthony Mahanna]

* Merge pull request #21 from arangoml/MLP-443. [Chris Woodward]

  MLP-443 | Bump python-arango

* Removes E721 overzealous instance checking rule. [Chris Woodward]

* MLP-443 | Bump python-arango. [Chris Woodward]

* Use `isinstance` [Anthony Mahanna]

* Initial commit. [Anthony Mahanna]

* Merge pull request #19 from arangoml/feature/batching. [Alex Geenen]

  Adapter batching

* Temp: remove request_timeout. [Anthony Mahanna]

  `mypy` is being silly

* Merge branch 'master' into feature/batching. [Anthony Mahanna]

* Merge pull request #17 from arangoml/feature/exceptions. [Alex Geenen]

  Add Strict Parameter & Handle Invalid Edges

* Wip: arangodb to pyg batching. [Anthony Mahanna]

* Merge branch 'master' into feature/exceptions. [Anthony Mahanna]

* Misc: cleanup. [Anthony Mahanna]

* Remove: 3.7 build workflow. [Anthony Mahanna]

  end of line as of june 27 2023

* Merge branch 'master' into feature/exceptions. [Anthony Mahanna]

* Housekeeping. [Anthony Mahanna]

* Temp: new build.yml workflow. [Anthony Mahanna]

  (temporarily retiring self-hosted runner implementation)

* Merge branch 'master' into feature/exceptions. [Alex Geenen]

* Merge pull request #18 from arangoml/documentation/lunch-and-learn-video. [Chris Woodward]

  new: Lunch & Learn video link in readme

* Merge branch 'master' into documentation/lunch-and-learn-video. [Anthony Mahanna]

* Merge pull request #14 from arangoml/actions/changelog. [Chris Woodward]

  changelog: release 1.1.0

* Add Strict Parameter & Handle Invalid Edges. [Alex Geenen]


## 1.1.0 (2022-09-21)

### New

* Adbpyg 1.1.0 notebook. [aMahanna]

* `test_full_cycle_homogeneous_with_preserve_adb_keys` [aMahanna]

* Address comments. [aMahanna]

* `set[str]` metagraph value type. [aMahanna]

* Full Cycle README section. [aMahanna]

* `preserve_adb_keys` refactor. [aMahanna]

* Test_adapter.py refactor. [aMahanna]

* Test cases to cover `preserve_adb_keys` [aMahanna]

* `pytest_exception_interact` [aMahanna]

* `preserve_adb_keys` docstring. [aMahanna]

* Lazy attempt at #4. [aMahanna]

* Self-hosted runners for Github Actions (#10) [Anthony Mahanna]

* Query/dataframe optimization. [aMahanna]

* Test_adb_partial_to_pyg. [aMahanna]

* Optional & partial edge collectiond data transfer (ArangoDB to PyG) [aMahanna]

* Notebook output file. [aMahanna]

  (for blog post purposes)

### Fix

* Black & mypy. [aMahanna]

* Flake8. [aMahanna]

* Black. [aMahanna]

* Docstring. [aMahanna]

* Default param value. [aMahanna]

* Flake8. [aMahanna]

* Map typings. [aMahanna]

* `test_adb_partial_to_pyg` RNG. [aMahanna]

* Typo. [aMahanna]

* Black. [aMahanna]

* Black. [aMahanna]

* Black. [aMahanna]

### Other

* Merge pull request #12 from arangoml/workflow-patch. [Chris Woodward]

  Update release.yml

* Update release.yml. [Chris Woodward]

  remove quotes

* Update release.yml. [Chris Woodward]

  updates build step to target specific python version

* Merge pull request #11 from arangoml/feature/adbpyg-map. [Chris Woodward]

  new: `preserve_adb_keys` in PyG to ArangoDB

* Update README.md. [aMahanna]

* Update README.md. [aMahanna]

* Update: documentation. [aMahanna]

* Cleanup: progress bars. [aMahanna]

* Cleanup: test_adapter. [aMahanna]

* Update README.md. [aMahanna]

* Update README.md. [aMahanna]

* Update test_adapter.py. [aMahanna]

* Revert "bump" [aMahanna]

  This reverts commit abf477bf20f98262f6b473c14e8e217e4eca0872.

* Bump. [aMahanna]

* Move: __build_tensor_from_dataframe. [aMahanna]

  also minor cleanup

* Update README.md. [aMahanna]

* Update `explicit_metagraph` docstring. [aMahanna]

* Update release.yml. [aMahanna]

* Cleanup. [aMahanna]

* Update: docstrings. [aMahanna]

* Update docstring. [aMahanna]

* Cleanup: test_adapter. [aMahanna]

* Debug: test `HeterogeneousTurnedHomogeneous` [aMahanna]

* Cleanup: `__finish_adb_dataframe` and `__build_dataframe_from_tensor` [aMahanna]

* Remove: `cudf` imports. [aMahanna]

  out of this PR's scope

* Temp: fix cudf to_dict error. [aMahanna]

* Debug: `pytest_exception_interact` [aMahanna]

* Temp: `# flake8: noqa` [aMahanna]

  (bypassing for now to see the passing tests)

* Cleanup: `pyg_keys` [aMahanna]

* Move: `preserve_adb_keys` [aMahanna]

* Cleanup: adapter.py. [aMahanna]

* Temp: disable (partial) feature validation in `assert_arangodb_data` [aMahanna]

  this needs to be cleaned up anyway

* Checkpoint. [aMahanna]

* Initial (experimental) commit. [aMahanna]

* Merge branch 'master' into feature/adbpyg-map. [aMahanna]

* Cleanup: `validate_adb_metagraph` [aMahanna]

* Update README.md. [aMahanna]

* Update README.md. [aMahanna]

* Cleanup. [aMahanna]

* Temp: disable HeterogeneousPartialEdgeCollectionImport. [aMahanna]

* Cleanup: abc. [aMahanna]

* Remove: temp conftest hack. [aMahanna]

* Cleanup build.yml & release.yml. [aMahanna]

* Update build.yml. [aMahanna]

* Update conftest.py. [aMahanna]

* Temp: black hack. [aMahanna]

* Temp: create py310 database for 3.10 testing. [aMahanna]

* Initial commit. [aMahanna]


## 1.0.0 (2022-07-29)

### New

* `adbpyg-adapter` MVP (#2) [Anthony Mahanna]

### Fix

* Release.yml typo. [Anthony Mahanna]

* Notebook. [aMahanna]

### Other

* Update README.md. [Anthony Mahanna]

* Update README.md. [Anthony Mahanna]

* Create adb_logo.png. [aMahanna]

* Revert "Revert "initial repo commit"" [aMahanna]

  This reverts commit f2633054057b61fabbf9b3bc3492cee7a292e041.

* Revert "initial repo commit" [aMahanna]

  This reverts commit 9323718c8d9d5015f7928b4dd643c07f316b1b2b.

* Initial repo commit. [aMahanna]


