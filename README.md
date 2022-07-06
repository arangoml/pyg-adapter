# ArangoDB-PyG Adapter

[![build](https://github.com/arangoml/pyg-adapter/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/arangoml/pyg-adapter/actions/workflows/build.yml)
[![CodeQL](https://github.com/arangoml/pyg-adapter/actions/workflows/analyze.yml/badge.svg?branch=master)](https://github.com/arangoml/pyg-adapter/actions/workflows/analyze.yml)
[![Coverage Status](https://coveralls.io/repos/github/arangoml/pyg-adapter/badge.svg?branch=master)](https://coveralls.io/github/arangoml/pyg-adapter)
[![Last commit](https://img.shields.io/github/last-commit/arangoml/pyg-adapter)](https://github.com/arangoml/pyg-adapter/commits/master)

[![PyPI version badge](https://img.shields.io/pypi/v/adbpyg-adapter?color=3775A9&style=for-the-badge&logo=pypi&logoColor=FFD43B)](https://pypi.org/project/adbpyg-adapter/)
[![Python versions badge](https://img.shields.io/pypi/pyversions/adbpyg-adapter?color=3776AB&style=for-the-badge&logo=python&logoColor=FFD43B)](https://pypi.org/project/adbpyg-adapter/)

[![License](https://img.shields.io/github/license/arangoml/pyg-adapter?color=9E2165&style=for-the-badge)](https://github.com/arangoml/pyg-adapter/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/static/v1?style=for-the-badge&label=code%20style&message=black&color=black)](https://github.com/psf/black)
[![Downloads](https://img.shields.io/badge/dynamic/json?style=for-the-badge&color=282661&label=Downloads&query=total_downloads&url=https://api.pepy.tech/api/projects/adbpyg-adapter)](https://pepy.tech/project/adbpyg-adapter)


<a href="https://www.arangodb.com/" rel="arangodb.com">![](./examples/assets/adb_logo.png)</a>
<a href="https://www.pyg.org/" rel="pyg.org"><img src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true" width=40% /></a>

The ArangoDB-PyG Adapter exports Graphs from ArangoDB, the multi-model database for graph & beyond, into PyTorch Geometric (PyG), ________, and vice-versa.


## About PyG

## Installation

#### Latest Release
```
pip install torch
pip install adbpyg-adapter
```
#### Current State
```
pip install torch
pip install git+https://github.com/arangoml/pyg-adapter.git
```

##  Quickstart

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arangoml/pyg-adapter/blob/master/examples/ArangoDB_PyG_Adapter.ipynb)

```py
from arango import ArangoClient  # Python-Arango driver
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset # Sample graph form PyG

from adbpyg_adapter import ADBPYG_Adapter

db = ArangoClient(hosts="http://localhost:8529").db("_system", username="root", password="")

adbpyg_adapter = ADBPYG_Adapter(db, loggin_lvl=1)

homo_data = FakeDataset(edge_dim=1)[0]
hetero_data = FakeHeteroDataset(edge_dim=2)[0]

# Use Case 1: PyG to ArangoDB
adbpyg_adapter.pyg_to_arangodb("FakeHomoData", homo_data)
adbpyg_adapter.pyg_to_arangodb("FakeHeteroData", hetero_data)

# Use Case 2.1: ArangoDB to PyG via Graph name
pyg_homo = adbpyg_adapter.arangodb_graph_to_pyg("FakeHomoData")
pyg_hetero = adbpyg_adapter.arangodb_graph_to_pyg("FakeHeteroData")

# Use Case 2.2: ArangoDB to PyG via Collection names
pyg_homo = adbpyg_adapter.arangodb_collections_to_pyg("FakeHomoData", v_cols={'FakeHomoData_N'}, e_cols={'FakeHomoData_E'})
pyg_hetero = adbpyg_adapter.arangodb_collections_to_pyg("FakeHeteroData", v_cols={'v0', 'v1', 'v2'}, e_cols={'e0'})

# Use Case 2.3: ArangoDB to PyG via Metagraph
homo_metagraph = {
    "vertexCollections": {
        "FakeHomoData_N": {"x": "x", "y": "y"},
    },
    "edgeCollections": {
        "FakeHomoData_E": {"edge_weight": "edge_weight"},
    },
}
new_homo_data = adbpyg_adapter.arangodb_to_pyg("FakeHomoData", homo_metagraph)

hetero_metagraph = {
    "vertexCollections": {
        "v0": {"x": "x", "y": "y"},
        "v1": {"x": "x"},
        "v2": {"x": "x"},
    },
    "edgeCollections": {
        "e0": {"edge_attr": "edge_attr"},
    },
}
pyg_hetero = adbpyg_adapter.arangodb_to_pyg("FakeHeteroData", hetero_metagraph)
```

##  Development & Testing (Not Yet Ready)

Prerequisite: `arangorestore`

1. `git clone https://github.com/arangoml/pyg-adapter.git`
2. `cd pyg-adapter`
3. (create virtual environment of choice)
4. `pip install -e .[dev]`
5. (create an ArangoDB instance with method of choice)
6. `pytest --url <> --dbName <> --username <> --password <>`

**Note**: A `pytest` parameter can be omitted if the endpoint is using its default value:
```python
def pytest_addoption(parser):
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="")
```
