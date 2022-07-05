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
pip install torch adbpyg-adapter
```
#### Current State
```
pip install torch git+https://github.com/arangoml/pyg-adapter.git
```

##  Quickstart

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arangoml/pyg-adapter/blob/master/examples/ArangoDB_PyG_Adapter.ipynb)

```py
import logging
from arango import ArangoClient  # Python-Arango driver
from torch_geometric.datasets import FakeHeteroDataset # Sample graph form PyG

from adbpyg_adapter import ADBPYG_Adapter

# Let's assume that the ArangoDB "fraud detection" dataset is imported to this endpoint
db = ArangoClient(hosts="http://localhost:8529").db("_system", username="root", password="")

adbpyg_adapter = ADBPYG_Adapter(db, logging_lvl=logging.DEBUG)

# Use Case 1.1: ArangoDB to PyG via Graph name
pyg_fraud_graph = adbpyg_adapter.arangodb_graph_to_pyg("fraud-detection")

# Use Case 1.2: ArangoDB to PyG via Collection names
pyg_fraud_graph_2 = adbpyg_adapter.arangodb_collections_to_pyg(
    "fraud-detection",
    {"account", "Class", "customer"},  # Vertex collections
    {"accountHolder", "Relationship", "transaction"},  # Edge collections
)

# Use Case 1.3: ArangoDB to PyG via Metagraph
metagraph = { 
    "vertexCollections": {
        "account": {'x': 'features', 'y': 'label'},
        "bank": {'x': 'features'},
        "customer": {'x': 'features'},
    },
    "edgeCollections": {
        "accountHolder": {},
        "transaction": {'edge_attr': 'features'},
    },
}
pyg_fraud_graph_3 = adbpyg_adapter.arangodb_to_pyg("fraud-detection", metagraph)

# Use Case 2: PyG to ArangoDB
pyg_hetero_graph = FakeHeteroDataset()[0]
adb_hetero_graph = adbpyg_adapter.pyg_to_arangodb("FakeHeteroGraph", pyg_hetero_graph)
```

##  Development & Testing

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
