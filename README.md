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

The ArangoDB-PyG Adapter exports Graphs from ArangoDB, the multi-model database for graph & beyond, into PyTorch Geometric (PyG), a PyTorch-based Graph Neural Network library, and vice-versa.

## About PyG

**PyG** *(PyTorch Geometric)* is a library built upon [PyTorch](https://pytorch.org/) to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

It consists of various methods for deep learning on graphs and other irregular structures, also known as *[geometric deep learning](http://geometricdeeplearning.com/)*, from a variety of published papers.
In addition, it consists of easy-to-use mini-batch loaders for operating on many small and single giant graphs, [multi GPU-support](https://github.com/pyg-team/pytorch_geometric/tree/master/examples/multi_gpu), [`DataPipe` support](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/datapipe.py), distributed graph learning via [Quiver](https://github.com/pyg-team/pytorch_geometric/tree/master/examples/quiver), a large number of common benchmark datasets (based on simple interfaces to create your own), the [GraphGym](https://pytorch-geometric.readthedocs.io/en/latest/notes/graphgym.html) experiment manager, and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

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
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset # Sample graphs form PyG

from adbpyg_adapter import ADBPyG_Adapter
from adbpyg_adapter.utils import IdentityEncoder, EnumEncoder

# Let's assume that the ArangoDB "IMDB" dataset is imported to this endpoint
db = ArangoClient(hosts="http://localhost:8529").db("_system", username="root", password="")

adbpyg_adapter = ADBPyG_Adapter(db)

# Use Case 1: PyG to ArangoDB
data = FakeHeteroDataset(edge_dim=2)[0] # data = FakeDataset(edge_dim=1)[0]
adbpyg_adapter.pyg_to_arangodb("FakeData", data)

# Use Case 2.1: ArangoDB to PyG via Graph name
pyg_g = adbpyg_adapter.arangodb_graph_to_pyg("FakeData")

# Use Case 2.2: ArangoDB to PyG via Collection names
pyg_g = adbpyg_adapter.arangodb_collections_to_pyg("FakeData", v_cols={"v0", "v1", "v2"}, e_cols={"e0"})

# Use Case 2.3: ArangoDB to PyG via Metagraph v1 (ArangoDB attributes are already formatted to PyG data standards)
metagraph_v1 = {
    "vertexCollections": {
        "v0": {"x": "x", "y": "y"},
        "v1": {"x": "x"},
        "v2": {"x": "x"},
    },
    "edgeCollections": {
        "e0": {"edge_attr": "edge_attr"},
    },
}
pyg_hetero = adbpyg_adapter.arangodb_to_pyg("FakeData", metagraph_v1)

# Use Case 2.4: ArangoDB to PyG via Metagraph v2 (ArangoDB attributes are transformed to fit PyG data standards via user-defined Encoders)
metagraph_v2 = {
    "vertexCollections": {
        "Movies": {
            "x": {"Action": IdentityEncoder(dtype=long)},
            "y": {"Comedy": IdentityEncoder(dtype=long)},
        },
        "Users": {
            "x": {
                "Gender": EnumEncoder(mapping={"M": 0, "F": 1}),
                "Age": IdentityEncoder(dtype=long),
            }
        },
    },
    "edgeCollections": {
        "Ratings": {
            "edge_weight": {
                "Rating": IdentityEncoder(dtype=long),
            }
        }
    },
}
pyg_imdb = adbpyg_adapter.arangodb_to_pyg("IMDB", metagraph_v2)
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
