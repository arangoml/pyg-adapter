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


<a href="https://www.arangodb.com/" rel="arangodb.com">![](https://raw.githubusercontent.com/arangoml/pyg-adapter/master/examples/assets/adb_logo.png)</a>
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
import torch
import pandas
from torch_geometric.datasets import FakeHeteroDataset

from arango import ArangoClient  # Python-Arango driver

from adbpyg_adapter import ADBPyG_Adapter, ADBPyG_Controller
from adbpyg_adapter.encoders import IdentityEncoder, CategoricalEncoder

# Load some fake PyG data for demo purposes
data = FakeHeteroDataset(
    num_node_types=2,
    num_edge_types=3,
    avg_num_nodes=20,
    avg_num_channels=3,  # avg number of features per node
    edge_dim=2,  # number of features per edge
    num_classes=3,  # number of unique label values
)[0]

# Let's assume that the ArangoDB "IMDB" dataset is imported to this endpoint
db = ArangoClient(hosts="http://localhost:8529").db("_system", username="root", password="")

adbpyg_adapter = ADBPyG_Adapter(db)
```

### PyG to ArangoDB
```py
# 1.1: PyG to ArangoDB
adb_g = adbpyg_adapter.pyg_to_arangodb("FakeData", data)

# 1.2: PyG to ArangoDB with a (completely optional) metagraph for customized adapter behaviour
def y_tensor_to_2_column_dataframe(pyg_tensor):
    """
    A user-defined function to create two
    ArangoDB attributes out of the 'y' label tensor

    NOTE: user-defined functions must return a Pandas Dataframe
    """
    label_map = {0: "Kiwi", 1: "Blueberry", 2: "Avocado"}

    df = pandas.DataFrame(columns=["label_num", "label_str"])
    df["label_num"] = pyg_tensor.tolist()
    df["label_str"] = df["label_num"].map(label_map)

    return df


metagraph = {
    "nodeTypes": {
        "v0": {
            "x": "features",  # 1) you can specify a string value for attribute renaming
            "y": y_tensor_to_2_column_dataframe,  # 2) you can specify a function for user-defined handling, as long as the function returns a Pandas DataFrame
        },
    },
    "edgeTypes": {
        ("v0", "e0", "v0"): {
            # 3) you can specify a list of strings for tensor dissasembly (if you know the number of node/edge features in advance)
            "edge_attr": [ "a", "b"]  
        },
    },
}


adb_g = adbpyg_adapter.pyg_to_arangodb("FakeData", data, metagraph, explicit_metagraph=False)

# 1.3: PyG to ArangoDB with the same (optional) metagraph, but with `explicit_metagraph=True`
# With `explicit_metagraph=True`, the node & edge types omitted from the metagraph will NOT be converted to ArangoDB.
# Only 'v0' and ('v0', 'e0', 'v0') will be brought over (i.e 'v1', ('v0', 'e0', 'v1'), ... are ignored)
adb_g = adbpyg_adapter.pyg_to_arangodb("FakeData", data, metagraph, explicit_metagraph=True)

# 1.4: PyG to ArangoDB with a Custom Controller  (more user-defined behavior)
class Custom_ADBPyG_Controller(ADBPyG_Controller):
    def _prepare_pyg_node(self, pyg_node: dict, col: str) -> dict:
        """Optionally modify a PyG node object before it gets inserted into its designated ArangoDB collection.

        :param pyg_node: The PyG node object to (optionally) modify.
        :param col: The ArangoDB collection the PyG node belongs to.
        :return: The PyG Node object
        """
        pyg_node["foo"] = "bar"
        return pyg_node

    def _prepare_pyg_edge(self, pyg_edge: dict, edge_type: tuple) -> dict:
        """Optionally modify a PyG edge object before it gets inserted into its designated ArangoDB collection.

        :param pyg_edge: The PyG edge object to (optionally) modify.
        :param edge_type: The Edge Type of the PyG edge. Formatted
            as (from_collection, edge_collection, to_collection)
        :return: The PyG Edge object
        """
        pyg_edge["bar"] = "foo"
        return pyg_edge


adb_g = ADBPyG_Adapter(db, Custom_ADBPyG_Controller()).pyg_to_arangodb("FakeData", data)
```

### ArangoDB to PyG
```py
# Start from scratch!
adbpyg_adapter.pyg_to_arangodb("FakeData", data, overwrite_graph=True, on_duplicate="replace")

# 2.1: ArangoDB to PyG via Graph name (does not transfer attributes)
pyg_g = adbpyg_adapter.arangodb_graph_to_pyg("FakeData")

# 2.2: ArangoDB to PyG via Collection names (does not transfer attributes)
pyg_g = adbpyg_adapter.arangodb_collections_to_pyg("FakeData", v_cols={"v0", "v1"}, e_cols={"e0"})

# 2.3: ArangoDB to PyG via Metagraph v1 (transfer attributes "as is", meaning they are already formatted to PyG data standards)
metagraph_v1 = {
    "vertexCollections": {
        # we instruct the adapter to create the "x" and "y" tensor data from the "x" and "y" ArangoDB attributes
        "v0": { "x": "x", "y": "y"},  
        "v1": {"x": "x"},
    },
    "edgeCollections": {
        "e0": {"edge_attr": "edge_attr"},
    },
}
pyg_g = adbpyg_adapter.arangodb_to_pyg("FakeData", metagraph_v1)

# 2.4: ArangoDB to PyG via Metagraph v2 (transfer attributes via user-defined encoders)
# For more info on user-defined encoders in PyG, see https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
metagraph_v2 = {
    "vertexCollections": {
        "Movies": {
            "x": {  # Build a feature matrix from the "Action" & "Drama" document attributes
                "Action": IdentityEncoder(dtype=torch.long),
                "Drama": IdentityEncoder(dtype=torch.long),
            },
            "y": "Comedy",
        },
        "Users": {
            "x": {
                "Gender": CategoricalEncoder(mapping={"M": 0, "F": 1}),
                "Age": IdentityEncoder(dtype=torch.long),
            }
        },
    },
    "edgeCollections": {
        "Ratings": {
            "edge_weight": "Rating"
        }
    },
}
pyg_g = adbpyg_adapter.arangodb_to_pyg("IMDB", metagraph_v2)

# 2.5: ArangoDB to PyG via Metagraph v3 (transfer attributes via user-defined functions)
def udf_v0_x(v0_df):
    # process v0_df here to return v0 "x" feature matrix
    # v0_df["x"] = ...
    return torch.tensor(v0_df["x"].to_list())


def udf_v1_x(v1_df):
    # process v1_df here to return v1 "x" feature matrix
    # v1_df["x"] = ...
    return torch.tensor(v1_df["x"].to_list())


metagraph_v3 = {
    "vertexCollections": {
        "v0": {
            "x": udf_v0_x,  # supports named functions
            "y": lambda df: torch.tensor(df["y"].to_list()),  # also supports lambda functions
        },
        "v1": {"x": udf_v1_x},
    },
    "edgeCollections": {
        "e0": {"edge_attr": (lambda df: torch.tensor(df["edge_attr"].to_list()))},
    },
}
pyg_g = adbpyg_adapter.arangodb_to_pyg("FakeData", metagraph_v3)
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
