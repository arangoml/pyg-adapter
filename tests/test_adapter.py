from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Union

import pytest
from pandas import DataFrame
from torch import Tensor, cat, long, tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType

from adbpyg_adapter import ADBPyG_Adapter
from adbpyg_adapter.encoders import CategoricalEncoder, IdentityEncoder
from adbpyg_adapter.exceptions import (
    ADBMetagraphError,
    InvalidADBEdgesError,
    PyGMetagraphError,
)
from adbpyg_adapter.typings import (
    ADBMap,
    ADBMetagraph,
    ADBMetagraphValues,
    Json,
    PyGMap,
    PyGMetagraph,
    PyGMetagraphValues,
)
from adbpyg_adapter.utils import validate_adb_metagraph, validate_pyg_metagraph

from .conftest import (
    Custom_ADBPyG_Controller,
    adbpyg_adapter,
    arango_restore,
    con,
    db,
    get_fake_hetero_graph,
    get_fake_homo_graph,
    get_karate_graph,
    get_social_graph,
    tracer,
    udf_key_df_to_tensor,
    udf_users_x_tensor_to_df,
    udf_v2_x_tensor_to_df,
    udf_x_df_to_tensor,
)


def test_validate_constructor() -> None:
    bad_db: Dict[str, Any] = dict()

    class Bad_ADBPyG_Controller:
        pass

    with pytest.raises(TypeError):
        ADBPyG_Adapter(bad_db)

    with pytest.raises(TypeError):
        ADBPyG_Adapter(db, Bad_ADBPyG_Controller())  # type:ignore[arg-type]


@pytest.mark.parametrize(
    "bad_metagraph",
    [  # empty metagraph
        ({}),
        # missing required parent key
        ({"edgeCollections": {}}),
        # empty sub-metagraph
        ({"vertexCollections": {}}),
        # bad collection name
        (
            {
                "vertexCollections": {
                    1: {},
                    # other examples include:
                    # True: {},
                    # ('a'): {}
                }
            }
        ),
        # bad collection metagraph
        (
            {
                "vertexCollections": {
                    "vcol_a": None,
                    # other examples include:
                    # "vcol_a": 1,
                    # "vcol_a": 'foo',
                }
            }
        ),
        # bad collection metagraph 2
        (
            {
                "vertexCollections": {
                    "vcol_a": {"a", "b", 3},
                    # other examples include:
                    # "vcol_a": 1,
                    # "vcol_a": 'foo',
                },
                "edgeCollections": {},
            }
        ),
        # bad meta_key
        (
            {
                "vertexCollections": {
                    "vcol_a": {
                        1: {},
                        # other example include:
                        # True: {},
                        # ("x"): {},
                    }
                }
            }
        ),
        # bad meta_val
        (
            {
                "vertexCollections": {
                    "vcol_a": {
                        "x": True,
                        # other example include:
                        # 'x': ('a'),
                        # 'x': ['a'],
                        # 'x': 5
                    }
                }
            }
        ),
        # bad meta_val encoder key
        ({"vertexCollections": {"vcol_a": {"x": {1: IdentityEncoder()}}}}),
        # bad meta_val encoder value
        (
            {
                "vertexCollections": {
                    "vcol_a": {
                        "x": {
                            "Action": True,
                            # other examples include:
                            # 'Action': {}
                            # 'Action': (lambda : 1)()
                        }
                    }
                }
            }
        ),
    ],
)
def test_validate_adb_metagraph(bad_metagraph: Dict[Any, Any]) -> None:
    with pytest.raises(ADBMetagraphError):
        validate_adb_metagraph(bad_metagraph)


@pytest.mark.parametrize(
    "bad_metagraph",
    [
        # bad node type
        (
            {
                "nodeTypes": {
                    ("a", "b", "c"): {},
                    # other examples include:
                    # 1: {},
                    # True: {}
                }
            }
        ),
        # bad edge type
        (
            {
                "edgeTypes": {
                    "b": {},
                    # other examples include:
                    # 1: {},
                    # True: {}
                }
            }
        ),
        # bad edge type 2
        (
            {
                "edgeTypes": {
                    ("a", "b", 3): {},
                    # other examples include:
                    # 1: {},
                    # True: {}
                }
            }
        ),
        # bad data type metagraph
        (
            {
                "nodeTypes": {
                    "ntype_a": None,
                    # other examples include:
                    # "ntype_a": 1,
                    # "ntype_a": 'foo',
                }
            }
        ),
        # bad data type metagraph 2
        ({"nodeTypes": {"ntype_a": {"a", "b", 3}}}),
        # bad meta_val
        (
            {
                "nodeTypes": {
                    "ntype_a'": {
                        "x": True,
                        # other example include:
                        # 'x': ('a'),
                        # 'x': (lambda: 1)(),
                    }
                }
            }
        ),
        # bad meta_val list
        (
            {
                "nodeTypes": {
                    "ntype_a'": {
                        "x": ["a", 3],
                        # other example include:
                        # 'x': ('a'),
                        # 'x': (lambda: 1)(),
                    }
                }
            }
        ),
    ],
)
def test_validate_pyg_metagraph(bad_metagraph: Dict[Any, Any]) -> None:
    with pytest.raises(PyGMetagraphError):
        validate_pyg_metagraph(bad_metagraph)


@pytest.mark.parametrize(
    "adapter, name, pyg_g, metagraph, \
        explicit_metagraph, overwrite_graph, batch_size, adb_import_kwargs",
    [
        (
            adbpyg_adapter,
            "Karate_1",
            get_karate_graph(),
            {"nodeTypes": {"Karate_1_N": {"x": "node_features"}}},
            False,
            False,
            33,
            {},
        ),
        (
            adbpyg_adapter,
            "Karate_2",
            get_karate_graph(),
            {"nodeTypes": {"Karate_2_N": {"x": "node_features"}}},
            True,
            False,
            1000,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_1",
            get_fake_homo_graph(avg_num_nodes=3),
            {"nodeTypes": {"FakeHomoGraph_1_N": {"y": "label"}}},
            False,
            False,
            1,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_2",
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=1),
            {},
            False,
            False,
            1000,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_3",
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=1),
            {},
            True,
            False,
            None,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_4",
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=2),
            {
                "nodeTypes": {"FakeHomoGraph_4_N": {"y": "label"}},
                "edgeTypes": {
                    ("FakeHomoGraph_4_N", "FakeHomoGraph_4_E", "FakeHomoGraph_4_N"): {
                        "edge_attr": "features"
                    }
                },
            },
            True,
            False,
            None,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_5",
            get_fake_homo_graph(avg_num_nodes=3),
            {
                "edgeTypes": {
                    ("FakeHomoGraph_5_N", "FakeHomoGraph_5_E", "FakeHomoGraph_5_N"): {}
                },
            },
            True,
            False,
            None,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_1",
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=1),
            {},
            False,
            False,
            1,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_2",
            get_fake_hetero_graph(avg_num_nodes=2),
            {"nodeTypes": {"v2": {"x": udf_v2_x_tensor_to_df}}},
            True,
            False,
            1000,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_3",
            get_fake_hetero_graph(avg_num_nodes=2),
            {"nodeTypes": {"v0": {"x", "y"}, "v2": {"x"}}},
            True,
            False,
            None,
            {},
        ),
        (
            adbpyg_adapter,
            "SocialGraph",
            get_social_graph(),
            {"nodeTypes": {"user": {"x": ["age", "gender"]}}},
            False,
            True,
            None,
            {},
        ),
    ],
)
def test_pyg_to_adb(
    adapter: ADBPyG_Adapter,
    name: str,
    pyg_g: Union[Data, HeteroData],
    metagraph: PyGMetagraph,
    explicit_metagraph: bool,
    overwrite_graph: bool,
    batch_size: Optional[int],
    adb_import_kwargs: Any,
) -> None:
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adapter.pyg_to_arangodb(
        name,
        pyg_g,
        metagraph,
        explicit_metagraph,
        overwrite_graph,
        batch_size,
        **adb_import_kwargs,
    )
    assert_pyg_to_adb(name, pyg_g, metagraph, explicit_metagraph)
    db.delete_graph(name, drop_collections=True)


def test_pyg_to_adb_ambiguity_error() -> None:
    d = Data(edge_index=tensor([[0, 1], [1, 0]]))
    with pytest.raises(ValueError):
        adbpyg_adapter.pyg_to_arangodb("graph", d)


def test_pyg_to_adb_with_controller() -> None:
    name = "Karate_3"
    data = get_karate_graph()
    db.delete_graph(name, drop_collections=True, ignore_missing=True)

    ADBPyG_Adapter(db, Custom_ADBPyG_Controller()).pyg_to_arangodb(name, data)

    for doc in db.collection(f"{name}_N"):
        assert "foo" in doc
        assert doc["foo"] == "bar"

    for edge in db.collection(f"{name}_E"):
        assert "bar" in edge
        assert edge["bar"] == "foo"

    db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize(
    "adapter, name, metagraph, pyg_g_old, batch_size",
    [
        (
            adbpyg_adapter,
            "Karate",
            {
                "vertexCollections": {
                    "Karate_N": {"x": "x", "y": "y"},
                },
                "edgeCollections": {
                    "Karate_E": {},
                },
            },
            get_karate_graph(),
            33,
        ),
        (
            adbpyg_adapter,
            "Homogoneous",
            {
                "vertexCollections": {
                    "Homogoneous_N": {"x": "x", "y": "y"},
                },
                "edgeCollections": {
                    "Homogoneous_E": {"edge_weight": "edge_weight"},
                },
            },
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=1),
            1,
        ),
        (
            adbpyg_adapter,
            "Heterogeneous",
            {
                "vertexCollections": {
                    "v0": {"x": "x", "y": "y"},
                    "v1": {"x": "x"},
                    "v2": {"x": "x"},
                },
                "edgeCollections": {
                    "e0": {"edge_attr": "edge_attr"},
                },
            },
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=2),
            1000,
        ),
        (
            adbpyg_adapter,
            "HeterogeneousSimpleMetagraph",
            {
                "vertexCollections": {
                    "v0": {"x", "y"},
                    "v1": {"x"},
                    "v2": {"x"},
                },
                "edgeCollections": {
                    "e0": {"edge_attr"},
                },
            },
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=2),
            None,
        ),
        (
            adbpyg_adapter,
            "HeterogeneousOverComplicatedMetagraph",
            {
                "vertexCollections": {
                    "v0": {"x": {"x": None}, "y": {"y": None}},
                    "v1": {"x": "x"},
                    "v2": {"x": {"x": None}},
                },
                "edgeCollections": {
                    "e0": {"edge_attr": {"edge_attr": None}},
                },
            },
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=2),
            None,
        ),
        (
            adbpyg_adapter,
            "HeterogeneousUserDefinedFunctions",
            {
                "vertexCollections": {
                    "v0": {
                        "x": (lambda df: tensor(df["x"].to_list())),
                        "y": (lambda df: tensor(df["y"].to_list())),
                    },
                    "v1": {"x": udf_x_df_to_tensor},
                    "v2": {"x": udf_key_df_to_tensor("x")},
                },
                "edgeCollections": {
                    "e0": {"edge_attr": (lambda df: tensor(df["edge_attr"].to_list()))},
                },
            },
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=2),
            None,
        ),
    ],
)
def test_adb_to_pyg(
    adapter: ADBPyG_Adapter,
    name: str,
    metagraph: ADBMetagraph,
    pyg_g_old: Optional[Union[Data, HeteroData]],
    batch_size: Optional[int],
) -> None:
    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.pyg_to_arangodb(name, pyg_g_old)

    pyg_g_new = adapter.arangodb_to_pyg(name, metagraph, batch_size=batch_size)
    assert_adb_to_pyg(pyg_g_new, metagraph)

    if pyg_g_old:
        db.delete_graph(name, drop_collections=True)


def test_adb_partial_to_pyg() -> None:
    # Generate a valid pyg_g graph
    pyg_g = get_fake_hetero_graph(avg_num_nodes=2, edge_dim=2)
    e_t = ("v0", "e0", "v0")
    while e_t not in pyg_g.edge_types:
        pyg_g = get_fake_hetero_graph(avg_num_nodes=2, edge_dim=2)

    name = "Heterogeneous"
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adbpyg_adapter.pyg_to_arangodb(name, pyg_g)

    metagraph: ADBMetagraph

    # Case 1: Partial edge collection import turns the graph homogeneous
    metagraph = {
        "vertexCollections": {
            "v0": {"x": "x", "y": "y"},
        },
        "edgeCollections": {
            "e0": {"edge_attr": "edge_attr"},
        },
    }

    pyg_g_new = adbpyg_adapter.arangodb_to_pyg(
        "HeterogeneousTurnedHomogeneous", metagraph
    )

    assert type(pyg_g_new) is Data
    assert pyg_g["v0"].x.tolist() == pyg_g_new.x.tolist()
    assert pyg_g["v0"].y.tolist() == pyg_g_new.y.tolist()
    assert pyg_g[e_t].edge_index.tolist() == pyg_g_new.edge_index.tolist()
    assert pyg_g[e_t].edge_attr.tolist() == pyg_g_new.edge_attr.tolist()

    # Case 2: Partial edge collection import keeps the graph heterogeneous
    metagraph = {
        "vertexCollections": {
            "v0": {"x": "x", "y": "y"},
            "v1": {"x": "x"},
        },
        "edgeCollections": {
            "e0": {"edge_attr": "edge_attr"},
        },
    }

    pyg_g_new = adbpyg_adapter.arangodb_to_pyg(
        "HeterogeneousWithOneLessNodeType", metagraph
    )

    assert type(pyg_g_new) is HeteroData
    assert set(pyg_g_new.node_types) == {"v0", "v1"}
    for n_type in pyg_g_new.node_types:
        for k, v in pyg_g_new[n_type].items():
            assert v.tolist() == pyg_g[n_type][k].tolist()

    for e_type in pyg_g_new.edge_types:
        for k, v in pyg_g_new[e_type].items():
            assert v.tolist() == pyg_g[e_type][k].tolist()

    db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize(
    "adapter, name, v_cols, e_cols, pyg_g_old",
    [
        (
            adbpyg_adapter,
            "SocialGraph",
            {"user", "game"},
            {"plays", "follows"},
            get_social_graph(),
        )
    ],
)
def test_adb_collections_to_pyg(
    adapter: ADBPyG_Adapter,
    name: str,
    v_cols: Set[str],
    e_cols: Set[str],
    pyg_g_old: Union[Data, HeteroData],
) -> None:
    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.pyg_to_arangodb(name, pyg_g_old)

    pyg_g_new = adapter.arangodb_collections_to_pyg(
        name,
        v_cols,
        e_cols,
    )

    # Manually set the number of nodes (since nodes are feature-less)
    for v_col in v_cols:
        if pyg_g_old:
            pyg_g_new[v_col].num_nodes = pyg_g_old[v_col].num_nodes
        else:
            pyg_g_new[v_col].num_nodes = db.collection(v_col).count()

    assert_adb_to_pyg(
        pyg_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    if pyg_g_old:
        db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize(
    "adapter, name, pyg_g_old",
    [
        (adbpyg_adapter, "Heterogeneous", get_fake_hetero_graph(avg_num_nodes=2)),
    ],
)
def test_adb_graph_to_pyg(
    adapter: ADBPyG_Adapter, name: str, pyg_g_old: Union[Data, HeteroData]
) -> None:
    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.pyg_to_arangodb(name, pyg_g_old)

    pyg_g_new = adapter.arangodb_graph_to_pyg(name)

    graph = db.graph(name)
    v_cols: Set[str] = graph.vertex_collections()
    edge_definitions: List[Json] = graph.edge_definitions()
    e_cols: Set[str] = {c["edge_collection"] for c in edge_definitions}

    # Manually set the number of nodes (since nodes are feature-less)
    for v_col in v_cols:
        if pyg_g_old:
            pyg_g_new[v_col].num_nodes = pyg_g_old[v_col].num_nodes
        else:
            pyg_g_new[v_col].num_nodes = db.collection(v_col).count()

    assert_adb_to_pyg(
        pyg_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    if pyg_g_old:
        db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize("adapter", [adbpyg_adapter])
def test_adb_graph_to_pyg_to_arangodb_with_missing_document_and_strict(
    adapter: ADBPyG_Adapter,
) -> None:
    name = "Karate_3"
    data = get_karate_graph()
    db.delete_graph(name, drop_collections=True, ignore_missing=True)

    ADBPyG_Adapter(db, tracer=tracer).pyg_to_arangodb(name, data)

    graph = db.graph(name)
    v_cols: Set[str] = graph.vertex_collections()
    edge_definitions: List[Json] = graph.edge_definitions()
    e_cols: Set[str] = {c["edge_collection"] for c in edge_definitions}

    for v_col in v_cols:
        vertex_collection = db.collection(v_col)
        vertex_collection.delete("0")

    metagraph: ADBMetagraph = {
        "vertexCollections": {col: {} for col in v_cols},
        "edgeCollections": {col: {} for col in e_cols},
    }

    with pytest.raises(InvalidADBEdgesError):
        adapter.arangodb_to_pyg(name, metagraph=metagraph, strict=True)

    db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize("adapter", [adbpyg_adapter])
def test_adb_graph_to_pyg_to_arangodb_with_missing_document_and_permissive(
    adapter: ADBPyG_Adapter,
) -> None:
    name = "Karate_3"
    data = get_karate_graph()
    db.delete_graph(name, drop_collections=True, ignore_missing=True)

    ADBPyG_Adapter(db, tracer=tracer).pyg_to_arangodb(name, data)

    graph = db.graph(name)
    v_cols: Set[str] = graph.vertex_collections()
    edge_definitions: List[Json] = graph.edge_definitions()
    e_cols: Set[str] = {c["edge_collection"] for c in edge_definitions}

    for v_col in v_cols:
        vertex_collection = db.collection(v_col)
        vertex_collection.delete("0")

    metagraph: ADBMetagraph = {
        "vertexCollections": {col: {} for col in v_cols},
        "edgeCollections": {col: {} for col in e_cols},
    }

    data = adapter.arangodb_to_pyg(name, metagraph=metagraph, strict=False)

    collection_count: int = db.collection(list(e_cols)[0]).count()
    assert len(data.edge_index[0]) < collection_count

    db.delete_graph(name, drop_collections=True)


def test_full_cycle_imdb_without_preserve_adb_keys() -> None:
    name = "imdb"
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    arango_restore(con, "tests/data/adb/imdb_dump")
    db.create_graph(
        name,
        edge_definitions=[
            {
                "edge_collection": "Ratings",
                "from_vertex_collections": ["Users"],
                "to_vertex_collections": ["Movies"],
            },
        ],
    )

    adb_to_pyg_metagraph: ADBMetagraph = {
        "vertexCollections": {
            "Movies": {
                "y": "Comedy",
                "x": {
                    "Action": IdentityEncoder(dtype=long),
                    "Drama": IdentityEncoder(dtype=long),
                    # etc....
                },
            },
            "Users": {
                "x": {
                    "Age": IdentityEncoder(dtype=long),
                    "Gender": CategoricalEncoder(),
                }
            },
        },
        "edgeCollections": {"Ratings": {"edge_weight": "Rating"}},
    }

    pyg_g = adbpyg_adapter.arangodb_to_pyg(name, adb_to_pyg_metagraph)
    assert_adb_to_pyg(pyg_g, adb_to_pyg_metagraph)

    pyg_to_adb_metagraph: PyGMetagraph = {
        "nodeTypes": {
            "Movies": {
                "y": "comedy",
                "x": ["action", "drama"],
            },
            "Users": {"x": udf_users_x_tensor_to_df},
        },
        "edgeTypes": {("Users", "Ratings", "Movies"): {"edge_weight": "rating"}},
    }
    adbpyg_adapter.pyg_to_arangodb(name, pyg_g, pyg_to_adb_metagraph, overwrite=True)
    assert_pyg_to_adb(name, pyg_g, pyg_to_adb_metagraph)

    db.delete_graph(name, drop_collections=True)


def test_full_cycle_homogeneous_with_preserve_adb_keys() -> None:
    d = get_fake_homo_graph(avg_num_nodes=20, num_channels=2)

    # Get Fake Data in ArangoDB
    name = "Homogeneous"
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adbpyg_adapter.pyg_to_arangodb(name, d)

    pyg_g = adbpyg_adapter.arangodb_graph_to_pyg(name, preserve_adb_keys=True)

    graph = db.graph(name)
    v_cols: Set[str] = graph.vertex_collections()
    edge_definitions: List[Json] = graph.edge_definitions()
    e_cols: Set[str] = {c["edge_collection"] for c in edge_definitions}

    metagraph: ADBMetagraph = {
        "vertexCollections": {col: {} for col in v_cols},
        "edgeCollections": {col: {} for col in e_cols},
    }
    assert_adb_to_pyg(pyg_g, metagraph, True)
    assert "_v_key" in pyg_g and "_e_key" in pyg_g

    num_nodes = d.num_nodes
    pyg_g["_v_key"].append(f"new-vertex-{num_nodes}")
    pyg_g.num_nodes = num_nodes + 1

    adbpyg_adapter.pyg_to_arangodb(name, pyg_g, on_duplicate="update")
    assert_pyg_to_adb(name, pyg_g, {}, False)
    assert db.collection("Homogeneous_N").get(f"new-vertex-{num_nodes}") is not None

    db.delete_graph(name, drop_collections=True, ignore_missing=True)

def test_full_cycle_imdb_with_preserve_adb_keys() -> None:
    name = "imdb"
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    arango_restore(con, "tests/data/adb/imdb_dump")
    db.create_graph(
        name,
        edge_definitions=[
            {
                "edge_collection": "Ratings",
                "from_vertex_collections": ["Users"],
                "to_vertex_collections": ["Movies"],
            },
        ],
    )

    adb_to_pyg_metagraph: ADBMetagraph = {
        "vertexCollections": {
            "Movies": {
                "y": "Comedy",  # { "Comedy": IdentityEncoder(dtype=long) }
                "x": {
                    "Action": IdentityEncoder(dtype=long),
                    "Drama": IdentityEncoder(dtype=long),
                    # etc....
                },
            },
            "Users": {
                "x": {
                    "Age": IdentityEncoder(dtype=long),
                    "Gender": CategoricalEncoder(),
                }
            },
        },
        "edgeCollections": {"Ratings": {"edge_weight": "Rating"}},
    }

    pyg_g = adbpyg_adapter.arangodb_to_pyg(
        name, adb_to_pyg_metagraph, preserve_adb_keys=True
    )
    assert_adb_to_pyg(pyg_g, adb_to_pyg_metagraph, True)

    # Add PyG User Node & update the _key property
    pyg_g["Users"].x = cat((pyg_g["Users"].x, tensor([[99, 1]])), 0)
    pyg_g["Users"]["_key"].append("new-user-944")

    # (coverage testing) Add _id property to Movies
    # There's no point in having both _key and _id at the same time,
    # but it is possible that a user prefers to have `preserve_adb_keys=False`,
    # and build their own _key or _id list. The following line tries to simulate
    # that while still adhering to the IMDB graph structure.
    pyg_g["Movies"]["_id"] = ["Movies/" + k for k in pyg_g["Movies"]["_key"]]

    pyg_to_adb_metagraph: PyGMetagraph = {
        "nodeTypes": {
            "Users": {"x": ["Age", "Gender"], "_key": "_key"},
            "Movies": {"_id"},  # Note: we can either use _id or _key here
        },
        "edgeTypes": {("Users", "Ratings", "Movies"): {"_key"}},
    }

    adbpyg_adapter.pyg_to_arangodb(
        name,
        pyg_g,
        pyg_to_adb_metagraph,
        explicit_metagraph=True,
        on_duplicate="update",
    )
    assert_pyg_to_adb(name, pyg_g, pyg_to_adb_metagraph, True)

    assert db.collection("Users").get("new-user-944") is not None

    db.delete_graph(name, drop_collections=True)

def assert_pyg_to_adb(
    name: str,
    pyg_g: Union[Data, HeteroData],
    metagraph: PyGMetagraph,
    explicit_metagraph: bool = False,
) -> None:
    is_homogeneous = type(pyg_g) is Data

    # Maps PyG Node ids to ArangoDB Vertex _keys
    pyg_map: PyGMap = defaultdict(dict)

    node_types: List[str]
    edge_types: List[EdgeType]
    explicit_metagraph = metagraph != {} and explicit_metagraph
    if explicit_metagraph:
        node_types = metagraph.get("nodeTypes", {}).keys()  # type: ignore
        edge_types = metagraph.get("edgeTypes", {}).keys()  # type: ignore

    elif is_homogeneous:
        n_type = name + "_N"
        node_types = [n_type]
        edge_types = [(n_type, name + "_E", n_type)]

    else:
        node_types = pyg_g.node_types
        edge_types = pyg_g.edge_types

    n_meta = metagraph.get("nodeTypes", {})
    for n_type in node_types:
        node_data = pyg_g if is_homogeneous else pyg_g[n_type]
        collection = db.collection(n_type)
        assert collection.count() == node_data.num_nodes

        df = DataFrame(collection.all())
        pyg_map[n_type] = df["_id"].to_dict()

        if "_key" in node_data:  # preserve_adb_keys = True
            assert node_data["_key"] == df["_key"].tolist()

        meta = n_meta.get(n_type, {})
        assert_pyg_to_adb_meta(df, meta, node_data, explicit_metagraph)

    e_meta = metagraph.get("edgeTypes", {})
    for e_type in edge_types:
        edge_data: EdgeStorage = pyg_g if is_homogeneous else pyg_g[e_type]
        from_col, e_col, to_col = e_type
        collection = db.collection(e_col)

        df = DataFrame(collection.all())
        df[["from_col", "from_key"]] = df["_from"].str.split(pat="/", n=1, expand=True)
        df[["to_col", "to_key"]] = df["_to"].str.split(pat="/", n=1, expand=True)

        et_df = df[(df["from_col"] == from_col) & (df["to_col"] == to_col)]
        assert len(et_df) == edge_data.num_edges

        from_nodes = edge_data.edge_index[0].tolist()
        to_nodes = edge_data.edge_index[1].tolist()

        if pyg_map[from_col]:
            assert [pyg_map[from_col][n] for n in from_nodes] == et_df["_from"].tolist()

        if pyg_map[to_col]:
            assert [pyg_map[to_col][n] for n in to_nodes] == et_df["_to"].tolist()

        if "_key" in edge_data:  # preserve_adb_keys = True
            assert edge_data["_key"] == et_df["_key"].tolist()

        meta = e_meta.get(e_type, {})
        assert_pyg_to_adb_meta(et_df, meta, edge_data, explicit_metagraph)


def assert_pyg_to_adb_meta(
    df: DataFrame,
    meta: Union[Set[str], Dict[Any, PyGMetagraphValues]],
    pyg_data: Union[NodeStorage, EdgeStorage],
    explicit_metagraph: bool,
) -> None:
    valid_meta: Dict[Any, PyGMetagraphValues]
    valid_meta = meta if type(meta) is dict else {m: m for m in meta}

    if explicit_metagraph:
        pyg_keys = set(valid_meta.keys())
    else:
        pyg_keys = set(k for k, _ in pyg_data.items())

    for k in pyg_keys:
        if k == "edge_index":
            continue

        meta_val = valid_meta.get(k, str(k))
        data = pyg_data[k]

        if type(data) is list and len(data) == len(df) and type(meta_val) is str:
            if meta_val in ["_v_key", "_e_key"]:  # Homogeneous situation
                meta_val = "_key"

            assert meta_val in df
            assert df[meta_val].tolist() == data

        if type(data) is Tensor and len(data) == len(df):
            if type(meta_val) is str:
                assert meta_val in df
                assert df[meta_val].tolist() == data.tolist()

            if type(meta_val) is list:
                assert all([e in df for e in meta_val])
                assert df[meta_val].values.tolist() == data.tolist()

            if callable(meta_val):
                udf_df = meta_val(data, DataFrame(index=range(len(data))))
                assert all([column in df for column in udf_df.columns])
                for column in udf_df.columns:
                    assert df[column].tolist() == udf_df[column].tolist()


def assert_adb_to_pyg(
    pyg_g: Union[Data, HeteroData],
    metagraph: ADBMetagraph,
    preserve_adb_keys: bool = False,
) -> None:
    is_homogeneous = (
        len(metagraph["vertexCollections"]) == 1
        and len(metagraph["edgeCollections"]) == 1
    )

    # Maps ArangoDB Vertex _keys to PyG Node ids
    adb_map: ADBMap = defaultdict(dict)

    for v_col, meta in metagraph["vertexCollections"].items():
        node_data: NodeStorage
        if is_homogeneous:
            node_data = pyg_g
        else:
            assert v_col in pyg_g.node_types
            node_data = pyg_g[v_col]

        collection = db.collection(v_col)
        assert node_data.num_nodes == collection.count()

        df = DataFrame(collection.all())
        adb_map[v_col] = {adb_id: pyg_id for pyg_id, adb_id in enumerate(df["_key"])}

        if preserve_adb_keys:
            k = "_v_key" if is_homogeneous else "_key"
            assert k in node_data

            data = df["_key"].tolist()
            assert len(data) == len(node_data[k])
            assert data == node_data[k]

        assert_adb_to_pyg_meta(meta, df, node_data)

    et_df: DataFrame
    v_cols: List[str] = list(metagraph["vertexCollections"].keys())
    for e_col, meta in metagraph["edgeCollections"].items():
        collection = db.collection(e_col)
        assert collection.count() <= pyg_g.num_edges

        df = DataFrame(collection.all())
        df[["from_col", "from_key"]] = df["_from"].str.split(pat="/", n=1, expand=True)
        df[["to_col", "to_key"]] = df["_to"].str.split(pat="/", n=1, expand=True)

        for (from_col, to_col), count in (
            df[["from_col", "to_col"]].value_counts().items()
        ):
            edge_type = (from_col, e_col, to_col)
            if from_col not in v_cols or to_col not in v_cols:
                continue

            edge_data: EdgeStorage
            if is_homogeneous:
                edge_data = pyg_g
            else:
                assert edge_type in pyg_g.edge_types
                edge_data = pyg_g[edge_type]

            assert count == edge_data.num_edges

            et_df = df[(df["from_col"] == from_col) & (df["to_col"] == to_col)]
            from_nodes = et_df["from_key"].map(adb_map[from_col]).tolist()
            to_nodes = et_df["to_key"].map(adb_map[to_col]).tolist()

            assert from_nodes == edge_data.edge_index[0].tolist()
            assert to_nodes == edge_data.edge_index[1].tolist()

            if preserve_adb_keys:
                k = "_e_key" if is_homogeneous else "_key"
                assert k in edge_data

                data = et_df["_key"].tolist()
                assert len(data) == len(edge_data[k])
                assert data == edge_data[k]

            assert_adb_to_pyg_meta(meta, et_df, edge_data)


def assert_adb_to_pyg_meta(
    meta: Union[str, Set[str], Dict[str, ADBMetagraphValues]],
    df: DataFrame,
    pyg_data: Union[NodeStorage, EdgeStorage],
) -> None:
    valid_meta: Dict[str, ADBMetagraphValues]
    valid_meta = meta if type(meta) is dict else {m: m for m in meta}

    for k, v in valid_meta.items():
        assert k in pyg_data
        assert type(pyg_data[k]) is Tensor

        t = pyg_data[k].tolist()
        if type(v) is str:
            data = df[v].tolist()
            assert len(data) == len(t)
            assert data == t

        if type(v) is dict:
            data = []
            for attr, encoder in v.items():
                if encoder is None:
                    data.append(tensor(df[attr].to_list()))
                if callable(encoder):
                    data.append(encoder(df[attr]))

            cat_data = cat(data, dim=-1).tolist()
            assert len(cat_data) == len(t)
            assert cat_data == t

        if callable(v):
            data = v(df).tolist()
            assert len(data) == len(t)
            assert data == t
