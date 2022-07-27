from typing import Any, Dict, List, Optional, Set, Union

import pytest
from arango.graph import Graph as ArangoGraph
from torch import Tensor, long, tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType

from adbpyg_adapter import ADBPyG_Adapter
from adbpyg_adapter.encoders import EnumEncoder, IdentityEncoder
from adbpyg_adapter.typings import ADBMetagraph, PyGMetagraph

from .conftest import (  # SequenceEncoder,
    adbpyg_adapter,
    arango_restore,
    con,
    db,
    get_fake_hetero_graph,
    get_fake_homo_graph,
    get_karate_graph,
    get_social_graph,
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
        ADBPyG_Adapter(db, Bad_ADBPyG_Controller())  # type: ignore


@pytest.mark.parametrize(
    "adapter, name, pyg_g, metagraph, \
        explicit_metagraph, overwrite_graph, import_options",
    [
        (
            adbpyg_adapter,
            "Karate_1",
            get_karate_graph(),
            {"nodeTypes": {"Karate_1_N": {"x": "node_features"}}},
            False,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "Karate_2",
            get_karate_graph(),
            {"nodeTypes": {"Karate_2_N": {"x": "node_features"}}},
            True,
            False,
            {"overwrite": True},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_1",
            get_fake_homo_graph(avg_num_nodes=3),
            {"nodeTypes": {"FakeHomoGraph_1_N": {"y": "label"}}},
            False,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_2",
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=1),
            {},
            False,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_3",
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=1),
            {},
            True,
            False,
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
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_1",
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=1),
            {},
            False,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_2",
            get_fake_hetero_graph(avg_num_nodes=2),
            {"nodeTypes": {"v2": {"x": udf_v2_x_tensor_to_df}}},
            True,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "SocialGraph",
            get_social_graph(),
            {"nodeTypes": {"user": {"x": ["age", "gender"]}}},
            False,
            False,
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
    import_options: Any,
) -> None:
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adb_g = adapter.pyg_to_arangodb(
        name, pyg_g, metagraph, explicit_metagraph, overwrite_graph, **import_options
    )
    assert_arangodb_data(name, pyg_g, adb_g, metagraph, explicit_metagraph)
    db.delete_graph(name, drop_collections=True, ignore_missing=True)

    if explicit_metagraph and "edgeTypes" not in metagraph:
        for n_type in metagraph.get("nodeTypes", {}).keys():
            db.delete_collection(n_type)


@pytest.mark.parametrize(
    "adapter, name, metagraph, pyg_g_old",
    [
        (
            adbpyg_adapter,
            "Karate",
            {
                "vertexCollections": {
                    "Karate_N": {"x": "x"},
                },
                "edgeCollections": {
                    "Karate_E": {},
                },
            },
            get_karate_graph(),
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
        ),
        (
            adbpyg_adapter,
            "HeterogeneousOverComplicated",
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
        ),
        (
            adbpyg_adapter,
            "HeterogeneousUDF",
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
        ),
    ],
)
def test_adb_to_pyg(
    adapter: ADBPyG_Adapter,
    name: str,
    metagraph: ADBMetagraph,
    pyg_g_old: Optional[Union[Data, HeteroData]],
) -> None:
    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.pyg_to_arangodb(name, pyg_g_old)

    pyg_g_new = adapter.arangodb_to_pyg(name, metagraph)
    assert_pyg_data(pyg_g_new, metagraph)

    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)


@pytest.mark.parametrize(
    "adapter, name, v_cols, e_cols, pyg_g_old",
    [(adbpyg_adapter, "SocialGraph", {"user", "game"}, {"plays"}, get_social_graph())],
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

    assert_pyg_data(
        pyg_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)


@pytest.mark.parametrize(
    "adapter, name, pyg_g_old",
    [
        (adbpyg_adapter, "FakeHeterogeneous", get_fake_hetero_graph(avg_num_nodes=2)),
    ],
)
def test_adb_graph_to_pyg(
    adapter: ADBPyG_Adapter, name: str, pyg_g_old: Union[Data, HeteroData]
) -> None:
    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)
        adapter.pyg_to_arangodb(name, pyg_g_old)

    arango_graph = db.graph(name)
    v_cols = arango_graph.vertex_collections()
    e_cols = {col["edge_collection"] for col in arango_graph.edge_definitions()}

    pyg_g_new = adapter.arangodb_graph_to_pyg(name)

    # Manually set the number of nodes (since nodes are feature-less)
    for v_col in v_cols:
        if pyg_g_old:
            pyg_g_new[v_col].num_nodes = pyg_g_old[v_col].num_nodes
        else:
            pyg_g_new[v_col].num_nodes = db.collection(v_col).count()

    assert_pyg_data(
        pyg_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    if pyg_g_old:
        db.delete_graph(name, drop_collections=True, ignore_missing=True)


def test_full_cycle_imdb() -> None:
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
                    "Gender": EnumEncoder(),
                }
            },
        },
        "edgeCollections": {"Ratings": {"edge_weight": "Rating"}},
    }

    pyg_g = adbpyg_adapter.arangodb_to_pyg(name, adb_to_pyg_metagraph)
    assert_pyg_data(pyg_g, adb_to_pyg_metagraph)

    pyg_to_adb_metagraph: PyGMetagraph = {
        "nodeTypes": {
            "Movies": {
                "y": "comedy",  # ["comedy"]
                "x": ["action", "drama"],
            },
            "Users": {"x": udf_users_x_tensor_to_df},  # ["age", "gender"],
        },
        "edgeTypes": {("Users", "Ratings", "Movies"): {"edge_weight": "rating"}},
    }
    adb_g = adbpyg_adapter.pyg_to_arangodb(
        name, pyg_g, pyg_to_adb_metagraph, overwrite=True
    )
    assert_arangodb_data(
        name, pyg_g, adb_g, pyg_to_adb_metagraph, skip_edge_assertion=True
    )

    db.delete_graph(name, drop_collections=True)


def assert_arangodb_data(
    name: str,
    pyg_g: Union[Data, HeteroData],
    adb_g: ArangoGraph,
    metagraph: PyGMetagraph,
    explicit_metagraph: bool = False,
    skip_edge_assertion: bool = False,
) -> None:
    is_homogeneous = type(pyg_g) is Data

    node_types: List[str]
    edge_types: List[EdgeType]
    if metagraph and explicit_metagraph:
        node_types = metagraph.get("nodeTypes", {}).keys()  # type: ignore
        edge_types = metagraph.get("edgeTypes", {}).keys()  # type: ignore
    elif is_homogeneous:
        node_types = [name + "_N"]
        edge_types = [(name + "_N", name + "_E", name + "_N")]
    else:
        node_types = pyg_g.node_types
        edge_types = pyg_g.edge_types

    x: Tensor
    y: Tensor

    n_type: str
    n_meta = metagraph.get("nodeTypes", {})
    for n_type in node_types:
        meta = n_meta.get(n_type, {})
        collection = db.collection(n_type)

        node_data: NodeStorage = pyg_g if is_homogeneous else pyg_g[n_type]
        num_nodes = node_data.num_nodes

        assert collection.count() == num_nodes

        # TODO: Remove str restriction
        has_node_feature_matrix = "x" in node_data and type(meta.get("x", "x")) is str
        has_node_target_label = (
            num_nodes == len(node_data.get("y", [])) and type(meta.get("y", "y")) is str
        )

        for i in range(num_nodes):
            vertex = collection.get(str(i))
            assert vertex

            if has_node_feature_matrix:
                meta_val = meta.get("x", "x")
                assert meta_val in vertex

                x = node_data.x[i]
                assert x.tolist() == vertex[meta_val]

            if has_node_target_label:
                meta_val = meta.get("y", "y")
                assert meta_val in vertex

                y = node_data.y[i]
                y_val: Any
                try:
                    y_val = y.item()
                except ValueError:
                    y_val = y.tolist()

                # TODO: remove this ugly hack
                if type(vertex[meta_val]) is list:
                    assert [y_val] == vertex[meta_val]
                else:
                    assert y_val == vertex[meta_val]

    edge_weight: Tensor
    edge_attr: Tensor
    e_type: EdgeType
    e_meta = metagraph.get("edgeTypes", {})
    for e_type in edge_types:
        meta = e_meta.get(e_type, {})
        from_col, e_col, to_col = e_type
        collection = db.collection(e_col)

        edge_data: EdgeStorage = pyg_g if is_homogeneous else pyg_g[e_type]
        num_edges: int = edge_data.num_edges

        # There can be multiple PyG edge types within
        # the same ArangoDB edge collection
        assert collection.count() >= num_edges

        if skip_edge_assertion:
            continue

        # TODO: Remove str restriction
        has_edge_weight_list = (
            "edge_weight" in edge_data
            and type(meta.get("edge_weight", "edge_weight")) is str
        )
        has_edge_feature_matrix = (
            "edge_attr" in edge_data and type(meta.get("edge_attr", "edge_attr")) is str
        )
        has_edge_target_label = (
            num_edges == len(edge_data.get("y", [])) and type(meta.get("y", "y")) is str
        )

        for i, (from_n, to_n) in enumerate(zip(*(edge_data.edge_index.tolist()))):
            edge = collection.find(
                {
                    "_from": f"{from_col}/{from_n}",
                    "_to": f"{to_col}/{to_n}",
                }
            ).next()

            assert edge

            if has_edge_weight_list:
                meta_val = meta.get("edge_weight", "edge_weight")
                assert meta_val in edge

                edge_weight = edge_data.edge_weight[i]
                assert edge_weight.item() == edge[meta_val]

            if has_edge_feature_matrix:
                meta_val = meta.get("edge_attr", "edge_attr")
                assert meta_val in edge

                edge_attr = edge_data.edge_attr[i]
                assert edge_attr.tolist() == edge[meta_val]

            if has_edge_target_label:
                meta_val = meta.get("y", "y")
                assert meta_val in edge

                y = edge_data.y[i]
                try:
                    y_val = y.item()
                except ValueError:
                    y_val = y.tolist()

                # TODO: remove this ugly hack
                if type(edge[meta_val]) is list:
                    assert [y_val] == edge[meta_val]
                else:
                    assert y_val == edge[meta_val]


def assert_pyg_data(pyg_g: Union[Data, HeteroData], metagraph: ADBMetagraph) -> None:
    is_homogeneous = (
        len(metagraph["vertexCollections"]) == 1
        and len(metagraph["edgeCollections"]) == 1
    )

    edge_type_map = dict()
    if is_homogeneous:
        v_col = list(metagraph["vertexCollections"].keys())[0]
        e_col = list(metagraph["edgeCollections"].keys())[0]
        edge_type_map[(v_col, e_col, v_col)] = 0
    else:
        for edge_type in pyg_g.edge_types:
            edge_type_map[edge_type] = 0

    # Maps ArangoDB IDs to PyG IDs
    adb_map = dict()

    y_val: Any  # ignore this for now
    for v_col, meta in metagraph["vertexCollections"].items():
        node_data: NodeStorage = pyg_g if is_homogeneous else pyg_g[v_col]
        num_nodes = node_data.num_nodes

        collection = db.collection(v_col)
        assert num_nodes == collection.count()

        # TODO: Remove str restriction to introduce Encoder verificiation
        has_node_feature_matrix = type(meta.get("x")) is str
        has_node_target_label = type(meta.get("y")) is str

        for i, doc in enumerate(collection):
            adb_map[doc["_id"]] = i

            if has_node_feature_matrix:
                x: Tensor = node_data.x[i]
                assert [float(num) for num in doc[meta["x"]]] == x.tolist()

            if has_node_target_label:
                y: Tensor = node_data.y[i]

                try:
                    y_val = y.item()
                except ValueError:
                    y_val = y.tolist()

                assert doc[meta["y"]] == y_val

    for e_col, meta in metagraph["edgeCollections"].items():
        collection = db.collection(e_col)
        collection_count = collection.count()
        assert collection_count == pyg_g.num_edges

        # TODO: Remove str restriction to introduce Encoder verificiation
        has_edge_weight_list = type(meta.get("edge_weight")) is str
        has_edge_feature_matrix = type(meta.get("edge_attr")) is str
        has_edge_target_label = type(meta.get("y")) is str

        for edge in collection:
            from_adb_col = str(edge["_from"]).split("/")[0]
            to_adb_col = str(edge["_to"]).split("/")[0]

            edge_type = (from_adb_col, e_col, to_adb_col)
            edge_data: EdgeStorage = pyg_g if is_homogeneous else pyg_g[edge_type]

            i = edge_type_map[edge_type]
            from_pyg_id: Tensor = edge_data.edge_index[0][i]
            to_pyg_id: Tensor = edge_data.edge_index[1][i]

            assert adb_map[edge["_from"]] == from_pyg_id.item()
            assert adb_map[edge["_to"]] == to_pyg_id.item()

            edge_type_map[edge_type] += 1

            if has_edge_weight_list:
                assert "edge_weight" in edge_data
                assert edge[meta["edge_weight"]] == edge_data.edge_weight[i].item()

            if has_edge_feature_matrix:
                assert "edge_attr" in edge_data
                assert edge[meta["edge_attr"]] == edge_data.edge_attr[i].tolist()

            if has_edge_target_label:
                assert "y" in edge_data

                y = edge_data.y[i]
                try:
                    y_val = y.item()
                except ValueError:
                    y_val = y.tolist()

                assert edge[meta["y"]] == y_val
