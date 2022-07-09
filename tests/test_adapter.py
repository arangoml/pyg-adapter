from collections import defaultdict
from typing import Any, Dict, Set, Union

import pytest
from arango.graph import Graph as ArangoGraph
from torch.functional import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage

from adbpyg_adapter import ADBPyG_Adapter
from adbpyg_adapter.typings import ArangoMetagraph, DEFAULT_PyG_METAGRAPH, PyGMetagraph

from .conftest import (
    adbpyg_adapter,
    db,
    get_fake_hetero_graph,
    get_fake_homo_graph,
    get_karate_graph,
    get_social_graph,
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
    "adapter, name, pyg_g, metagraph, overwrite_graph, import_options",
    [
        (
            adbpyg_adapter,
            "Karate",
            get_karate_graph(),
            DEFAULT_PyG_METAGRAPH,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_1",
            get_fake_homo_graph(avg_num_nodes=3),
            DEFAULT_PyG_METAGRAPH,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_2",
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=1),
            DEFAULT_PyG_METAGRAPH,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHomoGraph_3",
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=2),
            DEFAULT_PyG_METAGRAPH,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_1",
            get_fake_hetero_graph(avg_num_nodes=2),
            DEFAULT_PyG_METAGRAPH,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_2",
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=1),
            DEFAULT_PyG_METAGRAPH,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "FakeHeteroGraph_3",
            get_fake_hetero_graph(avg_num_nodes=2, edge_dim=2),
            DEFAULT_PyG_METAGRAPH,
            False,
            {},
        ),
        (
            adbpyg_adapter,
            "SocialGraph",
            get_social_graph(),
            DEFAULT_PyG_METAGRAPH,
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
    overwrite_graph: bool,
    import_options: Any,
) -> None:
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adb_g = adapter.pyg_to_arangodb(
        name, pyg_g, metagraph, overwrite_graph, **import_options
    )
    assert_arangodb_data(name, pyg_g, adb_g, metagraph)
    db.delete_graph(name, drop_collections=True)


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
            "FakeHomogoneous",
            {
                "vertexCollections": {
                    "FakeHomogoneous_N": {"x": "x", "y": "y"},
                },
                "edgeCollections": {
                    "FakeHomogoneous_E": {"edge_weight": "edge_weight"},
                },
            },
            get_fake_homo_graph(avg_num_nodes=3, edge_dim=1),
        ),
        (
            adbpyg_adapter,
            "FakeHeterogeneous",
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
    ],
)
def test_adb_to_pyg(
    adapter: ADBPyG_Adapter,
    name: str,
    metagraph: ArangoMetagraph,
    pyg_g_old: Union[Data, HeteroData],
) -> None:
    # re-create the ArangoDB graph since we delete the ones above
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adapter.pyg_to_arangodb(name, pyg_g_old)

    pyg_g_new = adapter.arangodb_to_pyg(name, metagraph)
    assert_pyg_data(pyg_g_new, metagraph)

    db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize(
    "adapter, name, v_cols, e_cols, pyg_g_old",
    [(adbpyg_adapter, "SocialGraph", {"user", "game"}, {"plays"}, get_social_graph())],
)
def test_adb_collections_to_dgl(
    adapter: ADBPyG_Adapter,
    name: str,
    v_cols: Set[str],
    e_cols: Set[str],
    pyg_g_old: Union[Data, HeteroData],
) -> None:
    # re-create the ArangoDB graph since we delete the ones above
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adapter.pyg_to_arangodb(name, pyg_g_old)

    pyg_g_new = adapter.arangodb_collections_to_pyg(
        name,
        v_cols,
        e_cols,
    )

    # Manually set the number of nodes (since nodes are feature-less)
    for v_col in v_cols:
        pyg_g_new[v_col].num_nodes = pyg_g_old[v_col].num_nodes

    assert_pyg_data(
        pyg_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    db.delete_graph(name, drop_collections=True)


@pytest.mark.parametrize(
    "adapter, name, pyg_g_old",
    [(adbpyg_adapter, "FakeHeterogeneous", get_fake_hetero_graph(avg_num_nodes=2))],
)
def test_adb_graph_to_dgl(
    adapter: ADBPyG_Adapter, name: str, pyg_g_old: Union[Data, HeteroData]
) -> None:
    # re-create the ArangoDB graph since we delete the ones above
    db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adapter.pyg_to_arangodb(name, pyg_g_old)

    arango_graph = db.graph(name)
    v_cols = arango_graph.vertex_collections()
    e_cols = {col["edge_collection"] for col in arango_graph.edge_definitions()}

    pyg_g_new = adapter.arangodb_graph_to_pyg(name)

    # Manually set the number of nodes (since nodes are feature-less)
    for v_col in v_cols:
        pyg_g_new[v_col].num_nodes = pyg_g_old[v_col].num_nodes

    assert_pyg_data(
        pyg_g_new,
        metagraph={
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        },
    )

    db.delete_graph(name, drop_collections=True)


def assert_arangodb_data(
    name: str,
    pyg_g: Union[Data, HeteroData],
    adb_g: ArangoGraph,
    metagraph: PyGMetagraph,
) -> None:
    is_homogeneous = type(pyg_g) is Data
    if is_homogeneous:
        edge_types = [(name + "_N", name + "_E", name + "_N")]
    else:
        edge_types = pyg_g.edge_types

    vertex_collections = adb_g.vertex_collections()

    x: Tensor
    y: Tensor
    for v_col in vertex_collections:
        collection = adb_g.vertex_collection(v_col)

        node_data: NodeStorage = pyg_g if is_homogeneous else pyg_g[v_col]
        num_nodes = node_data.num_nodes

        assert collection.count() == num_nodes

        has_node_feature_matrix = "x" in node_data
        has_node_target_label = num_nodes == len(node_data.get("y", []))

        for i in range(num_nodes):
            vertex = collection.get(str(i))
            assert vertex

            if has_node_feature_matrix:
                assert metagraph["x"] in vertex

                x = node_data.x[i]
                assert x.tolist() == vertex[metagraph["x"]]

            if has_node_target_label:
                assert metagraph["y"] in vertex

                y = node_data.y[i]
                y_val: Any
                try:
                    y_val = y.item()
                except ValueError:
                    y_val = y.tolist()

                assert y_val == vertex[metagraph["y"]]

        edge_weight: Tensor
        edge_attr: Tensor
        for edge_type in edge_types:
            from_col, e_col, to_col = edge_type
            collection = adb_g.edge_collection(e_col)

            edge_data: EdgeStorage = pyg_g if is_homogeneous else pyg_g[edge_type]
            num_edges: int = edge_data.num_edges

            # There can be multiple PyG edge types within
            # the same ArangoDB edge collection
            assert collection.count() >= num_edges

            has_edge_weight_list = "edge_weight" in edge_data
            has_edge_feature_matrix = "edge_attr" in edge_data
            has_edge_target_label = num_edges == len(edge_data.get("y", []))

            for i, (from_n, to_n) in enumerate(zip(*(edge_data.edge_index.tolist()))):
                edge = collection.find(
                    {
                        "_from": f"{from_col}/{from_n}",
                        "_to": f"{to_col}/{to_n}",
                    }
                ).next()

                assert edge

                if has_edge_weight_list:
                    assert metagraph["edge_weight"] in edge

                    edge_weight = edge_data.edge_weight[i]
                    assert edge_weight.item() == edge[metagraph["edge_weight"]]

                if has_edge_feature_matrix:
                    assert metagraph["edge_attr"] in edge

                    edge_attr = edge_data.edge_attr[i]
                    assert edge_attr.tolist() == edge[metagraph["edge_attr"]]

                if has_edge_target_label:
                    assert metagraph["y"] in edge

                    y = edge_data.y[i]
                    try:
                        y_val = y.item()
                    except ValueError:
                        y_val = y.tolist()

                    assert y_val == edge[metagraph["y"]]


def assert_pyg_data(pyg_g: Union[Data, HeteroData], metagraph: ArangoMetagraph) -> None:
    is_homogeneous = (
        len(metagraph["vertexCollections"]) == 1
        and len(metagraph["edgeCollections"]) == 1
    )

    y_val: Any  # ignore this for now
    for v_col, atribs in metagraph["vertexCollections"].items():
        node_data: NodeStorage = pyg_g if is_homogeneous else pyg_g[v_col]
        num_nodes = node_data.num_nodes

        collection = db.collection(v_col)
        assert num_nodes == collection.count()

        has_node_feature_matrix = "x" in atribs
        has_node_target_label = "y" in atribs

        for i, doc in enumerate(collection):
            if has_node_feature_matrix:
                x: Tensor = node_data.x[i]
                assert [float(num) for num in doc[atribs["x"]]] == x.tolist()

            if has_node_target_label:
                y: Tensor = node_data.y[i]

                try:
                    y_val = y.item()
                except ValueError:
                    y_val = y.tolist()

                assert doc[atribs["y"]] == y_val

    for e_col, atribs in metagraph["edgeCollections"].items():
        collection = db.collection(e_col)
        collection_count = collection.count()

        has_edge_weight_list = "edge_weight" in atribs
        has_edge_feature_matrix = "edge_attr" in atribs
        has_edge_target_label = "y" in atribs

        edge_data: EdgeStorage
        if is_homogeneous:
            edge_data = pyg_g
            assert collection_count == edge_data.num_edges

            for i, edge in enumerate(collection):
                from_adb_id = int(str(edge["_from"]).split("/")[1])
                to_adb_id = int(str(edge["_to"]).split("/")[1])

                from_pyg_id: Tensor = edge_data.edge_index[0][i]
                to_pyg_id: Tensor = edge_data.edge_index[1][i]

                assert from_adb_id == from_pyg_id.item()
                assert to_adb_id == to_pyg_id.item()

                if has_edge_weight_list:
                    assert "edge_weight" in edge_data
                    assert (
                        edge[atribs["edge_weight"]] == edge_data.edge_weight[i].item()
                    )

                if has_edge_feature_matrix:
                    assert "edge_attr" in edge_data
                    assert edge[atribs["edge_attr"]] == edge_data.edge_attr[i].tolist()

                if has_edge_target_label:
                    assert "y" in edge_data

                    y = edge_data.y[i]
                    try:
                        y_val = y.item()
                    except ValueError:
                        y_val = y.tolist()

                    assert edge[atribs["y"]] == y_val

        else:
            edge_type_map = defaultdict(list)
            for edge_type in pyg_g.edge_types:
                # There can be multiple PyG edge types within
                # the same ArangoDB edge collection
                assert collection_count >= pyg_g[edge_type].num_edges

                _, e_col, _ = edge_type
                edge_type_map[e_col].append(edge_type)

            edge_matches: Dict[str, bool] = {}
            for edge in collection:
                edge_matches[edge["_id"]] = False

                from_adb_col, from_adb_id = edge["_from"].split("/")
                to_adb_col, to_adb_id = edge["_to"].split("/")

                from_adb_id = int(str(edge["_from"]).split("/")[1])
                to_adb_id = int(str(edge["_to"]).split("/")[1])

                # Find the ArangoDB edge somewhere within the PyG graph
                for edge_type in edge_type_map[e_col]:
                    from_ntype, _, to_ntype = edge_type

                    if from_adb_col != from_ntype or to_adb_col != to_ntype:
                        continue

                    edge_data = pyg_g[edge_type]

                    for i, (from_pyg_id, to_pyg_id) in enumerate(
                        zip(*(edge_data.edge_index.tolist()))
                    ):
                        if (
                            int(from_adb_id) == from_pyg_id
                            and int(to_adb_id) == to_pyg_id
                        ):
                            edge_matches[edge["_id"]] = True

                            if has_edge_weight_list:
                                assert "edge_weight" in edge_data
                                assert (
                                    edge[atribs["edge_weight"]]
                                    == edge_data.edge_weight[i].item()
                                )

                            if has_edge_feature_matrix:
                                assert "edge_attr" in edge_data
                                assert (
                                    edge[atribs["edge_attr"]]
                                    == edge_data.edge_attr[i].tolist()
                                )

                            if has_edge_target_label:
                                assert "y" in edge_data

                                y = edge_data.y[i]
                                try:
                                    y_val = y.item()
                                except ValueError:
                                    y_val = y.tolist()

                                assert edge[atribs["y"]] == y_val

                            break

            assert collection_count == len(edge_matches.keys())
            assert all(
                [edge_matches[edge_id] is True for edge_id in edge_matches.keys()]
            )
