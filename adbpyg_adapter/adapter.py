#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Union

from arango.cursor import Cursor
from arango.database import Database
from arango.graph import Graph as ADBGraph
from arango.result import Result
from torch import tensor
from torch.functional import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType

from .abc import Abstract_ADBPYG_Adapter
from .typings import ArangoMetagraph, Json
from .utils import logger


class ADBPYG_Adapter(Abstract_ADBPYG_Adapter):
    """ArangoDB-PyG adapter.

    :param db: A python-arango database instance
    :type db: arango.database.Database
    :param logging_lvl: Defaults to logging.INFO. Other useful options are
        logging.DEBUG (more verbose), and logging.WARNING (less verbose).
    :type logging_lvl: str | int
    :raise ValueError: If invalid parameters
    """

    def __init__(
        self,
        db: Database,
        logging_lvl: Union[str, int] = logging.INFO,
    ):
        self.set_logging(logging_lvl)

        if issubclass(type(db), Database) is False:
            msg = "**db** parameter must inherit from arango.database.Database"
            raise TypeError(msg)

        self.__db = db

        logger.info(f"Instantiated ADBPYG_Adapter with database '{db.name}'")

    @property
    def db(self) -> Database:
        return self.__db  # pragma: no cover

    def set_logging(self, level: Union[int, str]) -> None:
        logger.setLevel(level)

    def arangodb_to_pyg(
        self, name: str, metagraph: ArangoMetagraph, **query_options: Any
    ) -> HeteroData:
        """Create a HeteroData graph from the user-defined metagraph.

        :param name: The PyG graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to PyG, along with the name of the node & edge feature matrices, and
            the target label.
        :type metagraph: adbpyg_adapter.typings.ArangoMetagraph
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG HeteroData
        :rtype: torch_geometric.data.HeteroData
        :raise ValueError: If missing required keys in metagraph

        Here is an example entry for parameter **metagraph**:

        .. code-block:: python
        {
            "vertexCollections": {
                "account": {'x': 'features', 'y': 'balance'},
                "bank": {'x': 'features'},
                "customer": {'x': 'features'},
            },
            "edgeCollections": {
                "accountHolder": {},
                "transaction": {'edge_attr': 'features'},
            },
        }
        """
        logger.debug(f"--arangodb_to_pyg('{name}')--")

        is_homogeneous = (
            len(metagraph["vertexCollections"]) == 1
            and len(metagraph["edgeCollections"]) == 1
        )

        # Maps ArangoDB vertex IDs to PyG node IDs
        adb_map: Dict[str, Json] = dict()

        data = Data() if is_homogeneous else HeteroData()
        x_feature_matrix: List[Any] = []
        y_target_label: List[Any] = []

        adb_v: Json
        for v_col, atribs in metagraph["vertexCollections"].items():
            logger.debug(f"Preparing '{v_col}' vertices")
            has_node_feature_matrix = "x" in atribs
            has_node_target_label = "y" in atribs

            for i, adb_v in enumerate(self.__fetch_adb_docs(v_col, query_options)):
                adb_id = adb_v["_id"]
                logger.debug(f"V{i}: {adb_id}")

                adb_map[adb_id] = {"id": i, "col": v_col}

                if has_node_feature_matrix:
                    x_feature_matrix.append(adb_v[atribs["x"]])

                if has_node_target_label:
                    y_target_label.append(adb_v[atribs["y"]])

            node_data: NodeStorage = data if is_homogeneous else data[v_col]
            node_data.num_nodes = i + 1

            if has_node_feature_matrix:
                logger.debug(f"Setting '{v_col}' node feature matrix")
                node_data.x = tensor(x_feature_matrix)
                x_feature_matrix.clear()

            if has_node_target_label:
                logger.debug(f"Setting '{v_col}' node target label")
                node_data.y = tensor(y_target_label)
                y_target_label.clear()

        adb_e: Json
        edge_dict: DefaultDict[EdgeType, DefaultDict[str, List[Any]]]
        for e_col, atribs in metagraph["edgeCollections"].items():
            logger.debug(f"Preparing '{e_col}' edges")
            has_edge_feature_matrix = "edge_attr" in atribs
            has_edge_target_label = "y" in atribs

            edge_dict = defaultdict(lambda: defaultdict(list))

            for i, adb_e in enumerate(self.__fetch_adb_docs(e_col, query_options)):
                logger.debug(f'E{i}: {adb_e["_id"]}')

                from_node = adb_map[adb_e["_from"]]
                to_node = adb_map[adb_e["_to"]]
                edge_type: EdgeType = (from_node["col"], e_col, to_node["col"])

                edge = edge_dict[edge_type]
                edge["from_nodes"].append(from_node["id"])
                edge["to_nodes"].append(to_node["id"])

                if has_edge_feature_matrix:
                    edge["edge_attr"].append(adb_e[atribs["edge_attr"]])

                if has_edge_target_label:
                    edge["y"].append(adb_e[atribs["y"]])

            for edge_type, edge in edge_dict.items():
                logger.debug(f"Setting {edge_type} edge index")

                edge_data: EdgeStorage = data if is_homogeneous else data[edge_type]
                edge_data.edge_index = tensor([edge["from_nodes"], edge["to_nodes"]])

                if has_edge_feature_matrix:
                    logger.debug(f"Setting {edge_type} edge feature matrix")
                    edge_data.edge_attr = tensor(edge_data["edge_attr"])

                if has_edge_target_label:
                    logger.debug(f"Setting {edge_type} edge target label")
                    edge_data.y = tensor(edge_data["y"])

        logger.info(f"Created PyG '{name}' Graph")
        return data

    def arangodb_collections_to_pyg(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        **query_options: Any,
    ) -> HeteroData:
        """Create a PyG graph from ArangoDB collections.

        :param name: The PyG graph name.
        :type name: str
        :param v_cols: A set of ArangoDB vertex collections to
            import to PyG.
        :type v_cols: Set[str]
        :param e_cols: A set of ArangoDB edge collections to import to PyG.
        :type e_cols: Set[str]
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG HeteroData
        :rtype: torch_geometric.data.HeteroData
        """
        metagraph: ArangoMetagraph = {
            "vertexCollections": {col: dict() for col in v_cols},
            "edgeCollections": {col: dict() for col in e_cols},
        }

        return self.arangodb_to_pyg(name, metagraph, **query_options)

    def arangodb_graph_to_pyg(self, name: str, **query_options: Any) -> HeteroData:
        """Create a PyG graph from an ArangoDB graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG HeteroData
        :rtype: torch_geometric.data.HeteroData
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_pyg(name, v_cols, e_cols, **query_options)

    def pyg_to_arangodb(
        self,
        name: str,
        pyg_g: Union[Data, HeteroData],
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ADBGraph:
        logger.debug(f"--pyg_to_arangodb('{name}')--")

        is_homogeneous = type(pyg_g) is Data
        if is_homogeneous:
            edge_types = [(name + "_N", name + "_E", name + "_N")]
        else:
            edge_types = pyg_g.edge_types

        edge_definitions = self.etypes_to_edefinitions(edge_types)

        if overwrite_graph:
            logger.debug("Overwrite graph flag is True. Deleting old graph.")
            self.__db.delete_graph(name, ignore_missing=True)

        if self.__db.has_graph(name):
            adb_graph = self.__db.graph(name)
        else:
            adb_graph = self.__db.create_graph(name, edge_definitions)

        adb_v_cols: List[str] = adb_graph.vertex_collections()

        v_col_docs: List[Json] = []  # to-be-inserted ArangoDB vertices
        for v_col in adb_v_cols:
            node_data: NodeStorage = pyg_g if is_homogeneous else pyg_g[v_col]
            num_nodes: int = node_data.num_nodes
            logger.debug(f"Preparing {num_nodes} '{v_col}' nodes")

            has_node_feature_matrix = "x" in node_data
            has_node_target_label = node_data.num_nodes == len(node_data.get("y", []))

            for i in range(num_nodes):
                logger.debug(f"N{i}: {i}")

                adb_vertex: Json = {"_key": str(6)}

                if has_node_feature_matrix:
                    node_features: Tensor = node_data.x[i]
                    adb_vertex["x"] = node_features.tolist()

                if has_node_target_label:
                    node_label: Tensor = node_data.y[i]
                    try:
                        adb_vertex["y"] = node_label.item()
                    except ValueError:
                        adb_vertex["y"] = node_label.tolist()

                v_col_docs.append(adb_vertex)

            self.__insert_adb_docs(v_col, v_col_docs, import_options)
            v_col_docs.clear()

        e_col_docs: List[Json] = []  # to-be-inserted ArangoDB edges
        for edge_type in edge_types:
            edge_data: EdgeStorage = pyg_g if is_homogeneous else pyg_g[edge_type]
            num_edges: int = edge_data.num_edges
            logger.debug(f"Preparing {num_edges} '{edge_type}' nodes")

            from_col, e_col, to_col = edge_type

            has_edge_feature_matrix = "edge_attr" in edge_data
            has_edge_target_label = edge_data.num_edges == len(edge_data.get("y", []))

            for i, (from_n, to_n) in enumerate(zip(*edge_data.edge_index.tolist())):
                logger.debug(f"E{i}: ({from_n}, {to_n})")

                adb_edge: Json = {
                    "_from": f"{from_col}/{str(from_n)}",
                    "_to": f"{to_col}/{str(to_n)}",
                }

                if has_edge_feature_matrix:
                    edge_features: Tensor = edge_data.edge_attr[i]
                    adb_edge["edge_attr"] = edge_features.tolist()

                if has_edge_target_label:
                    edge_label: Tensor = edge_data.y[i]
                    try:
                        adb_edge["y"] = edge_label.item()
                    except ValueError:
                        adb_edge["y"] = edge_label.tolist()

                e_col_docs.append(adb_edge)

            self.__insert_adb_docs(e_col, e_col_docs, import_options)
            e_col_docs.clear()

        logger.info(f"Created ArangoDB '{name}' Graph")
        return adb_graph

    def etypes_to_edefinitions(self, edge_types: List[EdgeType]) -> List[Json]:
        """Converts a PyG graph's edge_types property to ArangoDB graph edge_definitions

        :param edge_types: A list of string triplets (str, str, str) for
            source node type, edge type and destination node type.
        :type edge_types: List[torch_geometric.typing.EdgeType]
        :return: ArangoDB Edge Definitions
        :rtype: List[adbpyg_adapter.typings.Json]

        Here is an example of **edge_definitions**:

        .. code-block:: python
        [
            {
                "edge_collection": "teaches",
                "from_vertex_collections": ["Teacher"],
                "to_vertex_collections": ["Lecture"]
            }
        ]
        """

        edge_type_map: DefaultDict[str, DefaultDict[str, Set[str]]]
        edge_type_map = defaultdict(lambda: defaultdict(set))
        for edge_type in edge_types:
            from_col, e_col, to_col = edge_type
            edge_type_map[e_col]["from"].add(from_col)
            edge_type_map[e_col]["to"].add(to_col)

        edge_definitions: List[Json] = []
        for e_col, v_cols in edge_type_map.items():
            edge_definitions.append(
                {
                    "from_vertex_collections": list(v_cols["from"]),
                    "edge_collection": e_col,
                    "to_vertex_collections": list(v_cols["to"]),
                }
            )

        return edge_definitions

    def __fetch_adb_docs(self, col: str, query_options: Any) -> Result[Cursor]:
        """Fetches ArangoDB documents within a collection.

        :param col: The ArangoDB collection.
        :type col: str
        :param query_options: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type query_options: Any
        :return: Result cursor.
        :rtype: arango.cursor.Cursor
        """
        aql = f"""
            FOR doc IN {col}
                RETURN doc
        """

        return self.__db.aql.execute(aql, **query_options)

    def __insert_adb_docs(
        self, col: str, docs: List[Json], import_options: Any
    ) -> None:
        """Insert ArangoDB documents into their ArangoDB collection.

        :param col: The ArangoDB collection name
        :type col: str
        :param docs: To-be-inserted ArangoDB documents
        :type docs: List[Json]
        :param import_options: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        """
        logger.debug(f"Inserting {len(docs)} documents into '{col}'")
        result = self.__db.collection(col).import_bulk(docs, **import_options)
        logger.debug(result)
