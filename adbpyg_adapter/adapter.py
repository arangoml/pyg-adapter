#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Union

from arango.cursor import Cursor
from arango.database import Database
from arango.graph import Graph as ADBGraph
from arango.result import Result
from pandas import DataFrame
from torch import cat, tensor
from torch.functional import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType
from tqdm import tqdm

from adbpyg_adapter.controller import ADBPyG_Controller

from .abc import Abstract_ADBPyG_Adapter
from .typings import (
    ArangoMetagraph,
    DEFAULT_PyG_METAGRAPH,
    Json,
    PyGEncoder,
    PyGMetagraph,
)
from .utils import logger


class ADBPyG_Adapter(Abstract_ADBPyG_Adapter):
    """ArangoDB-PyG adapter.

    :param db: A python-arango database instance
    :type db: arango.database.Database
    :param controller: The ArangoDB-PyG controller, used to prepare
        nodes & edges before ArangoDB insertion, optionally re-defined
        by the user if needed (otherwise defaults to ADBPyG_Controller).
    :type controller: adbpyg_adapter.controller.ADBPyG_Controller
    :param logging_lvl: Defaults to logging.INFO. Other useful options are
        logging.DEBUG (more verbose), and logging.WARNING (less verbose).
    :type logging_lvl: str | int
    :raise ValueError: If invalid parameters
    """

    def __init__(
        self,
        db: Database,
        controller: ADBPyG_Controller = ADBPyG_Controller(),
        logging_lvl: Union[str, int] = logging.INFO,
    ):
        self.set_logging(logging_lvl)

        if issubclass(type(db), Database) is False:
            msg = "**db** parameter must inherit from arango.database.Database"
            raise TypeError(msg)

        if issubclass(type(controller), ADBPyG_Controller) is False:
            msg = "**controller** parameter must inherit from ADBPyG_Controller"
            raise TypeError(msg)

        self.__db = db
        self.__cntrl: ADBPyG_Controller = controller

        logger.info(f"Instantiated ADBPYG_Adapter with database '{db.name}'")

    @property
    def db(self) -> Database:
        return self.__db  # pragma: no cover

    def set_logging(self, level: Union[int, str]) -> None:
        logger.setLevel(level)

    def arangodb_to_pyg(
        self, name: str, metagraph: ArangoMetagraph, **query_options: Any
    ) -> Union[Data, HeteroData]:
        """Create a PyG graph from the user-defined metagraph. DOES carry
            over node/edge features/labels, based on **metagraph**.

        :param name: The PyG graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to PyG, along with the name of the node/edge feature matrices and
            the target label attribute names used in ArangoDB.
            See below for an example of **metagraph**
        :type metagraph: adbpyg_adapter.typings.ArangoMetagraph
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData

        Here is an example entry for parameter **metagraph**:

        .. code-block:: python
        {
            "vertexCollections": {
                "v0": {'x': 'v0_features', 'y': 'label'},
                "v1": {'x': 'v1_features'},
                "v2": {'x': 'v2_features'},
            },
            "edgeCollections": {
                "e0": {'edge_attr': 'e0_features'},
                "e1": {'edge_weight': 'edge_weight'},
            },
        }

        For example, this metagraph specifies that each document
        within the "v0" collection has a feature matrix named "v0_features",
        and also has a node label named "label". We map these keys to "x"
        and "y" to create the standard PyG object.


        {
            "vertexCollections": {
                "v0": {
                    'x': {
                        'a': IdentityEncoder(dtype=torch.long),
                        'b': SentenceEncoder()
                    },
                    'y': SentenceEncoder()
                },
                "v1": {'x': 'v1_features'},
                "v2": {'x': 'v2_features'},
            },
            "edgeCollections": {
                "e0": {'edge_attr': 'e0_features'},
                "e1": {'edge_weight': 'edge_weight'},
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

        for v_col, meta in metagraph["vertexCollections"].items():
            node_data: NodeStorage = data if is_homogeneous else data[v_col]
            logger.debug(f"Preparing '{v_col}' vertices")

            df = DataFrame(self.__fetch_adb_docs(v_col, query_options))
            adb_map.update({adb_id: pyg_id for pyg_id, adb_id in enumerate(df["_id"])})

            if "x" in meta:
                node_data.x = self.__build_pyg_data(meta["x"], df)

            if "y" in meta:
                node_data.y = self.__build_pyg_data(meta["y"], df)

        for e_col, meta in metagraph["edgeCollections"].items():
            logger.debug(f"Preparing '{e_col}' edges")

            df = DataFrame(self.__fetch_adb_docs(e_col, query_options))
            df["from_col"] = df["_from"].str.split("/").str[0]
            df["to_col"] = df["_to"].str.split("/").str[0]

            for (from_col, to_col), count in (
                df[["from_col", "to_col"]].value_counts().items()
            ):
                edge_type = (from_col, e_col, to_col)
                edge_data: EdgeStorage = data if is_homogeneous else data[edge_type]
                logger.debug(f"Preparing {count} '{edge_type}' edges")

                df_by_edge_type: DataFrame = df[
                    (df["from_col"] == from_col) & (df["to_col"] == to_col)
                ]

                from_nodes = [adb_map[adb_id] for adb_id in df_by_edge_type["_from"]]
                to_nodes = [adb_map[adb_id] for adb_id in df_by_edge_type["_to"]]

                edge_data.edge_index = tensor([from_nodes, to_nodes])

                if "edge_weight" in meta:
                    edge_data.edge_weight = self.__build_pyg_data(
                        meta["edge_weight"], df_by_edge_type
                    )

                if "edge_attr" in meta:
                    edge_data.edge_attr = self.__build_pyg_data(
                        meta["edge_attr"], df_by_edge_type
                    )

                if "y" in meta:
                    edge_data.y = self.__build_pyg_data(meta["y"], df_by_edge_type)

        logger.info(f"Created PyG '{name}' Graph")
        return data

    def arangodb_collections_to_pyg(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        **query_options: Any,
    ) -> HeteroData:
        """Create a PyG graph from ArangoDB collections. Does NOT carry
            over node/edge features/labels.

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
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        """
        metagraph: ArangoMetagraph = {
            "vertexCollections": {col: dict() for col in v_cols},
            "edgeCollections": {col: dict() for col in e_cols},
        }

        return self.arangodb_to_pyg(name, metagraph, **query_options)

    def arangodb_graph_to_pyg(self, name: str, **query_options: Any) -> HeteroData:
        """Create a PyG graph from an ArangoDB graph. Does NOT carry
            over node/edge features/labels.

        :param name: The ArangoDB graph name.
        :type name: str
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_pyg(name, v_cols, e_cols, **query_options)

    def pyg_to_arangodb(
        self,
        name: str,
        pyg_g: Union[Data, HeteroData],
        metagraph: PyGMetagraph = DEFAULT_PyG_METAGRAPH,
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
            has_node_target_label = num_nodes == len(node_data.get("y", []))

            for i in tqdm(
                range(num_nodes),
                desc=v_col,
                colour="CYAN",
                disable=logger.level != logging.INFO,
            ):
                logger.debug(f"N{i}: {i}")

                adb_vertex: Json = {"_key": str(i)}

                if has_node_feature_matrix:
                    node_features: Tensor = node_data.x[i]
                    adb_vertex[metagraph["x"]] = node_features.tolist()

                if has_node_target_label:
                    node_label: Tensor = node_data.y[i]
                    try:
                        adb_vertex[metagraph["y"]] = node_label.item()
                    except ValueError:
                        adb_vertex[metagraph["y"]] = node_label.tolist()

                self.__cntrl._prepare_arangodb_vertex(adb_vertex, v_col)
                v_col_docs.append(adb_vertex)

            self.__insert_adb_docs(v_col, v_col_docs, import_options)
            v_col_docs.clear()

        e_col_docs: List[Json] = []  # to-be-inserted ArangoDB edges
        for edge_type in edge_types:
            edge_data: EdgeStorage = pyg_g if is_homogeneous else pyg_g[edge_type]
            num_edges: int = edge_data.num_edges

            logger.debug(f"Preparing {num_edges} '{edge_type}' nodes")

            from_col, e_col, to_col = edge_type

            has_edge_weight_list = "edge_weight" in edge_data
            has_edge_feature_matrix = "edge_attr" in edge_data
            has_edge_target_label = num_edges == len(edge_data.get("y", []))

            for i, (from_n, to_n) in enumerate(
                tqdm(
                    zip(*(edge_data.edge_index.tolist())),
                    total=num_edges,
                    desc=str(edge_type),
                    colour="YELLOW",
                    disable=logger.level != logging.INFO,
                )
            ):
                logger.debug(f"E{i}: ({from_n}, {to_n})")

                adb_edge: Json = {
                    "_from": f"{from_col}/{str(from_n)}",
                    "_to": f"{to_col}/{str(to_n)}",
                }

                if has_edge_weight_list:
                    edge_weights: Tensor = edge_data.edge_weight[i]
                    adb_edge[metagraph["edge_weight"]] = edge_weights.item()

                if has_edge_feature_matrix:
                    edge_features: Tensor = edge_data.edge_attr[i]
                    adb_edge[metagraph["edge_attr"]] = edge_features.tolist()

                if has_edge_target_label:
                    edge_label: Tensor = edge_data.y[i]
                    try:
                        adb_edge[metagraph["y"]] = edge_label.item()
                    except ValueError:
                        adb_edge[metagraph["y"]] = edge_label.tolist()

                self.__cntrl._prepare_arangodb_edge(adb_edge, edge_type)
                e_col_docs.append(adb_edge)

            self.__insert_adb_docs(e_col, e_col_docs, import_options)
            e_col_docs.clear()

        logger.info(f"Created ArangoDB '{name}' Graph")
        return adb_graph

    def etypes_to_edefinitions(self, edge_types: List[EdgeType]) -> List[Json]:
        """Converts PyG edge_types to ArangoDB edge_definitions

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

    def __build_pyg_data(
        self, meta_val: Union[str, Dict[str, PyGEncoder]], df: DataFrame
    ) -> Tensor:
        meta_type = type(meta_val)

        if meta_type is str:
            return tensor(df[meta_val].to_list())

        elif meta_type is dict:
            data = []
            for attr, encoder in meta_val.items():  # type: ignore # (false positive)
                if encoder is None:
                    data.append(df[attr])
                else:
                    data.append(encoder(df[attr]))  # type: ignore

            return cat(data, dim=-1)

        else:
            msg = f"Invalid **meta_val** argument type: {meta_val}"
            raise TypeError(msg)

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
