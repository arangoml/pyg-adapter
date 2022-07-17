#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from types import FunctionType
from typing import Any, DefaultDict, Dict, List, Set, Union

from arango.cursor import Cursor
from arango.database import Database
from arango.graph import Graph as ADBGraph
from arango.result import Result
from pandas import DataFrame
from torch import Tensor, cat, tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType

from adbpyg_adapter.controller import ADBPyG_Controller

from .abc import Abstract_ADBPyG_Adapter
from .typings import (
    ArangoMetagraph,
    DEFAULT_PYG_KEY_MAP,
    Json,
    PyGEncoder,
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

        logger.info(f"Instantiated ADBPyG_Adapter with database '{db.name}'")

    @property
    def db(self) -> Database:
        return self.__db  # pragma: no cover

    @property
    def cntrl(self) -> Database:
        return self.__cntrl  # pragma: no cover

    def set_logging(self, level: Union[int, str]) -> None:
        logger.setLevel(level)

    def arangodb_to_pyg(
        self, name: str, metagraph: ArangoMetagraph, **query_options: Any
    ) -> Union[Data, HeteroData]:
        """Create a PyG graph from the user-defined metagraph. DOES carry
            over node/edge features/labels, via the **metagraph**.

        :param name: The PyG graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to PyG, along with collection-level specifications to indicate
            which ArangoDB attributes will become PyG features/labels.
            See below for examples of **metagraph**
        :type metagraph: adbpyg_adapter.typings.ArangoMetagraph
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData

        1) Here is an example entry for parameter **metagraph**:

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

        The metagraph above specifies that each document
        within the "v0" collection has a "pre-built" feature matrix
        named "v0_features", and also has a node label named "label".
        We map these keys to the "x" and "y" properties of a standard
        PyG graph.

        2) Here is another example entry for parameter **metagraph**:
        .. code-block:: python
        {
            "vertexCollections": {
                "Movies": {
                    "x": {
                        "movie title": SequenceEncoder(),
                        "Action": IdentityEncoder(),
                    }
                },
                "Users": {
                    "x": {
                        "Gender": EnumEncoder(),
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

        The metagraph above will build the "Movies" feature matrix
        using the 'movie title' & 'Action' attributes, by reling on
        the user-specified Encoders (see adbpyg_adapter.utils for examples).
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
                node_data.x = self.__build_tensor(meta["x"], df)

            if "y" in meta:
                node_data.y = self.__build_tensor(meta["y"], df)

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
                    edge_data.edge_weight = self.__build_tensor(
                        meta["edge_weight"], df_by_edge_type
                    )

                if "edge_attr" in meta:
                    edge_data.edge_attr = self.__build_tensor(
                        meta["edge_attr"], df_by_edge_type
                    )

                if "y" in meta:
                    edge_data.y = self.__build_tensor(meta["y"], df_by_edge_type)

        logger.info(f"Created PyG '{name}' Graph")
        return data

    def arangodb_collections_to_pyg(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        **query_options: Any,
    ) -> HeteroData:
        """Create a PyG graph from ArangoDB collections. Due to risk of
            ambiguity, this method DOES NOT transfer ArangoDB attributes to PyG.

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
        """Create a PyG graph from an ArangoDB graph. Due to risk of
            ambiguity, this method DOES NOT transfer ArangoDB attributes to PyG.

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
        pyg_key_map: Dict[str, str] = DEFAULT_PYG_KEY_MAP,
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ADBGraph:
        """Create an ArangoDB graph from a PyG graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param pyg_g: The existing PyG graph.
        :type pyg_g: Data | HeteroData
        :param pyg_key_map: An object mapping the PyG standard properties
            (i.e "x", "y", "edge_weight", "edge_attr") to user-defined
            strings, which will be used as the ArangoDB attribute names.
            If not specified, defaults to the built-in pyg_key_map.
            See below for an example of **pyg_key_map**.
        :type pyg_key_map: Dict[str, str]
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections.
        :type overwrite_graph: bool
        :param import_options: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type import_options: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph


        1) Here is an example entry for parameter **pyg_key_map**:

        .. code-block:: python
        {
            "x": "node_features",
            "y": "label",
            "edge_weight": "weight",
            "edge_attr": "edge_features"
        }

        Using the metagraph above will represent the "x" property of the
        PyG graph as "node_features" in ArangoDB, and the "y" property as
        "label".
        """
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

        # Define PyG data properties
        x: Tensor
        y: Tensor
        edge_weight: Tensor
        edge_attr: Tensor
        node_data: NodeStorage
        edge_data: EdgeStorage

        for v_col in adb_v_cols:
            node_data = pyg_g if is_homogeneous else pyg_g[v_col]
            num_nodes: int = node_data.num_nodes

            logger.debug(f"Preparing {num_nodes} '{v_col}' nodes")

            df = DataFrame([{"_key": str(i)} for i in range(num_nodes)])

            if "x" in node_data:
                x = node_data.x
                df[pyg_key_map["x"]] = x.tolist()

            if num_nodes == len(node_data.get("y", [])):
                y = node_data.y
                try:
                    df[pyg_key_map["y"]] = y.item()
                except ValueError:
                    df[pyg_key_map["y"]] = y.tolist()

            df = df.apply(lambda n: self.__cntrl._prepare_pyg_node(n, v_col), axis=1)
            self.__insert_adb_docs(v_col, df.to_dict("records"), import_options)

        for edge_type in edge_types:
            edge_data = pyg_g if is_homogeneous else pyg_g[edge_type]
            num_edges: int = edge_data.num_edges

            logger.debug(f"Preparing {num_edges} '{edge_type}' nodes")

            from_col, e_col, to_col = edge_type

            df = DataFrame(
                zip(*(edge_data.edge_index.tolist())), columns=["_from", "_to"]
            )
            df["_from"] = from_col + "/" + df["_from"].astype(str)
            df["_to"] = to_col + "/" + df["_to"].astype(str)

            if "edge_weight" in edge_data:
                df[pyg_key_map["edge_weight"]] = edge_data.edge_weight

            if "edge_attr" in edge_data:
                df[pyg_key_map["edge_attr"]] = edge_data.edge_attr.tolist()

            if num_edges == len(edge_data.get("y", [])):
                y = edge_data.y
                try:
                    df[pyg_key_map["y"]] = y.item()
                except ValueError:
                    df[pyg_key_map["y"]] = y.tolist()

            df = df.apply(lambda e: self.__cntrl._prepare_pyg_edge(e, e_col), axis=1)
            self.__insert_adb_docs(e_col, df.to_dict("records"), import_options)

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

    def __build_tensor(
        self, meta_val: Union[str, Dict[str, PyGEncoder], FunctionType], df: DataFrame
    ) -> Tensor:
        """Builds PyG-ready Tensors from a Pandas Dataframes, based on
        the nature of the user-defined metagraph.

        :param meta_val: The value mapped to the ArangoDB-PyG metagraph key.
            e.g the value of `metagraph['vertexCollections']['users']['x']`.
            The current accepted **meta_val** types are:
            1) str: return the DataFrame's **meta_val** column values as a Tensor
            2) dict: encode all `key` column values & concatenate as a Tensor
            3) function: execute a user-defined function to return a Tensor
        :type meta_val: str | dict | function
        :param df: The Pandas Dataframe representing ArangoDB data.
        :type df: pandas.DataFrame
        :return: A PyG-ready tensor
        :rtype: torch.Tensor
        """
        if type(meta_val) is str:
            return tensor(df[meta_val].to_list())

        elif type(meta_val) is dict:
            data = []
            for attr, encoder in meta_val.items():
                if encoder is None:
                    data.append(tensor(df[attr].to_list()))
                elif callable(encoder):
                    data.append(encoder(df[attr]))
                else:
                    msg = f"Invalid encoder for ArangoDB attribute '{attr}': {encoder}"
                    raise ValueError(msg)

            return cat(data, dim=-1)

        elif type(meta_val) is FunctionType:
            # user defined function that returns a tensor
            udf_tensor: Tensor = meta_val(df)
            return udf_tensor

        else:
            msg = f"""
                Invalid **meta_val** argument type: {meta_val}.
                Expected Union[str, Dict[str, PyGEncoder], FunctionType],
                got {type(meta_val)} instead.
            """
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
