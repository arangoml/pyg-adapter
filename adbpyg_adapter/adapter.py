#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Union

from arango.database import Database
from arango.graph import Graph as ADBGraph
from pandas import DataFrame
from torch import Tensor, cat, tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType

from .abc import Abstract_ADBPyG_Adapter
from .controller import ADBPyG_Controller
from .exceptions import ADBMetagraphError, PyGMetagraphError
from .typings import (
    ADBMap,
    ADBMetagraph,
    ADBMetagraphValues,
    Json,
    PyGMap,
    PyGMetagraph,
    PyGMetagraphValues,
)
from .utils import logger, progress, validate_adb_metagraph, validate_pyg_metagraph


class ADBPyG_Adapter(Abstract_ADBPyG_Adapter):
    """ArangoDB-PyG adapter.

    :param db: A python-arango database instance
    :type db: arango.database.Database
    :param controller: The ArangoDB-PyG controller, used to prepare
        nodes & edges before ArangoDB insertion, optionally re-defined
        by the user if needed. Defaults to `ADBPyG_Controller`.
    :type controller: adbpyg_adapter.controller.ADBPyG_Controller
    :param logging_lvl: Defaults to logging.INFO. Other useful options are
        logging.DEBUG (more verbose), and logging.WARNING (less verbose).
    :type logging_lvl: str | int
    :raise TypeError: If invalid parameter types
    """

    def __init__(
        self,
        db: Database,
        controller: ADBPyG_Controller = ADBPyG_Controller(),
        logging_lvl: Union[str, int] = logging.INFO,
    ):
        self.set_logging(logging_lvl)

        if not issubclass(type(db), Database):
            msg = "**db** parameter must inherit from arango.database.Database"
            raise TypeError(msg)

        if not issubclass(type(controller), ADBPyG_Controller):
            msg = "**controller** parameter must inherit from ADBPyG_Controller"
            raise TypeError(msg)

        self.__db = db
        self.__cntrl = controller

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
        self,
        name: str,
        metagraph: ADBMetagraph,
        preserve_adb_keys: bool = False,
        **query_options: Any,
    ) -> Union[Data, HeteroData]:
        """Create a PyG graph from ArangoDB data. DOES carry
            over node/edge features/labels, via the **metagraph**.

        :param name: The PyG graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to PyG, along with collection-level specifications to indicate
            which ArangoDB attributes will become PyG features/labels.

            The current supported **metagraph** values are:
                1) Set[str]: The set of PyG-ready ArangoDB attributes to store
                    in your PyG graph.

                2) Dict[str, str]: The PyG property name mapped to the ArangoDB
                    attribute name that stores your PyG ready data.

                3) Dict[str, Dict[str, None | Callable]]:
                    The PyG property name mapped to a dictionary, which maps your
                    ArangoDB attribute names to a callable Python Class
                    (i.e has a `__call__` function defined), or to None
                    (if the ArangoDB attribute is already a list of numerics).
                    NOTE: The `__call__` function must take as input a Pandas DataFrame,
                    and must return a PyTorch Tensor.

                4) Dict[str, Callable[[pandas.DataFrame], torch.Tensor]]:
                    The PyG property name mapped to a user-defined function
                    for custom behaviour. NOTE: The function must take as input
                    a Pandas DataFrame, and must return a PyTorch Tensor.

            See below for examples of **metagraph**.
        :type metagraph: adbpyg_adapter.typings.ADBMetagraph
        :param preserve_adb_keys: NOTE: EXPERIMENTAL FEATURE. USE AT OWN RISK.
            If True, preserves the ArangoDB Vertex & Edge _key values into
            the PyG graph. Users can then re-import their PyG graph into
            ArangoDB using the same _key values via the following method:

            .. code-block:: python
            adbpyg_adapter.pyg_to_arangodb(
                graph_name, pyg_graph, ..., on_duplicate="update"
            )

            NOTE: If your ArangoDB graph is Homogeneous, the ArangoDB keys will
            be preserved under `_v_key` & `_e_key` in your PyG graph. If your
            ArangoDB graph is Heterogeneous, the ArangoDB keys will be preserved
            under `_key` in your PyG graph.
        :type preserve_adb_keys: bool
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid metagraph.

        **metagraph** examples

        1)
        .. code-block:: python
        {
            "vertexCollections": {
                "v0": {'x', 'y'}, # equivalent to {'x': 'x', 'y': 'y'}
                "v1": {'x'},
                "v2": {'x'},
            },
            "edgeCollections": {
                "e0": {'edge_attr'},
                "e1": {'edge_weight'},
            },
        }

        The metagraph above specifies that each document
        within the "v0" ArangoDB collection has a "pre-built" feature matrix
        named "x", and also has a node label named "y".
        We map these keys to the "x" and "y" properties of the PyG graph.

        2)
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
        within the "v0" ArangoDB collection has a "pre-built" feature matrix
        named "v0_features", and also has a node label named "label".
        We map these keys to the "x" and "y" properties of the PyG graph.

        3)
        .. code-block:: python
        from adbpyg_adapter.encoders import IdentityEncoder, CategoricalEncoder

        {
            "vertexCollections": {
                "Movies": {
                    "x": {
                        "Action": IdentityEncoder(dtype=torch.long),
                        "Drama": IdentityEncoder(dtype=torch.long),
                        'Misc': None
                    },
                    "y": "Comedy",
                },
                "Users": {
                    "x": {
                        "Gender": CategoricalEncoder(),
                        "Age": IdentityEncoder(dtype=torch.long),
                    }
                },
            },
            "edgeCollections": {
                "Ratings": { "edge_weight": "Rating" }
            },
        }

        The metagraph above will build the "Movies" feature matrix 'x'
        using the ArangoDB 'Action', 'Drama' & 'misc' attributes, by relying on
        the user-specified Encoders (see adbpyg_adapter.encoders for examples).
        NOTE: If the mapped value is `None`, then it assumes that the ArangoDB attribute
        value is a list containing numerical values only.

        4)
        .. code-block:: python
        def udf_v0_x(v0_df):
            # process v0_df here to return v0 "x" feature matrix
            # ...
            return torch.tensor(v0_df["x"].to_list())

        def udf_v1_x(v1_df):
            # process v1_df here to return v1 "x" feature matrix
            # ...
            return torch.tensor(v1_df["x"].to_list())

        {
            "vertexCollections": {
                "v0": {
                    "x": udf_v0_x, # named functions
                    "y": (lambda df: tensor(df["y"].to_list())), # lambda functions
                },
                "v1": {"x": udf_v1_x},
                "v2": {"x": (lambda df: tensor(df["x"].to_list()))},
            },
            "edgeCollections": {
                "e0": {"edge_attr": (lambda df: tensor(df["edge_attr"].to_list()))},
            },
        }

        The metagraph above provides an interface for a user-defined function to
        build a PyG-ready Tensor from a DataFrame equivalent to the
        associated ArangoDB collection.
        """
        logger.debug(f"--arangodb_to_pyg('{name}')--")

        validate_adb_metagraph(metagraph)

        is_homogeneous = (
            len(metagraph["vertexCollections"]) == 1
            and len(metagraph["edgeCollections"]) == 1
        )

        # Maps ArangoDB Vertex _keys to PyG Node ids
        adb_map: ADBMap = defaultdict(dict)

        data = Data() if is_homogeneous else HeteroData()

        for v_col, meta in metagraph["vertexCollections"].items():
            logger.debug(f"Preparing '{v_col}' vertices")

            df = self.__fetch_adb_docs(v_col, meta == {}, query_options)
            adb_map[v_col] = {
                adb_id: pyg_id for pyg_id, adb_id in enumerate(df["_key"])
            }

            node_data: NodeStorage = data if is_homogeneous else data[v_col]
            self.__set_pyg_data(meta, node_data, df)

            if preserve_adb_keys:
                k = "_v_key" if is_homogeneous else "_key"
                node_data[k] = list(adb_map[v_col].keys())

        et_df: DataFrame
        v_cols: List[str] = list(metagraph["vertexCollections"].keys())
        for e_col, meta in metagraph.get("edgeCollections", {}).items():
            logger.debug(f"Preparing '{e_col}' edges")

            df = self.__fetch_adb_docs(e_col, meta == {}, query_options)
            df[["from_col", "from_key"]] = df["_from"].str.split("/", 1, True)
            df[["to_col", "to_key"]] = df["_to"].str.split("/", 1, True)

            for (from_col, to_col), count in (
                df[["from_col", "to_col"]].value_counts().items()
            ):
                edge_type = (from_col, e_col, to_col)
                if from_col not in v_cols or to_col not in v_cols:
                    logger.debug(f"Skipping {edge_type}")
                    continue  # partial edge collection import to pyg

                logger.debug(f"Preparing {count} '{edge_type}' edges")

                # Get the edge data corresponding to the current edge type
                et_df = df[(df["from_col"] == from_col) & (df["to_col"] == to_col)]
                adb_map[edge_type] = {
                    adb_id: pyg_id for pyg_id, adb_id in enumerate(et_df["_key"])
                }

                from_nodes = et_df["from_key"].map(adb_map[from_col]).tolist()
                to_nodes = et_df["to_key"].map(adb_map[to_col]).tolist()

                edge_data: EdgeStorage = data if is_homogeneous else data[edge_type]
                edge_data.edge_index = tensor([from_nodes, to_nodes])
                self.__set_pyg_data(meta, edge_data, et_df)

                if preserve_adb_keys:
                    k = "_e_key" if is_homogeneous else "_key"
                    edge_data[k] = list(adb_map[edge_type].keys())

        logger.info(f"Created PyG '{name}' Graph")
        return data

    def arangodb_collections_to_pyg(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        preserve_adb_keys: bool = False,
        **query_options: Any,
    ) -> Union[Data, HeteroData]:
        """Create a PyG graph from ArangoDB collections. Due to risk of
            ambiguity, this method DOES NOT transfer ArangoDB attributes to PyG.

        :param name: The PyG graph name.
        :type name: str
        :param v_cols: The set of ArangoDB vertex collections to import to PyG.
        :type v_cols: Set[str]
        :param e_cols: The set of ArangoDB edge collections to import to PyG.
        :type e_cols: Set[str]
        :param preserve_adb_keys: NOTE: EXPERIMENTAL FEATURE. USE AT OWN RISK.
            If True, preserves the ArangoDB Vertex & Edge _key values into
            the PyG graph. Users can then re-import their PyG graph into
            ArangoDB using the same _key values via the following method:

            .. code-block:: python
            adbpyg_adapter.pyg_to_arangodb(
                graph_name, pyg_graph, ..., on_duplicate="update"
            )

            NOTE: If your ArangoDB graph is Homogeneous, the ArangoDB keys will
            be preserved under `_v_key` & `_e_key` in your PyG graph. If your
            ArangoDB graph is Heterogeneous, the ArangoDB keys will be preserved
            under `_key` in your PyG graph.
        :type preserve_adb_keys: bool
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        metagraph: ADBMetagraph = {
            "vertexCollections": {col: dict() for col in v_cols},
            "edgeCollections": {col: dict() for col in e_cols},
        }

        return self.arangodb_to_pyg(name, metagraph, preserve_adb_keys, **query_options)

    def arangodb_graph_to_pyg(
        self, name: str, preserve_adb_keys: bool = False, **query_options: Any
    ) -> Union[Data, HeteroData]:
        """Create a PyG graph from an ArangoDB graph. Due to risk of
            ambiguity, this method DOES NOT transfer ArangoDB attributes to PyG.

        :param name: The ArangoDB graph name.
        :type name: str
        :param preserve_adb_keys: NOTE: EXPERIMENTAL FEATURE. USE AT OWN RISK.
            If True, preserves the ArangoDB Vertex & Edge _key values into
            the PyG graph. Users can then re-import their PyG graph into
            ArangoDB using the same _key values via the following method:

            .. code-block:: python
            adbpyg_adapter.pyg_to_arangodb(
                graph_name, pyg_graph, ..., on_duplicate="update"
            )

            NOTE: If your ArangoDB graph is Homogeneous, the ArangoDB keys will
            be preserved under `_v_key` & `_e_key` in your PyG graph. If your
            ArangoDB graph is Heterogeneous, the ArangoDB keys will be preserved
            under `_key` in your PyG graph.
        :type preserve_adb_keys: bool
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_pyg(
            name, v_cols, e_cols, preserve_adb_keys, **query_options
        )

    def pyg_to_arangodb(
        self,
        name: str,
        pyg_g: Union[Data, HeteroData],
        metagraph: PyGMetagraph = {},
        explicit_metagraph: bool = True,
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ADBGraph:
        """Create an ArangoDB graph from a PyG graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param pyg_g: The existing PyG graph.
        :type pyg_g: Data | HeteroData
        :param metagraph: An optional object mapping the PyG keys of
            the node & edge data to strings, list of strings, or user-defined
            functions. NOTE: Unlike the metagraph for ArangoDB to PyG, this
            one is optional.

            The current supported **metagraph** values are:
                1) Set[str]: The set of PyG data properties to store
                    in your ArangoDB database.

                2) Dict[str, str]: The PyG property name mapped to the ArangoDB
                    attribute name that will be used to store your PyG data in ArangoDB.

                3) List[str]: A list of ArangoDB attribute names that will break down
                    your tensor data, resulting in one ArangoDB attribute per feature.
                    Must know the number of node/edge features in advance to take
                    advantage of this metagraph value type.

                4) Dict[str, Callable[[pandas.DataFrame], torch.Tensor]]:
                    The PyG property name mapped to a user-defined function
                    for custom behaviour. NOTE: The function must take as input
                    a PyTorch Tensor, and must return a Pandas DataFrame.

            See below for an example of **metagraph**.
        :type metagraph: adbpyg_adapter.typings.PyGMetagraph
        :param explicit_metagraph: Whether to take the metagraph at face value or not.
            If False, node & edge types OMITTED from the metagraph will be
            brought over into ArangoDB. Also applies to node & edge attributes.
            Defaults to True.
        :type explicit_metagraph: bool
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections. Defaults to False.
        :type overwrite_graph: bool
        :param import_options: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type import_options: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        :raise adbpyg_adapter.exceptions.PyGMetagraphError: If invalid metagraph.

        **metagraph** example

        .. code-block:: python
        def y_tensor_to_2_column_dataframe(pyg_tensor):
            # A user-defined function to create two ArangoDB attributes
            # out of the 'y' label tensor
            label_map = {0: "Kiwi", 1: "Blueberry", 2: "Avocado"}

            df = pandas.DataFrame(columns=["label_num", "label_str"])
            df["label_num"] = pyg_tensor.tolist()
            df["label_str"] = df["label_num"].map(label_map)

            return df

        metagraph = {
            "nodeTypes": {
                "v0": {
                    "x": "features",  # 1)
                    "y": y_tensor_to_2_column_dataframe,  # 2)
                },
                "v1": {"x"} # 3)
            },
            "edgeTypes": {
                ("v0", "e0", "v0"): {"edge_attr": [ "a", "b"]}, # 4)
            },
        }

        The metagraph above accomplishes the following:
        1) Renames the PyG 'v0' 'x' feature matrix to 'features'
            when stored in ArangoDB.
        2) Builds a 2-column Pandas DataFrame from the 'v0' 'y' labels
            through a user-defined function for custom behaviour handling.
        3) Transfers the PyG 'v1' 'x' feature matrix under the same name.
        4) Dissasembles the 2-feature Tensor into two ArangoDB attributes,
            where each attribute holds one feature value.
        """
        logger.debug(f"--pyg_to_arangodb('{name}')--")

        validate_pyg_metagraph(metagraph)

        is_homogeneous = type(pyg_g) is Data
        if is_homogeneous and pyg_g.num_nodes == pyg_g.num_edges and not metagraph:
            msg = f"""
                Ambiguity Error: can't convert to ArangoDB,
                as the PyG graph has the same number
                of nodes & edges {pyg_g.num_nodes}.
                Please supply a PyG-ArangoDB metagraph to
                categorize your node & edge attributes.
            """
            raise ValueError(msg)

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

        if overwrite_graph:
            logger.debug("Overwrite graph flag is True. Deleting old graph.")
            self.__db.delete_graph(name, ignore_missing=True)

        if self.__db.has_graph(name):
            adb_graph = self.__db.graph(name)
        else:
            edge_definitions = self.etypes_to_edefinitions(edge_types)
            orphan_collections = self.ntypes_to_ocollections(node_types, edge_types)
            adb_graph = self.__db.create_graph(
                name, edge_definitions, orphan_collections
            )

        # Define PyG data properties
        node_data: NodeStorage
        edge_data: EdgeStorage

        n_meta = metagraph.get("nodeTypes", {})
        for n_type in node_types:
            node_data = pyg_g if is_homogeneous else pyg_g[n_type]

            meta = n_meta.get(n_type, {})
            empty_df = DataFrame(index=range(node_data.num_nodes))
            df = self.__set_adb_data(empty_df, meta, node_data, explicit_metagraph)

            if "_id" in df:
                pyg_map[n_type] = df["_id"].to_dict()
            else:
                if "_key" not in df:
                    df["_key"] = df.index.astype(str)

                pyg_map[n_type] = (n_type + "/" + df["_key"]).to_dict()

            if type(self.__cntrl) is not ADBPyG_Controller:
                f = lambda n: self.__cntrl._prepare_pyg_node(n, n_type)
                df = df.apply(f, axis=1)

            self.__insert_adb_docs(n_type, df, import_options)

        e_meta = metagraph.get("edgeTypes", {})
        for e_type in edge_types:
            edge_data = pyg_g if is_homogeneous else pyg_g[e_type]
            src_n_type, _, dst_n_type = e_type

            columns = ["_from", "_to"]
            meta = e_meta.get(e_type, {})
            df = DataFrame(zip(*(edge_data.edge_index.tolist())), columns=columns)
            df = self.__set_adb_data(df, meta, edge_data, explicit_metagraph)

            df["_from"] = (
                df["_from"].map(pyg_map[src_n_type])
                if pyg_map[src_n_type]
                else src_n_type + "/" + df["_from"].astype(str)
            )

            df["_to"] = (
                df["_to"].map(pyg_map[dst_n_type])
                if pyg_map[dst_n_type]
                else dst_n_type + "/" + df["_to"].astype(str)
            )

            if type(self.__cntrl) is not ADBPyG_Controller:
                f = lambda e: self.__cntrl._prepare_pyg_edge(e, e_type)
                df = df.apply(f, axis=1)

            self.__insert_adb_docs(e_type, df, import_options)

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

        if not edge_types:
            return []

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

    def ntypes_to_ocollections(
        self, node_types: List[str], edge_types: List[EdgeType]
    ) -> List[str]:
        """Converts PyG node_types to ArangoDB orphan collections, if any.

        :param node_types: A list of strings representing the PyG node types.
        :type node_types: List[str]
        :param edge_types: A list of string triplets (str, str, str) for
            source node type, edge type and destination node type.
        :type edge_types: List[torch_geometric.typing.EdgeType]
        :return: ArangoDB Orphan Collections
        :rtype: List[str]
        """

        non_orphan_collections = set()
        for from_col, _, to_col in edge_types:
            non_orphan_collections.add(from_col)
            non_orphan_collections.add(to_col)

        orphan_collections = set(node_types) ^ non_orphan_collections
        return list(orphan_collections)

    def __fetch_adb_docs(
        self, col: str, empty_meta: bool, query_options: Any
    ) -> DataFrame:
        """Fetches ArangoDB documents within a collection. Returns the
            documents in a DataFrame.

        :param col: The ArangoDB collection.
        :type col: str
        :param empty_meta: Set to True if the metagraph specification
            for **col** is empty.
        :type empty_meta: bool
        :param query_options: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type query_options: Any
        :return: A DataFrame representing the ArangoDB documents.
        :rtype: pandas.DataFrame
        """
        # Only return the entire document if **empty_meta** is False
        aql = f"""
            FOR doc IN @@col
                RETURN {
                    "{ _key: doc._key, _from: doc._from, _to: doc._to }"
                    if empty_meta
                    else "doc"
                }
        """

        with progress(
            f"(ADB → PyG): {col}",
            text_style="#8929C2",
            spinner_style="#40A6F5",
        ) as p:
            p.add_task("__fetch_adb_docs")

            return DataFrame(
                self.__db.aql.execute(
                    aql, count=True, bind_vars={"@col": col}, **query_options
                )
            )

    def __insert_adb_docs(
        self, doc_type: Union[str, EdgeType], df: DataFrame, kwargs: Any
    ) -> None:
        """Insert ArangoDB documents into their ArangoDB collection.

        :param doc_type: The node or edge type of the soon-to-be ArangoDB documents
        :type doc_type: str | tuple[str, str, str]
        :param df: To-be-inserted ArangoDB documents, formatted as a DataFrame
        :type df: pandas.DataFrame
        :param kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        """
        col = doc_type if type(doc_type) is str else doc_type[1]

        with progress(
            f"(PyG → ADB): {doc_type} ({len(df)})",
            text_style="#97C423",
            spinner_style="#994602",
        ) as p:
            p.add_task("__insert_adb_docs")

            docs = df.to_dict("records")
            result = self.__db.collection(col).import_bulk(docs, **kwargs)
            logger.debug(result)

    def __set_pyg_data(
        self,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        pyg_data: Union[Data, NodeStorage, EdgeStorage],
        df: DataFrame,
    ) -> None:
        """A helper method to build the PyG NodeStorage or EdgeStorage object
        for the PyG graph. Is responsible for preparing the input **meta** such
        that it becomes a dictionary, and building PyG-ready tensors from the
        ArangoDB DataFrame **df**.

        :param meta: The metagraph associated to the current ArangoDB vertex or
            edge collection. e.g metagraph['vertexCollections']['Users']
        :type meta: Set[str] |  Dict[str, adbpyg_adapter.typings.ADBMetagraphValues]
        :param pyg_data: The NodeStorage or EdgeStorage of the current
            PyG node or edge type.
        :type pyg_data: torch_geometric.data.storage.(NodeStorage | EdgeStorage)
        :param df: The DataFrame representing the ArangoDB collection data
        :type df: pandas.DataFrame
        """
        valid_meta: Dict[str, ADBMetagraphValues]
        valid_meta = meta if type(meta) is dict else {m: m for m in meta}

        for k, v in valid_meta.items():
            pyg_data[k] = self.__build_tensor_from_dataframe(df, k, v)

    def __set_adb_data(
        self,
        df: DataFrame,
        meta: Union[Set[str], Dict[Any, PyGMetagraphValues]],
        pyg_data: Union[Data, NodeStorage, EdgeStorage],
        explicit_metagraph: bool,
    ) -> DataFrame:
        """A helper method to build the ArangoDB Dataframe for the given
        collection. Is responsible for creating "sub-DataFrames" from PyG tensors
        or lists, and appending them to the main dataframe **df**. If the data
        does not adhere to the supported types, or is not of specific length,
        then it is silently skipped.

        :param df: The main ArangoDB DataFrame containing (at minimum)
            the vertex/edge _id or _key attribute.
        :type df: pandas.DataFrame
        :param meta: The metagraph associated to the
            current PyG node or edge type. e.g metagraph['nodeTypes']['v0']
        :type meta: Set[str] | Dict[Any, adbpyg_adapter.typings.PyGMetagraphValues]
        :param pyg_data: The NodeStorage or EdgeStorage of the current
            PyG node or edge type.
        :type pyg_data: torch_geometric.data.storage.(NodeStorage | EdgeStorage)
        :param explicit_metagraph: The value of **explicit_metagraph**
            in **pyg_to_arangodb**.
        :type explicit_metagraph: bool
        :return: The completed DataFrame for the (soon-to-be) ArangoDB collection.
        :rtype: pandas.DataFrame
        :raise ValueError: If an unsupported PyG data value is found.
        """
        logger.debug(
            f"__set_adb_data(df, {meta}, {type(pyg_data)}, {explicit_metagraph}"
        )

        valid_meta: Dict[Any, PyGMetagraphValues]
        valid_meta = meta if type(meta) is dict else {m: m for m in meta}

        if explicit_metagraph:
            pyg_keys = set(valid_meta.keys())
        else:
            # can't do keys() (not compatible with Homogeneous graphs)
            pyg_keys = set(k for k, _ in pyg_data.items())

        for k in pyg_keys:
            if k == "edge_index":
                continue

            data = pyg_data[k]
            meta_val = valid_meta.get(k, str(k))

            if type(meta_val) is str and type(data) is list and len(data) == len(df):
                if meta_val in ["_v_key", "_e_key"]:  # Homogeneous situation
                    meta_val = "_key"

                df = df.join(DataFrame(data, columns=[meta_val]))

            if type(data) is Tensor and len(data) == len(df):
                df = df.join(self.__build_dataframe_from_tensor(data, k, meta_val))

        return df

    def __build_tensor_from_dataframe(
        self,
        adb_df: DataFrame,
        meta_key: str,
        meta_val: ADBMetagraphValues,
    ) -> Tensor:
        """Constructs a PyG-ready Tensor from a DataFrame, based on
        the nature of the user-defined metagraph.

        :param adb_df: The DataFrame representing ArangoDB data.
        :type adb_df: pandas.DataFrame
        :param meta_key: The current ArangoDB-PyG metagraph key
        :type meta_key: str
        :param meta_val: The value mapped to **meta_key** to
            help convert **df** into a PyG-ready Tensor.
            e.g the value of `metagraph['vertexCollections']['users']['x']`.
        :type meta_val: adbpyg_adapter.typings.ADBMetagraphValues
        :return: A PyG-ready tensor equivalent to the dataframe
        :rtype: torch.Tensor
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid **meta_val**.
        """
        logger.debug(
            f"__build_tensor_from_dataframe(df, '{meta_key}', {type(meta_val)})"
        )

        if type(meta_val) is str:
            return tensor(adb_df[meta_val].to_list())

        if type(meta_val) is dict:
            data = []
            for attr, encoder in meta_val.items():
                if encoder is None:
                    data.append(tensor(adb_df[attr].to_list()))
                elif callable(encoder):
                    data.append(encoder(adb_df[attr]))
                else:  # pragma: no cover
                    msg = f"Invalid encoder for ArangoDB attribute '{attr}': {encoder}"
                    raise ADBMetagraphError(msg)

            return cat(data, dim=-1)

        if callable(meta_val):
            # **meta_val** is a user-defined that returns a tensor
            user_defined_result = meta_val(adb_df)

            if type(user_defined_result) is not Tensor:  # pragma: no cover
                msg = f"Invalid return type for function {meta_val} ('{meta_key}')"
                raise ADBMetagraphError(msg)

            return user_defined_result

        raise ADBMetagraphError(f"Invalid {meta_val} type")  # pragma: no cover

    def __build_dataframe_from_tensor(
        self,
        pyg_tensor: Tensor,
        meta_key: Any,
        meta_val: PyGMetagraphValues,
    ) -> DataFrame:
        """Builds a DataFrame from PyG Tensor, based on
        the nature of the user-defined metagraph.

        :param pyg_tensor: The Tensor representing PyG data.
        :type pyg_tensor: torch.Tensor
        :param meta_key: The current PyG-ArangoDB metagraph key
        :type meta_key: Any
        :param meta_val: The value mapped to the PyG-ArangoDB metagraph key to
            help convert **tensor** into a DataFrame.
            e.g the value of `metagraph['nodeTypes']['users']['x']`.
        :type meta_val: adbpyg_adapter.typings.PyGMetagraphValues
        :return: A DataFrame equivalent to the Tensor
        :rtype: pandas.DataFrame
        :raise adbpyg_adapter.exceptions.PyGMetagraphError: If invalid **meta_val**.
        """
        logger.debug(
            f"__build_dataframe_from_tensor(df, '{meta_key}', {type(meta_val)})"
        )

        if type(meta_val) is str:
            df = DataFrame(columns=[meta_val])
            df[meta_val] = pyg_tensor.tolist()
            return df

        if type(meta_val) is list:
            num_features = pyg_tensor.size()[1]
            if len(meta_val) != num_features:  # pragma: no cover
                msg = f"""
                    Invalid list length for **meta_val** ('{meta_key}'):
                    List length must match the number of
                    features found in the tensor ({num_features}).
                """
                raise PyGMetagraphError(msg)

            df = DataFrame(columns=meta_val)
            df[meta_val] = pyg_tensor.tolist()
            return df

        if callable(meta_val):
            # **meta_val** is a user-defined function that returns a dataframe
            user_defined_result = meta_val(pyg_tensor)

            if type(user_defined_result) is not DataFrame:  # pragma: no cover
                msg = f"Invalid return type for function {meta_val} ('{meta_key}')"
                raise PyGMetagraphError(msg)

            return user_defined_result

        raise PyGMetagraphError(f"Invalid {meta_val} type")  # pragma: no cover
