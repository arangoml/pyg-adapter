#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Union

try:
    # https://github.com/arangoml/pyg-adapter/issues/4
    from cudf import DataFrame
except ModuleNotFoundError:
    from pandas import DataFrame

from arango.database import Database
from arango.graph import Graph as ADBGraph
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

        # Maps ArangoDB vertex keys to PyG node IDs
        self.adb_map: ADBMap = defaultdict(lambda: defaultdict(dict))

        # Maps PyG node IDs to ArangoDB vertex keys
        self.pyg_map: PyGMap = defaultdict(lambda: defaultdict(dict))

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
        **query_options: Any,
    ) -> Union[Data, HeteroData]:
        """Create a PyG graph from ArangoDB data. DOES carry
            over node/edge features/labels, via the **metagraph**.

        :param name: The PyG graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to PyG, along with collection-level specifications to indicate
            which ArangoDB attributes will become PyG features/labels.
            See below for examples of **metagraph**.
        :type metagraph: adbpyg_adapter.typings.ADBMetagraph
        :param query_options: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type query_options: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid metagraph.

        The current supported **metagraph** values are:
            1) str: The name of the ArangoDB attribute that stores your PyG-ready data

            2) Dict[str, Callable[[(pandas | cudf).DataFrame], torch.Tensor] | None]:
                A dictionary mapping ArangoDB attributes to a callable Python Class
                (i.e has a `__call__` function defined), or to None
                (if the ArangoDB attribute is already a list of numerics).

            3) Callable[[(pandas | cudf).DataFrame], torch.Tensor]:
                A user-defined function for custom behaviour.
                NOTE: The function return type MUST be a tensor.

        1)
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
        We map these keys to the "x" and "y" properties of a standard
        PyG graph.

        2)
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
                "Ratings": {
                    "edge_weight": "Rating"
                }
            },
        }

        The metagraph above will build the "Movies" feature matrix 'x'
        using the ArangoDB 'Action', 'Drama' & 'misc' attributes, by relying on
        the user-specified Encoders (see adbpyg_adapter.encoders for examples).
        NOTE: If the mapped value is `None`, then it assumes that the ArangoDB attribute
        value is a list containing numerical values only.

        3)
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

        adb_map = self.adb_map[name]

        data = Data() if is_homogeneous else HeteroData()

        for v_col, meta in metagraph["vertexCollections"].items():
            logger.debug(f"Preparing '{v_col}' vertices")

            df = self.__fetch_adb_docs(v_col, meta == {}, query_options)
            adb_map[v_col] = {adb_id: pyg_id for pyg_id, adb_id in enumerate(df["_id"])}

            node_data: NodeStorage = data if is_homogeneous else data[v_col]
            for k, v in meta.items():
                node_data[k] = self.__build_tensor_from_dataframe(df, k, v)

        et_df: DataFrame
        v_cols: List[str] = list(metagraph["vertexCollections"].keys())
        for e_col, meta in metagraph.get("edgeCollections", {}).items():
            logger.debug(f"Preparing '{e_col}' edges")

            df = self.__fetch_adb_docs(e_col, meta == {}, query_options)
            df["from_col"] = df["_from"].str.split("/").str[0]
            df["to_col"] = df["_to"].str.split("/").str[0]

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
                    adb_id: pyg_id for pyg_id, adb_id in enumerate(et_df["_id"])
                }

                from_nodes = et_df["_from"].map(adb_map[from_col]).tolist()
                to_nodes = et_df["_to"].map(adb_map[to_col]).tolist()

                edge_data: EdgeStorage = data if is_homogeneous else data[edge_type]
                edge_data.edge_index = tensor([from_nodes, to_nodes])
                for k, v in meta.items():
                    edge_data[k] = self.__build_tensor_from_dataframe(et_df, k, v)

        logger.info(f"Created PyG '{name}' Graph")
        return data

    def arangodb_collections_to_pyg(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
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

        return self.arangodb_to_pyg(name, metagraph, **query_options)

    def arangodb_graph_to_pyg(
        self, name: str, **query_options: Any
    ) -> Union[Data, HeteroData]:
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
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        graph = self.__db.graph(name)
        v_cols = graph.vertex_collections()
        e_cols = {col["edge_collection"] for col in graph.edge_definitions()}

        return self.arangodb_collections_to_pyg(name, v_cols, e_cols, **query_options)

    def pyg_to_arangodb(
        self,
        name: str,
        pyg_g: Union[Data, HeteroData],
        metagraph: PyGMetagraph = {},
        explicit_metagraph: bool = True,
        overwrite_graph: bool = False,
        preserve_adb_keys: bool = False,  # TODO: explain
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
            one is optional. See below for an example of **metagraph**.
        :type metagraph: adbpyg_adapter.typings.PyGMetagraph
        :param explicit_metagraph: Whether to take the metagraph at face value or not.
            If False, node & edge types OMITTED from the metagraph will be
            brought over into ArangoDB. Defaults to True.
        :type explicit_metagraph: bool
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections. Defaults to False.
        :type overwrite_graph: bool
        :param preserve_adb_keys: NOTE: EXPERIMENTAL FEATURE. USE AT OWN RISK.
            If True, relies on **adbpyg_adapter.adb_map[**name**]** to map the
            PyG Node & Edge IDs back into ArangoDB Vertex & Edge IDs. Assumes that
            the user has a valid ADB Map for **name**.
            An ADB Map can be built by running ArangoDB to PyG operation for
            the same **name**, or by creating the map manually. Defaults to False.

            .. code-block:: python
            adbpyg_adapter.pyg_to_arangodb(
                name, pyg_g, preserve_adb_keys=True, on_duplicate="update"
            )
        :type preserve_adb_keys: bool
        :param import_options: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type import_options: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        :raise adbpyg_adapter.exceptions.PyGMetagraphError: If invalid metagraph.

        The current supported **metagraph** values are:
            1) str: The name of the ArangoDB attribute that will store your PyG data

            2) List[str]: A list of ArangoDB attribute names that will break down
                your tensor data to have one ArangoDB attribute per tensor value.

            3) Callable[[torch.Tensor], (pandas | cudf).DataFrame]:
                A user-defined function for custom behaviour.
                NOTE: The function return type MUST be a DataFrame (pandas or cudf).

        1) Here is an example entry for parameter **metagraph**:
        .. code-block:: python

        def v2_x_to_pandas_dataframe(t: Tensor):
            # The parameter **t** is the tensor representing
            # the feature matrix 'x' of the 'v2' node type.

            df = (pandas | cudf).DataFrame(columns=["v2_features"])
            df["v2_features"] = t.tolist()
            # do more things with df["v2_features"] here ...
            return df

        {
            "nodeTypes": {
                "v0": {'x': 'v0_features', 'y': 'label'}, # supports str as value
                "v1": {'x': ['x_0', 'x_1', ..., 'x_77']}, # supports list as value
                "v2": {'x': v2_x_to_pandas_dataframe}, # supports function as value
            },
            "edgeTypes": {
                ('v0', 'e0', 'v0'): {'edge_weight': 'v0_e0_v0_weight'}:
                ('v0', 'e0', 'v1'): {'edge_weight': 'v0_e0_v1_weight'},
                # etc...
            },
        }

        Using the metagraph above will store the v0 "x" feature matrix as
        "v0_features" in ArangoDB, and store the v0 "y" label tensor as
        "label". Furthemore, the v1 "x" feature matrix is broken down in order to
        associate one ArangoDB attribute per feature. Lastly, the v2 feature matrix
        is converted into a DataFrame via a user-defined function.
        """
        logger.debug(f"--pyg_to_arangodb('{name}')--")

        validate_pyg_metagraph(metagraph)

        is_homogeneous = type(pyg_g) is Data

        pyg_map = self.pyg_map[name]
        if preserve_adb_keys:
            if is_homogeneous:
                msg = """
                    **preserve_adb_keys** does not yet support
                    homogeneous graphs (i.e type(pyg_g) is Data).
                """
                raise ValueError(msg)

            if self.adb_map[name] == {}:
                msg = f"""
                    Parameter **preserve_adb_keys** was enabled,
                    but no ArangoDB Map was found for graph {name} in
                    **self.adb_map**.
                """
                raise ValueError(msg)

            # Build the reverse map
            for k, map in self.adb_map[name].items():
                pyg_map[k].update({pyg_id: adb_id for adb_id, pyg_id in map.items()})

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
            num_nodes = node_data.num_nodes

            df = DataFrame(index=range(num_nodes))
            if preserve_adb_keys:
                num_node_keys = len(pyg_map[n_type])

                if num_nodes != num_node_keys:
                    msg = f"""
                        {num_nodes} does not match
                        number of node keys in pyg_map
                        ({num_node_keys}) for {n_type}
                    """
                    raise ValueError(msg)

                df["_id"] = df.index.map(pyg_map[n_type])
            else:
                df["_key"] = df.index.astype(str)

            meta = n_meta.get(n_type, {})
            pyg_keys = (
                set(meta.keys())
                if explicit_metagraph
                else {k for k, _ in node_data.items()}  # can't do node_data.keys()
            )
            df = self.__finish_adb_dataframe(df, meta, pyg_keys, node_data)

            if type(self.__cntrl) is not ADBPyG_Controller:
                f = lambda n: self.__cntrl._prepare_pyg_node(n, n_type)
                df = df.apply(f, axis=1)

            self.__insert_adb_docs(n_type, df, import_options)

        e_meta = metagraph.get("edgeTypes", {})
        for e_type in edge_types:
            edge_data = pyg_g if is_homogeneous else pyg_g[e_type]
            from_col, _, to_col = e_type

            columns = ["_from", "_to"]
            df = DataFrame(zip(*(edge_data.edge_index.tolist())), columns=columns)
            if preserve_adb_keys:
                df["_id"] = df.index.map(pyg_map[e_type])
                df["_from"] = df["_from"].map(pyg_map[from_col])
                df["_to"] = df["_to"].map(pyg_map[to_col])
            else:
                df["_from"] = from_col + "/" + df["_from"].astype(str)
                df["_to"] = to_col + "/" + df["_to"].astype(str)

            meta = e_meta.get(e_type, {})
            pyg_keys = (
                set(meta.keys())
                if explicit_metagraph
                else {k for k, _ in edge_data.items()}  # can't do edge_data.keys()
            )
            df = self.__finish_adb_dataframe(df, meta, pyg_keys, edge_data)

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
        :rtype: (pandas | cudf).DataFrame
        """
        # Only return the entire document if **empty_meta** is False
        data = "{_id: doc._id, _from: doc._from, _to: doc._to}" if empty_meta else "doc"
        aql = f"""
            FOR doc IN @@col
                RETURN {data}
        """

        with progress(
            f"Export: {col}",
            text_style="#97C423",
            spinner_style="#7D3B04",
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
        :type df: (pandas | cudf).DataFrame
        :param kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        """
        col = doc_type if type(doc_type) is str else doc_type[1]

        # TODO: clean this up
        try:
            docs = df.to_dict("records")
        except TypeError:  # cudf does not support to_dict
            docs = df.to_pandas.to_dict("records")

        with progress(
            f"Import: {doc_type} ({len(docs)})",
            text_style="#825FE1",
            spinner_style="#3AA7F4",
        ) as p:
            p.add_task("__insert_adb_docs")

            result = self.__db.collection(col).import_bulk(docs, **kwargs)
            logger.debug(result)

    def __build_tensor_from_dataframe(
        self,
        adb_df: DataFrame,
        meta_key: str,
        meta_val: ADBMetagraphValues,
    ) -> Tensor:
        """Constructs a PyG-ready Tensor from a DataFrame, based on
        the nature of the user-defined metagraph.

        :param adb_df: The DataFrame representing ArangoDB data.
        :type adb_df: (pandas | cudf).DataFrame
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
        logger.debug(f"__build_tensor_from_dataframe(df, '{meta_key}', {meta_val})")

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

    def __finish_adb_dataframe(
        self,
        df: DataFrame,
        meta: Dict[Any, PyGMetagraphValues],
        pyg_keys: Set[Any],
        pyg_data: Union[NodeStorage, EdgeStorage],
    ) -> DataFrame:
        """A helper method to complete the ArangoDB Dataframe for the given
        collection. Is responsible for creating DataFrames from PyG tensors,
        and appending them to the main dataframe **df**.

        :param df: The main ArangoDB DataFrame containing (at minimum)
            the vertex/edge _id or _key attribute.
        :type df: (pandas | cudf).DataFrame
        :param meta: The metagraph associated to the
            current PyG node or edge type.
        :type meta: Dict[Any, adbpyg_adapter.typings.PyGMetagraphValues]
        :param pyg_keys: The set of PyG NodeStorage or EdgeStorage keys, retrieved
            either from the **meta** parameter (if **explicit_metagraph** is True),
            or from the **pyg_data** parameter (if **explicit_metagraph** is False).
        :type pyg_keys: Set[Any]
        :param pyg_data: The NodeStorage or EdgeStorage of the current
            PyG node or edge type.
        :type pyg_data: torch_geometric.data.storage.(NodeStorage | EdgeStorage)
        :return: The completed DataFrame for the (soon-to-be) ArangoDB collection.
        :rtype: (pandas | cudf).DataFrame
        """
        for k in pyg_keys:
            if k == "edge_index":
                continue

            t = pyg_data[k]
            if type(t) is Tensor and len(t) == len(df):
                v = meta.get(k, str(k))
                df = df.join(self.__build_dataframe_from_tensor(t, k, v))

        return df

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
        :rtype: (pandas | cudf).DataFrame
        :raise adbpyg_adapter.exceptions.PyGMetagraphError: If invalid **meta_val**.
        """
        logger.debug(f"__build_dataframe_from_tensor(df, '{meta_key}', {meta_val})")

        if type(meta_val) in [str, list]:
            columns = [meta_val] if type(meta_val) is str else meta_val

            df = DataFrame(columns=columns)
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
