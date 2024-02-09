#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from math import ceil
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from arango.cursor import Cursor
from arango.database import StandardDatabase
from arango.graph import Graph as ADBGraph
from pandas import DataFrame, Series
from rich.console import Group
from rich.live import Live
from rich.progress import Progress
from torch import Tensor, cat, tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType

from .abc import Abstract_ADBPyG_Adapter
from .controller import ADBPyG_Controller
from .exceptions import ADBMetagraphError, InvalidADBEdgesError, PyGMetagraphError
from .typings import (
    ADBMap,
    ADBMetagraph,
    ADBMetagraphValues,
    Json,
    PyGMap,
    PyGMetagraph,
    PyGMetagraphValues,
)
from .utils import (
    get_bar_progress,
    get_export_spinner_progress,
    get_import_spinner_progress,
    logger,
    validate_adb_metagraph,
    validate_pyg_metagraph,
)


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
        db: StandardDatabase,
        controller: ADBPyG_Controller = ADBPyG_Controller(),
        logging_lvl: Union[str, int] = logging.INFO,
    ):
        self.set_logging(logging_lvl)

        if not isinstance(db, StandardDatabase):
            msg = "**db** parameter must inherit from arango.database.StandardDatabase"
            raise TypeError(msg)

        if not isinstance(controller, ADBPyG_Controller):
            msg = "**controller** parameter must inherit from ADBPyG_Controller"
            raise TypeError(msg)

        self.__db = db
        self.__async_db = db.begin_async_execution(return_result=False)
        self.__cntrl = controller

        logger.info(f"Instantiated ADBPyG_Adapter with database '{db.name}'")

    @property
    def db(self) -> StandardDatabase:
        return self.__db  # pragma: no cover

    @property
    def cntrl(self) -> ADBPyG_Controller:
        return self.__cntrl  # pragma: no cover

    def set_logging(self, level: Union[int, str]) -> None:
        logger.setLevel(level)

    ###########################
    # Public: ArangoDB -> PyG #
    ###########################

    def arangodb_to_pyg(
        self,
        name: str,
        metagraph: ADBMetagraph,
        preserve_adb_keys: bool = False,
        strict: bool = True,
        **adb_export_kwargs: Any,
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
        :param strict: Set fault tolerance when loading a graph from ArangoDB. If set
            to false, this will ignore invalid edges (e.g. dangling/half edges).
        :type strict: bool
        :param adb_export_kwargs: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type adb_export_kwargs: Any
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

        # The PyG Data or HeteroData object
        data = Data() if is_homogeneous else HeteroData()

        v_cols: List[str] = list(metagraph["vertexCollections"].keys())

        ######################
        # Vertex Collections #
        ######################

        preserve_key = None
        if preserve_adb_keys:
            preserve_key = "_v_key" if is_homogeneous else "_key"

        for v_col, meta in metagraph["vertexCollections"].items():
            logger.debug(f"Preparing '{v_col}' vertices")

            node_data: NodeStorage = data if is_homogeneous else data[v_col]

            if preserve_key is not None:
                node_data[preserve_key] = []

            # 1. Fetch ArangoDB vertices
            v_col_cursor, v_col_size = self.__fetch_adb_docs(
                v_col, meta, **adb_export_kwargs
            )

            # 2. Process ArangoDB vertices
            self.__process_adb_cursor(
                "#8929C2",
                v_col_cursor,
                v_col_size,
                self.__process_adb_vertex_df,
                v_col,
                adb_map,
                meta,
                preserve_key,
                node_data=node_data,
            )

        ####################
        # Edge Collections #
        ####################

        preserve_key = None
        if preserve_adb_keys:
            preserve_key = "_e_key" if is_homogeneous else "_key"

        for e_col, meta in metagraph.get("edgeCollections", {}).items():
            logger.debug(f"Preparing '{e_col}' edges")

            # 1. Fetch ArangoDB edges
            e_col_cursor, e_col_size = self.__fetch_adb_docs(
                e_col, meta, **adb_export_kwargs
            )

            # 2. Process ArangoDB edges
            self.__process_adb_cursor(
                "#40A6F5",
                e_col_cursor,
                e_col_size,
                self.__process_adb_edge_df,
                e_col,
                adb_map,
                meta,
                preserve_key,
                data=data,
                v_cols=v_cols,
                strict=strict,
                is_homogeneous=is_homogeneous,
            )

        logger.info(f"Created PyG '{name}' Graph")
        return data

    def arangodb_collections_to_pyg(
        self,
        name: str,
        v_cols: Set[str],
        e_cols: Set[str],
        preserve_adb_keys: bool = False,
        strict: bool = True,
        **adb_export_kwargs: Any,
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
        :param strict: Set fault tolerance when loading a graph from ArangoDB. If set
            to false, this will ignore invalid edges (e.g. dangling/half edges).
        :type strict: bool
        :param adb_export_kwargs: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type adb_export_kwargs: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        metagraph: ADBMetagraph = {
            "vertexCollections": {col: dict() for col in v_cols},
            "edgeCollections": {col: dict() for col in e_cols},
        }

        return self.arangodb_to_pyg(
            name, metagraph, preserve_adb_keys, strict, **adb_export_kwargs
        )

    def arangodb_graph_to_pyg(
        self,
        name: str,
        preserve_adb_keys: bool = False,
        strict: bool = True,
        **adb_export_kwargs: Any,
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
        :param strict: Set fault tolerance when loading a graph from ArangoDB. If set
            to false, this will ignore invalid edges (e.g. dangling/half edges).
        :type strict: bool
        :param adb_export_kwargs: Keyword arguments to specify AQL query options when
            fetching documents from the ArangoDB instance. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.aql.AQL.execute
        :type adb_export_kwargs: Any
        :return: A PyG Data or HeteroData object
        :rtype: torch_geometric.data.Data | torch_geometric.data.HeteroData
        :raise adbpyg_adapter.exceptions.ADBMetagraphError: If invalid metagraph.
        """
        graph = self.__db.graph(name)
        v_cols: Set[str] = graph.vertex_collections()
        edge_definitions: List[Json] = graph.edge_definitions()
        e_cols: Set[str] = {c["edge_collection"] for c in edge_definitions}

        return self.arangodb_collections_to_pyg(
            name, v_cols, e_cols, preserve_adb_keys, strict, **adb_export_kwargs
        )

    ###########################
    # Public: PyG -> ArangoDB #
    ###########################

    def pyg_to_arangodb(
        self,
        name: str,
        pyg_g: Union[Data, HeteroData],
        metagraph: PyGMetagraph = {},
        explicit_metagraph: bool = True,
        overwrite_graph: bool = False,
        batch_size: Optional[int] = None,
        use_async: bool = False,
        **adb_import_kwargs: Any,
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
            If False, node & edge types OMITTED from the metagraph will still be
            brought over into ArangoDB. Also applies to node & edge attributes.
            Defaults to True.
        :type explicit_metagraph: bool
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections. Defaults to False.
        :type overwrite_graph: bool
        :param batch_size: Process the PyG Nodes & Edges in batches of size
            **batch_size**. Defaults to `None`, which processes each
            NodeStorage & EdgeStorage in one batch.
        :type batch_size: int
        :param use_async: Performs asynchronous ArangoDB ingestion if enabled.
            Defaults to False.
        :type use_async: bool
        :param adb_import_kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type adb_import_kwargs: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        :raise adbpyg_adapter.exceptions.PyGMetagraphError: If invalid metagraph.

        **metagraph** example

        .. code-block:: python
        def y_tensor_to_2_column_dataframe(pyg_tensor, adb_df):
            # A user-defined function to create two ArangoDB attributes
            # out of the 'y' label tensor
            label_map = {0: "Kiwi", 1: "Blueberry", 2: "Avocado"}

            adb_df["label_num"] = pyg_tensor.tolist()
            adb_df["label_str"] = adb_df["label_num"].map(label_map)

            return adb_df

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

        is_custom_controller = type(self.__cntrl) is not ADBPyG_Controller
        is_explicit_metagraph = metagraph != {} and explicit_metagraph
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

        # This maps the PyG Node ids to ArangoDB Vertex _keys
        pyg_map: PyGMap = defaultdict(dict)

        # Get the Node & Edge Types
        node_types, edge_types = self.__get_node_and_edge_types(
            name, pyg_g, metagraph, is_explicit_metagraph, is_homogeneous
        )

        # Create the ArangoDB Graph
        adb_graph = self.__create_adb_graph(
            name, overwrite_graph, node_types, edge_types
        )

        # Define the PyG data properties
        node_data: NodeStorage
        edge_data: EdgeStorage

        spinner_progress = get_import_spinner_progress("    ")

        #############
        # PyG Nodes #
        #############

        n_meta = metagraph.get("nodeTypes", {})
        for n_type in node_types:
            meta = n_meta.get(n_type, {})

            node_data = pyg_g if is_homogeneous else pyg_g[n_type]
            node_data_batch_size = batch_size or node_data.num_nodes

            start_index = 0
            end_index = min(node_data_batch_size, node_data.num_nodes)
            batches = ceil(node_data.num_nodes / node_data_batch_size)

            bar_progress = get_bar_progress(f"(PyG → ADB): '{n_type}'", "#97C423")
            bar_progress_task = bar_progress.add_task(n_type, total=node_data.num_nodes)

            with Live(Group(bar_progress, spinner_progress)):
                for _ in range(batches):
                    # 1. Process the Node batch
                    df = self.__process_pyg_node_batch(
                        n_type,
                        node_data,
                        meta,
                        pyg_map,
                        is_explicit_metagraph,
                        is_custom_controller,
                        start_index,
                        end_index,
                    )

                    bar_progress.advance(bar_progress_task, advance=len(df))

                    # 2. Insert the ArangoDB Node Documents
                    self.__insert_adb_docs(
                        spinner_progress, df, n_type, use_async, **adb_import_kwargs
                    )

                    # 3. Update the batch indices
                    start_index = end_index
                    end_index = min(
                        end_index + node_data_batch_size, node_data.num_nodes
                    )

        #############
        # PyG Edges #
        #############

        e_meta = metagraph.get("edgeTypes", {})
        for e_type in edge_types:
            meta = e_meta.get(e_type, {})

            edge_data = pyg_g if is_homogeneous else pyg_g[e_type]
            edge_data_batch_size = batch_size or edge_data.num_edges

            start_index = 0
            end_index = min(edge_data_batch_size, edge_data.num_edges)
            batches = ceil(edge_data.num_edges / edge_data_batch_size)

            bar_progress = get_bar_progress(f"(PyG → ADB): {e_type}", "#994602")
            bar_progress_task = bar_progress.add_task(e_type, total=edge_data.num_edges)

            with Live(Group(bar_progress, spinner_progress)):
                for _ in range(batches):
                    # 1. Process the Edge batch
                    df = self.__process_pyg_edge_batch(
                        e_type,
                        edge_data,
                        meta,
                        pyg_map,
                        is_explicit_metagraph,
                        is_custom_controller,
                        start_index,
                        end_index,
                    )

                    bar_progress.advance(bar_progress_task, advance=len(df))

                    # 2. Insert the ArangoDB Edge Documents
                    self.__insert_adb_docs(
                        spinner_progress, df, e_type[1], use_async, **adb_import_kwargs
                    )

                    # 3. Update the batch indices
                    start_index = end_index
                    end_index = min(
                        end_index + edge_data_batch_size, edge_data.num_edges
                    )

        logger.info(f"Created ArangoDB '{name}' Graph")
        return adb_graph

    ############################
    # Private: ArangoDB -> PyG #
    ############################

    def __fetch_adb_docs(
        self,
        col: str,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        **adb_export_kwargs: Any,
    ) -> Tuple[Cursor, int]:
        """ArangoDB -> PyG: Fetches ArangoDB documents within a collection.
        Returns the documents in a DataFrame.

        :param col: The ArangoDB collection.
        :type col: str
        :param meta: The MetaGraph associated to **col**
        :type meta: Set[str] | Dict[str, adbpyg_adapter.typings.ADBMetagraphValues]
        :param adb_export_kwargs: Keyword arguments to specify AQL query options
            when fetching documents from the ArangoDB instance.
        :type adb_export_kwargs: Any
        :return: A DataFrame representing the ArangoDB documents.
        :rtype: pandas.DataFrame
        """

        def get_aql_return_value(
            meta: Union[Set[str], Dict[str, ADBMetagraphValues]]
        ) -> str:
            """Helper method to formulate the AQL `RETURN` value based on
            the document attributes specified in **meta**
            """
            attributes = []

            if type(meta) is set:
                attributes = list(meta)

            elif type(meta) is dict:
                for value in meta.values():
                    if type(value) is str:
                        attributes.append(value)
                    elif type(value) is dict:
                        attributes.extend(list(value.keys()))
                    elif callable(value):
                        # Cannot determine which attributes to extract if UDFs are used
                        # Therefore we just return the entire document
                        return "doc"

            return f"""
                MERGE(
                    {{ _key: doc._key, _from: doc._from, _to: doc._to }},
                    KEEP(doc, {list(attributes)})
                )
            """

        col_size: int = self.__db.collection(col).count()

        with get_export_spinner_progress(f"ADB Export: '{col}' ({col_size})") as p:
            p.add_task(col)

            cursor: Cursor = self.__db.aql.execute(
                f"FOR doc IN @@col RETURN {get_aql_return_value(meta)}",
                bind_vars={"@col": col},
                **{**adb_export_kwargs, **{"stream": True}},
            )

            return cursor, col_size

    def __process_adb_cursor(
        self,
        progress_color: str,
        cursor: Cursor,
        col_size: int,
        process_adb_df: Callable[..., int],
        col: str,
        adb_map: ADBMap,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        preserve_key: Optional[str],
        **kwargs: Any,
    ) -> None:
        """ArangoDB -> PyG: Processes the ArangoDB Cursors for vertices and edges.

        :param progress_color: The progress bar color.
        :type progress_color: str
        :param cursor: The ArangoDB cursor for the current **col**.
        :type cursor: arango.cursor.Cursor
        :param col_size: The size of **col**.
        :type col_size: int
        :param process_adb_df: The function to process the cursor data
            (in the form of a Dataframe).
        :type process_adb_df: Callable
        :param col: The ArangoDB collection for the current **cursor**.
        :type col: str
        :param adb_map: The ArangoDB -> PyG map.
        :type adb_map: adbpyg_adapter.typings.ADBMap
        :param meta: The metagraph for the current **col**.
        :type meta: Set[str] | Dict[str, ADBMetagraphValues]
        :param preserve_key: The PyG key to preserve the ArangoDB _key values.
        :type preserve_key: Optional[str]
        :param kwargs: Additional keyword arguments to pass to **process_adb_df**.
        :type args: Any
        """

        progress = get_bar_progress(f"(ADB → PyG): '{col}'", progress_color)
        progress_task_id = progress.add_task(col, total=col_size)

        with Live(Group(progress)):
            i = 0
            while not cursor.empty():
                cursor_batch = len(cursor.batch())
                df = DataFrame([cursor.pop() for _ in range(cursor_batch)])

                i = process_adb_df(i, df, col, adb_map, meta, preserve_key, **kwargs)
                progress.advance(progress_task_id, advance=len(df))

                df.drop(df.index, inplace=True)

                if cursor.has_more():
                    cursor.fetch()

    def __process_adb_vertex_df(
        self,
        i: int,
        df: DataFrame,
        v_col: str,
        adb_map: ADBMap,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        preserve_key: Optional[str],
        node_data: NodeStorage,
    ) -> int:
        """ArangoDB -> PyG: Process the ArangoDB Vertex DataFrame
        into the PyG NodeStorage object.

        :param i: The last PyG Node id value.
        :type i: int
        :param df: The ArangoDB Vertex DataFrame.
        :type df: pandas.DataFrame
        :param v_col: The ArangoDB Vertex Collection.
        :type v_col: str
        :param adb_map: The ArangoDB -> PyG map.
        :type adb_map: adbpyg_adapter.typings.ADBMap
        :param meta: The metagraph for the current **v_col**.
        :type meta: Set[str] | Dict[str, ADBMetagraphValues]
        :param preserve_key: The PyG key to preserve the ArangoDB _key values.
        :type preserve_key: Optional[str]
        :param node_data: The PyG NodeStorage object.
        :type node_data: torch_geometric.data.NodeStorage
        :return: The last PyG Node id value.
        :rtype: int
        """
        # 1. Map each ArangoDB _key to a PyG node id
        for adb_key in df["_key"]:
            adb_map[v_col][adb_key] = i
            i += 1

        # 2. Set the PyG Node Data
        self.__set_pyg_data(meta, node_data, df)

        # 3. Maintain the ArangoDB _key values
        if preserve_key is not None:
            node_data[preserve_key].extend(list(df["_key"]))

        return i

    def __process_adb_edge_df(
        self,
        _: int,
        df: DataFrame,
        e_col: str,
        adb_map: ADBMap,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        preserve_key: Optional[str],
        data: Union[Data, HeteroData],
        v_cols: List[str],
        strict: bool,
        is_homogeneous: bool,
    ) -> int:
        """ArangoDB -> PyG: Process the ArangoDB Edge DataFrame
        into the PyG EdgeStorage object.

        :param _: Not used.
        :type _: int
        :param df: The ArangoDB Edge DataFrame.
        :type df: pandas.DataFrame
        :param e_col: The ArangoDB Edge Collection.
        :type e_col: str
        :param adb_map: The ArangoDB -> PyG map.
        :type adb_map: adbpyg_adapter.typings.ADBMap
        :param meta: The metagraph for the current **e_col**.
        :type meta: Set[str] | Dict[str, ADBMetagraphValues]
        :param preserve_key: The PyG key to preserve the ArangoDB _key values.
        :type preserve_key: Optional[str]
        :param data: The PyG Data or HeteroData object.
        :type data: torch_geometric.data.Data | torch_geometric.data.HeteroData
        :param v_cols: The list of ArangoDB Vertex Collections.
        :type v_cols: List[str]
        :param strict: Whether to raise an error if invalid edges are found.
        :type strict: bool
        :param is_homogeneous: Whether the ArangoDB graph is homogeneous.
        :type is_homogeneous: bool
        :return: The last PyG Edge id value. This is a useless return value,
            but is needed for type hinting.
        :rtype: int
        """
        # 1. Split the ArangoDB _from & _to IDs into two columns
        df[["from_col", "from_key"]] = self.__split_adb_ids(df["_from"])
        df[["to_col", "to_key"]] = self.__split_adb_ids(df["_to"])

        # 2. Iterate over each edge type
        for (from_col, to_col), count in (
            df[["from_col", "to_col"]].value_counts().items()
        ):
            edge_type = (from_col, e_col, to_col)
            edge_data: EdgeStorage = data if is_homogeneous else data[edge_type]

            # 3. Check for partial Edge Collection import
            if from_col not in v_cols or to_col not in v_cols:
                logger.debug(f"Skipping {edge_type}")
                continue

            logger.debug(f"Preparing {count} {edge_type} edges")

            # 4. Get the edge data corresponding to the current edge type
            et_df: DataFrame = df[
                (df["from_col"] == from_col) & (df["to_col"] == to_col)
            ]

            # 5. Map each ArangoDB from/to _key to the corresponding PyG node id
            # NOTE: map() is somehow converting int values to float...
            # So we rely on astype(int) to convert the float back to int,
            # but we also fill NaN values with -1 so that we can convert
            # the entire column to int without any issues. Need to revisit...
            from_n = et_df["from_key"].map(adb_map[from_col]).fillna(-1).astype(int)
            to_n = et_df["to_key"].map(adb_map[to_col]).fillna(-1).astype(int)

            # 6. Set/Update the PyG Edge Index
            edge_index = tensor(
                np.array([from_n.to_numpy(), to_n.to_numpy()]), dtype=torch.int64
            )

            empty_tensor = tensor([], dtype=torch.int64)
            existing_edge_index = edge_data.get("edge_index", empty_tensor)
            edge_data.edge_index = cat((existing_edge_index, edge_index), dim=1)

            # 7. Deal with invalid edges
            if torch.any(edge_data.edge_index == -1):
                if strict:
                    m = f"Invalid edges found in Edge Collection {e_col}, {from_col} -> {to_col}."  # noqa: E501
                    raise InvalidADBEdgesError(m)
                else:
                    # Remove the invalid edges
                    edge_data.edge_index = edge_data.edge_index[
                        :, ~torch.any(edge_data.edge_index == -1, dim=0)
                    ]

            # 8. Set the PyG Edge Data
            self.__set_pyg_data(meta, edge_data, et_df)

            # 9. Maintain the ArangoDB _key values
            if preserve_key is not None:
                if preserve_key not in edge_data:
                    edge_data[preserve_key] = []

                edge_data[preserve_key].extend(list(et_df["_key"]))

        return 1  # Useless return value, but needed for type hinting

    def __split_adb_ids(self, s: Series) -> Series:
        """AranogDB -> PyG: Helper method to split the ArangoDB IDs
        within a Series into two columns

        :param s: The Series containing the ArangoDB IDs.
        :type s: pandas.Series
        :return: A DataFrame with two columns: the ArangoDB Collection,
            and the ArangoDB _key.
        :rtype: pandas.Series
        """
        return s.str.split(pat="/", n=1, expand=True)

    def __set_pyg_data(
        self,
        meta: Union[Set[str], Dict[str, ADBMetagraphValues]],
        pyg_data: Union[Data, NodeStorage, EdgeStorage],
        df: DataFrame,
    ) -> None:
        """AranogDB -> PyG: A helper method to build the PyG NodeStorage or
        EdgeStorage object for the PyG graph. Is responsible for preparing
        the input **meta** such that it becomes a dictionary, and building
        PyG-ready tensors from the ArangoDB DataFrame **df**.

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
            t = self.__build_tensor_from_dataframe(df, k, v)

            if k not in pyg_data:
                pyg_data[k] = t
            elif isinstance(pyg_data[k], Tensor):
                pyg_data[k] = cat((pyg_data[k], t))
            else:  # pragma: no cover
                m = f"'{k}' key in PyG Data must point to a Tensor"
                raise TypeError(m)

    def __build_tensor_from_dataframe(
        self,
        adb_df: DataFrame,
        meta_key: str,
        meta_val: ADBMetagraphValues,
    ) -> Tensor:
        """ArangoDB -> PyG: Constructs a PyG-ready Tensor from a DataFrame,
        based on the nature of the user-defined metagraph.

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
        m = f"__build_tensor_from_dataframe(df, '{meta_key}', {type(meta_val)})"
        logger.debug(m)

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
            # **meta_val** is a user-defined function that returns a tensor
            user_defined_result = meta_val(adb_df)

            if type(user_defined_result) is not Tensor:  # pragma: no cover
                msg = f"Invalid return type for function {meta_val} ('{meta_key}')"
                raise ADBMetagraphError(msg)

            return user_defined_result

        raise ADBMetagraphError(f"Invalid {meta_val} type")  # pragma: no cover

    ############################
    # Private: PyG -> ArangoDB #
    ############################

    def __get_node_and_edge_types(
        self,
        name: str,
        pyg_g: Union[Data, HeteroData],
        metagraph: PyGMetagraph,
        is_explicit_metagraph: bool,
        is_homogeneous: bool,
    ) -> Tuple[List[str], List[EdgeType]]:
        """PyG -> ArangoDB: Returns the node & edge types of the PyG graph,
        based on the metagraph and whether the graph has default
        canonical etypes.

        :param name: The PyG graph name.
        :type name: str
        :param pyg_g: The existing PyG graph.
        :type pyg_g: Data | HeteroData
        :param metagraph: The PyG Metagraph.
        :type metagraph: adbpyg_adapter.typings.PyGMetagraph
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :param is_homogeneous: Whether the PyG graph is homogeneous or not.
        :type is_homogeneous: bool
        :return: The node & edge types of the PyG graph.
        :rtype: Tuple[List[str], List[torch_geometric.typing.EdgeType]]]
        """
        node_types: List[str]
        edge_types: List[EdgeType]

        if is_explicit_metagraph:
            node_types = metagraph.get("nodeTypes", {}).keys()  # type: ignore
            edge_types = metagraph.get("edgeTypes", {}).keys()  # type: ignore

        elif is_homogeneous:
            n_type = f"{name}_N"
            node_types = [n_type]
            edge_types = [(n_type, f"{name}_E", n_type)]

        else:
            node_types = pyg_g.node_types
            edge_types = pyg_g.edge_types

        return node_types, edge_types

    def __etypes_to_edefinitions(self, edge_types: List[EdgeType]) -> List[Json]:
        """PyG -> ArangoDB: Converts PyG edge_types to ArangoDB edge_definitions

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

    def __ntypes_to_ocollections(
        self, node_types: List[str], edge_types: List[EdgeType]
    ) -> List[str]:
        """PyG -> ArangoDB: Converts PyG node_types to ArangoDB
        orphan collections, if any.

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

    def __create_adb_graph(
        self,
        name: str,
        overwrite_graph: bool,
        node_types: List[str],
        edge_types: List[EdgeType],
    ) -> ADBGraph:
        """PyG -> ArangoDB: Creates the ArangoDB graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections. Defaults to False.
        :type overwrite_graph: bool
        :param node_types: A list of strings representing the PyG node types.
        :type node_types: List[str]
        :param edge_types: A list of string triplets (str, str, str) for
            source node type, edge type and destination node type.
        :type edge_types: List[torch_geometric.typing.EdgeType]
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph
        """
        if overwrite_graph:
            logger.debug("Overwrite graph flag is True. Deleting old graph.")
            self.__db.delete_graph(name, ignore_missing=True)

        if self.__db.has_graph(name):
            return self.__db.graph(name)

        edge_definitions = self.__etypes_to_edefinitions(edge_types)
        orphan_collections = self.__ntypes_to_ocollections(node_types, edge_types)

        return self.__db.create_graph(
            name,
            edge_definitions,
            orphan_collections,
        )

    def __process_pyg_node_batch(
        self,
        n_type: str,
        node_data: NodeStorage,
        meta: Union[Set[str], Dict[Any, PyGMetagraphValues]],
        pyg_map: PyGMap,
        is_explicit_metagraph: bool,
        is_custom_controller: bool,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """PyG -> ArangoDB: Processes the PyG Node batch
        into an ArangoDB DataFrame.

        :param n_type: The PyG node type.
        :type n_type: str
        :param node_data: The PyG NodeStorage object.
        :type node_data: torch_geometric.data.NodeStorage
        :param meta: The metagraph for the current **n_type**.
        :type meta: Set[str] | Dict[Any, adbpyg_adapter.typings.PyGMetagraphValues]
        :param pyg_map: The PyG -> ArangoDB map.
        :type pyg_map: adbpyg_adapter.typings.PyGMap
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :param is_custom_controller: Whether a custom controller is used.
        :type is_custom_controller: bool
        :param start_index: The start index of the current batch.
        :type start_index: int
        :param end_index: The end index of the current batch.
        :type end_index: int
        :return: The ArangoDB DataFrame representing the PyG Node batch.
        :rtype: pandas.DataFrame
        """
        # 1. Set the ArangoDB Node Data
        df = self.__set_adb_data(
            DataFrame(index=range(start_index, end_index)),
            meta,
            node_data,
            node_data.num_nodes,
            is_explicit_metagraph,
            start_index,
            end_index,
        )

        # 2. Update the PyG Map
        if "_id" in df:
            pyg_map[n_type].update(df["_id"].to_dict())
        else:
            df["_key"] = df.get("_key", df.index.astype(str))
            pyg_map[n_type].update((n_type + "/" + df["_key"]).to_dict())

        # 3. Apply the ArangoDB Node Controller (if provided)
        if is_custom_controller:
            df = df.apply(lambda n: self.__cntrl._prepare_pyg_node(n, n_type), axis=1)

        return df

    def __process_pyg_edge_batch(
        self,
        e_type: EdgeType,
        edge_data: EdgeStorage,
        meta: Union[Set[str], Dict[Any, PyGMetagraphValues]],
        pyg_map: PyGMap,
        is_explicit_metagraph: bool,
        is_custom_controller: bool,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """PyG -> ArangoDB: Processes the PyG Edge batch
        into an ArangoDB DataFrame.

        :param e_type: The PyG edge type.
        :type e_type: torch_geometric.typing.EdgeType
        :param edge_data: The PyG EdgeStorage object.
        :type edge_data: torch_geometric.data.EdgeStorage
        :param meta: The metagraph for the current **e_type**.
        :type meta: Set[str] | Dict[Any, adbpyg_adapter.typings.PyGMetagraphValues]
        :param pyg_map: The PyG -> ArangoDB map.
        :type pyg_map: adbpyg_adapter.typings.PyGMap
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :param is_custom_controller: Whether a custom controller is used.
        :type is_custom_controller: bool
        :param start_index: The start index of the current batch.
        :type start_index: int
        :param end_index: The end index of the current batch.
        :type end_index: int
        :return: The ArangoDB DataFrame representing the PyG Edge batch.
        :rtype: pandas.DataFrame
        """
        src_n_type, _, dst_n_type = e_type

        # 1. Fetch the Edge Index of the current batch
        edge_index = edge_data.edge_index[:, start_index:end_index]

        # 2. Set the ArangoDB Edge Data
        df = self.__set_adb_data(
            DataFrame(
                zip(*(edge_index.tolist())),
                index=range(start_index, end_index),
                columns=["_from", "_to"],
            ),
            meta,
            edge_data,
            edge_data.num_edges,
            is_explicit_metagraph,
            start_index,
            end_index,
        )

        # 3. Set the _from column
        df["_from"] = (
            df["_from"].map(pyg_map[src_n_type])
            if pyg_map[src_n_type]
            else src_n_type + "/" + df["_from"].astype(str)
        )

        # 4. Set the _to column
        df["_to"] = (
            df["_to"].map(pyg_map[dst_n_type])
            if pyg_map[dst_n_type]
            else dst_n_type + "/" + df["_to"].astype(str)
        )

        # 5. Apply the ArangoDB Edge Controller (if provided)
        if is_custom_controller:
            df = df.apply(lambda e: self.__cntrl._prepare_pyg_edge(e, e_type), axis=1)

        return df

    def __set_adb_data(
        self,
        df: DataFrame,
        meta: Union[Set[str], Dict[Any, PyGMetagraphValues]],
        pyg_data: Union[Data, NodeStorage, EdgeStorage],
        pyg_data_size: int,
        is_explicit_metagraph: bool,
        start_index: int,
        end_index: int,
    ) -> DataFrame:
        """PyG -> ArangoDB: A helper method to build the ArangoDB Dataframe for
        the given collection. Is responsible for creating "sub-DataFrames"
        from PyG tensors or lists, and appending them to the main dataframe
        **df**. If the data does not adhere to the supported types, or is
        not of specific length, then it is silently skipped.

        :param df: The main ArangoDB DataFrame containing (at minimum)
            the vertex/edge _id or _key attribute.
        :type df: pandas.DataFrame
        :param meta: The metagraph associated to the
            current PyG node or edge type. e.g metagraph['nodeTypes']['v0']
        :type meta: Set[str] | Dict[Any, adbpyg_adapter.typings.PyGMetagraphValues]
        :param pyg_data: The NodeStorage or EdgeStorage of the current
            PyG node or edge type.
        :type pyg_data: torch_geometric.data.storage.(NodeStorage | EdgeStorage)
        :param pyg_data_size: The size of the NodeStorage or EdgeStorage of the
            current PyG node or edge type.
        :type pyg_data_size: int
        :param is_explicit_metagraph: Take the metagraph at face value or not.
        :type is_explicit_metagraph: bool
        :param start_index: The starting index of the current batch to process.
        :type start_index: int
        :param end_index: The ending index of the current batch to process.
        :type end_index: int
        :type pyg_data: torch_geometric.data.storage.(NodeStorage | EdgeStorage)
        :return: The completed DataFrame for the (soon-to-be) ArangoDB collection.
        :rtype: pandas.DataFrame
        :raise ValueError: If an unsupported PyG data value is found.
        """
        logger.debug(
            f"__set_adb_data(df, {meta}, {type(pyg_data)}, {is_explicit_metagraph}"
        )

        valid_meta: Dict[Any, PyGMetagraphValues]
        valid_meta = meta if type(meta) is dict else {m: m for m in meta}

        pyg_keys = (
            set(valid_meta.keys())
            if is_explicit_metagraph
            # pyg_data.keys() is not compatible with Homogeneous graphs:
            else set(k for k, _ in pyg_data.items())
        )

        for meta_key in pyg_keys - {"edge_index"}:
            data = pyg_data[meta_key]
            meta_val = valid_meta.get(meta_key, str(meta_key))

            if (
                type(meta_val) is str
                and type(data) is list
                and len(data) == pyg_data_size
            ):
                meta_val = "_key" if meta_val in ["_v_key", "_e_key"] else meta_val
                df = df.join(DataFrame(data[start_index:end_index], columns=[meta_val]))

            if type(data) is Tensor and len(data) == pyg_data_size:
                df = df.join(
                    self.__build_dataframe_from_tensor(
                        data[start_index:end_index],
                        start_index,
                        end_index,
                        meta_key,
                        meta_val,
                    )
                )

        return df

    def __build_dataframe_from_tensor(
        self,
        pyg_tensor: Tensor,
        start_index: int,
        end_index: int,
        meta_key: Any,
        meta_val: PyGMetagraphValues,
    ) -> DataFrame:
        """PyG -> ArangoDB: Builds a DataFrame from PyG Tensor, based on
        the nature of the user-defined metagraph.

        :param pyg_tensor: The Tensor representing PyG data.
        :type pyg_tensor: torch.Tensor
        :param start_index: The starting index of the current batch to process.
        :type start_index: int
        :param end_index: The ending index of the current batch to process.
        :type end_index: int
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
            df = DataFrame(index=range(start_index, end_index), columns=[meta_val])
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

            df = DataFrame(index=range(start_index, end_index), columns=meta_val)
            df[meta_val] = pyg_tensor.tolist()
            return df

        if callable(meta_val):
            # **meta_val** is a user-defined function that populates
            # and returns the empty dataframe
            empty_df = DataFrame(index=range(start_index, end_index))
            user_defined_result = meta_val(pyg_tensor, empty_df)

            if not isinstance(user_defined_result, DataFrame):  # pragma: no cover
                msg = f"""
                    Invalid return type for function {meta_val} ('{meta_key}').
                    Function must return Pandas DataFrame.
                """
                raise PyGMetagraphError(msg)

            if (
                user_defined_result.index.start != start_index
                or user_defined_result.index.stop != end_index
            ):  # pragma: no cover
                msg = f"""
                    User Defined Function {meta_val} ('{meta_key}') must return
                    DataFrame with start index {start_index} & stop index {end_index}
                """
                raise PyGMetagraphError(msg)

            return user_defined_result

        raise PyGMetagraphError(f"Invalid {meta_val} type")  # pragma: no cover

    def __insert_adb_docs(
        self,
        spinner_progress: Progress,
        df: DataFrame,
        col: str,
        use_async: bool,
        **adb_import_kwargs: Any,
    ) -> None:
        """PyG -> ArangoDB: Insert ArangoDB documents into their ArangoDB collection.

        :param spinner_progress: The spinner progress bar.
        :type spinner_progress: rich.progress.Progress
        :param df: To-be-inserted ArangoDB documents, formatted as a DataFrame
        :type df: pandas.DataFrame
        :param col: The ArangoDB collection name.
        :type col: str
        :param use_async: Performs asynchronous ArangoDB ingestion if enabled.
        :type use_async: bool
        :param adb_import_kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :param adb_import_kwargs: Any
        """
        action = f"ADB Import: '{col}' ({len(df)})"
        spinner_progress_task = spinner_progress.add_task("", action=action)

        docs = df.to_dict("records")
        db = self.__async_db if use_async else self.__db
        result = db.collection(col).import_bulk(docs, **adb_import_kwargs)
        logger.debug(result)

        df.drop(df.index, inplace=True)

        spinner_progress.stop_task(spinner_progress_task)
        spinner_progress.update(spinner_progress_task, visible=False)
