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
from .typings import ADBMetagraph, Json, PyGEncoder, PyGMetagraph
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
        self, name: str, metagraph: ADBMetagraph, **query_options: Any
    ) -> Union[Data, HeteroData]:
        """Create a PyG graph from the user-defined metagraph. DOES carry
            over node/edge features/labels, via the **metagraph**.

        :param name: The PyG graph name.
        :type name: str
        :param metagraph: An object defining vertex & edge collections to import
            to PyG, along with collection-level specifications to indicate
            which ArangoDB attributes will become PyG features/labels.
            See below for examples of **metagraph**
        :type metagraph: adbpyg_adapter.typings.ADBMetagraph
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

        3) Here is a final example for parameter **metagraph**:
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
        build a PyG-ready Tensor from a Pandas DataFrame equivalent to the
        associated ArangoDB collection.
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
            logger.debug(f"Preparing '{v_col}' vertices")

            df = DataFrame(self.__fetch_adb_docs(v_col, query_options))
            adb_map.update({adb_id: pyg_id for pyg_id, adb_id in enumerate(df["_id"])})

            node_data: NodeStorage = data if is_homogeneous else data[v_col]
            for k, v in meta.items():
                node_data[k] = self.__build_tensor_from_dataframe(df, v)

        for e_col, meta in metagraph["edgeCollections"].items():
            logger.debug(f"Preparing '{e_col}' edges")

            df = DataFrame(self.__fetch_adb_docs(e_col, query_options))
            df["from_col"] = df["_from"].str.split("/").str[0]
            df["to_col"] = df["_to"].str.split("/").str[0]

            for (from_col, to_col), count in (
                df[["from_col", "to_col"]].value_counts().items()
            ):
                edge_type = (from_col, e_col, to_col)
                logger.debug(f"Preparing {count} '{edge_type}' edges")

                # Get the edge data corresponding to the current edge type
                et_df = df[(df["from_col"] == from_col) & (df["to_col"] == to_col)]
                from_nodes = [adb_map[_id] for _id in et_df["_from"]]
                to_nodes = [adb_map[_id] for _id in et_df["_to"]]

                edge_data: EdgeStorage = data if is_homogeneous else data[edge_type]
                edge_data.edge_index = tensor([from_nodes, to_nodes])
                for k, v in meta.items():
                    edge_data[k] = self.__build_tensor_from_dataframe(et_df, v)

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
        metagraph: ADBMetagraph = {
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
        metagraph: PyGMetagraph = {},
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ADBGraph:
        """Create an ArangoDB graph from a PyG graph.

        :param name: The ArangoDB graph name.
        :type name: str
        :param pyg_g: The existing PyG graph.
        :type pyg_g: Data | HeteroData
        :param metagraph: An optional object mapping the PyG keys of
            the node & edge data to ArangoDB key strings or user-defined
            functions. NOTE: Unlike the metagraph for ArangoDB to PyG, this
            one is optional. See below for an example of **metagraph**.
        :type metagraph: adbpyg_adapter.typings.PyGMetagraph
        :param overwrite_graph: Overwrites the graph if it already exists.
            Does not drop associated collections.
        :type overwrite_graph: bool
        :param import_options: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        :type import_options: Any
        :return: The ArangoDB Graph API wrapper.
        :rtype: arango.graph.Graph


        1) Here is an example entry for parameter **metagraph**:

        .. code-block:: python
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

        def v2_x_to_pandas_dataframe(t: Tensor):
            df = pandas.DataFrame(columns=["v2_features"])
            df["v2_features"] = t.tolist()
            # do more things with df["v2_features"] here ...
            return df

        Using the metagraph above will set a custom ArangoDB attribute key for
        the v0 "x" feature matrix ('v0_features'), and its "y" label ('label').
        Furthemore, the v1 "x" feature matrix is broken down in order to
        associate one ArangoDB attribute per feature. Lastly, the v2 feature matrix
        is converted into a DataFrame via a user-defined function.
        """
        logger.debug(f"--pyg_to_arangodb('{name}')--")

        is_homogeneous = type(pyg_g) is Data

        edge_types = (
            [(name + "_N", name + "_E", name + "_N")]
            if is_homogeneous
            else pyg_g.edge_types
        )
        edge_definitions = self.etypes_to_edefinitions(edge_types)

        if overwrite_graph:
            logger.debug("Overwrite graph flag is True. Deleting old graph.")
            self.__db.delete_graph(name, ignore_missing=True)

        adb_graph = (
            self.__db.graph(name)
            if self.__db.has_graph(name)
            else self.__db.create_graph(name, edge_definitions)
        )

        # Define PyG data properties
        node_data: NodeStorage
        edge_data: EdgeStorage

        v_col: str
        n_meta = metagraph.get("nodeTypes", {})
        for v_col in adb_graph.vertex_collections():
            node_data = pyg_g if is_homogeneous else pyg_g[v_col]
            df = DataFrame([{"_key": str(i)} for i in range(node_data.num_nodes)])

            meta = n_meta.get(v_col, {})
            for k, v in node_data.items():
                if type(v) is Tensor and len(v) == node_data.num_nodes:
                    df = df.join(self.__build_dataframe_from_tensor(v, meta.get(k, k)))

            if type(self.__cntrl) is not ADBPyG_Controller:
                f = lambda n: self.__cntrl._prepare_pyg_node(n, v_col)
                df = df.apply(f, axis=1)

            self.__insert_adb_docs(v_col, df.to_dict("records"), import_options)

        e_meta = metagraph.get("edgeTypes", {})
        for edge_type in edge_types:
            edge_data = pyg_g if is_homogeneous else pyg_g[edge_type]
            from_col, e_col, to_col = edge_type

            columns = ["_from", "_to"]
            df = DataFrame(zip(*(edge_data.edge_index.tolist())), columns=columns)
            df["_from"] = from_col + "/" + df["_from"].astype(str)
            df["_to"] = to_col + "/" + df["_to"].astype(str)

            meta = e_meta.get(edge_type, {})
            for k, v in edge_data.items():
                if type(v) is Tensor and len(v) == edge_data.num_edges:
                    df = df.join(self.__build_dataframe_from_tensor(v, meta.get(k, k)))

            if type(self.__cntrl) is not ADBPyG_Controller:
                f = lambda e: self.__cntrl._prepare_pyg_edge(e, e_col)
                df = df.apply(f, axis=1)

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

    def __build_tensor_from_dataframe(
        self, df: DataFrame, meta_val: Union[str, Dict[str, PyGEncoder], FunctionType]
    ) -> Tensor:
        """Constructs a PyG-ready Tensor from a Pandas Dataframe, based on
        the nature of the user-defined metagraph.

        :param df: The Pandas Dataframe representing ArangoDB data.
        :type df: pandas.DataFrame
        :param meta_val: The value mapped to the ArangoDB-PyG metagraph key to
            help convert **df** into a PyG-ready Tensor.
            e.g the value of `metagraph['vertexCollections']['users']['x']`.
            The current accepted **meta_val** types are:
            1) str: return the DataFrame's **meta_val** column values as a Tensor
            2) dict: encode all `key` column values & concatenate as a Tensor
            3) function: execute a user-defined function to return a Tensor
        :type meta_val: str | dict | function
        :return: A PyG-ready tensor equivalent to the dataframe
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
            # **meta_val** is a user-defined that returns a tensor
            user_defined_tensor: Tensor = meta_val(df)
            return user_defined_tensor

        else:
            msg = f"""
                Invalid **meta_val** argument type: {meta_val}.
                Expected Union[str, Dict[str, PyGEncoder], FunctionType],
                got {type(meta_val)} instead.
            """
            raise TypeError(msg)

    def __build_dataframe_from_tensor(
        self, tensor: Tensor, meta_val: Union[str, List[str], FunctionType]
    ) -> DataFrame:
        """Builds a Pandas DataFrame from PyG Tensor, based on
        the nature of the user-defined metagraph.

        :param tensor: The Tensor representing PyG data.
        :type tensor: torch.Tensor
        :param meta_val: The value mapped to the PyG-ArangoDB metagraph key to
            help convert **tensor** into a Pandas Dataframe.
            e.g the value of `metagraph['nodeTypes']['users']['x']`.
            The current accepted **meta_val** types are:
            1) str: return a 1-column DataFrame equivalent to the Tensor
            2) list[str]: return an N-column DataFrame equivalent to the Tensor
            3) func: return a DataFrame based on the Tensor via a user-defined function
        :type meta_val: str | dict | function

        :return: A Pandas DataFrame equivalent to the Tensor
        :rtype: pandas.DataFrame
        """
        if type(meta_val) in [str, list]:
            columns = [meta_val] if type(meta_val) is str else meta_val
            df = DataFrame(columns=columns)
            df[meta_val] = tensor.tolist()
            return df

        elif type(meta_val) is FunctionType:
            # **meta_val** is a user-defined function that returns a dataframe
            user_defined_dataframe: DataFrame = meta_val(tensor)
            return user_defined_dataframe

        else:
            msg = f"""
                Invalid **meta_val** argument type: {meta_val}.
                Expected Union[str, FunctionType],
                got {type(meta_val)} instead.
            """
            raise TypeError(msg)

    def __insert_adb_docs(self, col: str, docs: List[Json], kwargs: Any) -> None:
        """Insert ArangoDB documents into their ArangoDB collection.

        :param col: The ArangoDB collection name
        :type col: str
        :param docs: To-be-inserted ArangoDB documents
        :type docs: List[Json]
        :param kwargs: Keyword arguments to specify additional
            parameters for ArangoDB document insertion. Full parameter list:
            https://docs.python-arango.com/en/main/specs.html#arango.collection.Collection.import_bulk
        """
        logger.debug(f"Inserting {len(docs)} documents into '{col}'")
        result = self.__db.collection(col).import_bulk(docs, **kwargs)
        logger.debug(result)
