#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Any, List, Set, Union

from arango.graph import Graph as ArangoDBGraph
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from .typings import ArangoMetagraph, DEFAULT_PyG_METAGRAPH, Json, PyGMetagraph


class Abstract_ADBPyG_Adapter(ABC):
    def __init__(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def arangodb_to_pyg(
        self, name: str, metagraph: ArangoMetagraph, **query_options: Any
    ) -> HeteroData:
        raise NotImplementedError  # pragma: no cover

    def arangodb_collections_to_pyg(
        self, name: str, v_cols: Set[str], e_cols: Set[str], **query_options: Any
    ) -> HeteroData:
        raise NotImplementedError  # pragma: no cover

    def arangodb_graph_to_pyg(self, name: str, **query_options: Any) -> HeteroData:
        raise NotImplementedError  # pragma: no cover

    def pyg_to_arangodb(
        self,
        name: str,
        pyg_g: Union[Data, HeteroData],
        metagraph: PyGMetagraph = DEFAULT_PyG_METAGRAPH,
        overwrite_graph: bool = False,
        **import_options: Any,
    ) -> ArangoDBGraph:
        raise NotImplementedError  # pragma: no cover

    def etypes_to_edefinitions(self, edge_types: List[EdgeType]) -> List[Json]:
        raise NotImplementedError  # pragma: no cover

    def __fetch_adb_docs(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def __insert_adb_docs(self) -> None:
        raise NotImplementedError  # pragma: no cover


class Abstract_ADBPyG_Controller(ABC):
    def _prepare_arangodb_vertex(self, adb_vertex: Json, col: str) -> None:
        raise NotImplementedError  # pragma: no cover

    def _prepare_arangodb_edge(self, adb_edge: Json, col: str) -> None:
        raise NotImplementedError  # pragma: no cover
