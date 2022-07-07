__all__ = ["Json", "ArangoMetagraph", "PyGMetagraph", "DEFAULT_PyG_METAGRAPH"]

from typing import Any, Dict

Json = Dict[str, Any]
ArangoMetagraph = Dict[str, Dict[str, Dict[str, str]]]
PyGMetagraph = Dict[str, str]

DEFAULT_PyG_METAGRAPH = {
    "x": "x",
    "y": "y",
    "edge_attr": "edge_attr",
    "edge_weight": "edge_weight",
}
