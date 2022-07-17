__all__ = ["Json", "ArangoMetagraph", "PyGMetagraph", "DEFAULT_PyG_METAGRAPH"]

from types import FunctionType
from typing import Any, Dict, Union

Json = Dict[str, Any]
PyGEncoder = object
ArangoMetagraph = Dict[
    str, Dict[str, Dict[str, Union[str, Dict[str, PyGEncoder], FunctionType]]]
]
PyGMetagraph = Dict[str, str]

DEFAULT_PyG_METAGRAPH = {
    "x": "x",
    "y": "y",
    "edge_attr": "edge_attr",
    "edge_weight": "edge_weight",
}
