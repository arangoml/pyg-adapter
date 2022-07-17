__all__ = ["Json", "ArangoMetagraph", "DEFAULT_PYG_KEY_MAP"]

from types import FunctionType
from typing import Any, Dict, Union

Json = Dict[str, Any]
PyGEncoder = object
ArangoMetagraph = Dict[
    str, Dict[str, Dict[str, Union[str, Dict[str, PyGEncoder], FunctionType]]]
]

DEFAULT_PYG_KEY_MAP = {
    "x": "x",
    "y": "y",
    "edge_attr": "edge_attr",
    "edge_weight": "edge_weight",
}
