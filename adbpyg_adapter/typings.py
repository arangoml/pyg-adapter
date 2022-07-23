__all__ = ["Json", "ADBMetagraph", "PyGMetagraph"]

from types import FunctionType
from typing import Any, Dict, Tuple, Union

Json = Dict[str, Any]
PyGEncoder = object
ADBMetagraph = Dict[
    str, Dict[str, Dict[str, Union[str, Dict[str, PyGEncoder], FunctionType]]]
]

PyGMetagraph = Dict[str, Dict[Union[str, Tuple[str, str, str]], Any]]
