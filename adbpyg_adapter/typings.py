__all__ = ["Json", "ArangoMetagraph"]

from typing import Any, Dict

Json = Dict[str, Any]
ArangoMetagraph = Dict[str, Dict[str, Dict[str, str]]]
