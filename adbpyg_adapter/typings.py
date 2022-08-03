__all__ = [
    "Json",
    "ADBMetagraph",
    "ADBMetagraphValues",
    "PyGMetagraph",
    "PyGMetagraphValues",
    "ADBMap",
    "PyGMap",
]

from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Union

try:
    # https://github.com/arangoml/pyg-adapter/issues/4
    from cudf import DataFrame
except ModuleNotFoundError:
    from pandas import DataFrame

from torch import Tensor

Json = Dict[str, Any]

DataFrameToTensor = Callable[[DataFrame], Tensor]
TensorToDataFrame = Callable[[Tensor], DataFrame]

ADBEncoders = Dict[str, DataFrameToTensor]
ADBMetagraphValues = Union[str, DataFrameToTensor, ADBEncoders]
ADBMetagraph = Dict[str, Dict[str, Dict[str, ADBMetagraphValues]]]

PyGDataTypes = Union[str, Tuple[str, str, str]]
PyGMetagraphValues = Union[str, List[str], TensorToDataFrame]
PyGMetagraph = Dict[str, Dict[PyGDataTypes, Dict[Any, PyGMetagraphValues]]]

ADBMap = DefaultDict[str, DefaultDict[PyGDataTypes, Dict[str, int]]]
PyGMap = DefaultDict[str, DefaultDict[PyGDataTypes, Dict[int, str]]]
