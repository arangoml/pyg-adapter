__all__ = [
    "Json",
    "ADBMetagraph",
    "ADBMetagraphValues",
    "PyGMetagraph",
    "PyGMetagraphValues",
    "ADBMap",
    "PyGMap",
]

from typing import Any, Callable, DefaultDict, Dict, List, Set, Tuple, Union

from pandas import DataFrame
from torch import Tensor

Json = Dict[str, Any]

DataFrameToTensor = Callable[[DataFrame], Tensor]
TensorToDataFrame = Callable[[Tensor, DataFrame], DataFrame]

ADBEncoders = Dict[str, DataFrameToTensor]
ADBMetagraphValues = Union[str, DataFrameToTensor, ADBEncoders]
ADBMetagraph = Dict[str, Dict[str, Union[Set[str], Dict[str, ADBMetagraphValues]]]]

PyGDataTypes = Union[str, Tuple[str, str, str]]
PyGMetagraphValues = Union[str, List[str], TensorToDataFrame]
PyGMetagraph = Dict[
    str, Dict[PyGDataTypes, Union[Set[str], Dict[Any, PyGMetagraphValues]]]
]

ADBMap = DefaultDict[PyGDataTypes, Dict[str, int]]
PyGMap = DefaultDict[PyGDataTypes, Dict[int, str]]
