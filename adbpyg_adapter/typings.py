__all__ = [
    "Json",
    "ADBMetagraph",
    "ADBMetagraphValues",
    "PyGMetagraph",
    "PyGMetagraphValues",
]

from typing import Any, Callable, Dict, List, Tuple, Union

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
