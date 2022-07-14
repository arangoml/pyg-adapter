import logging
import os
from typing import Any, Dict, Optional

from pandas import DataFrame
from torch import Tensor, from_numpy, zeros

logger = logging.getLogger(__package__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    f"[%(asctime)s] [{os.getpid()}] [%(levelname)s] - %(name)s: %(message)s",
    "%Y/%m/%d %H:%M:%S %z",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class EnumEncoder(object):
    def __init__(self, mapping: Optional[Dict[Any, Any]] = None) -> None:
        self.mapping = mapping

    def __call__(self, df: DataFrame) -> Tensor:
        if self.mapping is None:
            enums = df.unique()
            self.mapping = {enum: i for i, enum in enumerate(enums)}

        x = zeros(len(df), 1)
        for i, col in enumerate(df.values):
            x[i, 0] = self.mapping[col]

        return x


class IdentityEncoder(object):
    def __init__(self, dtype: Any = None) -> None:
        self.dtype = dtype

    def __call__(self, df: DataFrame) -> Tensor:
        return from_numpy(df.values).view(-1, 1).to(self.dtype)
