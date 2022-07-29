#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# See https://pytorch-geometric.readthedocs.io/en/latest/notes/load_csv.html
# for an example on encoders.

from typing import Any, Dict, Optional

from pandas import DataFrame
from torch import Tensor, from_numpy, zeros


class IdentityEncoder(object):
    """Converts a list of floating-point values into a PyTorch tensor"""

    def __init__(self, dtype: Any = None) -> None:
        self.dtype = dtype

    def __call__(self, df: DataFrame) -> Tensor:
        return from_numpy(df.values).view(-1, 1).to(self.dtype)


class CategoricalEncoder(object):
    """Converts a list of values into a PyTorch tensor through a mapping"""

    def __init__(self, mapping: Optional[Dict[Any, Any]] = None) -> None:
        self.mapping = mapping

    def __call__(self, df: DataFrame) -> Tensor:
        if self.mapping is None:
            unique_vals = df.unique()
            self.mapping = {u_v: i for i, u_v in enumerate(unique_vals)}

        x = zeros(len(df), 1)
        for i, col in enumerate(df.values):
            x[i, 0] = self.mapping[col]

        return x
