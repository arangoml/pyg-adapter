import logging
import os
from types import FunctionType
from typing import Any, Dict

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__package__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    f"[%(asctime)s] [{os.getpid()}] [%(levelname)s] - %(name)s: %(message)s",
    "%Y/%m/%d %H:%M:%S %z",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def progress(
    text: str,
    text_style: str = "none",
    spinner_name: str = "aesthetic",
    spinner_style: str = "#5BC0DE",
    transient: bool = False,
) -> Progress:
    return Progress(
        TextColumn(text, style=text_style),
        SpinnerColumn(spinner_name, spinner_style),
        TimeElapsedColumn(),
        transient=transient,
    )


def validate_adb_metagraph(metagraph: Dict[Any, Dict[Any, Any]]) -> None:
    meta: Dict[Any, Any]

    for parent_key in ["vertexCollections", "edgeCollections"]:
        for col, meta in metagraph.get(parent_key, {}).items():
            if type(col) != str:
                raise TypeError(f"Invalid {parent_key} key: {col} must be str")

            for meta_val in meta.values():
                if type(meta_val) not in [str, dict, FunctionType]:
                    msg = f"""
                        Invalid metagraph value type in {meta}:
                        {meta_val} must be str | Dict[str, object] | FunctionType
                    """
                    raise TypeError(msg)

                if type(meta_val) == dict:
                    for k, v in meta_val.items():
                        if type(k) != str:
                            msg = f"""
                                Invalid ArangoDB attribute key type: {v} must be str
                            """
                            raise TypeError(msg)

                        if v is not None and not callable(v):
                            msg = f"""
                                Invalid PyG Encoder type: {v} must be None | callable()
                            """
                            raise TypeError(msg)


def validate_pyg_metagraph(metagraph: Dict[Any, Dict[Any, Any]]) -> None:
    meta: Dict[Any, Any]

    for node_type in metagraph.get("nodeTypes", {}).keys():
        if type(node_type) != str:
            msg = f"Invalid nodeTypes key: {node_type} is not str"
            raise TypeError(msg)

    for edge_type in metagraph.get("edgeTypes", {}).keys():
        if type(edge_type) != tuple:
            msg = f"Invalid edgeTypes key: {edge_type} must be Tuple[str, str, str]"
            raise TypeError(msg)
        else:
            for elem in edge_type:
                if type(elem) != str:
                    raise TypeError(f"{elem} in {edge_type} must be str")

    for parent_key in ["nodeTypes", "edgeTypes"]:
        for meta in metagraph.get(parent_key, {}).values():
            for meta_val in meta.values():
                if type(meta_val) not in [str, list, FunctionType]:
                    msg = f"""
                        Invalid metagraph value type in {meta}:
                        {meta_val} must be str | List[str] | FunctionType
                    """
                    raise TypeError(msg)

                if type(meta_val) == list:
                    for v in meta_val:
                        if type(v) != str:
                            msg = f"""
                                Invalid metagraph value type in {meta_val}:
                                {v} must be str
                            """
                            raise TypeError(msg)
