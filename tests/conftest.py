import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from arango import ArangoClient
from arango.database import StandardDatabase
from arango.http import DefaultHTTPClient
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from torch import no_grad, tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import Amazon, FakeDataset, FakeHeteroDataset, KarateClub

from adbpyg_adapter import ADBPyG_Adapter
from adbpyg_adapter.typings import Json

db: StandardDatabase
adbpyg_adapter: ADBPyG_Adapter
PROJECT_DIR = Path(__file__).parent.parent


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="")


def pytest_configure(config: Any) -> None:
    con = {
        "url": config.getoption("url"),
        "username": config.getoption("username"),
        "password": config.getoption("password"),
        "dbName": config.getoption("dbName"),
    }

    print("----------------------------------------")
    print("URL: " + con["url"])
    print("Username: " + con["username"])
    print("Password: " + con["password"])
    print("Database: " + con["dbName"])
    print("----------------------------------------")

    class NoTimeoutHTTPClient(DefaultHTTPClient):  # type: ignore
        REQUEST_TIMEOUT = None

    global db
    db = ArangoClient(hosts=con["url"], http_client=NoTimeoutHTTPClient()).db(
        con["dbName"], con["username"], con["password"], verify=True
    )

    global adbpyg_adapter
    adbpyg_adapter = ADBPyG_Adapter(db, logging_lvl=logging.DEBUG)

    # Restore fraud dataset via arangorestore
    arango_restore(con, "tests/data/adb/imdb_dump")

    # Create Fraud Detection Graph
    db.delete_graph("imdb-movies", ignore_missing=True)
    db.create_graph(
        "imdb-movies",
        edge_definitions=[
            {
                "edge_collection": "Ratings",
                "from_vertex_collections": ["Users"],
                "to_vertex_collections": ["Movies"],
            },
        ],
    )


def arango_restore(con: Json, path_to_data: str) -> None:
    restore_prefix = "./tools/" if os.getenv("GITHUB_ACTIONS") else ""
    protocol = "http+ssl://" if "https://" in con["url"] else "tcp://"
    url = protocol + con["url"].partition("://")[-1]
    # A small hack to work around empty passwords
    password = f"--server.password {con['password']}" if con["password"] else ""

    subprocess.check_call(
        f'chmod -R 755 ./tools/arangorestore && {restore_prefix}arangorestore \
            -c none --server.endpoint {url} --server.database {con["dbName"]} \
                --server.username {con["username"]} {password} \
                    --input-directory "{PROJECT_DIR}/{path_to_data}"',
        cwd=f"{PROJECT_DIR}/tests",
        shell=True,
    )


def get_karate_graph() -> Data:
    return KarateClub()[0]  # requires networkx dependency


def get_fake_homo_graph(**params: Any) -> Data:
    return FakeDataset(**params)[0]


def get_fake_hetero_graph(**params: Any) -> HeteroData:
    return FakeHeteroDataset(**params)[0]


# Arguably too large for testing purposes
def get_amazon_photos_graph() -> Data:
    path = f"{PROJECT_DIR}/tests/data/pyg"
    return Amazon(root=path, name="Photo")[0]


def get_social_graph() -> HeteroData:
    data = HeteroData()

    data[("user", "follows", "user")].edge_index = tensor([[0, 1], [1, 2]])
    data[("user", "follows", "game")].edge_index = tensor([[0, 1, 2], [0, 1, 2]])
    data[("user", "plays", "game")].edge_index = tensor([[3, 3], [1, 2]])

    data["user"].num_nodes = 4
    data["game"].num_nodes = 3
    data["user"].x = tensor([[21], [16], [38], [64]])
    data[("user", "plays", "game")].edge_attr = tensor([[3], [5]])

    return data


class SequenceEncoder(object):
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Any = None
    ) -> None:
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @no_grad()
    def __call__(self, df: DataFrame) -> Any:
        x = self.model.encode(
            df.values,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )
        return x.cpu()