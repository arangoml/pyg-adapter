from arango import ArangoClient
from torch_geometric.datasets import FakeHeteroDataset, FakeDataset
from adbpyg_adapter import ADBPYG_Adapter
from collections import defaultdict
import torch

db = ArangoClient(hosts="http://localhost:8529").db("_system", username="root", password="")
db.delete_graph("FakeHeteroData", drop_collections=True, ignore_missing=True)
db.delete_graph("FakeHomoData", drop_collections=True, ignore_missing=True)

adbpyg_adapter = ADBPYG_Adapter(db, 1)

homo_data = FakeDataset()[0]
hetero_data = FakeHeteroDataset()[0]

# adbpyg_adapter.pyg_to_arangodb("FakeHomoData", homo_data)
# adbpyg_adapter.pyg_to_arangodb("FakeHeteroData", hetero_data)

# homo_metagraph = {
#     "vertexCollections": {
#         "FakeHomoData_N": {"x": "x", "y": "y"},
#     },
#     "edgeCollections": {
#         "FakeHomoData_E": {"y": "y"},
#     },
# }
# new_homo_data = adbpyg_adapter.arangodb_graph_to_pyg("FakeHomoData")
# new_homo_data = adbpyg_adapter.arangodb_collections_to_pyg("FakeHomoData", v_cols={'FakeHomoData_N'}, e_cols={'FakeHomoData_E'})
# new_homo_data = adbpyg_adapter.arangodb_to_pyg("FakeHomoData", homo_metagraph)

# hetero_metagraph = {
#     "vertexCollections": {
#         "v0": {"x": "x", "y": "y"},
#         "v1": {"x": "x"},
#         "v2": {"x": "x"},
#     },
#     "edgeCollections": {
#         "e0": {},
#     },
# }
# new_hetero_data = adbpyg_adapter.arangodb_graph_to_pyg("FakeHeteroData")
# new_hetero_data = adbpyg_adapter.arangodb_collections_to_pyg("FakeHeteroData", v_cols={'v0', 'v1', 'v2'}, e_cols={'e0'})
# new_hetero_data = adbpyg_adapter.FakeHeteroData("FakeHeteroData", hetero_metagraph)


