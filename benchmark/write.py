import argparse
import json
import pathlib
import random
import time
from typing import Any, Dict, List

import numpy as np
import requests
import torch
from arango import ArangoClient
from torch_geometric.datasets import FakeHeteroDataset

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:
    m = """
        OpenTelemetry is not installed.
        Please install it with `pip install adbpyg-adapter[tracing]`
    """

    raise ImportError(m)

from adbpyg_adapter import ADBPyG_Adapter
from adbpyg_adapter.tracing import create_tracer

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class JaegerSpan:
    def __init__(
        self,
        spanID: str,
        operationName: str,
        duration: int,
        tags: list[dict[str, str]],
    ):
        self.spanID = spanID
        self.operationName = operationName
        self.duration = duration
        self.tags = {
            tag["key"]: tag["value"]
            for tag in tags
            if tag["key"] not in ["span.kind", "internal.span.format"]
        }
        self.children: dict[str, "JaegerSpan"] = {}

    def add_child(self, span_id: str, child: "JaegerSpan"):
        self.children[span_id] = child

    def to_dict(self):
        return {
            "spanID": self.spanID,
            "operationName": self.operationName,
            "duration": self.duration,
            "tags": self.tags,
            "children": [child.to_dict() for child in self.children.values()],
        }


class JaegerSpanTree:
    def __init__(self, jaeger_json_data: Dict[str, Any]):
        self.root_span = self.__build_span_tree(jaeger_json_data)

    def __build_span_tree(self, jaeger_json_data: Dict[str, Any]):
        sorted_spans = sorted(
            jaeger_json_data["data"][0]["spans"], key=lambda span: span["startTime"]
        )

        root_spans: List[JaegerSpan] = []
        span_dict: Dict[str, JaegerSpan] = {}
        span: Dict[str, Any]
        for span in sorted_spans:
            span_id = span["spanID"]
            span_dict[span["spanID"]] = JaegerSpan(
                span_id, span["operationName"], span["duration"], span["tags"]
            )

            references = span.get("references", [])
            if len(references) == 0:
                root_spans.append(span_dict[span_id])
                continue

            for ref in references:
                if ref["refType"] == "CHILD_OF":
                    parent_span = span_dict[ref["spanID"]]
                    parent_span.add_child(span_id, span_dict[span_id])

        assert len(root_spans) == 1
        return root_spans[0]

    def to_dict(self):
        return self.root_span.to_dict()

    def to_json_file(self, output: str):
        current_dir = pathlib.Path(__file__).parent.absolute()
        with open(f"{current_dir}/traces/{output}", "w") as file:
            file.write(json.dumps(self.to_dict(), indent=4))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", type=str, default="http://localhost:8529")
    parser.add_argument("--dbName", type=str, default="_system")
    parser.add_argument("--username", type=str, default="root")
    parser.add_argument("--password", type=str, default="test")
    parser.add_argument("--otlp_endpoint", type=str, default="localhost:4317")
    parser.add_argument(
        "--output_dir", type=str, default="branch", choices=["branch", "head"]
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def get_adapter(args) -> ADBPyG_Adapter:
    db = ArangoClient(hosts=args.url).db(
        args.dbName, username=args.username, password=args.password, verify=True
    )

    tracer = create_tracer(
        "adbpyg-adapter-benchmark",
        enable_console_tracing=False,
        span_exporters=[OTLPSpanExporter(endpoint=args.otlp_endpoint, insecure=True)],
    )

    return ADBPyG_Adapter(db, tracer=tracer)


def run_pyg_to_arangodb(adapter: ADBPyG_Adapter, name: str) -> None:
    data = FakeHeteroDataset(edge_dim=2)[0]
    adapter.db.delete_graph(name, drop_collections=True, ignore_missing=True)
    adapter.pyg_to_arangodb(name, data)


def run_arangodb_to_pyg(adapter: ADBPyG_Adapter, name: str) -> None:
    adapter.arangodb_to_pyg(
        name,
        {
            "vertexCollections": {
                "v0": {"x", "y"},
                "v1": {"x"},
                "v2": {"x"},
            },
            "edgeCollections": {
                "e0": {"edge_attr": "edge_attr"},
            },
        },
    )


def get_span_tree(operation: str, start_time: str) -> JaegerSpanTree:
    url = "http://localhost:16686/api/traces"
    params = {
        "service": "adbpyg-adapter-benchmark",
        "operation": operation,
        "tag": "name:FakeHeteroGraphBenchmark",
        "start": start_time,
    }

    response = requests.get(url, params=params)
    assert response.status_code == 200

    return JaegerSpanTree(response.json())


def main():
    # 1. Parse the arguments
    args = parse_args()

    # 2. Get the adapter
    adbpyg_adapter = get_adapter(args)

    # 3. Run the benchmark
    # TODO: Figure out why Jaeger is reporting the traces
    # in the **same** operation... (only a problem for benchmarking)
    name = "FakeHeteroGraphBenchmark"
    start_time = str(time.time()).replace(".", "")
    run_pyg_to_arangodb(adbpyg_adapter, name)
    run_arangodb_to_pyg(adbpyg_adapter, name)

    # Wait for OTLP Export
    time.sleep(5)

    # 4. Get the span trees
    pyg_to_arangodb_span_tree = get_span_tree("pyg_to_arangodb", start_time)
    arangodb_to_pyg_span_tree = get_span_tree("arangodb_to_pyg", start_time)

    # 5. Write the span trees to disk
    pyg_to_arangodb_span_tree.to_json_file(f"{args.output_dir}/pyg_to_arangodb.json")
    arangodb_to_pyg_span_tree.to_json_file(f"{args.output_dir}/arangodb_to_pyg.json")


if __name__ == "__main__":
    main()
