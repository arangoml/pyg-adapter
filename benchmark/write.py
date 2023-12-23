import argparse
import json
import pathlib
import random
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import requests
import torch
from arango import ArangoClient
from retry import retry
from torch_geometric.datasets import FakeHeteroDataset

# import uuid


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
        span_id: str,
        operation_name: str,
        start_time: int,
        duration: int,
        tags: list[dict[str, str]],
    ):
        self.span_id = span_id
        self.operation_name = operation_name
        self.start_time = start_time
        self.duration = duration
        self.tags = {
            tag["key"]: tag["value"]
            for tag in tags
            if tag["key"] not in ["span.kind", "internal.span.format"]
        }

        self.children: dict[str, "JaegerSpan"] = {}
        self.parent: "JaegerSpan" = None

    def add_child(self, span_id: str, child: "JaegerSpan"):
        self.children[span_id] = child

    def set_parent(self, parent: "JaegerSpan"):
        self.parent = parent

    def to_dict(self) -> dict[str, Any]:
        return {
            "spanID": self.span_id,
            "operationName": self.operation_name,
            "startTime": self.start_time,
            "duration": self.duration,
            "tags": self.tags,
            "children": [child.to_dict() for child in self.children.values()],
        }


class JaegerSpanTree:
    def __init__(
        self,
        jaeger_endpoint: str,
        service_name: str,
        operation_name: str,
        start_time: str,
        tags: Dict[str, Any] = {},
    ) -> None:
        self.jaeger_endpoint = jaeger_endpoint
        self.service_name = service_name
        self.operation_name = operation_name
        self.start_time = start_time
        self.tags = tags

        self.root_span: JaegerSpan = None
        self.span_id_to_span: Dict[str, JaegerSpan] = {}
        self.operation_name_to_span: Dict[str, List[JaegerSpan]] = defaultdict(list)

        self.__build_span_tree()
        print(f"Built span tree for {self.service_name}-{self.operation_name}")

    def get_spans_by_operation_name(self, operation_name: str) -> List[JaegerSpan]:
        return self.operation_name_to_span[operation_name]

    def get_span_by_span_id(self, span_id: str) -> JaegerSpan:
        return self.span_id_to_span[span_id]

    def get_span_tag_value(self, span_id: str, tag_key: str) -> str:
        return self.span_id_to_span[span_id].tags[tag_key]

    def __build_span_tree(self) -> None:
        for span in self.__fetch_sorted_spans():
            span_id: str = span["spanID"]
            operation_name: str = span["operationName"]

            span_object = JaegerSpan(
                span_id,
                operation_name,
                span["startTime"],
                span["duration"],
                span["tags"],
            )

            self.span_id_to_span[span_id] = span_object
            self.operation_name_to_span[operation_name].append(span_object)

            references = span.get("references", [])
            if len(references) == 0:
                if self.root_span is not None:
                    m = f"Found multiple root spans: {self.root_span.span_id} and {span_id}"
                    print(m)
                    raise Exception(m)

                self.root_span = self.span_id_to_span[span_id]
                continue

            for ref in references:
                if ref["refType"] == "CHILD_OF":
                    parent_span_id = ref["spanID"]
                    parent_span = self.span_id_to_span[parent_span_id]
                    child_span = self.span_id_to_span[span_id]

                    parent_span.add_child(span_id, child_span)
                    child_span.set_parent(parent_span)

    def __fetch_sorted_spans(self) -> List[Dict[str, Any]]:
        params = {
            "service": self.service_name,
            "operation": self.operation_name,
            "tag": [f"{k}:{v}" for k, v in self.tags.items()],
            "start": self.start_time,
        }

        traces = self.__get_jaeger_traces(f"{self.jaeger_endpoint}/api/traces", params)

        if len(traces) > 1:
            m = f"Found multiple traces for {params}"
            print(m)
            raise Exception(m)

        spans = traces[0]["spans"]
        return sorted(spans, key=lambda span: span["startTime"])

    @retry(tries=6, delay=2, backoff=2)
    def __get_jaeger_traces(
        self, url: str, params: dict[str, Any]
    ) -> List[dict[str, Any]]:
        response = requests.get(url, params=params)

        if response.status_code != 200:
            m = f"Failed to fetch traces for {params}: {response.status_code}"
            print(m)
            raise Exception(m)

        traces = response.json()["data"]
        if len(traces) == 0:
            m = f"No traces found for {params}"
            print(m)
            raise Exception(m)

        return traces

    def to_dict(self) -> dict[str, Any]:
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
    parser.add_argument("--password", type=str, default="")
    parser.add_argument("--jaeger_endpoint", type=str, default="http://localhost:16686")
    parser.add_argument("--otlp_endpoint", type=str, default="http://localhost:4317")
    parser.add_argument(
        "--output_dir", type=str, choices=["branch", "master"], required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def get_adapter(args, service_name: str) -> ADBPyG_Adapter:
    db = ArangoClient(hosts=args.url).db(
        args.dbName, username=args.username, password=args.password, verify=True
    )

    tracer = create_tracer(
        service_name,
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


def main():
    service_name = f"adbpyg-adapter-benchmark"

    # 1. Parse the arguments
    args = parse_args()

    # 2. Get the adapter
    adbpyg_adapter = get_adapter(args, service_name)

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
    pyg_to_arangodb_span_tree = JaegerSpanTree(
        args.jaeger_endpoint,
        service_name,
        "pyg_to_arangodb",
        start_time,
        {"name": name},
    )

    arangodb_to_pyg_span_tree = JaegerSpanTree(
        args.jaeger_endpoint,
        service_name,
        "arangodb_to_pyg",
        start_time,
        {"name": name},
    )

    # 5. Write the span trees to disk
    pyg_to_arangodb_span_tree.to_json_file(f"{args.output_dir}/pyg_to_arangodb.json")
    arangodb_to_pyg_span_tree.to_json_file(f"{args.output_dir}/arangodb_to_pyg.json")


if __name__ == "__main__":
    main()
