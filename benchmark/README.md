# Benchmarking

This directory contains the benchmarking scripts for the project.

1. `compare.py` compares the benchmarking results of two branches.
2. `write.py`: writes the benchmarking results to a file: 
```py
parser.add_argument("--url", type=str, default="http://localhost:8529")
parser.add_argument("--dbName", type=str, default="_system")
parser.add_argument("--username", type=str, default="root")
parser.add_argument("--password", type=str, default="")
parser.add_argument("--jaeger_endpoint", type=str, default="http://localhost:16686")
parser.add_argument("--otlp_endpoint", type=str, default="http://localhost:4317")
parser.add_argument(
    "--output_dir", type=str, choices=["branch", "master"], required=True
)
```

Results are stored in:
- `benchmark/master` for the master results
- `benchmark/branch` for the branch results (added to `.gitignore`)
- `benchmark/diff` for the diff between the branch and master results (added to `.gitignore`)