import json
import pathlib


class DiffSpan:
    def __init__(
        self,
        span_id: str,
        operation_name: str,
        master_duration: int,
        branch_duration: int,
    ):
        self.span_id = span_id
        self.operation_name = operation_name
        self.master_duration = master_duration
        self.branch_duration = branch_duration
        self.improvement = f"{round((1 - branch_duration / master_duration) * 100)}%"
        self.children: dict[str, "DiffSpan"] = {}

    def add_child(self, span_id: str, child: "DiffSpan"):
        self.children[span_id] = child

    def to_dict(self, include_children: bool = True):
        res = {
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "master_duration": self.master_duration,
            "branch_duration": self.branch_duration,
            "improvement": self.improvement,
        }

        if include_children:
            res["children"] = [child.to_dict() for child in self.children.values()]

        return res


class DiffTree:
    def __init__(self, master_trace: dict, branch_trace: dict):
        self.root_span = self.__build_diff_tree(master_trace, branch_trace)

    def __build_diff_tree(self, master_trace: dict, branch_trace: dict):
        assert master_trace["operationName"] == branch_trace["operationName"]

        diff_span = DiffSpan(
            master_trace["spanID"],
            master_trace["operationName"],
            master_trace["duration"],
            branch_trace["duration"],
        )

        # Recursively build the tree for child spans
        for head_child_span, branch_child_span in zip(
            master_trace["children"], branch_trace["children"]
        ):
            child_span = self.__build_diff_tree(head_child_span, branch_child_span)
            diff_span.add_child(head_child_span["spanID"], child_span)

        return diff_span

    def to_dict(self):
        return self.root_span.to_dict()

    def to_json_file(self, operation: str):
        current_dir = pathlib.Path(__file__).parent.absolute()
        with open(f"{current_dir}/diff/{operation}.json", "w") as file:
            file.write(json.dumps(self.to_dict(), indent=4))


def main():
    current_dir = pathlib.Path(__file__).parent.absolute()

    for operation in ["pyg_to_arangodb", "arangodb_to_pyg"]:
        master_trace = json.load(open(f"{current_dir}/traces/master/{operation}.json"))
        branch_trace = json.load(open(f"{current_dir}/traces/branch/{operation}.json"))

        diff_tree = DiffTree(master_trace, branch_trace)
        diff_tree.to_json_file(operation)

        print("-" * 50)
        print(json.dumps(diff_tree.root_span.to_dict(include_children=False), indent=4))
        print("-" * 50)


if __name__ == "__main__":
    main()
