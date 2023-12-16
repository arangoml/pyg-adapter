import json
import pathlib


class DiffSpan:
    def __init__(
        self,
        span_id: str,
        operation_name: str,
        main_duration: int,
        branch_duration: int,
    ):
        self.span_id = span_id
        self.operation_name = operation_name
        self.main_span_duration = main_duration
        self.branch_span_duration = branch_duration
        self.improvement = f"{round((1 - branch_duration / main_duration) * 100)}%"
        self.children: dict[str, "DiffSpan"] = {}

    def add_child(self, span_id: str, child: "DiffSpan"):
        self.children[span_id] = child

    def to_dict(self, include_children: bool = True):
        res = {
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "main_duration": self.main_span_duration,
            "branch_duration": self.branch_span_duration,
            "improvement": self.improvement,
        }

        if include_children:
            res["children"] = [child.to_dict() for child in self.children.values()]

        return res


class DiffTree:
    def __init__(self, main_trace: dict, branch_trace: dict):
        self.root_span = self.__build_diff_tree(main_trace, branch_trace)

    def __build_diff_tree(self, main_trace: dict, branch_trace: dict):
        assert main_trace["operationName"] == branch_trace["operationName"]

        diff_span = DiffSpan(
            main_trace["spanID"],
            main_trace["operationName"],
            main_trace["duration"],
            branch_trace["duration"],
        )

        # Recursively build the tree for child spans
        for head_child_span, branch_child_span in zip(
            main_trace["children"], branch_trace["children"]
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
        main_trace = json.load(open(f"{current_dir}/traces/main/{operation}.json"))
        branch_trace = json.load(open(f"{current_dir}/traces/branch/{operation}.json"))

        diff_tree = DiffTree(main_trace, branch_trace)
        diff_tree.to_json_file(operation)

        print("-" * 50)
        print(json.dumps(diff_tree.root_span.to_dict(include_children=False), indent=4))
        print("-" * 50)


if __name__ == "__main__":
    main()
