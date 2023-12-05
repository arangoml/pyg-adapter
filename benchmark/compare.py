import json
import pathlib


class DiffSpan:
    def __init__(
        self,
        spanID: str,
        operationName: str,
        duration_head: int,
        duration_branch: int,
        tags: list[dict[str, str]],
    ):
        self.spanID = spanID
        self.operationName = operationName
        self.duration_head = duration_head
        self.duration_branch = duration_branch
        self.improvement = f"{round((1 - duration_branch / duration_head) * 100)}%"
        self.tags = tags
        self.children: dict[str, "DiffSpan"] = {}

    def add_child(self, span_id: str, child: "DiffSpan"):
        self.children[span_id] = child

    def to_dict(self, include_children: bool = True):
        res = {
            "spanID": self.spanID,
            "operationName": self.operationName,
            "duration_head": self.duration_head,
            "duration_branch": self.duration_branch,
            "improvement": self.improvement,
            "tags": self.tags,
        }

        if include_children:
            res["children"] = [child.to_dict() for child in self.children.values()]

        return res


class DiffTree:
    def __init__(self, head_trace: dict, branch_trace: dict):
        self.root_span = self.__build_diff_tree(head_trace, branch_trace)

    def __build_diff_tree(self, head_trace: dict, branch_trace: dict):
        diff_span = DiffSpan(
            head_trace["spanID"],
            head_trace["operationName"],
            head_trace["duration"],
            branch_trace["duration"],
            head_trace["tags"],
        )

        # Recursively build the tree for child spans
        for head_child_span, branch_child_span in zip(
            head_trace["children"], branch_trace["children"]
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
        head_trace = json.load(open(f"{current_dir}/traces/head/{operation}.json"))
        branch_trace = json.load(open(f"{current_dir}/traces/branch/{operation}.json"))

        diff_tree = DiffTree(head_trace, branch_trace)
        diff_tree.to_json_file(operation)

        print("-" * 50)
        print(json.dumps(diff_tree.root_span.to_dict(include_children=False), indent=4))
        print("-" * 50)


if __name__ == "__main__":
    main()
