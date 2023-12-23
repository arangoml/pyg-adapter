import json
import pathlib
from typing import Optional


def sort_children_by_start_time(children):
    return sorted(children, key=lambda span: span["startTime"])


def compare_span(master_child: Optional[dict], branch_child: Optional[dict]):
    if master_child and branch_child:
        assert master_child.get("operationName") == branch_child.get("operationName")
        assert master_child.get("tags") == branch_child.get("tags")

    operation_name = (
        master_child.get("operationName")
        if master_child
        else branch_child.get("operationName")
    )

    master_duration = master_child.get("duration") if master_child else None
    branch_duration = branch_child.get("duration") if branch_child else None
    improvement = (
        f"{round((1 - branch_duration / master_duration) * 100)}%"
        if master_duration and branch_duration
        else None
    )

    comparison = {
        "operationName": operation_name,
        "master_duration": master_duration,
        "branch_duration": branch_duration,
        "improvement": improvement,
        "tags": master_child.get("tags") if master_child else branch_child.get("tags"),
        "children": [],
    }

    if master_child and branch_child:
        comparison["children"] = compare_children(
            master_child["children"], branch_child["children"]
        )

    return comparison


def match_children(
    master_child: dict,
    branch_child: dict,
    master_children: list[dict],
    branch_children: list[dict],
):
    # Attempt to find a matching child in Branch Children for the current Master Child
    for i, branch_candidate in enumerate(branch_children):
        if branch_candidate.get("operationName") == master_child.get("operationName"):
            branch_children.pop(i)
            return master_child, branch_candidate

    # Attempt to find a matching child in Master Children for the current Branch Child
    for i, master_candidate in enumerate(master_children):
        if master_candidate.get("operationName") == branch_child.get("operationName"):
            master_children.pop(i)
            return master_candidate, branch_child

    return master_child, branch_child


def compare_children(master_children: list[dict], branch_children: list[dict]):
    result = []
    master_children_sorted = sort_children_by_start_time(master_children)
    branch_children_sorted = sort_children_by_start_time(branch_children)

    while master_children_sorted or branch_children_sorted:
        master_child = master_children_sorted.pop(0) if master_children_sorted else None
        branch_child = branch_children_sorted.pop(0) if branch_children_sorted else None

        if (
            master_child
            and branch_child
            and master_child.get("operationName") != branch_child.get("operationName")
        ):
            # Find the matching pair if they are out of order
            master_child, branch_child = match_children(
                master_child,
                branch_child,
                master_children_sorted,
                branch_children_sorted,
            )

        result.append(compare_span(master_child, branch_child))

    return result


def compare_traces(master_trace: dict, branch_trace: dict):
    assert master_trace.get("operationName") == branch_trace.get("operationName")
    assert master_trace.get("tags") == branch_trace.get("tags")

    result = {
        "operationName": master_trace.get("operationName"),
        "master_duration": master_trace["duration"],
        "branch_duration": branch_trace["duration"],
        "improvement": f"{round((1 - branch_trace['duration'] / master_trace['duration']) * 100)}%",
        "tags": master_trace.get("tags"),
        "children": compare_children(
            master_trace["children"], branch_trace["children"]
        ),
    }

    return result


def main():
    current_dir = pathlib.Path(__file__).parent.absolute()

    root_span_diffs = {}
    for operation in ["pyg_to_arangodb", "arangodb_to_pyg"]:
        master_trace = json.load(open(f"{current_dir}/traces/master/{operation}.json"))
        branch_trace = json.load(open(f"{current_dir}/traces/branch/{operation}.json"))
        diff_trace = compare_traces(master_trace, branch_trace)

        with open(f"{current_dir}/diff/{operation}.json", "w") as file:
            file.write(json.dumps(diff_trace, indent=4))

        root_span_diffs[operation] = {
            "master_duration": diff_trace["master_duration"],
            "branch_duration": diff_trace["branch_duration"],
            "improvement": diff_trace["improvement"],
        }

    print("-" * 50)
    print(json.dumps(root_span_diffs, indent=4))
    print("-" * 50)

    return root_span_diffs


if __name__ == "__main__":
    main()
