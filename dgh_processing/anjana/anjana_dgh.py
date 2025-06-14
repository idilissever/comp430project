from __future__ import annotations
import csv
from pathlib import Path
from dgh_processing.dgh import DGH, DGHNode


def build_dgh_from_csv(
    path: Path,
    column_name: str,
    *,
    delimiter: str = ",",
    header: bool = False,
    root_token: str = "*",
) -> DGH:
    """
    CSV format like:
        17,"[15, 20[","[10, 20[","[0, 20[","[0, 40[","[0, 80[",*
    Each field after the first is the next ancestor; the last field is '*'.
    """
    dgh = DGH(column_name, root_value=root_token)
    cache: dict[str, DGHNode] = {root_token: dgh.root}

    def _node(val: str) -> DGHNode:
        if val not in cache:
            cache[val] = DGHNode(val)
        return cache[val]

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader, None)

        for raw in reader:
            # drop empty cells and whitespace
            parts = [x.strip() for x in raw if x.strip()]
            if not parts:
                continue

            # ensure trailing root sentinel removed
            if parts[-1] == root_token:
                parts = parts[:-1]

            # add edges: child -> parent for every consecutive pair
            for child_val, parent_val in zip(parts, parts[1:] + [root_token]):
                child, parent = _node(child_val), _node(parent_val)

                # Skip if already correctly linked
                if child.parent is parent:
                    continue

                # Only add the child if it's not already a child of this parent
                if child not in parent.children:
                    parent.add_child(child)

    return dgh
