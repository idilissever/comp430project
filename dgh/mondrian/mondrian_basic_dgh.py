from __future__ import annotations
from pathlib import Path
from dgh.dgh import DGH, DGHNode, add_node


def _populate_dgh(dgh: DGH, spec_lines: list[str]) -> None:
    """Populate a DGH in-place from semicolon-delimited lines."""
    cache: dict[str, DGHNode] = {"*": dgh.root}

    for raw in spec_lines:
        parts: list[str] = [p for p in raw.strip().split(";") if p]
        # ignore trailing "*" sentinel
        if parts[-1] == "*":
            parts = parts[:-1]

        # walk from root downward, (grand)parent first
        parent: DGHNode = dgh.root
        for token in reversed(parts):
            if token not in cache:
                cache[token] = add_node(parent, token)
            parent = cache[token]


def build_dgh_from_file(path: Path, column_name: str) -> DGH:
    """Return a fully populated DGH built from one spec file."""
    spec_lines = path.read_text().splitlines()
    dgh = DGH(column_name)
    _populate_dgh(dgh, spec_lines)
    return dgh
