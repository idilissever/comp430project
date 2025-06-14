from __future__ import annotations
from typing import Any, Optional


class DGHNode:
    """
    Node in a Domain Generalization Hierarchy (DGH).
    """

    def __init__(self, value: Any) -> None:
        self.value: Any = value
        self.parent: Optional[DGHNode] = None
        self.children: list[DGHNode] = []

    def add_child(self, child_node: DGHNode) -> None:
        """
        Add an existing DGHNode as a child of this node.
        Sets the child's parent to this node.
        """
        if child_node.parent is not None:
            # Only remove if the child is actually in the parent's children list
            if child_node in child_node.parent.children:
                child_node.parent.children.remove(child_node)

        child_node.parent = self
        self.children.append(child_node)

    def is_root(self) -> bool:
        """
        Return True if this node is the root (has no parent).
        """
        return self.parent is None

    def depth(self) -> int:
        """
        Calculate and return the depth of this node within the tree.
        Root has depth 0; each level down increases depth by 1.
        """
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth

    def ancestors(self) -> list[DGHNode]:
        """
        Return a list of ancestors for this node, starting from itself up to the root.
        """
        ancestors = []
        node = self
        while node is not None:
            ancestors.append(node)
            node = node.parent
        return ancestors

    def leaf_count(self) -> int:
        """
        Return the number of leaf nodes in the subtree rooted at this node.
        A leaf node has no children.
        """
        if not self.children:
            return 1
        return sum(child.leaf_count() for child in self.children)

    def __repr__(self) -> str:
        return f"DGHNode({self.value!r})"

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on value and position in tree.
        """
        if not isinstance(other, DGHNode):
            return NotImplemented
        return self.value == other.value and self.parent == other.parent

    def __hash__(self) -> int:
        """
        Make nodes hashable for use in sets/dicts.
        """
        return hash((self.value, id(self.parent) if self.parent else None))


def add_node(parent_node: DGHNode, value: Any) -> DGHNode:
    """
    Create a new node with the given value under parent_node.
    Returns the newly created DGHNode.
    """
    new_node = DGHNode(value)
    parent_node.add_child(new_node)
    return new_node


def node_depth(node: DGHNode) -> int:
    """
    Return the depth of the specified node.
    """
    return node.depth()


def most_recent_common_ancestor(node1: DGHNode, node2: DGHNode) -> Optional[DGHNode]:
    """
    Find and return the Most Recent Common Ancestor of node1 and node2.
    If no common ancestor is found, returns None.
    """
    if node1 == node2:
        return node1

    ancestors1 = set(node1.ancestors())
    node = node2
    while node is not None:
        if node in ancestors1:
            return node
        node = node.parent
    return None


class DGH:
    """
    Generalization Hierarchy tree class (DGH).
    Manages a tree of DGHNode objects and provides utility methods such as most recent common ancestor and depth calculation.
    """

    def __init__(self, column_name: str, root_value: Any = "any") -> None:
        """
        Initialize the DGH with a single root node. Default value is "any".
        """
        self.root = DGHNode(root_value)
        self.column_name = column_name

    def find_node_by_value(
        self, value: Any, start_node: Optional[DGHNode] = None
    ) -> Optional[DGHNode]:
        """
        Find and return the first node with the given value using DFS.
        If start_node is None, searches from root.
        """
        if start_node is None:
            start_node = self.root

        if start_node.value == str(value):
            return start_node

        for child in start_node.children:
            result = self.find_node_by_value(value, child)
            if result is not None:
                return result
        return None

    def get_all_nodes(self) -> list[DGHNode]:
        """
        Return a list of all nodes in the tree using DFS traversal.
        """
        nodes = []

        def dfs(node: DGHNode) -> None:
            nodes.append(node)
            for child in node.children:
                dfs(child)

        dfs(self.root)
        return nodes

    def __repr__(self) -> str:
        return f"DGH(root={self.root!r})"
