"""Depth-limited collapsing of control-flow bodies."""

from dataclasses import dataclass

from ..graph import Graph


@dataclass
class CollapseSubgraphsTransformation:
    """Keep at most max_depth body levels, preserving owner nodes and ports.

    The supplied graph has depth zero. Collapsed bodies are removed from the
    visualization graph; only their names remain as presentation metadata.
    """

    max_depth: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int) or self.max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")

    def apply(self, graph: Graph, *, recursive: bool = True) -> None:
        pending = [(graph, 0)]
        while pending:
            scope, depth = pending.pop()
            for node in scope.nodes.values():
                if depth >= self.max_depth:
                    node.viz.collapsed_subgraphs.extend(node.subgraphs)
                    node.subgraphs.clear()
                elif recursive:
                    pending.extend((body.graph, depth + 1) for body in node.subgraphs.values())
