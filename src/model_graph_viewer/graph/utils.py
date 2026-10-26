"""Scope traversal and node-ID bookkeeping, separate from the graph records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .records import Graph


def walk_graphs(graph: Graph) -> Iterator[Graph]:
    """Visit the root, then its bodies in order, using an explicit stack.

    Each body has its own graph container and scope-local operation IDs.
    """
    pending = [graph]
    while pending:
        scope = pending.pop()
        yield scope
        children = []
        for node in scope.nodes.values():
            for body in node.subgraphs.values():
                children.append(body.graph)
        pending.extend(reversed(children))


def reserve_node_ids(graph: Graph) -> None:
    """Remember visible and hidden-source IDs when a graph is constructed.

    The counter only increases, even when nodes or annotations are removed.
    """
    for node in graph.nodes.values():
        graph._next_id = max(graph._next_id, node.id + 1)
        for annotation in node.viz.input_annotations:
            graph._next_id = max(graph._next_id, annotation.edge.source + 1)


def allocate_node_id(graph: Graph) -> int:
    """Reserve a fresh ID for a new node, using the scope's saved counter.

    Construction remembers existing IDs in one scan. Transformations creating nodes use
    this function to advance the counter without rescanning the graph.
    """
    while graph._next_id in graph.nodes:
        graph._next_id += 1
    node_id = graph._next_id
    graph._next_id += 1
    return node_id
