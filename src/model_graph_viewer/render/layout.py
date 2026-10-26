"""Small topology-based preferences for Graphviz; never change model connections."""

from graphlib import CycleError, TopologicalSorter

from ..graph import Edge, Graph


def continuation_edges(graph: Graph) -> set[Edge]:
    """Favour each source's longest downstream branch to reduce sideways zigzags.

    This is a drawing heuristic, not a semantic main-path detector. Cyclic scopes
    use Graphviz's normal weights; feedback mappings are drawn separately.
    """
    outgoing: dict[int, list[Edge]] = {node_id: [] for node_id in graph.nodes}
    for edge in graph.edges:
        outgoing[edge.source].append(edge)
    try:
        # Successors are prerequisites here, so sinks come before their sources.
        order = tuple(TopologicalSorter({
            source: {edge.target for edge in edges} for source, edges in outgoing.items()
        }).static_order())
    except CycleError:
        return set()

    depth: dict[int, int] = {}
    preferred = set()
    for node_id in order:
        edges = outgoing[node_id]
        if edges:
            edge = max(edges, key=lambda edge: depth[edge.target])
            preferred.add(edge)
            depth[node_id] = depth[edge.target] + 1
        else:
            depth[node_id] = 0
    return preferred
