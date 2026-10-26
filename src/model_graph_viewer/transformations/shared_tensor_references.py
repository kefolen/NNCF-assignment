"""Replace high-fan-out wires with linked input references."""

from collections import defaultdict
from dataclasses import dataclass

from ..graph import Graph, HiddenInputAnnotation


@dataclass
class SharedTensorReferencesTransformation:
    """Replace high-fan-out wires with explicit, linked input references.

    Sources remain visible. Each reference retains the canonical source ID and
    output port, target input port and edge tensor metadata. Constants are left
    for HideConstantsTransformation regardless of pipeline order. This is a presentation
    rewrite, not a computation or an assertion of equivalent tensor values.
    """

    min_consumers: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.min_consumers, bool) or not isinstance(self.min_consumers, int) or self.min_consumers < 2:
            raise ValueError("min_consumers must be an integer of at least 2")

    def apply(self, graph: Graph) -> None:
        consumers: dict[tuple[int, int], set[int]] = defaultdict(set)
        for edge in graph.edges:
            if graph.nodes[edge.source].op_type != "Constant":
                consumers[edge.source, edge.source_port].add(edge.target)
        selected = {source for source, targets in consumers.items() if len(targets) >= self.min_consumers}

        visible_edges = []
        for edge in graph.edges:
            if (edge.source, edge.source_port) not in selected:
                visible_edges.append(edge)
                continue
            graph.nodes[edge.target].viz.input_annotations.append(
                HiddenInputAnnotation(edge, kind="reference", shared=True)
            )
        graph.edges = visible_edges
