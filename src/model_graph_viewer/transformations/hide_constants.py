"""Replace Constant wires with consumer input annotations."""

from collections import defaultdict

from ..graph import Graph, HiddenInputAnnotation


class HideConstantsTransformation:
    """Replace Constant edges with metadata at the original consumer input ports."""

    def apply(self, graph: Graph) -> None:
        constant_ids = {key for key, node in graph.nodes.items() if node.op_type == "Constant"}
        consumers: dict[int, set[int]] = defaultdict(set)
        for edge in graph.edges:
            if edge.source in constant_ids:
                consumers[edge.source].add(edge.target)

        visible_edges = []
        for edge in graph.edges:
            if edge.source not in constant_ids:
                visible_edges.append(edge)
                continue
            graph.nodes[edge.target].viz.input_annotations.append(
                HiddenInputAnnotation(
                    edge,
                    shared=len(consumers[edge.source]) > 1,
                )
            )
        graph.edges = visible_edges

        for node_id in constant_ids:
            del graph.nodes[node_id]
