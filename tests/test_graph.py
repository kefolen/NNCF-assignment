"""Structural invariants of the framework-independent multigraph."""

from model_graph_viewer.graph import (
    Edge,
    Graph,
    HiddenInputAnnotation,
    Node,
    TensorMetadata,
    allocate_node_id,
)
from model_graph_viewer.transformations import HideConstantsTransformation, apply_transformations


def branching_graph() -> Graph:
    tensor = TensorMetadata("fp32", "[?, 4]")
    return Graph(
        nodes={
            0: Node(0, "duplicate", "Split", outputs=[tensor, TensorMetadata()]),
            1: Node(1, "duplicate", "Add", input_count=2, outputs=[tensor]),
            2: Node(2, "unused", "UnfamiliarOperation"),
        },
        edges=[Edge(0, 0, 1, 0, tensor), Edge(0, 1, 1, 1)],
    )


def test_hidden_provenance_and_removed_nodes_keep_ids_reserved() -> None:
    graph = branching_graph()
    graph.edges.pop()
    graph.nodes[80] = Node(80, "weight", "Constant", outputs=[TensorMetadata()])
    graph.edges.append(Edge(80, 0, 1, 1))
    graph = Graph(graph.nodes, graph.edges)
    apply_transformations(graph, [HideConstantsTransformation()])
    assert 80 not in graph.nodes
    assert graph.nodes[1].viz.input_annotations[0].edge.source == 80
    del graph.nodes[2]
    assert allocate_node_id(graph) == 81
    assert allocate_node_id(graph) == 82

    graph.nodes[1].viz.input_annotations = [HiddenInputAnnotation(Edge(0, 1, 1, 1), kind="reference")]
    assert allocate_node_id(graph) == 83


def test_constructor_reserves_provenance_without_a_visible_source() -> None:
    node = Node(2, "consumer", "Result", input_count=1)
    node.viz.input_annotations.append(HiddenInputAnnotation(Edge(15, 0, 2, 0)))
    graph = Graph({2: node})
    assert allocate_node_id(graph) == 16
