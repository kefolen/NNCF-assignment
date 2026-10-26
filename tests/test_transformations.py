from copy import deepcopy

import pytest

from model_graph_viewer import (
    CollapseSubgraphsTransformation, Edge, Graph, HideConstantsTransformation, HighlightTransformation,
    Node, SharedTensorReferencesTransformation, TensorMetadata,
    apply_transformations,
)


FLOAT = TensorMetadata("fp32", "[?, 2]")


def node(node_id, name="operation", op_type="Unknown", inputs=0, outputs=1):
    return Node(node_id, name, op_type, input_count=inputs, outputs=[FLOAT] * outputs)


def test_highlight_matches_all_duplicate_names_without_touching_topology():
    graph = Graph({0: node(0, "same"), 1: node(1, "same"), 2: node(2, "Same")}, [])
    graph.nodes[2].viz.border_color = "blue"
    graph.nodes[0].attributes["original"] = "kept"
    apply_transformations(graph, [HighlightTransformation(["same", "absent"])] * 2)
    assert [n.viz.border_color for n in graph.nodes.values()] == ["red", "red", "blue"]
    assert graph.nodes[0].attributes == {"original": "kept"}
    assert graph.edges == []


def test_constant_hiding_preserves_shared_ports_parallel_connections_and_outputs():
    graph = Graph(
        {
            0: node(0, op_type="Parameter", outputs=2),
            1: node(1, op_type="Constant"),
            2: node(2, inputs=4),
            3: node(3, op_type="Result", inputs=1, outputs=0),
            4: node(4, op_type="Constant"),  # isolated
        },
        [Edge(0, 0, 2, 0, FLOAT), Edge(0, 1, 2, 2, FLOAT),
         Edge(1, 0, 2, 1, FLOAT), Edge(1, 0, 2, 3), Edge(1, 0, 3, 0, FLOAT)],
    )
    original = list(graph.edges)
    apply_transformations(graph, [HideConstantsTransformation()])
    assert set(graph.nodes) == {0, 2, 3}
    assert [(e.source_port, e.target_port) for e in graph.edges] == [(0, 0), (1, 2)]
    annotations = graph.nodes[2].viz.input_annotations
    assert [a.edge for a in annotations] == original[2:4]
    assert all(a.shared for a in annotations)
    assert annotations[0].edge is original[2]
    assert annotations[0].edge.tensor is FLOAT
    assert annotations[1].edge.tensor == TensorMetadata()
    assert graph.nodes[3].viz.input_annotations[0].edge == original[4]
    once = deepcopy(graph)
    apply_transformations(graph, [HideConstantsTransformation()])
    assert graph == once


def test_repeated_use_on_one_consumer_is_not_shared():
    graph = Graph(
        {0: node(0, op_type="Constant"), 1: node(1, inputs=2)},
        [Edge(0, 0, 1, 0), Edge(0, 0, 1, 1)],
    )
    apply_transformations(graph, [HideConstantsTransformation()])
    assert set(graph.nodes) == {1}
    assert graph.edges == []
    annotations = graph.nodes[1].viz.input_annotations
    assert [a.edge.target_port for a in annotations] == [0, 1]
    assert all(not a.shared for a in annotations)


def test_pipeline_order_repetitions_empty_pipeline_and_failed_stage():
    graph = Graph({0: node(0, "weight", "Constant"), 1: node(1, inputs=1)}, [Edge(0, 0, 1, 0)])
    unchanged = deepcopy(graph)
    apply_transformations(graph, [])
    assert graph == unchanged
    apply_transformations(graph, [HighlightTransformation(["weight"]), HideConstantsTransformation()])
    assert graph.nodes[1].viz.border_color is None
    assert graph.nodes[1].viz.input_annotations[0].edge.source == 0

    class AppendName:
        def apply(self, graph):
            graph.nodes[1].name += "!"

    class Fail:
        def apply(self, graph):
            raise RuntimeError("transformation failed")

    apply_transformations(graph, [AppendName(), HighlightTransformation(["operation!"]), AppendName()])
    assert graph.nodes[1].name == "operation!!"
    assert graph.nodes[1].viz.border_color == "red"
    with pytest.raises(RuntimeError, match="transformation failed"):
        apply_transformations(graph, [Fail(), AppendName()])
    assert graph.nodes[1].name == "operation!!"


def test_shared_references_retain_exact_dependency_multiset_and_canonical_sources():
    graph = Graph(
        {0: node(0, "duplicate", outputs=2), 1: node(1, "duplicate"),
         2: node(2, inputs=3), 3: node(3, inputs=1), 4: node(4, inputs=1)},
        [Edge(0, 0, 2, 0, FLOAT), Edge(0, 0, 2, 1, FLOAT), Edge(0, 0, 3, 0, FLOAT),
         Edge(0, 1, 2, 2, FLOAT), Edge(1, 0, 4, 0)],
    )
    original = list(graph.edges)
    apply_transformations(graph, [HighlightTransformation(["duplicate"]), SharedTensorReferencesTransformation(2)])
    assert graph.edges == original[3:]
    assert set(graph.nodes) == set(range(5))
    assert graph.nodes[0].viz.border_color == graph.nodes[1].viz.border_color == "red"
    restored = list(graph.edges)
    for target in graph.nodes.values():
        for a in target.viz.input_annotations:
            assert a.kind == "reference"
            restored.append(a.edge)
    key = lambda e: (e.source, e.source_port, e.target, e.target_port)
    assert sorted(restored, key=key) == sorted(original, key=key)
    once = deepcopy(graph)
    apply_transformations(graph, [SharedTensorReferencesTransformation(2)])
    assert graph == once


def test_reference_threshold_counts_distinct_consumers_per_output_and_skips_constants():
    graph = Graph(
        {0: node(0, outputs=2), 1: node(1, op_type="Constant"),
         2: node(2, inputs=4), 3: node(3, inputs=2)},
        [Edge(0, 0, 2, 0), Edge(0, 0, 2, 1), Edge(0, 1, 2, 2),
         Edge(0, 1, 3, 0), Edge(1, 0, 2, 3), Edge(1, 0, 3, 1)],
    )
    other_order = deepcopy(graph)
    apply_transformations(graph, [SharedTensorReferencesTransformation(2), HideConstantsTransformation()])
    apply_transformations(other_order, [HideConstantsTransformation(), SharedTensorReferencesTransformation(2)])
    assert graph.edges == [Edge(0, 0, 2, 0), Edge(0, 0, 2, 1)]
    for view in (graph, other_order):
        assert 1 not in view.nodes
        assert sorted((a.kind, a.edge.source, a.edge.source_port, a.edge.target_port)
                      for n in view.nodes.values() for a in n.viz.input_annotations) == [
            ("constant", 1, 0, 1), ("constant", 1, 0, 3),
            ("reference", 0, 1, 0), ("reference", 0, 1, 2),
        ]


def test_transformation_configuration_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="collection"):
        HighlightTransformation("name")
    with pytest.raises(ValueError, match="at least one"):
        HighlightTransformation([])
    with pytest.raises(ValueError, match="at least 2"):
        SharedTensorReferencesTransformation(1)
    for invalid_depth in (-1, 1.5, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            CollapseSubgraphsTransformation(invalid_depth)

    graph = Graph({0: node(0)}, [])
    original = deepcopy(graph)
    apply_transformations(graph, [CollapseSubgraphsTransformation()])
    assert graph == original  # a flat model has no bodies to collapse
