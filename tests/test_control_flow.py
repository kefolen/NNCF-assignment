"""Real SDK control-flow interfaces, recursive transformations, and graph transformations."""

from copy import deepcopy
from dataclasses import asdict, replace

import openvino as ov
from openvino import opset13 as op
import pytest

from model_graph_viewer import (
    CollapseSubgraphsTransformation, Edge, HideConstantsTransformation,
    HighlightTransformation, Node,
    SharedTensorReferencesTransformation, allocate_node_id,
    apply_transformations,
)
from model_graph_viewer.loader.openvino import _read_graph
from model_graph_viewer.graph import walk_graphs


def make_if(condition, value, *, nested=False):
    left = op.parameter([2], ov.Type.f32, name="body_input")
    right = op.parameter([2], ov.Type.f32, name="body_input")
    then_value = (
        make_if(op.constant(True, ov.Type.boolean).output(0), left.output(0))
        if nested else op.add(left, op.constant([1., 2.], ov.Type.f32), name="inside")
    )
    then_body = ov.Model([then_value], [left])
    else_body = ov.Model([op.multiply(right, op.constant([2., 3.], ov.Type.f32), name="inside")], [right])
    branch = op.if_op(condition)
    branch.set_friendly_name("branch")
    branch.set_then_body(then_body)
    branch.set_else_body(else_body)
    branch.set_input(value, left, right)
    branch.set_output(then_body.get_results()[0], else_body.get_results()[0])
    return branch


def if_model(*, nested=False):
    condition = op.parameter([], ov.Type.boolean, name="condition")
    data = op.parameter([2], ov.Type.f32, name="data")
    return ov.Model([make_if(condition.output(0), data.output(0), nested=nested)], [condition, data])


@pytest.fixture(params=["If", "Loop", "TensorIterator"])
def control_model(request):
    if request.param == "If":
        return if_model()
    if request.param == "Loop":
        data = op.parameter([2], ov.Type.f32, name="data")
        iteration = op.parameter([], ov.Type.i64, name="iteration")
        state = op.parameter([2], ov.Type.f32, name="state")
        updated = op.add(state, op.constant([1., 2.], ov.Type.f32), name="inside")
        condition = op.less(iteration, op.constant(2, ov.Type.i64))
        loop = op.loop(op.constant(3, ov.Type.i64), op.constant(True, ov.Type.boolean))
        loop.set_function(ov.Model([condition, updated], [iteration, state]))
        loop.set_special_body_ports([0, 0])
        loop.set_merged_input(state, data.output(0), updated.output(0))
        return ov.Model([loop.get_iter_value(updated.output(0), -1)], [data])
    data = op.parameter([3, 2], ov.Type.f32, name="sequence")
    sliced = op.parameter([1, 2], ov.Type.f32, name="slice")
    state = op.parameter([1, 2], ov.Type.f32, name="state")
    updated = op.add(sliced, state, name="inside")
    iterator = op.tensor_iterator()
    iterator.set_body(ov.Model([updated], [sliced, state]))
    iterator.set_sliced_input(sliced, data.output(0), 0, 1, 1, -1, 0)
    iterator.set_merged_input(state, op.constant([[0., 0.]], ov.Type.f32).output(0), updated.output(0))
    concatenated = iterator.get_concatenated_slices(updated.output(0), 0, 1, 1, -1, 0)
    return ov.Model([concatenated, iterator.get_iter_value(updated.output(0), -1)], [data])


def test_control_flow_interfaces_and_recursive_transformations(control_model):
    graph, _ = _read_graph(control_model)
    original = asdict(graph)
    owner = next(node for node in graph.nodes.values() if node.subgraphs)
    sdk_owner = next(node for node in control_model.get_ordered_ops() if node.get_type_name() == owner.op_type)
    body_models = (
        [sdk_owner.get_then_body(), sdk_owner.get_else_body()]
        if owner.op_type == "If" else [sdk_owner.get_function()]
    )
    for body, sdk_body in zip(owner.subgraphs.values(), body_models):
        assert len(body.graph.nodes) == len(sdk_body.get_ordered_ops())
    assert len(list(walk_graphs(graph))) == 1 + len(body_models)
    for body in owner.subgraphs.values():
        assert all(body.graph.nodes[m.body_node_id].op_type == "Parameter" for m in body.inputs)
        assert all(body.graph.nodes[m.body_node_id].op_type == "Result" for m in body.outputs)
    if owner.op_type == "If":
        assert set(owner.subgraphs) == {"then", "else"}
        assert [body.inputs[0].parent_port for body in owner.subgraphs.values()] == [1, 1]
    else:
        body = owner.subgraphs["body"]
        merged = next(mapping for mapping in body.inputs if mapping.kind == "merged")
        assert body.graph.nodes[merged.feedback_result_id].op_type == "Result"
        if owner.op_type == "Loop":
            assert body.graph.nodes[body.iteration_parameter_id].name == "iteration"
            assert body.graph.nodes[body.condition_result_id].op_type == "Result"
            assert body.outputs[0].attributes == {"iteration": -1}
        else:
            sliced = next(mapping for mapping in body.inputs if mapping.kind == "sliced")
            assert sliced.attributes == {"start": 0, "stride": 1, "part_size": 1, "end": -1, "axis": 0}
            assert [mapping.kind for mapping in body.outputs] == ["concatenated", "value"]
            assert body.outputs[0].attributes == sliced.attributes

    mappings = [(deepcopy(body.inputs), deepcopy(body.outputs)) for body in owner.subgraphs.values()]
    apply_transformations(graph, [
        HighlightTransformation(["inside"]), HideConstantsTransformation(),
        SharedTensorReferencesTransformation(2),
    ])
    for scope in walk_graphs(graph):
        assert all(node.op_type != "Constant" for node in scope.nodes.values())
        assert all(node.viz.border_color == "red" for node in scope.nodes.values() if node.name == "inside")
    assert [(body.inputs, body.outputs) for body in owner.subgraphs.values()] == mappings
    assert asdict(_read_graph(control_model)[0]) == original


def test_nested_bodies_and_explicit_scope_policy():
    graph, _ = _read_graph(if_model(nested=True))
    assert len(list(walk_graphs(graph))) == 5
    apply_transformations(graph, [HighlightTransformation(["inside"])], recursive=False)
    assert all(node.viz.border_color is None for scope in walk_graphs(graph) for node in scope.nodes.values())

    class Rename:
        def apply(self, scope):
            for node in scope.nodes.values():
                node.name += "!"

    apply_transformations(graph, (stage for stage in [Rename(), HighlightTransformation(["inside!"])]))
    matches = [node for scope in walk_graphs(graph) for node in scope.nodes.values() if node.name == "inside!"]
    assert len(matches) == 3 and all(node.viz.border_color == "red" for node in matches)

    unchanged = deepcopy(graph)
    apply_transformations(graph, [CollapseSubgraphsTransformation(1)], recursive=False)
    assert graph == unchanged
    apply_transformations(graph, [CollapseSubgraphsTransformation(0)], recursive=False)
    assert len(list(walk_graphs(graph))) == 1


@pytest.mark.parametrize("max_depth", [0, 1, 2, 5])
def test_collapse_depth_preserves_retained_scopes_ports_and_metadata(max_depth):
    model = if_model(nested=True)
    graph, _ = _read_graph(model)
    original = deepcopy(graph)
    direct = deepcopy(graph)
    CollapseSubgraphsTransformation(max_depth).apply(direct)
    apply_transformations(graph, [CollapseSubgraphsTransformation(max_depth)])
    assert graph == direct

    original_scopes = dict(scoped_graphs(original))
    remaining_scopes = dict(scoped_graphs(graph))
    assert set(remaining_scopes) == {
        prefix for prefix in original_scopes if prefix.count("_b") <= max_depth
    }
    for prefix, scope in remaining_scopes.items():
        before = original_scopes[prefix]
        assert scope.edges == before.edges
        assert scope.nodes.keys() == before.nodes.keys()
        for node_id, node in scope.nodes.items():
            old = before.nodes[node_id]
            collapsed = prefix.count("_b") >= max_depth
            expected_names = list(old.subgraphs) if collapsed else []
            expected_viz = replace(old.viz, collapsed_subgraphs=expected_names)
            assert replace(node, subgraphs={}) == replace(old, subgraphs={}, viz=expected_viz)
            assert set(node.subgraphs) == (set() if collapsed else set(old.subgraphs))
            for name, body in node.subgraphs.items():
                assert body.inputs == old.subgraphs[name].inputs
                assert body.outputs == old.subgraphs[name].outputs

    once = deepcopy(graph)
    apply_transformations(graph, [CollapseSubgraphsTransformation(max_depth)])
    assert graph == once
    if max_depth >= 2:
        assert graph == original
    assert _read_graph(model)[0] == original


def test_collapsed_control_flow_preserves_dependencies(control_model):
    graph, _ = _read_graph(control_model)
    owner = next(node for node in graph.nodes.values() if node.subgraphs)
    body_names = list(owner.subgraphs)
    consumer_id = allocate_node_id(graph)
    graph.nodes[consumer_id] = Node(consumer_id, "extra consumer", "Result", input_count=1)
    graph.edges.append(Edge(owner.id, 0, consumer_id, 0, owner.outputs[0]))
    original_edges = list(graph.edges)

    apply_transformations(graph, [
        HighlightTransformation([owner.name]), CollapseSubgraphsTransformation(),
        HideConstantsTransformation(), SharedTensorReferencesTransformation(2),
    ])
    assert not owner.subgraphs
    assert owner.viz.collapsed_subgraphs == body_names
    dependencies = graph.edges + [
        annotation.edge for node in graph.nodes.values() for annotation in node.viz.input_annotations
    ]
    key = lambda edge: (edge.source, edge.source_port, edge.target, edge.target_port)
    assert sorted(dependencies, key=key) == sorted(original_edges, key=key)


def scoped_graphs(graph, prefix=""):
    yield prefix, graph
    for node in graph.nodes.values():
        for index, body in enumerate(node.subgraphs.values()):
            yield from scoped_graphs(body.graph, f"{prefix}n{node.id}_b{index}_")
