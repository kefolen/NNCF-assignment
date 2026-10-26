"""OpenVINO parsing fidelity, safe metadata extraction, and path failures."""

from pathlib import Path

import openvino as ov
from openvino import opset13 as op
import pytest

from model_graph_viewer.graph import TensorMetadata, walk_graphs
from model_graph_viewer.loader import ModelLoadError
from model_graph_viewer.loader.openvino import OpenVINOLoader, _read_graph


def test_ports_duplicate_names_attributes_and_independence() -> None:
    parameter = op.parameter(ov.PartialShape([ov.Dimension(1, 8), 4]), ov.Type.f32, name="duplicate")
    split = op.split(parameter, op.constant(1, ov.Type.i64), 2, name="split")
    added = op.add(split.output(0), split.output(1), name="duplicate")
    model = ov.Model([added.output(0), split.output(1)], [parameter])
    before = str(model)
    original_attributes = added.get_attributes()

    graph, _ = _read_graph(model)
    duplicates = [node for node in graph.nodes.values() if node.name == "duplicate"]
    assert len(duplicates) == 2
    assert len({node.id for node in duplicates}) == 2
    split_node = next(node for node in graph.nodes.values() if node.op_type == "Split")
    added_node = next(node for node in duplicates if node.op_type == "Add")
    assert len(split_node.outputs) == 2
    assert split_node.outputs == [TensorMetadata("fp32", "[1..8, 2]")] * 2
    assert added_node.attributes["auto_broadcast"] == "numpy"
    assert {
        (edge.source_port, edge.target_port)
        for edge in graph.edges
        if edge.source == split_node.id and edge.target == added_node.id
    } == {(0, 0), (1, 1)}
    for edge in graph.edges:
        assert edge.tensor is graph.nodes[edge.source].outputs[edge.source_port]

    added_node.name = "changed visualization name"
    added_node.attributes["auto_broadcast"] = "changed visualization attribute"
    parameter_node = next(node for node in graph.nodes.values() if node.op_type == "Parameter")
    parameter_node.attributes["shape"][1] = 999
    graph.edges.clear()
    assert added.get_friendly_name() == "duplicate"
    assert added.get_attributes() == original_attributes
    assert parameter.get_attributes()["shape"][1] == 4
    assert str(model) == before
    assert added.input(0).get_source_output().get_index() == 0
    assert added.input(1).get_source_output().get_index() == 1


def test_scalar_unknown_dimensions_bounds_and_dynamic_rank() -> None:
    cases = {
        "scalar": (ov.PartialShape([]), "[]"),
        "dynamic_dimension": (ov.PartialShape([-1, 4]), "[?, 4]"),
        "bounded_dimension": (ov.PartialShape([ov.Dimension(2, 9), 4]), "[2..9, 4]"),
        "lower_bound": (ov.PartialShape([ov.Dimension(3, -1)]), "[3..?]"),
        "dynamic_rank": (ov.PartialShape.dynamic(), "rank=?"),
    }
    parameters = [op.parameter(shape, ov.Type.bf16, name=name) for name, (shape, _) in cases.items()]
    graph, _ = _read_graph(ov.Model(parameters, parameters))
    for node in graph.nodes.values():
        if node.op_type == "Parameter":
            assert node.outputs == [TensorMetadata("bf16", cases[node.name][1])]


def test_constant_extraction_never_requests_values_or_generic_attributes(monkeypatch) -> None:
    constant = op.constant([1, 2, 3], ov.Type.i64)
    model = ov.Model([constant], [])

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor payload accessor was called")

    constant_type = type(constant)
    for method in ("get_attributes", "get_data", "get_vector", "get_tensor_view"):
        monkeypatch.setattr(constant_type, method, forbidden)
    monkeypatch.setattr(constant_type, "data", property(forbidden))
    monkeypatch.setattr(constant_type, "tensor_view", property(forbidden))
    graph, _ = _read_graph(model)
    constant_node = next(node for node in graph.nodes.values() if node.op_type == "Constant")
    assert constant_node.input_count == 0
    assert constant_node.attributes == {}
    assert constant_node.outputs == [TensorMetadata("i64", "[3]")]

    branch = op.if_op(op.parameter([], ov.Type.boolean).output(0))
    then_body = ov.Model([constant], [])
    else_body = ov.Model([constant], [])
    branch.set_then_body(then_body)
    branch.set_else_body(else_body)
    branch.set_output(then_body.get_results()[0], else_body.get_results()[0])
    graph, _ = _read_graph(ov.Model([branch], [branch.input_value(0).get_node()]))
    assert sum(node.op_type == "Constant" for scope in walk_graphs(graph) for node in scope.nodes.values()) == 2


def test_missing_metadata_and_bounded_rank_are_explicit() -> None:
    class BoundedRankShape:
        # OpenVINO 2026.4.1 drops bounds in PartialShape.dynamic(Dimension(...)).
        # Exercise the metadata contract when a source does expose those bounds.
        rank = ov.Dimension(1, 4)

    class Output:
        def __init__(self, shape=None):
            self.shape = shape

        def get_element_type(self):
            return None

        def get_partial_shape(self):
            return self.shape

    class TopLevelOperation:
        def get_instance_id(self):
            return 24

        def get_type_name(self):
            return "MetadataFixture"

        def get_friendly_name(self):
            return "conditional"

        def get_attributes(self):
            return {"ordinary": [1, True, "text"]}

        def get_input_size(self):
            return 0

        def outputs(self):
            return [Output(), Output(BoundedRankShape())]

        def inputs(self):
            return []

    class Model:
        def get_ordered_ops(self):
            return [TopLevelOperation()]

    graph, _ = _read_graph(Model())
    assert graph.nodes[0].outputs == [TensorMetadata(), TensorMetadata(shape="rank=1..4")]
    assert graph.nodes[0].attributes["ordinary"] == [1, True, "text"]


def test_weightless_ir_loads_without_bin(tmp_path: Path) -> None:
    parameter = op.parameter([2], ov.Type.f32, name="input")
    model = ov.Model([op.relu(parameter)], [parameter])
    model_path = tmp_path / "weightless.xml"
    ov.serialize(model, str(model_path))
    model_path.with_suffix(".bin").unlink()
    graph = OpenVINOLoader().load(model_path)
    assert {node.op_type for node in graph.nodes.values()} == {"Parameter", "Relu", "Result"}
    assert len(graph.edges) == 2


def test_unexpected_programming_errors_keep_their_type(monkeypatch, tmp_path: Path) -> None:
    from model_graph_viewer.loader import openvino as openvino_loader

    parameter = op.parameter([2], ov.Type.f32)
    model = ov.Model([parameter], [parameter])
    model_path = tmp_path / "model.xml"
    ov.serialize(model, str(model_path))

    def programming_error(_output):
        raise TypeError("internal extraction bug")

    monkeypatch.setattr(openvino_loader, "_tensor_metadata", programming_error)
    with pytest.raises(TypeError, match="internal extraction bug"):
        _read_graph(model)[0]
    with pytest.raises(TypeError, match="internal extraction bug"):
        OpenVINOLoader().load(model_path)


def test_file_conversion_errors_include_path_and_preserve_cause(monkeypatch, tmp_path: Path) -> None:
    from model_graph_viewer.loader import openvino as openvino_loader

    parameter = op.parameter([2], ov.Type.f32)
    model_path = tmp_path / "model.xml"
    ov.serialize(ov.Model([parameter], [parameter]), str(model_path))
    failure = ModelLoadError("Unsupported body input description")

    def conversion_error(_model):
        raise failure

    monkeypatch.setattr(openvino_loader, "_read_graph", conversion_error)
    with pytest.raises(ModelLoadError, match="Unsupported body input description") as caught:
        OpenVINOLoader().load(model_path)
    assert str(model_path) in str(caught.value)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("model.onnx", None, "Expected an OpenVINO IR .xml"),
        ("missing.xml", None, "does not exist"),
        ("broken.xml", "<invalid>", "Could not load OpenVINO IR"),
    ],
)
def test_path_errors_include_actionable_context(tmp_path: Path, filename, contents, message) -> None:
    model_path = tmp_path / filename
    if contents is not None:
        model_path.write_text(contents)
    with pytest.raises(ModelLoadError, match=message) as caught:
        OpenVINOLoader().load(model_path)
    assert filename in str(caught.value)
