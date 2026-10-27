from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import openvino as ov
import pytest
from openvino import opset13 as ops

from model_graph_viewer import (
    Edge, Graph, GraphTransformation, HideConstantsTransformation,
    HighlightTransformation, ModelLoader, Node, OpenVINOLoader, TensorMetadata, visualize,
)
from model_graph_viewer.interfaces.cli import main


@pytest.fixture
def small_model(tmp_path):
    data = ops.parameter(ov.PartialShape([ov.Dimension(1, 8), 4]), ov.Type.f32, name="input")
    split = ops.split(data, ops.constant(1, ov.Type.i64), 2, name="split")
    weights = ops.constant([[1.0, 0.0], [0.0, 1.0]], ov.Type.f32)
    left = ops.matmul(split.output(0), weights, False, False, name="branch")
    right = ops.matmul(split.output(1), weights, False, False, name="branch")
    twice = ops.add(left, left, name="parallel")
    model = ov.Model([ops.add(twice, right, name="sum")], [data], "branching")
    path = tmp_path / "model.xml"
    ov.save_model(model, path, compress_to_fp16=False)
    return model, path


def test_end_to_end_api_and_cli(small_model, tmp_path, capsys, monkeypatch):
    _, model_path = small_model
    output = tmp_path / "graph.svg"
    sources = []
    run = subprocess.run

    def record_run(*args, **kwargs):
        sources.append(kwargs["input"])
        return run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_run)
    result = visualize(
        model_path, output, [HighlightTransformation(["branch"]), HideConstantsTransformation()],
        show_dtypes=True, mclimit=1, nslimit=2, nslimit1=4, ordering="out",
    )
    root = ET.parse(output).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert sum(element.get("stroke") == "red" for element in root.iter()) >= 2
    assert "shared:" in "".join(root.itertext())
    assert any(element.tag.endswith("}text") and element.text == "fp32" for element in root.iter())
    assert result == output
    assert all(option in sources[0] for option in ("mclimit=1", "nslimit=2", "nslimit1=4", "ordering=out"))

    assert main([
        "--model", str(model_path), "--output", str(output),
        "--transformations", "hide-constants", "--show-dtypes",
        "--mclimit", "2", "--nslimit", "3", "--nslimit1", "5", "--ordering", "in",
    ]) == 0
    root = ET.parse(output).getroot()
    assert root.tag.endswith("svg")
    assert any(element.tag.endswith("}text") and element.text == "fp32" for element in root.iter())
    assert all(option in sources[1] for option in ("mclimit=2.0", "nslimit=3.0", "nslimit1=5.0", "ordering=in"))
    assert capsys.readouterr().err.strip() == str(output)


def test_cli_displays_constant_metadata_and_operation_attributes_by_default(small_model, tmp_path):
    _, model_path = small_model
    output = tmp_path / "constants.svg"
    assert main(["--model", str(model_path), "--output", str(output)]) == 0
    root = ET.parse(output).getroot()
    constant_labels = [
        [item.text for item in node.iter() if item.tag.endswith("}text")]
        for node in root.iter() if node.get("class") == "node"
        and any((item.text or "").startswith("Constant #") for item in node.iter())
    ]
    assert sorted(label[1:] for label in constant_labels) == [["fp32[2, 2]"], ["i64[]"]]
    matmul_labels = [
        [item.text for item in node.iter() if item.tag.endswith("}text")]
        for node in root.iter() if node.get("class") == "node"
        and any((item.text or "").startswith("MatMul #") for item in node.iter())
    ]
    assert len(matmul_labels) == 2
    assert all("transpose_a: False" in label and "transpose_b: False" in label for label in matmul_labels)


@pytest.mark.parametrize("flags, expected_references", [
    ([], 0),
    (["--transformations", "shared-tensor-references"], 0),
    (["--transformations", "shared-tensor-references", "--reference-min-consumers", "3"], 3),
    (["--transformations", "shared-tensor-references", "--reference-min-consumers", "4"], 0),
])
def test_cli_shared_references_are_opt_in_with_a_configurable_threshold(tmp_path, flags, expected_references):
    data = ops.parameter([4], ov.Type.f32, name="shared_input")
    branches = [ops.relu(data, name=f"branch_{index}") for index in range(3)]
    model_path = tmp_path / "shared.xml"
    ov.save_model(ov.Model(branches, [data]), model_path, compress_to_fp16=False)
    output = tmp_path / "shared.svg"

    assert main(["--model", str(model_path), "--output", str(output), *flags]) == 0
    root = ET.parse(output).getroot()
    links = [
        element.get("{http://www.w3.org/1999/xlink}href", "")
        for element in root.iter()
    ]
    assert sum(link.startswith("#") for link in links) == expected_references
    assert sum(element.get("class") == "edge" for element in root.iter()) == 6 - expected_references


@pytest.mark.parametrize("model_path, output_name", [
    ("models/classifier.xml", "classifier.svg"),
    ("models/tinyllama-fp16/openvino_model.xml", "tinyllama-fp16.svg"),
    ("openvino_model.xml", "openvino_model.svg"),
])
def test_model_based_default_output(small_model, tmp_path, monkeypatch, model_path, output_name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("model_graph_viewer.interfaces.api.render_svg", lambda graph, path, **kwargs: Path(path))
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(small_model[0], path, compress_to_fp16=False)

    assert visualize(path) == Path(output_name)
    assert main(["--model", str(path)]) == 0


@pytest.mark.parametrize("flags", [
    ["--transformations", "missing"], ["--transformations", "highlight"], ["--highlight", "branch"],
    ["--transformations", "hide-constants,"], ["--reference-min-consumers", "8"],
    ["--transformations", "shared-tensor-references", "--reference-min-consumers", "1"],
    ["--subgraph-depth", "1"],
    ["--transformations", "collapse-subgraphs", "--subgraph-depth", "-1"],
    ["--transformations", "collapse-subgraphs", "--subgraph-depth", "1.5"],
    ["--render-timeout", "nan"],
    ["--mclimit", "-1"], ["--nslimit", "nan"], ["--nslimit1", "inf"],
    ["--ordering", "left"],
])
def test_cli_rejects_invalid_options_before_loading(flags, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--model", "unused.xml", "--output", str(tmp_path / "unused.svg"), *flags])
    assert error.value.code == 2


def test_cli_expected_load_error(tmp_path, capsys):
    assert main(["--model", "missing.xml", "--output", str(tmp_path / "unused.svg")]) == 1
    assert "missing.xml" in capsys.readouterr().err


def test_custom_loader_and_transformation_use_only_the_independent_graph(tmp_path, monkeypatch):
    monkeypatch.setattr("model_graph_viewer.interfaces.api.render_svg", lambda graph, path, **kwargs: Path(path))
    tensor = TensorMetadata("fp32", "[2]")
    graph = Graph(
        nodes={
            0: Node(0, "input", "Parameter", outputs=[tensor]),
            1: Node(1, "weights", "Constant", outputs=[tensor]),
            2: Node(2, "sum", "Add", input_count=2, outputs=[tensor]),
        },
        edges=[Edge(0, 0, 2, 0, tensor), Edge(1, 0, 2, 1, tensor)],
    )
    calls = []

    class FixtureLoader:
        def load(self, path: str | Path) -> Graph:
            calls.append(path)
            return graph

    class RenameTransformation:
        def apply(self, graph: Graph) -> None:
            graph.nodes[2].name = "renamed"

    # Structural interfaces require no inheritance or registration in the API.
    loader: ModelLoader = FixtureLoader()
    transformations: list[GraphTransformation] = [
        RenameTransformation(), HighlightTransformation(["renamed"]), HideConstantsTransformation(),
    ]
    result = visualize(
        "another-format", tmp_path / "custom.svg", transformations=transformations, loader=loader,
    )
    assert calls == ["another-format"]
    assert result == tmp_path / "custom.svg"
    assert set(graph.nodes) == {0, 2}
    assert graph.nodes[2].name == "renamed"
    assert graph.nodes[2].viz.border_color == "red"
    assert graph.edges == [Edge(0, 0, 2, 0, tensor)]
    annotation = graph.nodes[2].viz.input_annotations[0]
    assert annotation.edge == Edge(1, 0, 2, 1, tensor)
