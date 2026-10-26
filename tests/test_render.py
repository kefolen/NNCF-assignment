"""Renderer contracts: ports, readable metadata, references, scopes, and export failures."""

from copy import deepcopy
import re
import subprocess
import xml.etree.ElementTree as ET

import pytest

from model_graph_viewer import (
    BodyPortMapping, CollapseSubgraphsTransformation, ControlFlowBody, Edge, Graph,
    HideConstantsTransformation, HighlightTransformation,
    HiddenInputAnnotation, Node, RenderError, SharedTensorReferencesTransformation,
    TensorMetadata, apply_transformations, render_svg, to_dot,
)
from model_graph_viewer.loader.openvino import _read_graph
from model_graph_viewer.render import settings
from test_control_flow import control_model, if_model, scoped_graphs


def flat_graph():
    tensor = TensorMetadata("fp32", "[?, 2..8]")
    scalar = TensorMetadata("i64", "[]")
    return Graph(
        nodes={
            0: Node(0, 'long scope/duplicate <& "имя">', "Unfamiliar", outputs=[tensor, scalar]),
            1: Node(
                1, 'long scope/duplicate <& "имя">', "Consumer", input_count=3,
                attributes={"quoted <key>": '"value" & ' + "long " * 30},
                outputs=[TensorMetadata()],
            ),
        },
        edges=[Edge(0, 0, 1, 0, tensor), Edge(0, 1, 1, 1, scalar), Edge(0, 0, 1, 2, tensor)],
    )


def test_layout_overrides_preserve_defaults_edge_order_and_graph(monkeypatch):
    graph = flat_graph()
    before = deepcopy(graph)
    default = to_dot(graph)
    assert not {"mclimit", "nslimit", "nslimit1", "ordering"} & default.graph_attr.keys()

    monkeypatch.setattr(settings, "LARGE_GRAPH_THRESHOLD", 1)
    large_default = to_dot(graph)
    for name, value in settings.LARGE_GRAPH.items():
        assert large_default.graph_attr[name] == value

    tuned = to_dot(graph, mclimit=1, nslimit=2, nslimit1=4, ordering="out")
    assert {name: tuned.graph_attr[name] for name in ("mclimit", "nslimit", "nslimit1", "ordering")} == {
        "mclimit": "1", "nslimit": "2", "nslimit1": "4", "ordering": "out",
    }
    # Partial overrides leave other defaults intact, including a later default call.
    partial = to_dot(graph, mclimit=0)
    assert partial.graph_attr["mclimit"] == "0"
    assert partial.graph_attr["nslimit"] == settings.LARGE_GRAPH["nslimit"]
    assert partial.graph_attr["nslimit1"] == settings.LARGE_GRAPH["nslimit1"]
    assert to_dot(graph).graph_attr == large_default.graph_attr
    # Ordering constrains the same emitted port connections; no topology is rewritten.
    assert tuned.body == large_default.body
    assert graph == before


@pytest.mark.parametrize("options, message", [
    ({"mclimit": -1}, "mclimit"),
    ({"nslimit": float("nan")}, "nslimit"),
    ({"nslimit1": float("inf")}, "nslimit1"),
    ({"ordering": "left"}, "ordering"),
])
def test_invalid_layout_options_preserve_existing_svg(tmp_path, options, message):
    output = tmp_path / "existing.svg"
    output.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        render_svg(flat_graph(), output, **options)
    assert output.read_text(encoding="utf-8") == "original"


def test_ports_metadata_escaping_and_styles_in_real_svg(tmp_path):
    graph = flat_graph()
    HighlightTransformation([graph.nodes[0].name]).apply(graph)
    dot = to_dot(graph).source
    assert "n0:o0:s -> n1:i0:n" in dot
    assert "n0:o1:s -> n1:i1:n" in dot
    assert "n0:o0:s -> n1:i2:n" in dot

    output = render_svg(graph, tmp_path / "nested" / "view.svg")
    root = ET.parse(output).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    nodes = [element for element in root.iter() if element.get("class") == "node"]
    assert {element.get("id") for element in nodes} == {"n0", "n1"}
    assert sum(element.get("class") == "edge" for element in root.iter()) == 3
    assert all(any(element.get("stroke") == "red" for element in node.iter()) for node in nodes)
    assert any(element.get("fill") == settings.DEFAULT_FILL for element in root.iter())
    text = "".join(element.text or "" for element in root.iter() if element.tag.endswith("}text"))
    tooltips = [element.get("{http://www.w3.org/1999/xlink}title", "") for element in root.iter()]
    assert 'duplicate <& "имя">' not in text
    assert "long scope/" not in text
    assert any(graph.nodes[0].name in tooltip for tooltip in tooltips)
    assert any("i64[]" in tooltip for tooltip in tooltips)
    assert any("fp32[?, 2..8]" in tooltip for tooltip in tooltips)
    assert any("unknown shape unknown" in tooltip for tooltip in tooltips)
    assert "fp32" not in text and "quoted <key>" not in text
    assert all(any(element.tag.endswith("}path") for element in node.iter()) for node in nodes)
    full_attribute = "quoted <key>: " + str(graph.nodes[1].attributes["quoted <key>"])
    assert any(full_attribute in tooltip for tooltip in tooltips)
    labels = {
        node.get("id"): [item.text for item in node.iter() if item.tag.endswith("}text")]
        for node in nodes
    }
    assert labels["n0"] == ["Unfamiliar #0", "out1", "out2"]
    assert labels["n1"] == ["in1", "in2", "in3", "Consumer #1"]
    edge_tooltips = {
        element.get("{http://www.w3.org/1999/xlink}title")
        for edge in root.iter() if edge.get("class") == "edge"
        for element in edge.iter() if element.tag.endswith("}a")
    }
    assert edge_tooltips == {
        "Unfamiliar #0:out1 -> Consumer #1:in1 (dtype: fp32)",
        "Unfamiliar #0:out2 -> Consumer #1:in2 (dtype: i64)",
        "Unfamiliar #0:out1 -> Consumer #1:in3 (dtype: fp32)",
    }
    # Drawing IDs remain SVG identities, but input cells and node backgrounds
    # override Graphviz's default hover text with a blank title.
    for node in nodes:
        background = next(element for element in node.iter() if element.tag.endswith("}a"))
        assert not background.get("{http://www.w3.org/1999/xlink}title", "").strip()
        for anchor in (element for element in node.iter() if element.tag.endswith("}a")):
            if any((item.text or "").startswith("in") for item in anchor.iter() if item.tag.endswith("}text")):
                assert not anchor.get("{http://www.w3.org/1999/xlink}title", "").strip()


def test_visible_output_dtypes_preserve_ports_metadata_and_graph(tmp_path):
    graph = flat_graph()
    graph.nodes[2] = Node(2, "scalar", "Constant", outputs=[TensorMetadata("fp16", "[]")])
    graph.nodes[3] = Node(3, "no outputs", "Sink", input_count=1)
    graph.edges.append(Edge(2, 0, 3, 0))  # Missing edge dtype stays unknown despite a typed source.
    before = deepcopy(graph)
    default = to_dot(graph).source
    typed = to_dot(graph, show_dtypes=True).source
    assert [line for line in typed.splitlines() if " -> " in line] == [
        line for line in default.splitlines() if " -> " in line
    ]

    root = ET.parse(render_svg(graph, tmp_path / "dtypes.svg", show_dtypes=True)).getroot()
    labels = {
        node.get("id"): [item.text for item in node.iter() if item.tag.endswith("}text")]
        for node in root.iter() if node.get("class") == "node"
    }
    assert labels["n0"] == ["Unfamiliar #0", "out1: fp32", "out2: i64"]
    assert labels["n1"] == ["in1", "in2", "in3", "Consumer #1", "unknown"]
    assert labels["n2"] == ["Constant #2", "fp16"]
    assert labels["n3"] == ["Sink #3"]
    assert any(
        element.get("{http://www.w3.org/1999/xlink}title") == "Constant #2 -> Sink #3 (dtype: unknown)"
        for element in root.iter()
    )
    assert any(
        "fp32[?, 2..8]" in element.get("{http://www.w3.org/1999/xlink}title", "")
        for element in root.iter()
    )
    assert graph == before
    assert to_dot(graph).source == default


def test_hidden_inputs_and_reference_links_have_no_input_tooltips(tmp_path):
    tensor = TensorMetadata("fp16", "[4]")
    graph = Graph(
        nodes={
            0: Node(0, "source", "Parameter", outputs=[tensor]),
            1: Node(1, "weights", "Constant", outputs=[tensor]),
            2: Node(2, "consumer", "Add", input_count=3),
            3: Node(3, "consumer", "Add", input_count=2),
        },
        edges=[
            Edge(0, 0, 2, 0, tensor), Edge(0, 0, 2, 2, tensor), Edge(0, 0, 3, 0, tensor),
            Edge(1, 0, 2, 1, tensor), Edge(1, 0, 3, 1, tensor),
        ],
    )
    apply_transformations(graph, [HideConstantsTransformation(), SharedTensorReferencesTransformation(2)])
    before_render = deepcopy(graph)
    root = ET.parse(render_svg(graph, tmp_path / "annotations.svg")).getroot()
    assert graph == before_render
    ids = {element.get("id") for element in root.iter()}
    assert "n1" not in ids
    links = [element.get("{http://www.w3.org/1999/xlink}href") for element in root.iter()]
    assert links.count("#n0") == 3 and "n0" in ids
    text = "".join(root.itertext())
    assert "in2: fp16[4] (shared: 1)" in text
    assert "in3: ref #0" in text and "ref #0:" not in text
    reference_links = [element for element in root.iter() if element.get("{http://www.w3.org/1999/xlink}href")]
    assert all(not element.get("{http://www.w3.org/1999/xlink}title", "").strip() for element in reference_links)
    # Long Constant metadata and shared IDs must remain visible without tooltips.
    shape = "[" + ", ".join(["1"] * 30) + "]"
    graph.nodes[2].viz.input_annotations.append(HiddenInputAnnotation(
        Edge(12345, 0, 2, 3, TensorMetadata("i64", shape)), shared=True,
    ))
    graph.nodes[2].input_count = 4
    long_root = ET.parse(render_svg(graph, tmp_path / "long-input.svg")).getroot()
    for element in long_root.iter():
        tooltip = element.get("{http://www.w3.org/1999/xlink}title", "")
        assert not tooltip.startswith("in")
    long_input = next(
        element for element in long_root.iter() if element.tag.endswith("}a")
        and any((item.text or "").startswith("in4:") for item in element if item.tag.endswith("}text"))
    )
    lines = [item.text for item in long_input if item.tag.endswith("}text")]
    assert " ".join(lines) == f"in4: i64{shape} (shared: 12345)"
    consumers = [element for element in root.iter() if element.get("id") in {"n2", "n3"}]
    assert all(
        any(element.get("fill", "").lower() == settings.NODE_FILL["Add"].lower() for element in node.iter())
        for node in consumers
    )
    assert not any(element.get("class") == "edge" for element in root.iter())


def test_continuation_preference_preserves_connections_and_family_colours(tmp_path):
    tensor = TensorMetadata("fp32", "[4]")
    types = ["Parameter", "Add", "MatMul", "Reshape", "Swish", "Result", "Assign", "Unfamiliar"]
    graph = Graph(
        {i: Node(i, kind, kind, input_count=2, outputs=[tensor]) for i, kind in enumerate(types)},
        [Edge(a, 0, b, p, tensor) for a, b, p in [
            (0, 1, 0), (1, 5, 0), (0, 2, 0), (2, 3, 1), (3, 4, 0), (4, 5, 1), (3, 6, 0),
        ]],
    )
    before = deepcopy(graph)
    source = to_dot(graph).source
    main = next(line for line in source.splitlines() if "n0:o0:s -> n2:i0:n" in line)
    side = next(line for line in source.splitlines() if "n0:o0:s -> n1:i0:n" in line)
    assert f"weight={settings.CONTINUATION_WEIGHT}" in main
    assert f'weight={settings.EDGE["weight"]}' in side
    root = ET.parse(render_svg(graph, tmp_path / "families.svg")).getroot()
    assert sum(element.get("class") == "edge" for element in root.iter()) == len(graph.edges)
    for node in (element for element in root.iter() if element.get("class") == "node"):
        kind = graph.nodes[int(node.get("id")[1:])].op_type
        colour = settings.NODE_FILL.get(kind, settings.DEFAULT_FILL).lower()
        assert any(element.get("fill", "").lower() == colour for element in node.iter())
    assert graph == before

    # Cyclic custom graphs still render all edges without topology preferences.
    graph.edges.append(Edge(4, 0, 2, 1, tensor))
    root = ET.parse(render_svg(graph, tmp_path / "cycle.svg")).getroot()
    assert sum(element.get("class") == "edge" for element in root.iter()) == len(graph.edges)


def test_large_graph_splines_avoid_an_intermediate_node(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LARGE_GRAPH_THRESHOLD", 1)
    tensor = TensorMetadata("fp32", "[4]")
    graph = Graph(
        {
            0: Node(0, "input", "Parameter", outputs=[tensor]),
            1: Node(1, "middle", "Relu", input_count=1, outputs=[tensor]),
            2: Node(2, "merge", "Add", input_count=2, outputs=[tensor]),
        },
        [Edge(0, 0, 1, 0, tensor), Edge(1, 0, 2, 0, tensor), Edge(0, 0, 2, 1, tensor)],
    )
    root = ET.parse(render_svg(graph, tmp_path / "routing.svg")).getroot()
    namespace = {"s": "http://www.w3.org/2000/svg"}
    middle = root.find('.//s:g[@id="n1"]//s:path', namespace)
    outline = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", middle.get("d"))]
    left, right = min(outline[::2]), max(outline[::2])
    top, bottom = min(outline[1::2]), max(outline[1::2])
    bypass = next(
        edge for edge in root.findall('.//s:g[@class="edge"]', namespace)
        if edge.find("s:title", namespace).text == "n0:s->n2:n"
    )
    path = bypass.find(".//s:path", namespace).get("d")
    assert bypass.find(".//s:a", namespace).get("{http://www.w3.org/1999/xlink}title") == (
        "Parameter #0 -> Add #2:in2 (dtype: fp32)"
    )
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path)]
    start = numbers[:2]
    for offset in range(2, len(numbers), 6):
        a, b, end = [numbers[index:index + 2] for index in range(offset, offset + 6, 2)]
        for step in range(101):
            t = step / 100
            x, y = [
                (1 - t) ** 3 * start[axis] + 3 * (1 - t) ** 2 * t * a[axis]
                + 3 * (1 - t) * t ** 2 * b[axis] + t ** 3 * end[axis]
                for axis in (0, 1)
            ]
            assert not (left < x < right and top < y < bottom)
        start = end


@pytest.mark.parametrize("show_dtypes", [False, True])
def test_body_mappings_and_loop_semantics(control_model, tmp_path, show_dtypes):
    graph, _ = _read_graph(control_model)
    apply_transformations(graph, [HideConstantsTransformation()])
    owner = next(node for node in graph.nodes.values() if node.subgraphs)
    dot = to_dot(graph, show_dtypes=show_dtypes).source
    root = ET.parse(render_svg(graph, tmp_path / "bodies.svg", show_dtypes=show_dtypes)).getroot()
    text = "".join(root.itertext())
    namespace = {"s": "http://www.w3.org/2000/svg"}
    boundary = root.find(f'.//s:g[@id="n{owner.id}_outputs"]', namespace)
    edge_tooltips = {
        element.get("{http://www.w3.org/1999/xlink}title", "")
        for edge in root.findall('.//s:g[@class="edge"]', namespace)
        for element in edge.findall('.//s:a', namespace)
    }
    assert edge_tooltips and all(" -> " in tooltip for tooltip in edge_tooltips)
    # Mapping edges enter from above; outgoing tensors leave below the same port row.
    expected_outputs = []
    for port, tensor in enumerate(owner.outputs):
        label = f"out{port + 1}" if len(owner.outputs) > 1 else ""
        if show_dtypes:
            label = f"{label}: {tensor.dtype}" if label else tensor.dtype
        if label:
            expected_outputs.append(label)
    assert [item.text for item in boundary.findall(".//s:text", namespace)] == expected_outputs
    assert any(
        f"{owner.op_type} #{owner.id} outputs" in item.get("{http://www.w3.org/1999/xlink}title", "")
        for item in boundary.iter()
    )
    for body_index, (body_name, body) in enumerate(owner.subgraphs.items()):
        child = f"n{owner.id}_b{body_index}_"
        body_label = f"{owner.op_type} #{owner.id}/{body_name}"
        if show_dtypes:
            parameter = next(node for node in body.graph.nodes.values() if node.op_type == "Parameter")
            parameter_svg = root.find(f'.//s:g[@id="{child}n{parameter.id}"]', namespace)
            assert parameter.outputs[0].dtype in [
                item.text for item in parameter_svg.findall(".//s:text", namespace)
            ]
        for mapping in body.inputs:
            source = f"n{owner.id}:i{mapping.parent_port}:n"
            target = f"{child}n{mapping.body_node_id}:o0:w"
            assert f"{source} -> {target}" in dot
            owner_port = f":in{mapping.parent_port + 1}" if owner.input_count > 1 else ""
            assert (
                f"{owner.op_type} #{owner.id}{owner_port} -> {body_label}/Parameter #{mapping.body_node_id}"
                f" (dtype: {body.graph.nodes[mapping.body_node_id].outputs[0].dtype})"
            ) in edge_tooltips
            if mapping.feedback_result_id is not None:
                source = f"{child}n{mapping.feedback_result_id}:i0:e"
                target = f"{child}n{mapping.body_node_id}:o0:e"
                assert f"{source} -> {target}" in dot
                assert (
                    f"{body_label}/Result #{mapping.feedback_result_id} -> {body_label}/Parameter #{mapping.body_node_id}"
                    f" (dtype: {body.graph.nodes[mapping.body_node_id].outputs[0].dtype})"
                ) in edge_tooltips
        for mapping in body.outputs:
            source = f"{child}n{mapping.body_node_id}:i0:e"
            target = f"n{owner.id}_outputs:o{mapping.parent_port}:n"
            assert f"{source} -> {target}" in dot
            owner_port = f":out{mapping.parent_port + 1}" if len(owner.outputs) > 1 else ""
            assert (
                f"{body_label}/Result #{mapping.body_node_id} -> {owner.op_type} #{owner.id}{owner_port}"
                f" (dtype: {owner.outputs[mapping.parent_port].dtype})"
            ) in edge_tooltips
    if owner.op_type == "If":
        assert f"If #{owner.id}: then" in text and f"If #{owner.id}: else" in text
    else:
        assert "next iteration" in text and "merged (initial value)" in text
        if owner.op_type == "Loop":
            assert "current iteration" in text and "continue condition" in text
            assert (
                f"Loop #{owner.id} -> {body_label}/Parameter #{body.iteration_parameter_id}"
            ) in edge_tooltips
            assert (
                f"{body_label}/Result #{body.condition_result_id} -> Loop #{owner.id}"
            ) in edge_tooltips
        else:
            assert "sliced" in text and "concatenated" in text and "axis=0" in text


def test_constant_only_body_has_layout_constraints():
    tensor = TensorMetadata("fp32", "[]")
    body_graph = Graph(
        {0: Node(0, "constant", "Constant", outputs=[tensor]), 1: Node(1, "result", "Result", input_count=1)},
        [Edge(0, 0, 1, 0, tensor)],
    )
    body = ControlFlowBody(body_graph, outputs=[BodyPortMapping(0, 1, "value")])
    graph = Graph({0: Node(0, "branch", "If", outputs=[tensor], subgraphs={"then": body})})
    source = to_dot(graph).source
    # With no parent-to-body input mapping, these constraints establish vertical order.
    assert "n0 -> n0_b0_n0 [style=invis]" in source
    assert "n0_b0_n1 -> n0_outputs [style=invis]" in source


@pytest.mark.parametrize("collapse", [False, True])
def test_nested_reference_targets_are_unique_and_resolve(tmp_path, collapse):
    graph, _ = _read_graph(if_model(nested=True))
    if collapse:
        CollapseSubgraphsTransformation().apply(graph)
    for prefix, scope in scoped_graphs(graph):
        edge = next((edge for edge in scope.edges if scope.nodes[edge.source].op_type == "If"), scope.edges[0])
        scope.edges.remove(edge)
        scope.nodes[edge.target].viz.input_annotations.append(HiddenInputAnnotation(edge, kind="reference"))
    root = ET.parse(render_svg(graph, tmp_path / "references.svg")).getroot()
    ids = [element.get("id") for element in root.iter() if element.get("id")]
    assert len(ids) == len(set(ids))
    links = [element.get("{http://www.w3.org/1999/xlink}href") for element in root.iter()]
    assert all(link[1:] in ids for link in links if link)
    assert all(
        f"{prefix}n{node.id}" in ids
        for prefix, scope in scoped_graphs(graph) for node in scope.nodes.values()
    )
    owner = next(node for node in graph.nodes.values() if node.op_type == "If")
    expected_owner_target = f"#n{owner.id}" + ("" if collapse else "_outputs")
    assert expected_owner_target in links


@pytest.mark.parametrize("failure, message", [
    (FileNotFoundError("dot"), "Native Graphviz"),
    (subprocess.TimeoutExpired("dot", 1), "timed out"),
    (subprocess.CalledProcessError(1, "dot", stderr="layout failed"), "layout failed"),
])
def test_native_failure_preserves_existing_svg(tmp_path, monkeypatch, failure, message):
    output = tmp_path / "existing.svg"
    output.write_text("original", encoding="utf-8")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RenderError, match=message) as caught:
        render_svg(flat_graph(), output, timeout=1)
    assert caught.value.__cause__ is failure
    assert output.read_text(encoding="utf-8") == "original"
