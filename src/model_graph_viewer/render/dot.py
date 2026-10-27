"""Draw operations, retained body scopes, and their port connections."""

import math
from typing import Literal

import graphviz

from ..graph import BodyPortMapping, ControlFlowBody, Graph, Node, walk_graphs
from . import settings
from .labels import node_label, output_label, output_node_id, port_label
from .layout import continuation_edges


def to_dot(
    graph: Graph, *, show_dtypes: bool = False,
    mclimit: float | None = None, nslimit: float | None = None,
    nslimit1: float | None = None, ordering: Literal["in", "out"] | None = None,
) -> graphviz.Digraph:
    """Build DOT with Constant dtype/shape and optional dtypes on other outputs.

    Layout overrides default to None, keeping the size-based defaults.
    mclimit scales crossing minimization; nslimit and nslimit1 bound coordinate
    and rank optimization iterations per node. Ordering constrains incoming or
    outgoing edges to their emission order. It does not reorder the graph.
    """
    limits = {"mclimit": mclimit, "nslimit": nslimit, "nslimit1": nslimit1}
    for name, value in limits.items():
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if ordering not in (None, "in", "out"):
        raise ValueError("ordering must be 'in', 'out', or None")

    dot = graphviz.Digraph("model", engine=settings.ENGINE, format="svg", strict=False)
    dot.graph_attr.update(settings.GRAPH)
    dot.node_attr.update(settings.NODE)
    dot.edge_attr.update(settings.EDGE)
    if sum(len(scope.nodes) for scope in walk_graphs(graph)) >= settings.LARGE_GRAPH_THRESHOLD:
        dot.graph_attr.update(settings.LARGE_GRAPH)
    dot.graph_attr.update({name: str(value) for name, value in limits.items() if value is not None})
    if ordering is not None:
        dot.graph_attr["ordering"] = ordering
    if any(node.subgraphs for node in graph.nodes.values()):
        dot.attr(label="Control flow: then/else are alternatives; loop bodies are shown once.")
    _add_scope(dot, graph, show_dtypes=show_dtypes)
    return dot


def _add_scope(
    dot: graphviz.Digraph, graph: Graph, prefix: str = "", *, show_dtypes: bool = False,
) -> None:
    for node in graph.nodes.values():
        name = f"{prefix}n{node.id}"
        dot.node(name, label=node_label(node, graph, prefix, show_dtypes=show_dtypes), id=name)
        if not node.subgraphs:
            continue
        if node.outputs:
            output = output_node_id(node, prefix)
            dot.node(output, label=output_label(node, show_dtypes=show_dtypes), id=output)
        for index, (body_name, body) in enumerate(node.subgraphs.items()):
            child = f"{name}_b{index}_"
            with dot.subgraph(name=f"cluster_{child}") as cluster:
                cluster.attr(
                    label=graphviz.escape(f"{node.op_type} #{node.id}: {body_name}"),
                    **settings.BODY,
                )
                _add_scope(cluster, body.graph, child, show_dtypes=show_dtypes)
            _add_interface(dot, node, body, prefix, child, body_name)

            # Keep disconnected body roots below the owner and sinks above its outputs.
            targets = {edge.target for edge in body.graph.edges}
            sources = {edge.source for edge in body.graph.edges}
            for node_id in body.graph.nodes:
                if node_id not in targets:
                    dot.edge(name, f"{child}n{node_id}", **settings.LAYOUT_EDGE)
                if node.outputs and node_id not in sources:
                    dot.edge(
                        output_node_id(body.graph.nodes[node_id], child),
                        output_node_id(node, prefix), **settings.LAYOUT_EDGE,
                    )

    preferred = continuation_edges(graph)
    for edge in graph.edges:
        source_node = graph.nodes[edge.source]
        target_node = graph.nodes[edge.target]
        source = output_node_id(source_node, prefix)
        source_port = port_label("out", edge.source_port, len(source_node.outputs))
        target_port = port_label("in", edge.target_port, target_node.input_count)
        dtype = edge.tensor.dtype if edge.tensor.dtype is not None else "unknown"
        tooltip = (
            f"{_endpoint_label(source_node, source_port)} -> {_endpoint_label(target_node, target_port)}"
            f" (dtype: {dtype})"
        )
        dot.edge(
            f"{source}:o{edge.source_port}:s", f"{prefix}n{edge.target}:i{edge.target_port}:n",
            tooltip=graphviz.escape(tooltip),
            weight=settings.CONTINUATION_WEIGHT if edge in preferred else settings.EDGE["weight"],
        )


def _endpoint_label(node: Node, port: str = "") -> str:
    label = f"{node.op_type} #{node.id}"
    return f"{label}:{port}" if port else label


def _mapping_text(mapping: BodyPortMapping) -> str:
    kind = "merged (initial value)" if mapping.kind == "merged" else mapping.kind
    details = ", ".join(f"{key}={value}" for key, value in mapping.attributes.items())
    return kind + ("; " + details if details else "")


def _connect(
    dot: graphviz.Digraph, source: str, target: str, label: str, tooltip: str, kind: str = "mapping",
    *, dtype: str | None = None,
) -> None:
    if kind != "control":
        tooltip += f" (dtype: {dtype if dtype is not None else 'unknown'})"
    dot.edge(
        source, target, label=graphviz.escape(label), tooltip=graphviz.escape(tooltip),
        **settings.INTERFACE_EDGES[kind],
    )


def _add_interface(
    dot: graphviz.Digraph, owner: Node, body: ControlFlowBody, prefix: str, child: str, body_name: str,
) -> None:
    name = f"{prefix}n{owner.id}"
    owner_label = _endpoint_label(owner)
    body_label = f"{owner_label}/{body_name}"
    for mapping in body.inputs:
        parameter_node = body.graph.nodes[mapping.body_node_id]
        parameter = f"{child}n{mapping.body_node_id}:o0"
        parameter_label = f"{body_label}/{_endpoint_label(parameter_node)}"
        port = port_label("in", mapping.parent_port, owner.input_count)
        label = (f"{port}: " if port else "") + _mapping_text(mapping)
        _connect(
            dot, f"{name}:i{mapping.parent_port}:n", f"{parameter}:w",
            label, f"{_endpoint_label(owner, port)} -> {parameter_label}", dtype=parameter_node.outputs[0].dtype,
        )
        if mapping.feedback_result_id is not None:
            result_label = f"{body_label}/{_endpoint_label(body.graph.nodes[mapping.feedback_result_id])}"
            _connect(
                dot, f"{child}n{mapping.feedback_result_id}:i0:e", f"{parameter}:e",
                "next iteration", f"{result_label} -> {parameter_label}", "feedback",
                dtype=parameter_node.outputs[0].dtype,
            )
    for mapping in body.outputs:
        result_label = f"{body_label}/{_endpoint_label(body.graph.nodes[mapping.body_node_id])}"
        port = port_label("out", mapping.parent_port, len(owner.outputs))
        label = (f"{port}: " if port else "") + _mapping_text(mapping)
        _connect(
            dot, f"{child}n{mapping.body_node_id}:i0:e", f"{name}_outputs:o{mapping.parent_port}:n",
            label, f"{result_label} -> {_endpoint_label(owner, port)}", dtype=owner.outputs[mapping.parent_port].dtype,
        )
    if body.iteration_parameter_id is not None:
        parameter_label = f"{body_label}/{_endpoint_label(body.graph.nodes[body.iteration_parameter_id])}"
        _connect(
            dot, name, f"{child}n{body.iteration_parameter_id}:o0:w",
            "current iteration", f"{owner_label} -> {parameter_label}", "control",
        )
    if body.condition_result_id is not None:
        result_label = f"{body_label}/{_endpoint_label(body.graph.nodes[body.condition_result_id])}"
        _connect(
            dot, f"{child}n{body.condition_result_id}:i0:e", name,
            "continue condition", f"{result_label} -> {owner_label}", "control",
        )
