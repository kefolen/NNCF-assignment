"""HTML labels, tensor descriptions, and source anchors for the independent graph."""

import html
import textwrap

from ..graph import Graph, Node, TensorMetadata
from . import settings


def tensor_text(tensor: TensorMetadata) -> str:
    dtype = tensor.dtype if tensor.dtype is not None else "unknown"
    shape = tensor.shape if tensor.shape is not None else "shape unknown"
    return dtype + ("" if shape.startswith("[") else " ") + shape


def port_label(direction: str, port: int, count: int) -> str:
    return f"{direction}{port + 1}" if count > 1 else ""


def output_node_id(node: Node, prefix: str) -> str:
    """Expanded owners have a separate drawing node containing their output ports."""
    suffix = "_outputs" if node.subgraphs and node.outputs else ""
    return f"{prefix}n{node.id}{suffix}"


def _attributes(attributes: dict[str, str]) -> str:
    return " ".join(
        f'{name}="{html.escape(value, quote=True)}"' for name, value in attributes.items()
    )


def _cell(text: str, **attributes: str) -> str:
    lines = text.split("\n")
    visible = "\n".join(
        line if len(line) <= settings.TEXT_LIMIT else line[:settings.TEXT_LIMIT - 1] + "…"
        for line in lines
    )
    content = html.escape(visible, quote=True).replace("\n", "<BR/>")
    if visible != text:
        attributes.setdefault("TITLE", text)
    return f'<TD {_attributes(attributes)}>{content}</TD>'


def _table(rows: list[str], attributes: dict[str, str]) -> str:
    return f'<TABLE {_attributes(attributes)}>{"".join(rows)}</TABLE>'


def _port_row(cells: list[str]) -> str:
    if not cells:
        return ""
    ports = _table([f'<TR>{"".join(cells)}</TR>'], settings.PORT_TABLE)
    return f'<TR><TD {_attributes(settings.PORT_CONTAINER)}>{ports}</TD></TR>'


def _outputs(node: Node, show_dtypes: bool) -> str:
    if len(node.outputs) == 1 and not show_dtypes and node.op_type != "Constant":
        return _port_row([_cell("", PORT="o0")])
    cells = []
    for port, tensor in enumerate(node.outputs):
        output = port_label("out", port, len(node.outputs))
        label = output
        if node.op_type == "Constant":
            label = textwrap.fill(
                (f"{output}: " if output else "") + tensor_text(tensor),
                width=settings.TEXT_LIMIT,
            )
        elif show_dtypes:
            dtype = tensor.dtype if tensor.dtype is not None else "unknown"
            label = f"{output}: {dtype}" if output else dtype
        tooltip = (f"{output}: " if output else "") + tensor_text(tensor)
        cells.append(_cell(label, TITLE=tooltip, PORT=f"o{port}"))
    return _port_row(cells)


def node_label(node: Node, graph: Graph, prefix: str, *, show_dtypes: bool = False) -> str:
    annotations = {item.edge.target_port: item for item in node.viz.input_annotations}
    inputs = []
    # Empty titles fall back to Graphviz's label; whitespace suppresses input tooltips.
    for port in range(node.input_count):
        label = port_label("in", port, node.input_count)
        prefix_text = f"{label}: " if label else ""
        annotation = annotations.get(port)
        if annotation is None:
            inputs.append(_cell(label, TITLE=" ", PORT=f"i{port}"))
            continue
        edge = annotation.edge
        if annotation.kind == "reference":
            source = graph.nodes[edge.source]
            output = port_label("out", edge.source_port, len(source.outputs))
            suffix = f":{output}" if output else ""
            text = f"{prefix_text}ref #{source.id}{suffix}"
            inputs.append(_cell(
                text, TITLE=" ", PORT=f"i{port}", HREF=f"#{output_node_id(source, prefix)}",
            ))
        else:
            metadata = tensor_text(edge.tensor)
            shared = f" (shared: {edge.source})" if annotation.shared else ""
            text = textwrap.fill(f"{prefix_text}{metadata}{shared}", width=settings.TEXT_LIMIT)
            inputs.append(_cell(text, TITLE=" ", PORT=f"i{port}"))

    title = f"{node.op_type} #{node.id}"
    details = [f"{name}: {value}" for name, value in node.attributes.items()]
    if len(node.outputs) == 1:
        details.append(tensor_text(node.outputs[0]))
    tooltip = "\n".join([title, node.name, *details])
    rows = [_port_row(inputs), f"<TR>{_cell(title, TITLE=tooltip)}</TR>"]
    for name, value in node.attributes.items():
        text = textwrap.fill(f"{name}: {value}", width=settings.TEXT_LIMIT)
        rows.append(f'<TR>{_cell(text, ALIGN="LEFT")}</TR>')
    if node.viz.collapsed_subgraphs:
        rows.append(f'<TR>{_cell("collapsed bodies: " + ", ".join(node.viz.collapsed_subgraphs))}</TR>')
    if not node.subgraphs:
        rows.append(_outputs(node, show_dtypes))
    border = node.viz.border_color or settings.DEFAULT_BORDER
    fill = settings.NODE_FILL.get(node.op_type, settings.DEFAULT_FILL)
    return f'<{_table(rows, {**settings.TABLE, "COLOR": border, "BGCOLOR": fill})}>'


def output_label(node: Node, *, show_dtypes: bool = False) -> str:
    tooltip = f"{node.op_type} #{node.id} outputs\n{node.name}"
    if len(node.outputs) == 1:
        tooltip += "\n" + tensor_text(node.outputs[0])
    appearance = {
        **settings.TABLE, "COLOR": settings.OUTPUT_BORDER, "BGCOLOR": settings.OUTPUT_FILL,
        "TITLE": tooltip,
    }
    return f'<{_table([_outputs(node, show_dtypes)], appearance)}>'
