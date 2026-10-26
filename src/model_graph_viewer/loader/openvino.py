"""Convert OpenVINO scopes and tensor ports into independent graph records."""

from pathlib import Path
from typing import Any

import openvino

from ..graph import (
    BodyPortMapping, ControlFlowBody, Edge, Graph, Node, TensorMetadata,
)
from .base import ModelLoadError


def _dimension_description(dimension: Any) -> str:
    minimum = dimension.get_min_length()
    maximum = dimension.get_max_length()
    if minimum == maximum:
        return str(minimum)
    if minimum == 0 and maximum < 0:
        return "?"
    return f"{minimum}..{'?' if maximum < 0 else maximum}"


def _shape_description(shape: Any) -> str | None:
    if shape is None:
        return None
    if shape.rank.is_dynamic:
        return f"rank={_dimension_description(shape.rank)}"
    return "[" + ", ".join(_dimension_description(dimension) for dimension in shape) + "]"


def _dtype_description(element_type: Any) -> str | None:
    if element_type is None:
        return None
    name = element_type.get_type_name()
    return {"f16": "fp16", "f32": "fp32", "f64": "fp64"}.get(name, name)


def _tensor_metadata(output: Any) -> TensorMetadata:
    return TensorMetadata(
        dtype=_dtype_description(output.get_element_type()),
        shape=_shape_description(output.get_partial_shape()),
    )


def _boundary_id(operations: list[Any], index: int, node_ids: dict[int, int]) -> int:
    """Translate a Parameter/Result list index into a body node ID."""
    if not 0 <= index < len(operations):
        raise ModelLoadError(f"Body interface index {index} is out of range")
    return node_ids[operations[index].get_instance_id()]


def _read_body(
    model: Any, input_descriptions: list[Any], output_descriptions: list[Any],
    special_ports: tuple[int, ...] = (),
) -> ControlFlowBody:
    """Copy a body and its parent-port mappings, including loop feedback."""
    graph, node_ids = _read_graph(model)
    parameters, results = model.get_parameters(), model.get_results()
    body = ControlFlowBody(graph)
    slice_fields = ("start", "stride", "part_size", "end", "axis")
    for description in input_descriptions:
        type_name = description.get_type_info().name
        feedback = None
        attributes = {}
        if type_name == "InvariantInputDescription":
            kind = "invariant"
        elif type_name == "SliceInputDescription":
            kind = "sliced"
            attributes = {name: int(getattr(description, name)) for name in slice_fields}
        elif type_name == "MergedInputDescription":
            kind = "merged"
            feedback = _boundary_id(results, description.body_value_index, node_ids)
        else:
            raise ModelLoadError(f"Unsupported body input description: {type_name}")
        body.inputs.append(BodyPortMapping(
            parent_port=description.input_index,
            body_node_id=_boundary_id(parameters, description.body_parameter_index, node_ids),
            kind=kind,
            feedback_result_id=feedback,
            attributes=attributes,
        ))
    for description in output_descriptions:
        type_name = description.get_type_info().name
        if type_name == "BodyOutputDescription":
            kind = "value"
            fields = ("iteration",)
        elif type_name == "ConcatOutputDescription":
            kind = "concatenated"
            fields = slice_fields
        else:
            raise ModelLoadError(f"Unsupported body output description: {type_name}")
        body.outputs.append(BodyPortMapping(
            parent_port=description.output_index,
            body_node_id=_boundary_id(results, description.body_value_index, node_ids),
            kind=kind,
            attributes={name: int(getattr(description, name)) for name in fields},
        ))
    if special_ports:
        # Loop uses body Parameter/Result indexes; negative values mean absent
        iteration, condition = special_ports
        if iteration >= 0:
            body.iteration_parameter_id = _boundary_id(parameters, iteration, node_ids)
        if condition >= 0:
            body.condition_result_id = _boundary_id(results, condition, node_ids)
    return body


def _read_graph(model: Any) -> tuple[Graph, dict[int, int]]:
    operations = model.get_ordered_ops()
    operation_ids = {operation.get_instance_id(): index for index, operation in enumerate(operations)}
    if len(operation_ids) != len(operations):
        raise ModelLoadError("OpenVINO returned duplicate operation identities")

    nodes: dict[int, Node] = {}
    for node_id, operation in enumerate(operations):
        op_type = operation.get_type_name()
        # get_attributes() returns detached Python metadata. Constants bypass it
        # because their generic attributes can expose tensor values.
        nodes[node_id] = Node(
            id=node_id,
            name=operation.get_friendly_name(),
            op_type=op_type,
            attributes={} if op_type == "Constant" else operation.get_attributes(),
            input_count=operation.get_input_size(),
            outputs=[_tensor_metadata(output) for output in operation.outputs()],
        )
        # Special handling for operations that contain subgraphs
        if op_type == "If":
            for index, (name, body_model) in enumerate((
                ("then", operation.get_then_body()), ("else", operation.get_else_body()),
            )):
                nodes[node_id].subgraphs[name] = _read_body(
                    body_model, operation.get_input_descriptions(index), operation.get_output_descriptions(index),
                )
        elif op_type in {"Loop", "TensorIterator"}:
            special = tuple(operation.get_special_body_ports()) if op_type == "Loop" else ()
            nodes[node_id].subgraphs["body"] = _read_body(
                operation.get_function(), operation.get_input_descriptions(),
                operation.get_output_descriptions(), special,
            )

    edges: list[Edge] = []
    for target_id, operation in enumerate(operations):
        for input_port in operation.inputs():
            source_output = input_port.get_source_output()
            source_identity = source_output.get_node().get_instance_id()
            if source_identity not in operation_ids:
                raise ModelLoadError(
                    f"Input {input_port.get_index()} of node {target_id} has a source outside its graph scope"
                )
            source_id = operation_ids[source_identity]
            source_port = source_output.get_index()
            edges.append(Edge(
                source=source_id, source_port=source_port,
                target=target_id, target_port=input_port.get_index(),
                tensor=nodes[source_id].outputs[source_port],
            ))
    return Graph(nodes, edges), operation_ids


class OpenVINOLoader:
    """Load OpenVINO IR files using the ``ModelLoader`` contract."""

    def load(self, path: str | Path) -> Graph:
        """Read an IR ``.xml`` file and its SDK-resolved weights."""
        model_path = Path(path)
        if model_path.suffix.lower() != ".xml":
            raise ModelLoadError(f"Expected an OpenVINO IR .xml file: {model_path}")
        if not model_path.is_file():
            raise ModelLoadError(f"OpenVINO IR file does not exist or is not a file: {model_path}")

        try:
            model = openvino.Core().read_model(str(model_path))
            return _read_graph(model)[0]
        except (OSError, RuntimeError) as error:
            raise ModelLoadError(f"Could not load OpenVINO IR {model_path}: {error}") from error
