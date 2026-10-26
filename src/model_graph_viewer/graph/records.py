"""Framework-independent data structures for the visualization graph.

Records describing fixed connections and metadata are frozen. Graph containers
and visualization settings remain editable by transformations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .utils import reserve_node_ids


@dataclass(frozen=True)
class TensorMetadata:
    """Dtype and shape descriptions; None means unavailable. No tensor values."""

    dtype: str | None = None
    shape: str | None = None


@dataclass(frozen=True)
class Edge:
    """One connection between a source output and a destination input."""

    source: int
    source_port: int
    target: int
    target_port: int
    tensor: TensorMetadata = field(default_factory=TensorMetadata)


@dataclass(frozen=True)
class HiddenInputAnnotation:
    """An original edge displayed at its consumer instead of as a wire.

    Constant sources may be removed. References connect non-Constant nodes,
    and their sources remain visible.
    """

    edge: Edge
    kind: str = "constant"
    shared: bool = False


@dataclass
class Visualization:
    """Presentation settings kept separate from operation attributes."""

    border_color: str | None = None
    input_annotations: list[HiddenInputAnnotation] = field(default_factory=list)
    collapsed_subgraphs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BodyPortMapping:
    """Map an owner port to a Parameter or Result in a body graph.

    Direction comes from the body's inputs or outputs list. An input may also
    receive its next-iteration value from a body Result.
    """

    parent_port: int
    body_node_id: int
    kind: str
    feedback_result_id: int | None = None
    attributes: dict[str, int] = field(default_factory=dict)


@dataclass
class ControlFlowBody:
    """A child graph and its owner-port mappings; control IDs are body-local."""

    graph: Graph
    inputs: list[BodyPortMapping] = field(default_factory=list)
    outputs: list[BodyPortMapping] = field(default_factory=list)
    iteration_parameter_id: int | None = None
    condition_result_id: int | None = None


@dataclass
class Node:
    """An operation with a scope-local ID, port metadata and optional bodies."""

    id: int
    name: str
    op_type: str
    attributes: dict[str, object] = field(default_factory=dict)
    input_count: int = 0
    outputs: list[TensorMetadata] = field(default_factory=list)
    viz: Visualization = field(default_factory=Visualization)
    subgraphs: dict[str, ControlFlowBody] = field(default_factory=dict)


@dataclass
class Graph:
    """One scope: operations keyed by ID and their visible port connections.

    Names may repeat, and two nodes may have several connections on different
    ports. Child scopes live in Node.subgraphs. The ID counter remembers removed
    sources so future nodes cannot reuse their provenance IDs.
    """

    nodes: dict[int, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _next_id: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        reserve_node_ids(self)
