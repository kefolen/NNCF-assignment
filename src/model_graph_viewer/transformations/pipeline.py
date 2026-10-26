"""Transformation contract and ordered execution across graph scopes."""

from typing import Iterable, Protocol

from ..graph import Graph, walk_graphs
from .collapse_subgraphs import CollapseSubgraphsTransformation


class GraphTransformation(Protocol):
    def apply(self, graph: Graph) -> None:
        """Change a visualization graph in place."""
        ...


def apply_transformations(
    graph: Graph,
    transformations: Iterable[GraphTransformation] = (),
    *,
    recursive: bool = True,
) -> None:
    """Apply each stage to all scopes in preorder, preserving order/repetitions.

    Set ``recursive=False`` to transform only the supplied scope. Ordinary
    transformations handle one scope; CollapseSubgraphsTransformation handles
    depth itself and runs once, relative to the supplied graph.
    There is no rollback for custom transformations. A failure never proceeds to rendering.
    Unexpected exceptions retain their type and receive a stage note.
    """
    for index, transformation in enumerate(transformations, start=1):
        stage = f"transformation {index} ({type(transformation).__name__})"
        try:
            if isinstance(transformation, CollapseSubgraphsTransformation):
                transformation.apply(graph, recursive=recursive)
            else:
                scopes = walk_graphs(graph) if recursive else (graph,)
                for scope in scopes:
                    transformation.apply(scope)
        except Exception as error:
            if hasattr(error, "add_note"):
                error.add_note(f"Failed during {stage}")
            raise
