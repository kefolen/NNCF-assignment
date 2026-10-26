"""Highlight operations selected by exact friendly names."""

from dataclasses import dataclass
from typing import Iterable

from ..graph import Graph


@dataclass
class HighlightTransformation:
    """Give every exact friendly-name match a red border, including duplicates."""

    names: Iterable[str]

    def __post_init__(self) -> None:
        if isinstance(self.names, str):
            raise ValueError("HighlightTransformation expects a collection of names, not one string")
        self.names = frozenset(self.names)
        if not self.names:
            raise ValueError("HighlightTransformation requires at least one name")

    def apply(self, graph: Graph) -> None:
        for node in graph.nodes.values():
            if node.name in self.names:
                node.viz.border_color = "red"
