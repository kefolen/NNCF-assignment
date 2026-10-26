"""Format-independent contract for loading visualization graphs."""

from pathlib import Path
from typing import Protocol

from ..graph import Graph


class ModelLoadError(RuntimeError):
    """A model file could not be read or converted by a loader."""


class ModelLoader(Protocol):
    def load(self, path: str | Path) -> Graph:
        """Return an independent graph with unique IDs, original ports and metadata.

        Normalize constant sources to ``Constant`` and retain no framework objects
        or tensor payloads. Preserve nested bodies and their interface mappings.
        """
        ...
