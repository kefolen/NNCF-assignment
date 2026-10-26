"""Export the independent visualization graph to one SVG using native Graphviz."""

import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Literal

from ..graph import Graph
from . import settings
from .dot import to_dot


class RenderError(RuntimeError):
    """Graphviz could not produce the requested SVG."""


def render_svg(
    graph: Graph, path: str | Path, *, timeout: float | None = None, show_dtypes: bool = False,
    mclimit: float | None = None, nslimit: float | None = None,
    nslimit1: float | None = None, ordering: Literal["in", "out"] | None = None,
) -> Path:
    """Render once, then replace the destination. Timeout limits native layout only.

    show_dtypes adds visible output dtypes. Layout options are per-call Graphviz
    overrides, as described in to_dot.
    """
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Graphviz timeout must be finite and greater than zero")
    output = Path(path)
    source = to_dot(
        graph, show_dtypes=show_dtypes,
        mclimit=mclimit, nslimit=nslimit, nslimit1=nslimit1, ordering=ordering,
    ).source
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as folder:
            svg = Path(folder) / "graph.svg"
            try:
                subprocess.run(
                    [settings.ENGINE, "-Tsvg", "-o", str(svg)],
                    input=source, encoding="utf-8", capture_output=True, timeout=timeout, check=True,
                )
            except FileNotFoundError as error:
                raise RenderError(
                    f"Native Graphviz '{settings.ENGINE}' was not found. "
                    "Install Graphviz and add its bin directory to PATH."
                ) from error
            except subprocess.TimeoutExpired as error:
                raise RenderError(
                    f"Graphviz timed out after {timeout} seconds while rendering '{output}'."
                ) from error
            except subprocess.CalledProcessError as error:
                raise RenderError(
                    f"Graphviz failed for '{output}' (exit {error.returncode}): {error.stderr.strip()}"
                ) from error
            os.replace(svg, output)
    except OSError as error:
        raise RenderError(f"Cannot write SVG '{output}': {error}") from error
    return output
