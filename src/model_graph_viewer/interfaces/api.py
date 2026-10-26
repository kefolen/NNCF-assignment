"""Public orchestration; CLI argument parsing lives elsewhere."""

from pathlib import Path
from typing import Iterable, Literal

from ..loader import ModelLoader, OpenVINOLoader
from ..transformations import GraphTransformation, apply_transformations
from ..render import render_svg


def visualize(
    model_path: str | Path,
    output_path: str | Path | None = None,
    transformations: Iterable[GraphTransformation] = (),
    loader: ModelLoader = OpenVINOLoader(),
    *,
    render_timeout: float | None = None,
    show_dtypes: bool = False,
    mclimit: float | None = None,
    nslimit: float | None = None,
    nslimit1: float | None = None,
    ordering: Literal["in", "out"] | None = None,
) -> Path:
    """Load, transform and export one expanded SVG; return its path.

    A different format needs only an object with ``load(path) -> Graph``.
    Without an output path, write a model-named SVG in the current directory.
    Set show_dtypes to display output dtypes on nodes. Graphviz layout limits
    and ordering are per-call overrides; see ``to_dot``.
    """
    if output_path is None:
        source = Path(model_path)
        model_name = source.stem
        if model_name == "openvino_model":
            model_name = source.parent.name or model_name
        output_path = Path(f"{model_name}.svg")

    graph = loader.load(model_path)
    apply_transformations(graph, transformations)
    return render_svg(
        graph, output_path, timeout=render_timeout, show_dtypes=show_dtypes,
        mclimit=mclimit, nslimit=nslimit, nslimit1=nslimit1, ordering=ordering,
    )
