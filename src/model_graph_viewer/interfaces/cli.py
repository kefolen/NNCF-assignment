"""Small explicit transformation registry and argument parsing on top of the Python API."""

import argparse
import math
import sys
from pathlib import Path
from typing import Callable

from .api import visualize
from ..loader import ModelLoadError
from ..transformations import (
    CollapseSubgraphsTransformation, GraphTransformation,
    HideConstantsTransformation, HighlightTransformation, SharedTensorReferencesTransformation,
)
from ..render import RenderError


TRANSFORMATION_REGISTRY: dict[str, Callable[[argparse.Namespace], GraphTransformation]] = {
    "highlight": lambda args: HighlightTransformation(args.highlight),
    "hide-constants": lambda args: HideConstantsTransformation(),
    "shared-tensor-references": lambda args: SharedTensorReferencesTransformation(
        args.reference_min_consumers if args.reference_min_consumers is not None else 8
    ),
    "collapse-subgraphs": lambda args: CollapseSubgraphsTransformation(
        args.subgraph_depth if args.subgraph_depth is not None else 0
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an OpenVINO IR graph as a port-aware SVG.")
    parser.add_argument("--model", type=Path, required=True, help="OpenVINO IR .xml path")
    parser.add_argument(
        "--output", type=Path,
        help="SVG path (default: model-named SVG in the current directory); parent directories are created",
    )
    parser.add_argument(
        "--transformations", default="", metavar="LIST",
        help="Ordered comma-separated names: " + ", ".join(TRANSFORMATION_REGISTRY),
    )
    parser.add_argument("--highlight", action="append", default=[], metavar="NAME", help="Exact friendly name; repeatable")
    parser.add_argument(
        "--reference-min-consumers", type=int, metavar="N",
        help="Minimum distinct consumers per source output; requires shared-tensor-references (default: 8, minimum: 2)",
    )
    parser.add_argument("--subgraph-depth", type=int, metavar="X", help="Body levels to keep expanded with collapse-subgraphs (default: 0)")
    parser.add_argument("--render-timeout", type=float, metavar="SECONDS", help="Optional positive native dot time limit")
    parser.add_argument("--show-dtypes", action="store_true", help="Display each node's output dtype at its output port")
    parser.add_argument("--mclimit", type=float, metavar="SCALE", help="Crossing minimization effort scale (Graphviz mclimit)")
    parser.add_argument("--nslimit", type=float, metavar="N", help="Coordinate optimization iterations per node (Graphviz nslimit)")
    parser.add_argument("--nslimit1", type=float, metavar="N", help="Rank optimization iterations per node (Graphviz nslimit1)")
    parser.add_argument("--ordering", choices=("in", "out"), help="Constrain incoming/outgoing edges to their emitted order")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.transformations.split(",")] if args.transformations else []
    unknown = [name for name in names if name not in TRANSFORMATION_REGISTRY]
    if unknown:
        parser.error(f"unknown transformation: {', '.join(repr(name) for name in unknown)}")
    if "highlight" in names and not args.highlight:
        parser.error("--transformations highlight requires at least one --highlight NAME")
    if args.highlight and "highlight" not in names:
        parser.error("--highlight requires the highlight transformation in --transformations")
    if args.reference_min_consumers is not None:
        if "shared-tensor-references" not in names:
            parser.error("--reference-min-consumers requires the shared-tensor-references transformation")
        if args.reference_min_consumers < 2:
            parser.error("--reference-min-consumers must be at least 2")
    if args.subgraph_depth is not None:
        if "collapse-subgraphs" not in names:
            parser.error("--subgraph-depth requires the collapse-subgraphs transformation")
        if args.subgraph_depth < 0:
            parser.error("--subgraph-depth must be nonnegative")
    if args.render_timeout is not None and (not math.isfinite(args.render_timeout) or args.render_timeout <= 0):
        parser.error("--render-timeout must be a finite positive number")
    for name in ("mclimit", "nslimit", "nslimit1"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name} must be finite and nonnegative")

    transformations = [TRANSFORMATION_REGISTRY[name](args) for name in names]
    try:
        output = visualize(
            args.model, args.output, transformations,
            render_timeout=args.render_timeout, show_dtypes=args.show_dtypes,
            mclimit=args.mclimit, nslimit=args.nslimit, nslimit1=args.nslimit1, ordering=args.ordering,
        )
    except (ModelLoadError, RenderError, OSError) as error:
        print(f"model-graph-viewer: {error}", file=sys.stderr)
        return 1
    print(output, file=sys.stderr)
    return 0
