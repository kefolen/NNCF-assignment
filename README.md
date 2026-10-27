# Model graph viewer

Model graph viewer helps you explore the structure of neural networks and trace
connections between their operations. It creates SVG diagrams from OpenVINO IR models,
showing operations, tensor shapes, data types, and connections. Configurable transformations
let you highlight selected operations and simplify crowded diagrams while preserving dependency information.

Available as a Python library and a command-line tool, it lets you create different views without modifying the underlying model.

Browse the [SVG examples](Benchmarking_Report.md#representative-svgs) for TinyLlama, BERT, GPT-2 and a
small branching model, including before-and-after transformation views.

## Getting started

The library requires Python 3.10 or newer, OpenVINO, the Python Graphviz package
and the native Graphviz executable. The verified environment uses Python
3.12.2 and native Graphviz 16.1.0 on Windows 11. Install native Graphviz with
your system package manager or installer, then add its `bin` directory to
`PATH` so that `dot -V` works.

With Graphviz available, run these commands from the repository root in
PowerShell. They create an environment, install the project and prepare a
small example model before rendering its first SVG.

If you already have an active virtual environment, skip its creation and use
`python` instead of `.venv/Scripts/python.exe` in the commands below.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,inspect,benchmark]" -c requirements.txt
dot -V
.venv/Scripts/python.exe scripts/prepare_models.py branching
.venv/Scripts/python.exe -m model_graph_viewer --model models/branching/openvino_model.xml --output benchmarks/output/branching/raw.svg
```

Open the resulting SVG in a browser, then run the next command to highlight
the operations named `branch` and hide Constant nodes. The example contains
two operations with that name, so both receive a red border.

```powershell
.venv/Scripts/python.exe -m model_graph_viewer --model models/branching/openvino_model.xml --output benchmarks/output/branching/highlight-hide.svg --transformations highlight,hide-constants --highlight branch
```

The optional installation groups add tests, Netron inspection and benchmark
tools. A normal installation with `pip install -e .` only installs the viewer's
Python dependencies. On Linux or macOS, use `.venv/bin/python` in the examples
instead of `.venv/Scripts/python.exe`.


## CLI

The CLI accepts an IR `.xml` file and an ordered list of transformations.
OpenVINO finds the corresponding weight file when the IR uses one, while an
empty transformation list preserves the full view. The installed command
`model-graph-viewer` and `python -m model_graph_viewer` provide the same options.

### General options

| Option | Purpose |
| --- | --- |
| `--model PATH` | Required OpenVINO IR `.xml` input file. |
| `--output PATH` | Output SVG path; parent directories are created. Defaults to a model-named file in the current directory. |
| `--transformations LIST` | Comma-separated transformation names, applied in the listed order. Defaults to no transformations. |
| `--show-dtypes` | Show output data types beside all output ports. Constant dtype and shape are always visible; other output dtypes are hidden by default. |
| `--render-timeout SECONDS` | Positive time limit for the native Graphviz call. Higher values allow longer layouts; the default is no time limit. |
| `--mclimit SCALE` | Scale the effort spent reducing edge crossings. Higher values allow more searching for fewer crossings, potentially taking longer. |
| `--nslimit N` | Set the coordinate optimization iteration limit per node. Higher values allow more refinement of horizontal node positions, potentially taking longer. |
| `--nslimit1 N` | Set the rank optimization iteration limit per node. Ranks are the diagram's vertical layers; higher values allow more work on assigning nodes to layers, potentially taking longer. |
| `--ordering {in,out}` | Preserve the emitted order of incoming (`in`) or outgoing (`out`) edges at each node. If omitted, Graphviz chooses their order. |
| `--help` | Show the available CLI options and exit. |

Constant nodes display their dtype and shape by default, for example
`fp16[64, 3, 7, 7]` or `i64[]` for a scalar. When Constants are hidden, this
metadata appears at their consumer inputs instead. Tensor values are never
displayed.

Operation attributes, such as Convolution strides and padding, are shown below
the operation heading. Long attribute text wraps onto multiple lines.

Port labels use one-based numbering: `in1` and `out1` correspond to model port
index `0`. Labels are omitted when an operation has only one input or output.
The graph records retain the original zero-based port indices.

The three layout effort options (`--mclimit`, `--nslimit`, `--nslimit1`) accept
finite nonnegative numbers, including fractions. Higher values can improve a
layout but do not guarantee a clearer diagram; lower values generally favor
speed. For graphs with at least 1,000 visible operations their defaults are `0.25`, `0.25` and `1`, respectively.
Smaller graphs use Graphviz's own defaults.

If `--output` is omitted, the CLI writes a model-named SVG in the current
directory. For example, `models/classifier.xml` produces `classifier.svg`,
while `models/tinyllama-fp16/openvino_model.xml` produces `tinyllama-fp16.svg`.

### Transformation-specific options

| Option | Required transformation | Effect |
| --- | --- | --- |
| `--highlight NAME` | `highlight` | Give every operation with this exact, case-sensitive friendly name a red border. At least one name is required; repeat the option to select more names. |
| `--reference-min-consumers N` | `shared-tensor-references` | Minimum distinct consuming operations per source output before replacing its wires with references. Default: 8; minimum: 2. Lower values replace more wires; higher values keep more wires visible. |
| `--subgraph-depth N` | `collapse-subgraphs` | Number of control-flow body levels to keep expanded. Default: 0, showing only the root graph. Higher values show more nested detail and can produce larger diagrams; must be a nonnegative integer. |

Select the required transformation in `--transformations` before using its
options. Unknown transformation names, options used without their required
transformation, and invalid numeric values are reported before model loading.

The normal visualization API and CLI have no render time limit unless one is
provided.

## Python API

The `visualize` function provides the same workflow in Python and returns the
SVG path. Its transformation list is empty by default, so the caller chooses
which changes appear in the view.

```python
from model_graph_viewer import (
    HideConstantsTransformation,
    HighlightTransformation,
    visualize,
)

output = visualize(
    "models/branching/openvino_model.xml",
    "benchmarks/output/branching/api-example.svg",
    transformations=[
        HighlightTransformation(["branch"]),
        HideConstantsTransformation(),
    ],
    render_timeout=30,
)
print(output)
```

For closer inspection, the loading, transformation and rendering steps can
also be called separately. This lets a caller examine the graph before
choosing the transformations and exporting the result.

```python
from model_graph_viewer import (
    HideConstantsTransformation,
    OpenVINOLoader,
    SharedTensorReferencesTransformation,
    apply_transformations,
    render_svg,
)

graph = OpenVINOLoader().load("models/tinyllama-fp16/openvino_model.xml")
apply_transformations(
    graph,
    [
        HideConstantsTransformation(),
        SharedTensorReferencesTransformation(min_consumers=8),
    ],
)
output = render_svg(
    graph,
    "benchmarks/output/tinyllama/shared-references.svg",
    timeout=3600,
)
```

Transformations edit this independent graph in place, so keeping both an
editable original view and an editable transformed view requires two loads or
an explicit graph copy.

The `to_dot` function returns a Graphviz `Digraph` when DOT inspection is
needed. Like `render_svg` and `visualize`, it accepts `show_dtypes`, `mclimit`,
`nslimit`, `nslimit1` and `ordering`. These options affect one drawing and leave
the shared rendering defaults unchanged.

## Transformations

A transformation is an operation on the visualization graph. Here is the table of implemented ones.

| CLI name | Python class | Effect |
| --- | --- | --- |
| `highlight` | `HighlightTransformation` | Give every node with a selected friendly name a red border. |
| `hide-constants` | `HideConstantsTransformation` | Replace Constant nodes and their outgoing wires with input annotations. |
| `shared-tensor-references` | `SharedTensorReferencesTransformation` | Replace high-fan-out wires with linked references to visible source outputs. |
| `collapse-subgraphs` | `CollapseSubgraphsTransformation` | Keep control-flow bodies expanded only up to a chosen depth. |


### Shared references

In TinyLlama, a few shared outputs feed operations across many decoder layers.
Their long wires cross much of the diagram, obscure local connections and
make Graphviz layout expensive. Hiding Constants removes nodes but leaves
these non-Constant dependencies in place.

`shared-tensor-references` replaces these wires with a source reference at each
consumer input. The source stays visible; each reference links back to it and
retains the source ID, both port indexes, dtype and shape. Selection counts
distinct consumers per output (default: eight), independently of operation
names or model architecture.

In the measured TinyLlama export, this replaces 264 wires while keeping all
1,911 non-Constant operations. Adding references after Constant hiding reduces
mean pipeline time from 1,322,390.90 to 1,442.02 ms in the
[benchmark](Benchmarking_Report.md#effect-of-shared-references).
The diagram still needs scrolling, and tracing a replaced wire requires
following its source link.

Compare the [before](examples/tinyllama/hide-constants.svg)
and [after](examples/tinyllama/shared-references.svg) views, or
recreate them:

```powershell
.venv/Scripts/python.exe -m model_graph_viewer --model models/tinyllama-fp16/openvino_model.xml --output benchmarks/output/tinyllama/hide-constants.svg --transformations hide-constants --render-timeout 3600
.venv/Scripts/python.exe -m model_graph_viewer --model models/tinyllama-fp16/openvino_model.xml --output benchmarks/output/tinyllama/shared-references.svg --transformations hide-constants,shared-tensor-references --reference-min-consumers 8 --render-timeout 3600
```

### Control-flow bodies

Expanding every conditional branch and loop body can make the outer model
structure hard to see. `collapse-subgraphs` provides an overview by hiding
bodies beyond `--subgraph-depth`, while retaining the owning `If`, `Loop` or
`TensorIterator`, its external connections and the names of collapsed bodies.

The depth setting lets you keep useful nesting levels without showing every
internal operation. It applies to existing control-flow bodies, so flat models
are unchanged. Omitted bodies are removed from the visualization graph;
reload the model to restore their detail.

## Architecture

```text
OpenVINO IR -> loader -> independent Graph -> transformations -> Graphviz -> SVG
```

| Component | Responsibility |
| --- | --- |
| [graph](src/model_graph_viewer/graph) | Framework-independent nodes, edges, tensor metadata and control-flow records. |
| [loader](src/model_graph_viewer/loader) | Convert source models into graph records. |
| [transformations](src/model_graph_viewer/transformations) | Apply configured changes to the visualization graph. |
| [render](src/model_graph_viewer/render) | Build port-aware DOT and export SVG through Graphviz. |
| [interfaces](src/model_graph_viewer/interfaces) | Python API, CLI and explicit transformation registry. |
| [benchmarks](benchmarks) | Measure components separately, outside the installed library. |

Scope-local IDs distinguish operations with duplicate names. Edges keep both
port indexes and parallel connections; tensor metadata distinguishes scalar,
dynamic and missing shapes. Operation attributes and visualization settings
are separate.

Small dataclasses and in-place transformations avoid repeated graph copies.
Dictionaries, sets and batched edge filtering keep loading and the built-in
transformations linear in graph size. Only the loader handles OpenVINO
objects; tensor payloads are never extracted for visualization.

## Extensibility

To add a transformation, implement `apply(graph)` and pass an instance in the
`transformations` list. The pipeline applies it to each retained graph scope.

```python
from model_graph_viewer import Graph


class MarkConvolutionsTransformation:
    def apply(self, graph: Graph) -> None:
        for node in graph.nodes.values():
            if node.op_type == "Convolution":
                node.viz.border_color = "blue"
```

Expose it in the CLI through `TRANSFORMATION_REGISTRY` in
[interfaces/cli.py](src/model_graph_viewer/interfaces/cli.py).
Use `allocate_node_id(graph)` for new nodes and preserve ports, annotations
and body mappings when changing connections.

To support another model format, implement `load(path) -> Graph` and pass
`loader=MyLoader()` to `visualize`. Return framework-independent records with
unique IDs, original ports, tensor metadata and any control-flow interfaces;
normalize constant sources to `Constant`. Existing transformations and
rendering can then be reused.

## Testing

From the repository root, with the `dev` and `benchmark` dependencies installed:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Tests cover loading, ports, shapes, duplicate names, transformations,
control-flow mappings, API/CLI behavior, benchmarking and SVG export.
Native Graphviz is required; tests create small models locally and download
nothing.

## Performance

Measured on 25 October 2026: Windows 11, Intel Core i7-1360P, 31.65 GiB RAM;
Python 3.12.2, OpenVINO 2026.4.1, Python Graphviz 0.21 and native Graphviz 16.1.0.
Times are mean pipeline durations in milliseconds over five runs per model
and view, including loading, transformations and SVG export.

| Model                        | Operations | Connections | Full detail, ms | Shared references, ms |
| ---------------------------- | ---------: | ----------: | --------------: | --------------------: |
| TinyLlama 1.1B Chat, FP16    |      3,168 |       3,741 |    1,291,950.86 |              1,442.02 |
| BERT base SST-2, sparse INT8 |      2,425 |       2,840 |        9,235.33 |              1,225.36 |
| GPT-2, FP32                  |      1,587 |       1,877 |      174,410.53 |              1,662.05 |

Sizes are before transformations. Full detail shows output dtypes; the other
views hide them. Shared references also hide Constants and use a threshold of
eight consumers. Graphviz dominates the longer cases; the benefit depends on
graph structure and view settings.

The small examples are the branching fixture (11 operations / 13 connections)
and MobileNetV2 (263 / 272). Sources and preparation commands are in the
[model preparation](benchmarks/README.md#prepare-models). See the
[benchmark report](Benchmarking_Report.md) for stage timings, confidence
intervals, memory usage and measurement limits. To rerun the three-model study:

```powershell
.venv/Scripts/python.exe -m benchmarks.run --prepare --repeats 5 --timeout 3600
```

## Known limitations

- OpenVINO IR is the only built-in format. The graph shows tensor-port
  dependencies and static control-flow structure, without running the model
  or inferring extra runtime state dependencies.
- Large layouts can take minutes and produce tall SVGs. Shared references
  require following links; collapsing bodies removes detail until a reload.
- Results cover three flat models on one machine, with at most 3,168 operations.
  `--models synthetic5000` provides a separate 5,001-node scale check outside
  this study.
- Possible improvements include block summaries and interactive navigation.
