# Model preparation and benchmarks

Benchmark results, environment and measurement details are in
[Benchmarking_Report.md](../Benchmarking_Report.md). Models stay in ignored
`models/`; [models.json](models.json) records their sources and hashes.
Representative SVGs are included in [examples](../examples/); the
[benchmark report](../Benchmarking_Report.md#representative-svgs) links each view.

## Prepare models

From the repository root, after installing the viewer:

```powershell
.venv/Scripts/python.exe scripts/prepare_models.py branching
.venv/Scripts/python.exe scripts/prepare_models.py mobilenet
```

`branching` generates a small example locally. `mobilenet` downloads a
checksum-verified ONNX checkpoint and converts it with OpenVINO.

### TinyLlama

Use a separate Python 3.12 environment for the optional export dependencies:

```powershell
python -m venv .venv-export
.venv-export/Scripts/python.exe -m pip install openvino==2026.4.1 openvino-tokenizers==2026.4.1.0 optimum-intel==2.2.0 optimum==2.3.0 torch==2.14.1 transformers==5.5.4 huggingface-hub==1.21.0
.venv-export/Scripts/python.exe scripts/prepare_models.py tinyllama
```

The script pins the source revision and exports stateful FP16 IR. To record
an existing export without rebuilding it:

```powershell
.venv/Scripts/python.exe scripts/prepare_models.py tinyllama --record-only
```

Preparation updates `models.json`. The measured TinyLlama export predates the
pinned script; its inferred source revision and file hashes remain in that
manifest. A new export can differ from the measured input.

## Run benchmarks

```powershell
.venv/Scripts/python.exe -m pip install -e ".[benchmark]" -c requirements.txt
.venv/Scripts/python.exe -m benchmarks.run --prepare --repeats 5 --timeout 3600
```

This expects a local TinyLlama export and downloads missing pinned BERT/GPT-2
IRs. `--models` and `--variants` select cases; `--help` lists all options.
The timeout and interval options accept seconds. Raw JSON timing fields also
use seconds; the report displays milliseconds.

Each run creates `benchmarks/output/run-<UTC timestamp>/` with results, SVGs
and progress updates every 5,000 ms. Machine-readable measurements and progress
files are local artifacts, excluded from Git along with historical
`benchmarks/results/` files. The submitted benchmark results are summarized in
the Markdown report.

## Reproduce SVG examples

With the model inputs prepared, run these commands from the repository root to
regenerate the pretrained-model SVGs without repeating the benchmark study:

```powershell
$models = @{
    tinyllama = "models/tinyllama-fp16/openvino_model.xml"
    "bert-int8" = "models/bert-int8/openvino_model.xml"
    gpt2 = "models/gpt2/openvino_model.xml"
}
foreach ($name in $models.Keys) {
    .venv/Scripts/python.exe -m model_graph_viewer --model $models[$name] --output "examples/$name/raw.svg" --show-dtypes --render-timeout 3600
    .venv/Scripts/python.exe -m model_graph_viewer --model $models[$name] --output "examples/$name/hide-constants.svg" --transformations hide-constants --render-timeout 3600
    .venv/Scripts/python.exe -m model_graph_viewer --model $models[$name] --output "examples/$name/shared-references.svg" --transformations hide-constants,shared-tensor-references --reference-min-consumers 8 --render-timeout 3600
}
```

The [getting-started guide](../README.md#getting-started) reproduces the
branching fixture's original and highlighted, Constant-hidden views.
Large raw and Constant-hidden layouts can take many minutes. A new TinyLlama
export can differ from the measured input, and SVG layouts can vary with
dependency versions.

## Optional control-flow example

The pinned [Silero VAD export](https://huggingface.co/yangsu0423/silero_vad_op18_ifless_openvino/tree/d9d0650cfbc07793c2d0751f0676094c52a6baf6)
contains `If` bodies (218 operations). Download its XML and BIN for manual
inspection; it is outside the benchmark study. The source record has no
license or export command. Recorded SHA256 checksums:

```text
XML 4443c59ed31a64b1fc576c2a07c6f3d7b7d0832e8a35c9bea8a2ea05c9643f4f
BIN 2fe857791279ca7196743717abf92f5c12dfcebebcf61e4ae282324703540a08
```
