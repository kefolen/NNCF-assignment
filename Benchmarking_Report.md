# Benchmarking report

Measurements: three pretrained OpenVINO models, three
views and five runs per case (45 successful measurements). The main benefit
comes from shared references on TinyLlama and GPT-2; BERT changes little.
The tables below summarize the measurements; reproduction commands are included.

## Models and views

| Model | Export | Operations | Connections |
| --- | --- | ---: | ---: |
| TinyLlama 1.1B Chat | Stateful FP16 | 3,168 | 3,741 |
| BERT base SST-2 | Sparse INT8 | 2,425 | 2,840 |
| GPT-2 | FP32 | 1,587 | 1,877 |

Counts are before transformations. Pinned sources:
[TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6),
[BERT](https://huggingface.co/OpenVINO/bert-base-uncased-sst2-unstructured80-int8-ov/tree/6a92100dddab08748c89b7a6d1a3936c78e54d1b),
and [GPT-2](https://huggingface.co/IlyasMoutawwakil/gpt2-openvino/tree/4c9f3db43ce614e79ac53465517fbded78ec4390).
The TinyLlama revision is inferred from its local source cache; its existing
export was not rebuilt. See the
[model preparation](benchmarks/README.md#tinyllama) for export commands and provenance.

| View (`--variants`) | Transformations | Output dtypes shown |
| --- | --- | --- |
| Full detail (`raw`) | None | Yes |
| Hide Constants (`baseline`) | Hide Constants | No |
| Shared references (`improved`) | Hide Constants, then shared references at eight distinct consumers per output | No |

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 11, build 26200 |
| Hardware | Intel Core i7-1360P, 16 logical CPUs, 31.65 GiB RAM |
| Python / OpenVINO | 3.12.2 / 2026.4.1 |
| Python / native Graphviz | 0.21 / 16.1.0 |
| Measurement packages | psutil 7.2.2, SciPy 1.18.1 |

## Measurement method

- Each run uses a fresh Python worker, executed sequentially and timed with
  `time.perf_counter`. Downloads and input hashing happen before timing.
  Disk caches, desktop activity and power settings are uncontrolled; no render
  warmups are performed. No inference, training or model compilation is run.
- Total time includes loading and graph construction, transformations, DOT
  construction, native Graphviz, SVG export and graph statistics. Python
  startup/imports and result JSON writing are excluded. Native Graphviz time
  includes subprocess startup, layout and SVG export.
- The 95% Student-t intervals estimate the mean from five measurements. They
  assume independent, approximately normal timings.
- Peak RSS is sampled every 200 ms over the worker and its children,
  including Graphviz. The table reports the mean of five sampled peaks.
  Shared pages can be counted twice and short peaks can be missed.

## Final total times

95% Student-t intervals are reported as mean ± margin.

| Model     | View              |     Mean, ms | 95% CI margin, ms | Mean peak RSS, MiB |
| --------- | ----------------- | -----------: | ----------------: | -----------------: |
| TinyLlama | Full detail       | 1,291,950.86 |      ± 292,762.60 |             542.47 |
| TinyLlama | Hide Constants    | 1,322,390.90 |      ± 120,850.99 |             476.96 |
| TinyLlama | Shared references |     1,442.02 |          ± 626.06 |             204.10 |
| BERT      | Full detail       |     9,235.33 |        ± 1,452.35 |             221.12 |
| BERT      | Hide Constants    |     1,114.48 |          ± 434.13 |             195.61 |
| BERT      | Shared references |     1,225.36 |          ± 520.27 |             192.67 |
| GPT-2     | Full detail       |   174,410.53 |       ± 36,741.67 |             294.00 |
| GPT-2     | Hide Constants    |    27,932.05 |       ± 10,938.97 |             229.57 |
| GPT-2     | Shared references |     1,662.05 |        ± 1,019.57 |             188.91 |

## Time spent in each stage

All mean durations are in milliseconds.
Other work covers DOT serialization, remaining file operations, statistics
and stage notifications.

| Model     | View              | Load, ms | Transform, ms | Build DOT, ms | Native Graphviz, ms | Other, ms |
| --------- | ----------------- | -------: | ------------: | ------------: | ------------------: | --------: |
| TinyLlama | Full detail       |   186.95 |         0.005 |        142.84 |        1,291,606.96 |     14.11 |
| TinyLlama | Hide Constants    |   166.01 |         2.482 |         91.59 |        1,322,108.94 |     21.88 |
| TinyLlama | Shared references |   197.32 |         4.178 |        105.26 |            1,115.04 |     20.21 |
| BERT      | Full detail       |   121.47 |         0.004 |        100.48 |            8,998.47 |     14.91 |
| BERT      | Hide Constants    |   128.18 |         1.919 |         67.40 |              899.19 |     17.80 |
| BERT      | Shared references |   142.01 |         4.183 |         96.62 |              962.00 |     20.55 |
| GPT-2     | Full detail       |   121.87 |         0.005 |         93.13 |          174,179.72 |     15.81 |
| GPT-2     | Hide Constants    |   104.78 |         1.265 |         54.02 |           27,754.97 |     17.02 |
| GPT-2     | Shared references |   116.34 |         3.055 |         81.92 |            1,441.23 |     19.50 |

Graphviz accounts for over 99.9% of TinyLlama's full-detail and Constant-hidden
time. The transformations themselves take only milliseconds.

## Effect of shared references

References retain the original dependency and port metadata in linked input
annotations. They replace wires, keeping the source operations visible.

| Model | Operations retained | Wires before references | Wires after references | Wires replaced |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama | 1,911 | 2,429 | 2,165 | 264 |
| BERT | 1,292 | 1,509 | 1,497 | 12 |
| GPT-2 | 915 | 1,122 | 1,062 | 60 |

TinyLlama's maximum visible fan-out falls from 44 to 4, and connections spanning
more than half the drawing's height fall from 101 to zero. Its width drops from
13,003 to 2,793 points, but its height stays near 82,000 points. BERT's reference
view is wider than its Constant-hidden view, so fewer wires do not always mean
a smaller or faster diagram.

## Representative SVGs

| Model | Full detail | Hide Constants | Shared references |
| --- | --- | --- | --- |
| TinyLlama | [SVG](examples/tinyllama/raw.svg) | [SVG](examples/tinyllama/hide-constants.svg) | [SVG](examples/tinyllama/shared-references.svg) |
| BERT | [SVG](examples/bert-int8/raw.svg) | [SVG](examples/bert-int8/hide-constants.svg) | [SVG](examples/bert-int8/shared-references.svg) |
| GPT-2 | [SVG](examples/gpt2/raw.svg) | [SVG](examples/gpt2/hide-constants.svg) | [SVG](examples/gpt2/shared-references.svg) |

Open the SVGs in a browser to use their tooltips and source-reference links.
The branching fixture also has an [original view](examples/branching/raw.svg)
and a [highlighted, Constant-hidden view](examples/branching/highlight-hide.svg).
Both operations named `branch` receive a red border, and their shared Constant
is identified by the same source ID at both inputs. See the
[reproduction commands](benchmarks/README.md#reproduce-svg-examples) to regenerate
the supplied views.

## Reproduce the benchmark

From the repository root with Python 3.12 and native Graphviz on `PATH`:

```powershell
python -m venv .venv
dot -V
.venv/Scripts/python.exe -m pip install -e ".[benchmark]" -c requirements.txt
.venv/Scripts/python.exe -m benchmarks.run --prepare --repeats 5 --timeout 3600
```

TinyLlama must already exist at `models/tinyllama-fp16/openvino_model.xml`;
[model preparation](benchmarks/README.md#tinyllama) covers its export. `--prepare`
downloads missing pinned BERT and GPT-2 IR files. The `--timeout` option accepts seconds and applies to each
native Graphviz call.

A new `benchmarks/output/run-<UTC timestamp>/` directory stores results and
SVGs; `progress.json` updates every 5,000 ms. Machine-readable results remain
local and are excluded from Git. Additional runner options are described in
[benchmarks/README.md](benchmarks/README.md).

## Interpretation limits

- Results cover three flat exports on one machine, with at most 3,168
  operations. The separate `synthetic5000` case is outside this study.
- Full detail also shows dtypes. Graphs with at least 1,000 visible operations
  use `mclimit=0.25`, `nslimit=0.25`, `nslimit1=1`; smaller views use native
  defaults. GPT-2 crosses this threshold after Constant hiding. Compare the
  two simplified views to isolate the shared-reference transformation.
- Timers measure Graphviz as a whole, without profiling its internal phases.
  The large improvements are observations for these exports and settings,
  not guarantees for other models.
