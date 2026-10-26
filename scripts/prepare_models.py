"""Prepare pinned, local example IRs; model files are excluded from version control.

The branching and MobileNet commands need only the viewer's OpenVINO dependency.
The optional TinyLlama export needs optimum-intel and its export dependencies.
Preparation is separate from visualization. Tests do not download or export
pretrained models.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import openvino as ov
from openvino import opset13 as ops


ROOT = Path(__file__).resolve().parents[1]
ONNX_REVISION = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"
ONNX_PATH = "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
ONNX_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
TINYLLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def sha256(path: Path) -> str:
    """Hash files in bounded chunks, including the large TinyLlama weights."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if sha256(temporary) != expected_sha256:
            raise ValueError(f"Downloaded file checksum does not match: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_branching(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    features = ops.parameter([ov.Dimension(1, 8), 4], ov.Type.f32, name="features")
    axis = ops.constant(1, ov.Type.i64, name="split_axis")
    split = ops.split(features, axis, 2, name="split")
    weights = ops.constant([0.5, 1.5], ov.Type.f32, name="shared_bias")
    # Duplicate friendly names are intentional; ports and IDs identify nodes.
    left_add = ops.add(split.output(0), weights, name="branch")
    right_add = ops.add(split.output(1), weights, name="branch")
    left = ops.relu(left_add, name="left")
    right = ops.relu(right_add, name="right")
    # Two connections between one source/destination pair exercise multigraphs.
    repeated = ops.add(left, left, name="repeated_input")
    joined = ops.concat([repeated, right], 1, name="joined")
    result = ops.result(joined, name="output")
    model = ov.Model([result], [features], "branching_fixture")
    output = destination / "openvino_model.xml"
    ov.save_model(model, output, compress_to_fp16=False)
    return output


def prepare_mobilenet(destination: Path, source_directory: Path) -> Path:
    source = source_directory / "mobilenetv2-12.onnx"
    url = f"https://media.githubusercontent.com/media/onnx/models/{ONNX_REVISION}/{ONNX_PATH}"
    download(url, source, ONNX_SHA256)
    destination.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(source)
    output = destination / "openvino_model.xml"
    ov.save_model(model, output, compress_to_fp16=False)
    return output


def prepare_tinyllama(destination: Path) -> Path:
    # Optional export environment only. A pinned revision avoids moving main.
    from optimum.exporters.openvino import main_export
    from optimum.intel.openvino import OVConfig

    main_export(
        model_name_or_path=TINYLLAMA_ID,
        revision=TINYLLAMA_REVISION,
        output=destination,
        task="text-generation-with-past",
        framework="pt",
        library_name="transformers",
        ov_config=OVConfig(dtype="fp16"),
        stateful=True,
        convert_tokenizer=True,
        trust_remote_code=False,
    )
    return destination / "openvino_model.xml"


def file_record(path: Path) -> dict:
    try:
        relative_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        relative_path = str(path)
    return {"path": relative_path, "bytes": path.stat().st_size, "sha256": sha256(path)}


def inspect_ir(path: Path) -> dict:
    """Read XML structure/metadata only; never decode tensor payloads."""
    root = ET.parse(path).getroot()
    layers = root.findall("./layers/layer")
    types = Counter(layer.get("type") for layer in layers)
    runtime = root.find("./rt_info/Runtime_version")
    if runtime is None:
        runtime = root.find("./rt_info/info[@name='OpenVINO Runtime']")
    packages = root.find("./rt_info/optimum")
    return {
        "nodes": len(layers),
        "edges": len(root.findall("./edges/edge")),
        "operation_types": dict(sorted(types.items())),
        "runtime_version": runtime.get("value") if runtime is not None else None,
        "export_packages": {item.tag: item.get("value") for item in packages}
        if packages is not None
        else {},
    }


def record_model(name: str, models: Path, manifest: Path, *, prepared: bool) -> None:
    directories = {
        "branching": "branching",
        "mobilenet": "mobilenetv2",
        "tinyllama": "tinyllama-fp16",
    }
    directory = models / directories[name]
    xml = directory / "openvino_model.xml"
    sources = {
        "branching": {
            "origin": "Locally generated with OpenVINO operations; no pretrained weights.",
            "revision": "scripts/prepare_models.py",
            "license": "Repository example code; no third-party checkpoint.",
            "export": "OpenVINO save_model(compress_to_fp16=False)",
        },
        "mobilenet": {
            "origin": f"https://github.com/onnx/models/blob/{ONNX_REVISION}/{ONNX_PATH}",
            "revision": ONNX_REVISION,
            "license": "Apache-2.0 (MobileNet model README)",
            "license_source": f"https://github.com/onnx/models/blob/{ONNX_REVISION}/validated/vision/classification/mobilenet/README.md#license",
            "source_sha256": ONNX_SHA256,
            "export": "OpenVINO convert_model(ONNX); save_model(compress_to_fp16=False)",
        },
        "tinyllama": {
            "origin": f"https://huggingface.co/{TINYLLAMA_ID}",
            "revision": TINYLLAMA_REVISION,
            "license": "Apache-2.0 (pinned Hugging Face model card)",
            "license_source": f"https://huggingface.co/{TINYLLAMA_ID}/blob/{TINYLLAMA_REVISION}/README.md",
            "export": "Existing OpenVINO IR; original export command not verified.",
            "provenance_note": "Source ID and revision describe the expected pinned input. Recording an existing IR hashes files and reads XML metadata; it does not establish its original source revision or export command.",
        },
    }
    if name == "tinyllama" and prepared:
        sources[name]["export"] = "Pinned Optimum Intel main_export: text-generation-with-past, OVConfig(dtype='fp16'), stateful=True, convert_tokenizer=True"
        sources[name]["provenance_note"] = "Exported by scripts/prepare_models.py with the explicit source revision above. Generated bytes can depend on exporter versions and operation names."
    data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    data["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    data["count_scope"] = "Top-level serialized IR XML layers and tensor edges, before visualization transformations."
    data.setdefault("models", {})[name] = {
        **sources[name],
        "prepare_command": f"python scripts/prepare_models.py {name}",
        "files": [file_record(xml), file_record(xml.with_suffix(".bin"))],
        "xml_graph": inspect_ir(xml),
    }
    versions = {}
    for package in ("openvino", "graphviz", "pytest", "netron"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    data["recording_environment"] = {"python": platform.python_version(), "packages": versions}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {name}: {xml}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("branching", "mobilenet", "tinyllama"))
    parser.add_argument("--record-only", action="store_true", help="Inspect and hash an existing export without exporting it again.")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks" / "models.json")
    arguments = parser.parse_args()
    models = arguments.models.resolve()
    if not arguments.record_only:
        if arguments.model == "branching":
            prepare_branching(models / "branching")
        elif arguments.model == "mobilenet":
            prepare_mobilenet(models / "mobilenetv2", models / "sources")
        else:
            prepare_tinyllama(models / "tinyllama-fp16")
    record_model(arguments.model, models, arguments.manifest, prepared=not arguments.record_only)


if __name__ == "__main__":
    main()
