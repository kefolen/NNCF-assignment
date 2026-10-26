"""The documented preparation command works without optional inspection tools."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys

from model_graph_viewer import OpenVINOLoader


def test_prepare_branching_without_optional_tools(tmp_path, monkeypatch):
    installed_version = importlib.metadata.version

    def version(package):
        if package in {"pytest", "netron"}:
            raise importlib.metadata.PackageNotFoundError(package)
        return installed_version(package)

    monkeypatch.setattr(importlib.metadata, "version", version)
    manifest = tmp_path / "models.json"
    monkeypatch.setattr(sys, "argv", [
        "prepare_models.py", "branching", "--models", str(tmp_path / "models"),
        "--manifest", str(manifest),
    ])
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_models.py"
    runpy.run_path(str(script), run_name="__main__")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    record = data["models"]["branching"]
    graph = OpenVINOLoader().load(tmp_path / "models/branching/openvino_model.xml")
    assert record["xml_graph"]["nodes"] == len(graph.nodes) == 11
    assert record["xml_graph"]["edges"] == len(graph.edges) == 13
    assert set(data["recording_environment"]["packages"]) == {"openvino", "graphviz"}
    for item in record["files"]:
        assert item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
