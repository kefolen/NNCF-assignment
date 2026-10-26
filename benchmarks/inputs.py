"""Pinned pretrained IR downloads, kept outside timed render attempts."""

import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

from .cases import MODELS


SOURCES = {
    "gpt2": {
        "repository": "IlyasMoutawwakil/gpt2-openvino",
        "revision": "4c9f3db43ce614e79ac53465517fbded78ec4390",
        "license": "mit",
        "weights_sha256": "1ad777709dff9324f8c82a979a7febbd820560187407f86860dc0a7194931ade",
    },
    "bert-int8": {
        "repository": "OpenVINO/bert-base-uncased-sst2-unstructured80-int8-ov",
        "revision": "6a92100dddab08748c89b7a6d1a3936c78e54d1b",
        "license": "apache-2.0",
        "weights_sha256": "6bc46883c54fe9cf7d9ed6d172ee9622273d1c081c186aae0da2755d30bc0861",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_model(root: Path, name: str, *, download: bool) -> dict:
    """Download only XML/weights, verify pinned weights, and record actual inputs."""
    xml = root / MODELS[name]
    source = dict(SOURCES.get(name, {}))
    if name == "tinyllama":
        source = json.loads((root / "benchmarks/models.json").read_text(encoding="utf-8"))["models"][name]
    expected = {"openvino_model.bin": source.get("weights_sha256")}
    for item in source.get("files", []):
        expected[Path(item["path"]).name] = item["sha256"]
    files = []
    for path in (xml, xml.with_suffix(".bin")):
        if download and name in SOURCES and not path.exists():
            url = f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}/{path.name}?download=true"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            # A broken download stays a .part file and is never treated as an IR.
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
            if expected.get(path.name) and sha256(temporary) != expected[path.name]:
                raise ValueError(f"Downloaded checksum mismatch: {url}")
            temporary.replace(path)
        digest = sha256(path)
        if expected.get(path.name) and digest != expected[path.name]:
            raise ValueError(f"Checksum mismatch: {path}; remove the file and prepare it again")
        files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    document = ET.parse(xml).getroot()
    return {
        "source": source, "files": files,
        "xml_top_level_nodes": len(document.findall("./layers/layer")),
        "xml_top_level_edges": len(document.findall("./edges/edge")),
        "preparation": "Downloaded pre-exported IR; no inference, training or re-export" if name in SOURCES else "Existing local IR",
    }
