"""Benchmark statistics, environment details, summaries, and SVG checks."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import xml.etree.ElementTree as ET

from scipy.stats import t

from model_graph_viewer.graph import Graph
from model_graph_viewer.graph import walk_graphs
from model_graph_viewer.render import to_dot


def count_graph(graph: Graph) -> dict[str, int]:
    """Count retained scopes and annotations; exclude body/control mappings."""
    counts = {"nodes": 0, "edges": 0, "hidden_inputs": 0, "references": 0}
    for scope in walk_graphs(graph):
        counts["nodes"] += len(scope.nodes)
        counts["edges"] += len(scope.edges)
        for node in scope.nodes.values():
            for annotation in node.viz.input_annotations:
                key = "references" if annotation.kind == "reference" else "hidden_inputs"
                counts[key] += 1
    return counts


def environment():
    hardware = {}
    if os.name == "nt":
        command = (
            "@{cpu=(Get-CimInstance Win32_Processor).Name; "
            "ram_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory} | ConvertTo-Json -Compress"
        )
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
        if result.returncode == 0:
            hardware = json.loads(result.stdout)
    version = subprocess.run(["dot", "-V"], capture_output=True, text=True)
    return {
        "os": platform.platform(), "python": platform.python_version(),
        "cpu": hardware.get("cpu", platform.processor()),
        "ram_bytes": hardware.get("ram_bytes"),
        "logical_cpus": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in ("openvino", "graphviz", "model-graph-viewer", "psutil", "scipy")},
        "native_graphviz": (version.stderr or version.stdout).strip(),
        "empty_graph_dot_attributes": dict(to_dot(Graph()).graph_attr),
    }


def mean_interval(values: list[float]) -> dict:
    """Two-sided Student-t interval for the arithmetic mean, not individual runs."""
    count = len(values)
    mean = statistics.mean(values) if values else None
    half_width = (float(t.ppf(0.975, count - 1)) * statistics.stdev(values) / count ** 0.5
                  if count >= 2 else None)
    return {
        "n": count, "mean_seconds": mean, "confidence_level": 0.95,
        "method": "Student t, two-sided, df=n-1",
        "lower_seconds": mean - half_width if half_width is not None else None,
        "upper_seconds": mean + half_width if half_width is not None else None,
    }


def summaries(runs):
    groups = {}
    for run in runs:
        group = groups.setdefault(f"{run['model']}/{run['variant']}", {"completed": 0, "failed": 0, "timings": {}})
        group["completed" if run["status"] == "complete" else "failed"] += 1
    for key, group in groups.items():
        completed = [run for run in runs if f"{run['model']}/{run['variant']}" == key and run["status"] == "complete"]
        for metric in sorted({name for run in completed for name in run["timings"]}):
            values = [run["timings"][metric] for run in completed if metric in run["timings"]]
            group["timings"][metric] = {
                "mean": statistics.mean(values), "median": statistics.median(values),
                "min": min(values), "max": max(values),
            }
        group["total_time_ci95"] = mean_interval([run["timings"]["total_seconds"] for run in completed])
        step_totals = {}
        for run in completed:
            for name, seconds in run.get("steps", {}).items():
                step_totals[name] = step_totals.get(name, 0.0) + seconds
        total = sum(step_totals.values())
        group["step_percent_of_total"] = {name: seconds / total * 100 for name, seconds in step_totals.items()} if total else {}
        peaks = [run["memory"]["peak_sampled"]["tree_rss_bytes"] for run in completed if "memory" in run]
        if peaks:
            group["peak_tree_rss_bytes"] = {
                "mean": statistics.mean(peaks), "median": statistics.median(peaks),
                "min": min(peaks), "max": max(peaks),
            }
    return groups


def svg_statistics(runs, root_dir: Path):
    """Validate representative SVGs, retaining dimensions useful for comparison."""
    results = {}
    for run in runs:
        name = run["output"]
        if run["status"] != "complete" or name in results:
            continue
        root = ET.parse(root_dir / name).getroot()
        if root.tag != "{http://www.w3.org/2000/svg}svg":
            raise ValueError(f"Expected an SVG document: {name}")
        counts = {
            "nodes": sum(element.get("class") == "node" for element in root.iter()),
            "edges": sum(element.get("class") == "edge" for element in root.iter()),
        }
        if counts != {key: run["after"][key] for key in counts}:
            raise ValueError(f"SVG topology count mismatch: {name}: {counts}")
        results[name] = {
            "view_box": root.get("viewBox"), "counts": counts,
            "bytes": (root_dir / name).stat().st_size,
            "canonical_reference_links": sum(
                element.get("{http://www.w3.org/1999/xlink}href", "").startswith("#")
                for element in root.iter()
            ),
        }
    return results
