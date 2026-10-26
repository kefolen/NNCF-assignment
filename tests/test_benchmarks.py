"""Benchmark reporting stays outside the visualization workflow."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import openvino as ov
from openvino import opset13 as ops
import pytest

import benchmarks.run as runner
from benchmarks.metrics import count_graph, mean_interval, summaries
from benchmarks.monitor import eta_seconds, monitor_worker, write_json
from model_graph_viewer import (
    BodyPortMapping, CollapseSubgraphsTransformation, ControlFlowBody,
    Edge, Graph, HiddenInputAnnotation, Node,
)


def test_counts_include_retained_bodies_and_separate_annotations_from_edges():
    body = Graph(
        nodes={
            0: Node(0, "value", "Parameter"),
            1: Node(1, "sum", "Add", input_count=3),
            2: Node(2, "output", "Result", input_count=1),
        },
        edges=[Edge(0, 0, 1, 0), Edge(1, 0, 2, 0)],
    )
    body.nodes[1].viz.input_annotations = [
        HiddenInputAnnotation(Edge(7, 0, 1, 1)),
        HiddenInputAnnotation(Edge(0, 0, 1, 2), kind="reference"),
    ]
    owner = Node(1, "loop", "Loop", input_count=2, subgraphs={
        "body": ControlFlowBody(
            body, inputs=[BodyPortMapping(0, 0, "invariant")],
            outputs=[BodyPortMapping(0, 2, "iteration")],
        ),
    })
    owner.viz.input_annotations = [HiddenInputAnnotation(Edge(9, 0, 1, 1))]
    graph = Graph({0: Node(0, "input", "Parameter"), 1: owner}, [Edge(0, 0, 1, 0)])

    before = count_graph(graph)
    CollapseSubgraphsTransformation(0).apply(graph)

    assert before == {"nodes": 5, "edges": 3, "hidden_inputs": 2, "references": 1}
    assert count_graph(graph) == {"nodes": 2, "edges": 1, "hidden_inputs": 1, "references": 0}


@pytest.mark.parametrize("outcome", ["success", "missing", "timeout"])
def test_benchmark_command_runs_fresh_workers_and_records_results(tmp_path, outcome):
    # Run a standalone copy so the command cannot replace benchmark outputs/results.
    shutil.copytree(
        Path(runner.__file__).parent, tmp_path / "benchmarks",
        ignore=shutil.ignore_patterns("results", "output", "__pycache__"),
    )
    if outcome != "missing":
        data = ops.parameter([2], ov.Type.f32, name="input")
        summed = ops.add(data, ops.constant([1., 2.], ov.Type.f32), name="branch")
        model = ov.Model([summed], [data])
        model_path = tmp_path / "models/branching/openvino_model.xml"
        model_path.parent.mkdir(parents=True)
        ov.save_model(model, model_path, compress_to_fp16=False)

    output = tmp_path / "experiment"
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.run", "--models", "branching", "--repeats", "2",
         "--timeout", "0.000001" if outcome == "timeout" else "10", "--output-dir", str(output),
         "--sample-interval", "0.02", "--progress-interval", "0.05"],
        cwd=tmp_path, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == (0 if outcome == "success" else 1), completed.stdout + completed.stderr
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == report["status"]
    assert progress["elapsed_seconds"] > 0
    if outcome == "missing":
        assert report["status"] == "failed"
        assert "openvino_model.xml" in report["error"]
        assert report["runs"] == []
        return
    runs = report["runs"]
    assert len(runs) == 6
    assert len({run["pid"] for run in runs}) == 6
    assert all(run["pid"] != os.getpid() for run in runs)
    for run in runs:
        summary = report["summaries"][f"branching/{run['variant']}"]
        assert run["memory"]["peak_sampled"]["tree_rss_bytes"] > 0
        assert run["worker_process_seconds"] >= run["timings"]["total_seconds"]
        if outcome == "success":
            assert run["status"] == "complete"
            assert run["before"] == {"nodes": 4, "edges": 3, "hidden_inputs": 0, "references": 0}
            expected = run["before"] if run["variant"] == "raw" else {
                "nodes": 3, "edges": 2, "hidden_inputs": 1, "references": 0,
            }
            assert run["after"] == expected
            assert Path(run["output"]).read_text(encoding="utf-8").lstrip().startswith("<?xml")
            assert set(run["timings"]) == {
                "loading_seconds", "transformation_seconds", "rendering_seconds",
                "dot_construction_seconds", "native_dot_seconds", "total_seconds",
            }
            assert all(value >= 0 for value in run["timings"].values())
            assert sum(run["steps"].values()) == pytest.approx(run["timings"]["total_seconds"])
            assert all(value >= 0 for value in run["steps"].values())
            assert summary["completed"] == 2
            assert summary["total_time_ci95"]["n"] == 2
            assert summary["total_time_ci95"]["lower_seconds"] is not None
            assert sum(summary["step_percent_of_total"].values()) == pytest.approx(100)
        else:
            assert run["status"] == "failed"
            assert run["failed_stage"] == "rendering"
            assert "timed out" in run["error"]
            assert "native_dot_seconds" in run["timings"]
            assert summary["completed"] == 0
            assert summary["failed"] == 2
            assert summary["total_time_ci95"]["n"] == 0
            assert summary["total_time_ci95"]["lower_seconds"] is None
    assert progress["completed_attempts"] == 6
    assert progress["eta_seconds"] == 0
    assert progress["current"] is None


def test_student_t_interval_and_failed_run_exclusion():
    interval = mean_interval([1., 2., 3., 4., 5.])
    assert interval["mean_seconds"] == 3
    assert interval["lower_seconds"] == pytest.approx(1.0367568385)
    assert interval["upper_seconds"] == pytest.approx(4.9632431615)
    assert mean_interval([])["lower_seconds"] is None
    assert mean_interval([1.])["lower_seconds"] is None
    runs = [
        {"model": "test", "variant": "raw", "status": "complete", "timings": {"total_seconds": value}}
        for value in (1., 3.)
    ]
    runs.append({"model": "test", "variant": "raw", "status": "failed", "timings": {"total_seconds": 1000.}})
    group = summaries(runs)["test/raw"]
    assert group["total_time_ci95"]["mean_seconds"] == 2
    assert group["total_time_ci95"]["n"] == 2
    assert group["failed"] == 1


def test_monitor_updates_during_worker_and_samples_native_children(tmp_path):
    stage_path = tmp_path / "stage.json"
    write_json(stage_path, {"stage": "native_graphviz"})
    updates = []
    code, memory, duration = monitor_worker(
        [sys.executable, "-c", "import subprocess, sys; subprocess.run([sys.executable, '-c', 'import time; time.sleep(0.8)'])"],
        root=tmp_path, stage_path=stage_path, update=lambda *values: updates.append(values),
        sample_interval=0.02, progress_interval=0.1,
    )
    assert code == 0
    assert duration >= 0.8
    assert len(updates) >= 3
    assert updates[-1][1] > updates[0][1]
    peaks = memory["peak_sampled"]
    assert peaks["worker_rss_bytes"] > 0
    assert peaks["children_rss_bytes"] > 0
    assert peaks["tree_rss_bytes"] >= peaks["children_rss_bytes"]
    assert memory["stage_peak_tree_rss_bytes"]["native_graphviz"] > 0
    tasks = [{"model": "a", "variant": "raw"}, {"model": "a", "variant": "raw"}]
    assert eta_seconds(tasks, [], 2, 10) == 38
    assert eta_seconds(tasks, [{**tasks[0], "worker_process_seconds": 10}], 2, 10) == 8


def test_retry_failed_preserves_source_and_retries_only_failed_repeat_ids(tmp_path):
    shutil.copytree(
        Path(runner.__file__).parent, tmp_path / "benchmarks",
        ignore=shutil.ignore_patterns("results", "output", "__pycache__"),
    )
    data = ops.parameter([2], ov.Type.f32, name="input")
    model = ov.Model([ops.add(data, ops.constant([1., 2.], ov.Type.f32))], [data])
    model_path = tmp_path / "models/branching/openvino_model.xml"
    model_path.parent.mkdir(parents=True)
    ov.save_model(model, model_path, compress_to_fp16=False)
    source = tmp_path / "original.json"
    write_json(source, {
        "status": "complete_with_failures",
        "configuration": {"repeats": 5, "timeout_seconds": 1800},
        "runs": [
            {"model": "branching", "variant": "raw", "repeat": 1, "status": "complete"},
            {"model": "branching", "variant": "baseline", "repeat": 1, "status": "failed"},
            {"model": "branching", "variant": "raw", "repeat": 2, "status": "failed"},
            {"model": "branching", "variant": "baseline", "repeat": 2, "status": "failed"},
        ],
    })
    original_bytes = source.read_bytes()
    output = tmp_path / "retry"
    completed = subprocess.run(
        [sys.executable, "-m", "benchmarks.run", "--retry-failed", str(source),
         "--output-dir", str(output)],
        cwd=tmp_path, capture_output=True, text=True, timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert source.read_bytes() == original_bytes
    assert report["status"] == "complete"
    assert report["retry_source"] == {"results_file": str(source), "timeout_seconds": 1800}
    assert report["configuration"]["timeout_seconds"] == 3600
    assert [(run["variant"], run["repeat"]) for run in report["runs"]] == [
        ("baseline", 1), ("raw", 2), ("baseline", 2),
    ]
    assert all(run["status"] == "complete" and run["render_timeout_seconds"] == 3600 for run in report["runs"])
    assert all(Path(run["output"]).is_file() for run in report["runs"])
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["planned_attempts"] == progress["completed_attempts"] == 3
