"""Repeated real-model benchmarks with live progress, memory and 95% mean CIs.

Run ``python -m benchmarks.run --prepare`` from the repository root.
All measurements remain outside the installed visualization library.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

from model_graph_viewer.loader import OpenVINOLoader
from model_graph_viewer.transformations import apply_transformations
import model_graph_viewer.render as renderer

from .cases import DETAIL_LEVELS, MODELS, configured_transformations, synthetic_graph
from .inputs import prepare_model
from .metrics import count_graph, environment, summaries
from .monitor import eta_seconds, monitor_worker, progress_snapshot, write_json


ROOT = Path(__file__).resolve().parents[1]


def run_worker(args) -> None:
    stage_path = args.output_dir / "current-stage.json"
    result_path = args.output_dir / "runs" / f"{args.task_id}.json"
    record = {"model": args.model, "pid": os.getpid(), "status": "failed", "timings": {}}
    timings = record["timings"]
    started = stage_started = perf_counter()
    stage = "preparation" if args.prepare_worker else "loading"

    def enter(name: str) -> float:
        write_json(stage_path, {"stage": name, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        return perf_counter()

    try:
        stage_started = enter(stage)
        if args.prepare_worker:
            record["input"] = prepare_model(ROOT, args.model, download=args.prepare)
        else:
            output = args.output_dir / "svg" / args.model / args.variant / f"repeat-{args.repeat:02d}.svg"
            options = DETAIL_LEVELS[args.variant]
            transformations = configured_transformations(args.variant)
            record.update({
                "variant": args.variant, "repeat": args.repeat, "output": str(output),
                "render_timeout_seconds": args.timeout,
                "render_options": options, "transformations": [type(item).__name__ for item in transformations],
                "min_reference_consumers": 8 if args.variant == "improved" else None,
            })
            graph = synthetic_graph() if args.model == "synthetic5000" else OpenVINOLoader().load(ROOT / MODELS[args.model])
            timings["loading_seconds"] = perf_counter() - stage_started
            record["before"] = count_graph(graph)
            stage = "transformation"
            stage_started = enter(stage)
            apply_transformations(graph, transformations)
            timings["transformation_seconds"] = perf_counter() - stage_started
            record["after"] = count_graph(graph)
            stage = "rendering"
            stage_started = enter(stage)
            original_to_dot = renderer.to_dot
            original_run = subprocess.run

            def measured_dot(*values, **kwargs):
                begin = enter("dot_construction")
                try:
                    dot = original_to_dot(*values, **kwargs)
                    record["dot_graph_attributes"] = dict(dot.graph_attr)
                    return dot
                finally:
                    timings["dot_construction_seconds"] = perf_counter() - begin
                    enter("render_other")

            def measured_native(*values, **kwargs):
                begin = enter("native_graphviz")
                try:
                    return original_run(*values, **kwargs)
                finally:
                    timings["native_dot_seconds"] = perf_counter() - begin
                    enter("render_other")

            # Profile the real renderer without generating DOT or rendering twice.
            with patch.object(renderer, "to_dot", measured_dot), patch.object(renderer.subprocess, "run", measured_native):
                renderer.render_svg(graph, output, timeout=args.timeout, **options)
            timings["rendering_seconds"] = perf_counter() - stage_started
            record["svg_bytes"] = output.stat().st_size
        record["status"] = "complete"
    except Exception as error:
        # This process boundary persists partial evidence and lets the suite continue.
        timings[f"{stage}_elapsed_seconds"] = perf_counter() - stage_started
        record["failed_stage"] = stage
        record["error"] = f"{type(error).__name__}: {error}"
    timings["total_seconds"] = perf_counter() - started
    if not args.prepare_worker and record["status"] == "complete":
        record["steps"] = {
            "loading": timings["loading_seconds"], "transformation": timings["transformation_seconds"],
            "dot_construction": timings["dot_construction_seconds"], "native_graphviz": timings["native_dot_seconds"],
            "render_other": timings["rendering_seconds"] - timings["dot_construction_seconds"] - timings["native_dot_seconds"],
            "bookkeeping": timings["total_seconds"] - timings["loading_seconds"] - timings["transformation_seconds"] - timings["rendering_seconds"],
        }
    write_json(result_path, record)


def run_suite(args, previous_report: dict | None = None) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "results.json").exists() or (args.output_dir / "progress.json").exists():
        raise FileExistsError(f"Use a new benchmark output directory: {args.output_dir}")
    (args.output_dir / "runs").mkdir()
    started = perf_counter()
    tasks = []
    if previous_report is not None:
        tasks = [{key: run[key] for key in ("model", "variant", "repeat")}
                 for run in previous_report["runs"] if run["status"] != "complete"]
    else:
        rng = random.Random(args.seed)
        # Balanced rounds give every case a first result before later repetitions.
        for repeat in range(1, args.repeats + 1):
            round_tasks = [{"model": model, "variant": variant, "repeat": repeat}
                           for model in args.models for variant in args.variants]
            rng.shuffle(round_tasks)
            tasks.extend(round_tasks)
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "configuration": {"models": args.models, "variants": args.variants, "repeats": args.repeats,
                          "timeout_seconds": args.timeout, "seed": args.seed, "tasks": tasks,
                          "memory_sample_interval_seconds": args.sample_interval,
                          "progress_interval_seconds": args.progress_interval},
        "conditions": [
            "Fresh sequential worker per attempt; OS disk caches are not flushed; no inference or training.",
            ("Retry only failed case/repeat IDs in their original failure order; previous results are preserved."
             if previous_report is not None else
             "Balanced rounds with seeded random case order reduce ordering effects. No untimed render warmup."),
            "Preparation (downloads and input hashing), Python startup/imports and memory monitoring are outside pipeline total_seconds.",
            "worker_process_seconds includes startup/imports; total_seconds covers loading through SVG export and graph statistics.",
            "rendering_seconds contains dot_construction_seconds and native_dot_seconds: do not add those nested values together.",
            "steps are disjoint: render_other includes DOT serialization/file operations, bookkeeping includes statistics and stage notifications.",
            "Memory is sampled absolute RSS of the worker plus live descendants (including Graphviz); coordinator excluded. Shared pages may be double-counted and short peaks missed.",
            "Per-stage memory peaks cover stages observed by the sampler. Windows OS peak working sets are also retained separately; summed process lifetime peaks need not be simultaneous.",
            "95% Student-t intervals describe the arithmetic mean total time of successful runs only, assume independent approximately normal sample means, and do not describe individual-run bounds.",
            "Timeouts/failures are retained separately and excluded from successful-time CIs; a CI with failures is conditional on success. Fewer than two successes yields null interval bounds.",
            "ETA uses observed case medians, falling back to the overall observed median, initially timeout+10 seconds per attempt. It is provisional, not a completion guarantee; preparation ETA is unknown.",
            "Other desktop activity and CPU power state are uncontrolled; few repetitions give only a rough uncertainty estimate.",
        ],
        "inputs": {}, "preparation": [], "runs": [], "summaries": {},
    }
    if previous_report is not None:
        report["retry_source"] = {
            "results_file": str(args.retry_failed),
            "timeout_seconds": previous_report["configuration"]["timeout_seconds"],
        }
    progress_path = args.output_dir / "progress.json"
    report_path = args.output_dir / "results.json"
    phase = "initializing"
    current = None

    def update(stage="startup", elapsed=0.0, sample=None, peaks=None):
        finished = len(report["runs"])
        eta = eta_seconds(tasks, report["runs"], elapsed, args.timeout) if phase == "benchmarking" else None
        write_json(progress_path, progress_snapshot(
            started, status=report["status"], phase=phase,
            current={**current, "stage": stage, "elapsed_seconds": elapsed,
                     "memory_current": sample, "memory_peak_sampled": peaks} if current else None,
            completed=finished, failed=sum(run["status"] != "complete" for run in report["runs"]),
            planned=len(tasks), eta=0.0 if report["status"].startswith("complete") else eta,
            prepared_models=len(report["inputs"]), planned_models=len(args.models),
            results_file=str(report_path),
        ))

    def execute(task: dict, *, preparing=False) -> dict:
        task_id = f"prepare-{task['model']}" if preparing else f"{task['model']}-{task['variant']}-{task['repeat']:02d}"
        stage_path = args.output_dir / "current-stage.json"
        stage_path.unlink(missing_ok=True)
        command = [sys.executable, "-u", "-m", "benchmarks.run", "--worker", "--model", task["model"],
                   "--task-id", task_id, "--output-dir", str(args.output_dir), "--timeout", str(args.timeout)]
        if preparing:
            command.append("--prepare-worker")
            if args.prepare:
                command.append("--prepare")
        else:
            command.extend(["--variant", task["variant"], "--repeat", str(task["repeat"])])
        print(f"Starting {task_id}", flush=True)
        code, memory, elapsed = monitor_worker(
            command, root=ROOT, stage_path=stage_path, update=update,
            sample_interval=args.sample_interval, progress_interval=args.progress_interval,
        )
        path = args.output_dir / "runs" / f"{task_id}.json"
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            **task, "status": "failed", "timings": {}, "failed_stage": "worker_process",
            "error": f"Worker exited {code} without a result; see stderr.log",
        }
        record.update({"memory": memory, "worker_process_seconds": elapsed, "worker_exit_code": code})
        write_json(path, record)
        print(f"Finished {task_id}: {record['status']} ({elapsed:.1f}s)", flush=True)
        return record

    update()
    write_json(report_path, report)
    try:
        report["environment"] = environment()
        phase = "preparing_models"
        for model in args.models:
            if model == "synthetic5000":
                report["inputs"][model] = {"kind": "synthetic; not a real model"}
                continue
            current = {"model": model}
            record = execute(current, preparing=True)
            report["preparation"].append(record)
            if record["status"] != "complete":
                raise RuntimeError(f"Input preparation failed for {model}: {record['error']}")
            report["inputs"][model] = record["input"]
            write_json(report_path, report)
        phase = "benchmarking"
        for current in tasks:
            record = execute(current)
            report["runs"].append(record)
            report["summaries"] = summaries(report["runs"])
            write_json(report_path, report)
            update()
        report["status"] = "complete" if all(run["status"] == "complete" for run in report["runs"]) else "complete_with_failures"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        current = None
        phase = "finished"
        update()
    print(f"Saved {report_path}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[*MODELS, "synthetic5000"], default=["tinyllama", "bert-int8", "gpt2"])
    parser.add_argument("--variants", nargs="+", choices=DETAIL_LEVELS, default=list(DETAIL_LEVELS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600, help="Native Graphviz timeout per attempt, seconds (default: 3600)")
    parser.add_argument("--retry-failed", type=Path, help="Retry only failed attempts from a previous results.json, keeping their repeat IDs")
    parser.add_argument("--prepare", action="store_true", help="Download missing pinned BERT/GPT-2 IRs before timing")
    parser.add_argument("--output-dir", type=Path, help="New directory for results, progress, per-attempt records and SVGs")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--sample-interval", type=float, default=0.2, help="Process-tree RSS sampling interval, seconds")
    parser.add_argument("--progress-interval", type=float, default=5, help="Progress JSON refresh interval, seconds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--task-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 for confidence intervals")
    for name in ("timeout", "sample_interval", "progress_interval"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if len(set(args.models)) != len(args.models) or len(set(args.variants)) != len(args.variants):
        parser.error("--models and --variants must not contain duplicates")
    previous_report = None
    if args.retry_failed is not None:
        args.retry_failed = args.retry_failed.resolve()
        try:
            previous_report = json.loads(args.retry_failed.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"Cannot read retry source {args.retry_failed}: {error}")
        if previous_report["status"] == "running":
            parser.error("The retry source is still running; wait for it to finish")
        failed = [run for run in previous_report["runs"] if run["status"] != "complete"]
        if not failed:
            parser.error("The retry source contains no failed attempts")
        args.models = list(dict.fromkeys(run["model"] for run in failed))
        args.variants = list(dict.fromkeys(run["variant"] for run in failed))
        args.repeats = previous_report["configuration"]["repeats"]
    args.output_dir = (args.output_dir or ROOT / "benchmarks/output" / datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")).resolve()
    if args.worker:
        run_worker(args)
    else:
        run_suite(args, previous_report)


if __name__ == "__main__":
    main()
