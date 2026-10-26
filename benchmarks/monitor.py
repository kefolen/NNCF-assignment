"""Atomic progress snapshots and sampled memory for a worker and native dot."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter, sleep

import psutil


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows can briefly deny replacement while a progress reader has it open.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(0.01)


def memory_sample(process: psutil.Process) -> dict[str, int]:
    """Sum RSS at one instant. Shared pages may be counted in both processes."""
    worker_rss = child_rss = worker_os_peak = child_os_peak = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []
    for current in processes:
        try:
            info = current.memory_info()
        except psutil.NoSuchProcess:
            continue
        if current.pid == process.pid:
            worker_rss = info.rss
            worker_os_peak = getattr(info, "peak_wset", 0)
        else:
            child_rss += info.rss
            child_os_peak += getattr(info, "peak_wset", 0)
    return {
        "worker_rss_bytes": worker_rss, "children_rss_bytes": child_rss,
        "tree_rss_bytes": worker_rss + child_rss,
        "worker_os_peak_rss_bytes": worker_os_peak,
        "children_os_peak_rss_bytes": child_os_peak,
    }


def eta_seconds(tasks: list[dict], runs: list[dict], current_seconds: float, timeout: float) -> float:
    """Use observed case medians; unseen cases use the overall median or budget."""
    durations = [run["worker_process_seconds"] for run in runs]
    fallback = statistics.median(durations) if durations else timeout + 10
    remaining = 0.0
    for index, task in enumerate(tasks[len(runs):]):
        same_case = [run["worker_process_seconds"] for run in runs
                     if (run["model"], run["variant"]) == (task["model"], task["variant"])]
        estimate = statistics.median(same_case) if same_case else fallback
        remaining += max(0.0, estimate - (current_seconds if index == 0 else 0))
    return remaining


def monitor_worker(command: list[str], *, root: Path, stage_path: Path, update,
                   sample_interval: float, progress_interval: float) -> tuple[int, dict, float]:
    started = perf_counter()
    peaks = {}
    stage_peaks = {}
    samples = 0
    last_update = -progress_interval
    stage = "startup"
    with subprocess.Popen(command, cwd=root) as child:
        process = psutil.Process(child.pid)
        try:
            while True:
                try:
                    stage = json.loads(stage_path.read_text(encoding="utf-8"))["stage"]
                except (FileNotFoundError, PermissionError):
                    # Missing at startup, or briefly locked by Windows replacement;
                    # keep the previous stage until the next memory sample.
                    pass
                sample = memory_sample(process)
                samples += 1
                for name, value in sample.items():
                    peaks[name] = max(peaks.get(name, 0), value)
                stage_peaks[stage] = max(stage_peaks.get(stage, 0), sample["tree_rss_bytes"])
                elapsed = perf_counter() - started
                if elapsed - last_update >= progress_interval:
                    update(stage, elapsed, sample, peaks)
                    last_update = elapsed
                if child.poll() is not None:
                    break
                sleep(sample_interval)
        finally:
            if child.poll() is None:
                # Also release Graphviz if this coordinator is interrupted.
                for descendant in process.children(recursive=True):
                    descendant.kill()
                child.kill()
                child.wait()
    return child.returncode, {
        "peak_sampled": peaks, "stage_peak_tree_rss_bytes": stage_peaks,
        "samples": samples, "sample_interval_seconds": sample_interval,
    }, perf_counter() - started


def progress_snapshot(started: float, *, status: str, phase: str, current: dict | None,
                      completed: int, failed: int, planned: int, eta: float | None,
                      **extra) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "updated_at_utc": now.isoformat(), "coordinator_pid": psutil.Process().pid,
        "status": status, "phase": phase, "elapsed_seconds": perf_counter() - started,
        "completed_attempts": completed, "failed_attempts": failed,
        "successful_attempts": completed - failed,
        "planned_attempts": planned, "finished_fraction": completed / planned,
        "current": current, "eta_seconds": eta,
        "estimated_finish_utc": (now + timedelta(seconds=eta)).isoformat() if eta is not None else None,
        **extra,
    }
