from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .config import CampaignConfig, write_json_atomic


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _nearest(samples: list[dict[str, Any]], unix_s: float) -> dict[str, Any]:
    return min(samples, key=lambda sample: abs(float(sample["unix_s"]) - unix_s))


def _integrate_power(samples: list[dict[str, Any]], start: float, end: float) -> float:
    chosen = [sample for sample in samples if start <= float(sample["unix_s"]) <= end]
    if len(chosen) < 2:
        return 0.0
    total = 0.0
    for previous, current in zip(chosen, chosen[1:]):
        dt = float(current["unix_s"]) - float(previous["unix_s"])
        p0 = previous.get("power_instant_w")
        p1 = current.get("power_instant_w")
        if p0 is not None and p1 is not None and dt > 0:
            total += (float(p0) + float(p1)) * 0.5 * dt
    return total


METRIC_RE = re.compile(r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$")


def _queue_stats(service_rows: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    queue_values: list[float] = []
    for row in service_rows:
        if not start <= float(row["unix_s"]) <= end:
            continue
        for line in row.get("metrics", []):
            match = METRIC_RE.match(line)
            if match and match.group("name") == "sglang:num_queue_reqs":
                queue_values.append(float(match.group("value")))
    return {
        "samples": len(queue_values),
        "median": median(queue_values) if queue_values else None,
        "nonzero_fraction": (
            sum(value > 0 for value in queue_values) / len(queue_values)
            if queue_values
            else None
        ),
        "maximum": max(queue_values) if queue_values else None,
    }


def _health_counters(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    counters: dict[str, str] = {}

    def visit(element: ET.Element, prefix: str) -> None:
        current = f"{prefix}/{element.tag}"
        children = list(element)
        if not children:
            tag = element.tag.lower()
            if any(
                keyword in tag
                for keyword in ("error", "replay", "remap", "retired", "recovery")
            ):
                counters[current] = (element.text or "").strip()
            return
        for child in children:
            visit(child, current)

    for index, gpu in enumerate(root.findall("gpu")):
        uuid = gpu.findtext("uuid") or f"gpu-{index}"
        visit(gpu, uuid)
    return counters


def validate_run(run_dir: Path, config: CampaignConfig) -> dict[str, Any]:
    requests = _jsonl(run_dir / "requests.jsonl")
    gpu_rows = _jsonl(run_dir / "gpu.jsonl")
    service_rows = _jsonl(run_dir / "service.jsonl")
    health_before = _health_counters(run_dir / "gpu-health-before.xml")
    health_after = _health_counters(run_dir / "gpu-health-after.xml")
    replay = json.loads((run_dir / "replay_summary.json").read_text(encoding="utf-8"))
    start = float(replay["run_start_unix_s"])
    cohort_end_s = replay.get("cohort_end_s")
    end = start + (float(cohort_end_s) if cohort_end_s is not None else float(replay["elapsed_s"]))

    by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gpu_rows:
        by_gpu[str(row["gpu_uuid"])].append(row)
    energy_counter_j = 0.0
    integrated_power_j = 0.0
    maximum_gap = 0.0
    expected_samples = 0.0
    observed_samples = 0
    gpu_details: dict[str, Any] = {}
    for uuid, samples in by_gpu.items():
        samples.sort(key=lambda item: float(item["unix_s"]))
        boundary_start = _nearest(samples, start)
        boundary_end = _nearest(samples, end)
        counter_delta = float(boundary_end["energy_j"]) - float(boundary_start["energy_j"])
        integrated = _integrate_power(samples, start, end)
        gaps = [
            float(current["unix_s"]) - float(previous["unix_s"])
            for previous, current in zip(samples, samples[1:])
            if start <= float(previous["unix_s"]) <= end
        ]
        device_gap = max(gaps, default=0.0)
        in_window = [sample for sample in samples if start <= float(sample["unix_s"]) <= end]
        expected = max(1.0, (end - start) / float(config.measurement["gpu_sample_seconds"]))
        energy_counter_j += counter_delta
        integrated_power_j += integrated
        maximum_gap = max(maximum_gap, device_gap)
        expected_samples += expected
        observed_samples += len(in_window)
        gpu_details[uuid] = {
            "energy_counter_j": counter_delta,
            "integrated_power_j": integrated,
            "maximum_gap_s": device_gap,
            "samples": len(in_window),
            "expected_samples": expected,
        }

    crosscheck = (
        abs(integrated_power_j - energy_counter_j) / energy_counter_j
        if energy_counter_j > 0
        else math.inf
    )
    coverage = observed_samples / expected_samples if expected_samples else 0.0
    forbidden_clock_event = any(
        row.get(name)
        for row in gpu_rows
        for name in (
            "clock_event_hw_slowdown",
            "clock_event_sw_thermal",
            "clock_event_hw_thermal",
            "clock_event_power_brake",
        )
    )
    successful = [request for request in requests if request.get("success")]
    output_tokens = sum(int(request.get("output_tokens") or 0) for request in successful)
    elapsed = end - start
    latency = {
        name: {
            "p50": _percentile(
                [float(row[name]) for row in successful if row.get(name) is not None], 0.50
            ),
            "p95": _percentile(
                [float(row[name]) for row in successful if row.get(name) is not None], 0.95
            ),
            "p99": _percentile(
                [float(row[name]) for row in successful if row.get(name) is not None], 0.99
            ),
        }
        for name in ("ttft_ms", "tpot_ms", "e2e_ms")
    }

    checks = {
        "all_requests_successful": len(successful) == len(requests) and bool(requests),
        "watchdog_not_hit": not bool(replay.get("watchdog_hit")),
        "two_gpu_telemetry": len(by_gpu) == 2,
        "gpu_sample_coverage": coverage
        >= float(config.measurement["minimum_gpu_sample_coverage"]),
        "gpu_maximum_gap": maximum_gap
        <= float(config.measurement["maximum_gpu_gap_seconds"]),
        "energy_crosscheck": crosscheck
        <= float(config.measurement["energy_crosscheck_relative_error"]),
        "gpu_health_counters_unchanged": health_before == health_after,
        "no_forbidden_clock_events": not forbidden_clock_event,
    }
    summary = {
        "valid": all(checks.values()),
        "checks": checks,
        "request_count": len(requests),
        "success_count": len(successful),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in successful),
        "output_tokens": output_tokens,
        "elapsed_s": elapsed,
        "throughput_requests_s": len(successful) / elapsed if elapsed > 0 else None,
        "throughput_output_tokens_s": output_tokens / elapsed if elapsed > 0 else None,
        "energy_j": energy_counter_j,
        "joules_per_request": energy_counter_j / len(successful) if successful else None,
        "joules_per_1000_output_tokens": (
            energy_counter_j * 1000.0 / output_tokens if output_tokens else None
        ),
        "integrated_power_j": integrated_power_j,
        "energy_crosscheck_relative_error": crosscheck,
        "gpu_sample_coverage": coverage,
        "gpu_maximum_gap_s": maximum_gap,
        "gpu": gpu_details,
        "gpu_health_before": health_before,
        "gpu_health_after": health_after,
        "queue": _queue_stats(service_rows, start, end),
        "latency_ms": latency,
    }
    write_json_atomic(run_dir / "validity.json", summary)
    return summary
