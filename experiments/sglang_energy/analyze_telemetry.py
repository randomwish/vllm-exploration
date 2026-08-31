#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROMETHEUS_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def parse_key_values(value: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for line in value.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                output[safe_name(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return output


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for position in order[index:end]:
            ranks[position] = average_rank
        index = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rank(left), rank(right))


def workload_for(label: str) -> str:
    for workload in ("PX87", "PX50", "PX0", "BAL", "PF", "DEC"):
        if re.search(rf"(?:^|-){workload}(?:-|$)", label):
            return workload
    if label.startswith("s1-"):
        return "BAL"
    if label.startswith("s2-"):
        return "PF"
    if label.startswith("s3-"):
        return "PX0"
    return "unknown"


def run_window(run_root: Path) -> tuple[float, float]:
    replay = json.loads((run_root / "replay_summary.json").read_text(encoding="utf-8"))
    start = float(replay["run_start_unix_s"])
    cohort_end = replay.get("cohort_end_s")
    elapsed = float(cohort_end) if cohort_end is not None else float(replay["elapsed_s"])
    return start, start + elapsed


def in_window(rows: Iterable[dict[str, Any]], start: float, end: float):
    return [row for row in rows if start <= float(row.get("unix_s", -1)) <= end]


def aggregate_numeric_rows(
    target: dict[str, Any], prefix: str, rows: list[dict[str, Any]], skip: set[str]
) -> None:
    values: dict[str, list[float]] = defaultdict(list)
    booleans: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        for key, raw in row.items():
            if key in skip:
                continue
            if isinstance(raw, bool):
                booleans[key].append(raw)
                continue
            value = finite_number(raw)
            if value is not None:
                values[key].append(value)
    for key, items in values.items():
        target[f"{prefix}_mean_{safe_name(key)}"] = statistics.mean(items)
        target[f"{prefix}_max_{safe_name(key)}"] = max(items)
    for key, items in booleans.items():
        target[f"{prefix}_fraction_{safe_name(key)}"] = sum(items) / len(items)


def aggregate_service(
    target: dict[str, Any], rows: list[dict[str, Any]], elapsed: float
) -> None:
    prometheus: dict[str, list[float]] = defaultdict(list)
    memory_current: list[float] = []
    cpu_snapshots: list[dict[str, float]] = []
    memory_event_snapshots: list[dict[str, float]] = []
    network: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    target["service_sample_count"] = len(rows)
    target["service_metrics_error_count"] = sum(
        row.get("metrics_error") is not None for row in rows
    )
    timestamps = sorted(
        value
        for row in rows
        if (value := finite_number(row.get("unix_s"))) is not None
    )
    if len(timestamps) >= 2:
        target["service_maximum_gap_s"] = max(
            right - left for left, right in zip(timestamps, timestamps[1:])
        )
    for row in rows:
        for line in row.get("metrics", []):
            match = PROMETHEUS_RE.match(line)
            if match:
                prometheus[safe_name(match.group("name"))].append(
                    float(match.group("value"))
                )
        cgroup = row.get("cgroup") or {}
        current = finite_number(cgroup.get("memory_current"))
        if current is not None:
            memory_current.append(current)
        maximum = finite_number(cgroup.get("memory_max"))
        if maximum is not None:
            target["cgroup_memory_max_bytes"] = maximum
        cpu_max = str(cgroup.get("cpu_max", "")).split()
        if len(cpu_max) == 2:
            quota = finite_number(cpu_max[0])
            period = finite_number(cpu_max[1])
            if quota is not None and period not in (None, 0):
                target["cgroup_cpu_quota_usec"] = quota
                target["cgroup_cpu_period_usec"] = period
                target["cgroup_cpu_limit_cores"] = quota / period
        if cgroup.get("cpu_stat"):
            cpu_snapshots.append(parse_key_values(cgroup["cpu_stat"]))
        if cgroup.get("memory_events"):
            memory_event_snapshots.append(parse_key_values(cgroup["memory_events"]))
        for interface, counters in (row.get("network") or {}).items():
            for key, raw in counters.items():
                value = finite_number(raw)
                if value is not None:
                    network[safe_name(interface)][safe_name(key)].append(value)

    for name, values in prometheus.items():
        target[f"service_mean_{name}"] = statistics.mean(values)
        target[f"service_max_{name}"] = max(values)
    if memory_current:
        target["cgroup_mean_memory_current_bytes"] = statistics.mean(memory_current)
        target["cgroup_max_memory_current_bytes"] = max(memory_current)
    if len(cpu_snapshots) >= 2:
        for key in sorted(set(cpu_snapshots[0]) & set(cpu_snapshots[-1])):
            delta = cpu_snapshots[-1][key] - cpu_snapshots[0][key]
            target[f"cgroup_delta_cpu_{key}"] = delta
            target[f"cgroup_rate_cpu_{key}_per_s"] = delta / elapsed
    if len(memory_event_snapshots) >= 2:
        for key in sorted(
            set(memory_event_snapshots[0]) & set(memory_event_snapshots[-1])
        ):
            target[f"cgroup_delta_memory_event_{key}"] = (
                memory_event_snapshots[-1][key] - memory_event_snapshots[0][key]
            )
    for interface, counters in network.items():
        for key, values in counters.items():
            if len(values) >= 2:
                delta = values[-1] - values[0]
                target[f"network_delta_{interface}_{key}"] = delta
                target[f"network_rate_{interface}_{key}_per_s"] = delta / elapsed


def aggregate_run(run_root: Path) -> dict[str, Any]:
    validity = json.loads((run_root / "validity.json").read_text(encoding="utf-8"))
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    start, end = run_window(run_root)
    elapsed = end - start
    label = run_root.name.split("-", 1)[1]
    settings = config.get("server_settings") or {}
    output: dict[str, Any] = {
        "run_id": run_root.name,
        "label": label,
        "workload": workload_for(label),
        "valid": bool(validity["valid"]),
        "radix_cache": settings.get("radix_cache"),
        "max_running_requests": settings.get("max_running_requests"),
        "chunked_prefill_size": settings.get("chunked_prefill_size"),
        "elapsed_s": elapsed,
    }
    output["prefix_reuse_fraction"] = {
        "PX0": 0.0,
        "PX50": 0.5,
        "PX87": 0.875,
    }.get(output["workload"])
    for key in (
        "request_count",
        "success_count",
        "input_tokens",
        "output_tokens",
        "throughput_requests_s",
        "throughput_output_tokens_s",
        "energy_j",
        "joules_per_request",
        "joules_per_1000_output_tokens",
        "integrated_power_j",
        "energy_crosscheck_relative_error",
        "gpu_sample_coverage",
        "gpu_maximum_gap_s",
    ):
        output[key] = validity.get(key)
    output["total_average_power_w"] = validity["energy_j"] / elapsed
    for family, values in validity.get("latency_ms", {}).items():
        for percentile, value in values.items():
            output[f"{family}_{percentile}"] = value
    for key, value in validity.get("queue", {}).items():
        output[f"queue_{key}"] = value
    for key, value in validity.get("checks", {}).items():
        output[f"check_{key}"] = bool(value)

    gpu_rows = in_window(jsonl(run_root / "gpu.jsonl"), start, end)
    output["gpu_sample_row_count"] = len(gpu_rows)
    by_device: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in gpu_rows:
        index = int(row["gpu_index"])
        by_device[index].append(row)
    output["gpu_device_count"] = len(by_device)
    energy_deltas: list[float] = []
    mean_powers: list[float] = []
    mean_busy: list[float] = []
    device_maximum_gaps: list[float] = []
    for device_rows in by_device.values():
        device_rows.sort(key=lambda row: float(row["unix_s"]))
        energies = [
            value
            for row in device_rows
            if (value := finite_number(row.get("energy_j"))) is not None
        ]
        if len(energies) >= 2:
            energy_deltas.append(energies[-1] - energies[0])
        powers = [
            value
            for row in device_rows
            if (value := finite_number(row.get("power_average_w"))) is not None
        ]
        if powers:
            mean_powers.append(statistics.mean(powers))
        busy = [
            value
            for row in device_rows
            if (value := finite_number(row.get("gpu_busy_pct"))) is not None
        ]
        if busy:
            mean_busy.append(statistics.mean(busy))
        timestamps = [float(row["unix_s"]) for row in device_rows]
        if len(timestamps) >= 2:
            device_maximum_gaps.append(
                max(right - left for left, right in zip(timestamps, timestamps[1:]))
            )
    if energy_deltas:
        output["gpu_counter_energy_j"] = sum(energy_deltas)
    if mean_powers and statistics.mean(mean_powers):
        output["gpu_power_imbalance_fraction"] = (
            max(mean_powers) - min(mean_powers)
        ) / statistics.mean(mean_powers)
    if mean_busy and statistics.mean(mean_busy):
        output["gpu_busy_imbalance_fraction"] = (
            max(mean_busy) - min(mean_busy)
        ) / statistics.mean(mean_busy)
    if device_maximum_gaps:
        output["gpu_raw_maximum_gap_s"] = max(device_maximum_gaps)
    aggregate_numeric_rows(
        output,
        "gpu",
        gpu_rows,
        {"unix_s", "monotonic_s", "gpu_index", "energy_j"},
    )
    service_rows = in_window(jsonl(run_root / "service.jsonl"), start, end)
    aggregate_service(output, service_rows, elapsed)

    auxiliary_gpm = in_window(jsonl(run_root / "gpm.jsonl"), start, end)
    if auxiliary_gpm:
        aggregate_numeric_rows(
            output,
            "gpm_aux",
            auxiliary_gpm,
            {"unix_s", "monotonic_s", "interval_s", "gpu_index", "energy_j"},
        )
    hbm_rows = list(jsonl(run_root / "hbm.jsonl"))
    output["hbm_row_count"] = len(hbm_rows)
    output["hbm_successful_row_count"] = sum(bool(row.get("gpus")) for row in hbm_rows)
    output["hbm_error_row_count"] = sum(bool(row.get("error")) for row in hbm_rows)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "run_id",
        "label",
        "workload",
        "valid",
        "radix_cache",
        "prefix_reuse_fraction",
    ]
    preferred = [column for column in preferred if column in columns]
    columns = preferred + [column for column in columns if column not in preferred]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["valid"]]
    scopes = {
        "all-valid": valid,
        "balanced-valid": [row for row in valid if row["workload"] == "BAL"],
        "prefix-valid": [
            row for row in valid if row["workload"] in {"PX0", "PX50", "PX87"}
        ],
    }
    targets = (
        "joules_per_request",
        "throughput_requests_s",
        "ttft_ms_p95",
        "e2e_ms_p95",
    )
    feature_prefixes = ("gpu_", "service_", "cgroup_", "network_", "queue_")
    output: list[dict[str, Any]] = []
    for scope, scope_rows in scopes.items():
        features = sorted(
            {
                key
                for row in scope_rows
                for key in row
                if key.startswith(feature_prefixes)
            }
        )
        for target in targets:
            for feature in features:
                pairs: list[tuple[float, float]] = []
                for row in scope_rows:
                    left = finite_number(row.get(feature))
                    right = finite_number(row.get(target))
                    if left is not None and right is not None:
                        pairs.append((left, right))
                if len(pairs) < 5:
                    continue
                coefficient = spearman(
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                )
                if coefficient is None:
                    continue
                output.append(
                    {
                        "scope": scope,
                        "feature": feature,
                        "target": target,
                        "n": len(pairs),
                        "spearman_rho": coefficient,
                    }
                )
    return sorted(
        output,
        key=lambda row: (row["scope"], row["target"], -abs(row["spearman_rho"])),
    )


def availability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = sorted({key for row in rows for key in row})
    output = []
    for column in columns:
        available = sum(row.get(column) not in (None, "") for row in rows)
        output.append(
            {
                "field": column,
                "available_runs": available,
                "total_runs": len(rows),
                "availability_fraction": available / len(rows),
            }
        )
    return output


def selected_relationship_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["valid"]]
    scopes = {
        "all-valid": valid,
        "balanced-valid": [row for row in valid if row["workload"] == "BAL"],
        "prefix-valid": [
            row for row in valid if row["workload"] in {"PX0", "PX50", "PX87"}
        ],
    }
    requested = {
        "all-valid": [
            ("total_average_power_w", "throughput_requests_s"),
            ("gpu_mean_power_average_w", "throughput_requests_s"),
            ("gpu_mean_gpu_busy_pct", "throughput_requests_s"),
            ("gpu_mean_tensor_activity_pct", "throughput_requests_s"),
            ("gpu_mean_dram_activity_pct", "throughput_requests_s"),
        ],
        "balanced-valid": [
            ("throughput_requests_s", "joules_per_request"),
            ("queue_nonzero_fraction", "ttft_ms_p95"),
            ("queue_median", "ttft_ms_p95"),
            ("queue_median", "e2e_ms_p95"),
            ("service_mean_sglang_num_queue_reqs", "ttft_ms_p95"),
        ],
        "prefix-valid": [
            ("radix_cache", "service_mean_sglang_cache_hit_rate"),
            ("prefix_reuse_fraction", "service_mean_sglang_cache_hit_rate"),
            ("service_mean_sglang_cache_hit_rate", "joules_per_request"),
            ("service_mean_sglang_cache_hit_rate", "ttft_ms_p95"),
            ("service_mean_sglang_cache_hit_rate", "e2e_ms_p95"),
            ("service_mean_sglang_cache_hit_rate", "gpu_mean_dram_activity_pct"),
            ("service_mean_sglang_cache_hit_rate", "gpu_mean_tensor_activity_pct"),
        ],
    }
    output: list[dict[str, Any]] = []
    for scope, pairs_to_check in requested.items():
        for feature, target in pairs_to_check:
            pairs: list[tuple[float, float]] = []
            for row in scopes[scope]:
                raw_left = row.get(feature)
                raw_right = row.get(target)
                left = float(raw_left) if isinstance(raw_left, bool) else finite_number(raw_left)
                right = (
                    float(raw_right)
                    if isinstance(raw_right, bool)
                    else finite_number(raw_right)
                )
                if left is not None and right is not None:
                    pairs.append((left, right))
            coefficient = (
                spearman(
                    [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                )
                if len(pairs) >= 3
                else None
            )
            output.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "target": target,
                    "n": len(pairs),
                    "spearman_rho": coefficient,
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate all recorded SGLang campaign telemetry by run"
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [
        aggregate_run(path)
        for path in sorted((args.results / "runs").iterdir())
        if path.is_dir() and (path / "validity.json").exists()
    ]
    correlations = correlation_rows(runs)
    selected_relationships = selected_relationship_rows(runs)
    availability = availability_rows(runs)
    write_csv(args.output / "telemetry-run-aggregates.csv", runs)
    write_csv(args.output / "telemetry-correlations.csv", correlations)
    write_csv(args.output / "selected-relationships.csv", selected_relationships)
    write_csv(args.output / "telemetry-availability.csv", availability)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "valid_runs": sum(row["valid"] for row in runs),
                "aggregate_fields": len({key for row in runs for key in row}),
                "correlations": len(correlations),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
