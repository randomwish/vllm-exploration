from __future__ import annotations

import math
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_MAP_ENTRY = re.compile(
    r"@(runqlat_count_1s|runqlat_sum_us_1s|runqlat_max_us_1s|"
    r"runqlat_ge_100us_1s|runqlat_ge_1ms_1s|runqlat_ge_10ms_1s|"
    r"cpu_changes_1s|cpu_migrations_1s|sched_migrate_task_1s|"
    r"futex_wait_count_1s|futex_wait_sum_us_1s|futex_wait_max_us_1s|"
    r"futex_wait_ge_100us_1s|futex_wait_ge_1ms_1s|"
    r"futex_wait_ge_10ms_1s|futex_wake_calls_1s)"
    r"\[(\d+)\]:\s+(\d+)"
)
_LEGACY_HISTOGRAM_HEADER = re.compile(r"@runqlat_us_250ms\[(\d+)\]:")
_LEGACY_HISTOGRAM_ROW = re.compile(
    r"^\s*\[(\d+),\s*(\d+)\)\s+(\d+)\s+\|"
)


def summarize_runqlat(
    output: str,
    measurement_start_bpf_clock: float | None,
    measurement_end_bpf_clock: float | None,
) -> dict[str, Any]:
    """Summarize scheduler wait for llama-server inside the measured window."""
    maps: dict[str, dict[int, int]] = defaultdict(dict)
    for name, bucket_text, value_text in _MAP_ENTRY.findall(output):
        maps[name][int(bucket_text)] = int(value_text)
    if maps.get("runqlat_count_1s"):
        return _summarize_aggregates(
            maps,
            measurement_start_bpf_clock,
            measurement_end_bpf_clock,
        )
    return _summarize_legacy_histogram(
        output,
        measurement_start_bpf_clock,
        measurement_end_bpf_clock,
    )


def summarize_phase_alignment(
    output: str,
    report_path: Path,
    measurement_start_bpf_clock: float | None,
    measurement_end_bpf_clock: float | None,
    measurement_start_unix: float | None,
) -> dict[str, Any]:
    """Align one-second kernel aggregates with successful request phases."""
    if (
        measurement_start_bpf_clock is None
        or measurement_end_bpf_clock is None
        or measurement_start_unix is None
    ):
        return {"valid": False, "error": "measurement clock anchors are missing"}

    maps = _parse_maps(output)
    if not maps.get("runqlat_count_1s"):
        return {"valid": False, "error": "one-second eBPF aggregates are missing"}

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        benchmark = report["benchmarks"][-1]
        successful = benchmark["requests"]["successful"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return {"valid": False, "error": f"GuideLLM requests unavailable: {exc}"}

    intervals: list[tuple[float, float, float]] = []
    for request in successful:
        if not isinstance(request, dict):
            continue
        request_start = request.get("request_start_time")
        request_end = request.get("request_end_time")
        ttft_ms = request.get("time_to_first_token_ms")
        if not all(
            isinstance(value, (int, float))
            for value in (request_start, request_end, ttft_ms)
        ):
            continue
        first_token = float(request_start) + float(ttft_ms) / 1000.0
        intervals.append(
            (float(request_start), first_token, float(request_end))
        )

    if not intervals:
        return {"valid": False, "error": "no successful request intervals found"}

    all_buckets = set().union(*(values.keys() for values in maps.values()))
    window_buckets = {
        bucket
        for bucket in all_buckets
        if _inside_window(
            bucket,
            1.0,
            measurement_start_bpf_clock,
            measurement_end_bpf_clock,
        )
    }
    phase_buckets: dict[str, set[int]] = {
        "prefill": set(),
        "decode": set(),
        "idle": set(),
    }
    overlaps = 0
    for bucket in window_buckets:
        bucket_bpf_clock = bucket + 0.5
        bucket_unix = measurement_start_unix + (
            bucket_bpf_clock - measurement_start_bpf_clock
        )
        in_prefill = any(start <= bucket_unix < first for start, first, _ in intervals)
        in_decode = any(first <= bucket_unix <= end for _, first, end in intervals)
        if in_prefill:
            phase_buckets["prefill"].add(bucket)
        if in_decode:
            phase_buckets["decode"].add(bucket)
        if in_prefill and in_decode:
            overlaps += 1
        if not in_prefill and not in_decode:
            phase_buckets["idle"].add(bucket)

    return {
        "valid": True,
        "method": (
            "one-second bucket centers mapped from CLOCK_BOOTTIME to Unix time; "
            "prefill runs from request start to first token and decode runs from "
            "first token to request end"
        ),
        "futex_timing_semantics": (
            "blocking-capable futex syscall duration is assigned to its exit bucket"
        ),
        "successful_request_intervals": len(intervals),
        "overlapping_phase_buckets": overlaps,
        "phases": {
            name: _summarize_selected_aggregates(maps, buckets)
            for name, buckets in phase_buckets.items()
        },
    }


def _parse_maps(output: str) -> dict[str, dict[int, int]]:
    maps: dict[str, dict[int, int]] = defaultdict(dict)
    for name, bucket_text, value_text in _MAP_ENTRY.findall(output):
        maps[name][int(bucket_text)] = int(value_text)
    return maps


def _inside_window(
    bucket: int,
    bucket_seconds: float,
    start: float | None,
    end: float | None,
) -> bool:
    center = (bucket + 0.5) * bucket_seconds
    return start is not None and end is not None and start <= center <= end


def _summarize_aggregates(
    maps: dict[str, dict[int, int]],
    start: float | None,
    end: float | None,
) -> dict[str, Any]:
    all_buckets = set().union(*(values.keys() for values in maps.values()))
    selected = {
        bucket
        for bucket in all_buckets
        if _inside_window(bucket, 1.0, start, end)
    }
    return _summarize_selected_aggregates(maps, selected)


def _summarize_selected_aggregates(
    maps: dict[str, dict[int, int]], selected: set[int]
) -> dict[str, Any]:
    samples = sum(maps["runqlat_count_1s"].get(bucket, 0) for bucket in selected)
    latency_sum = sum(
        maps["runqlat_sum_us_1s"].get(bucket, 0) for bucket in selected
    )
    maximum = max(
        (maps["runqlat_max_us_1s"].get(bucket, 0) for bucket in selected),
        default=None,
    )

    def total(name: str) -> int:
        return sum(maps[name].get(bucket, 0) for bucket in selected)

    def threshold(name: str, denominator: int) -> tuple[int, float | None]:
        count = sum(maps[name].get(bucket, 0) for bucket in selected)
        return count, count / denominator if denominator else None

    over_100us, fraction_over_100us = threshold(
        "runqlat_ge_100us_1s", samples
    )
    over_1ms, fraction_over_1ms = threshold("runqlat_ge_1ms_1s", samples)
    over_10ms, fraction_over_10ms = threshold("runqlat_ge_10ms_1s", samples)
    cpu_change_map = (
        "cpu_changes_1s" if maps.get("cpu_changes_1s") else "cpu_migrations_1s"
    )
    cpu_changes, cpu_change_fraction = threshold(cpu_change_map, samples)
    scheduler_migrations = total("sched_migrate_task_1s")

    futex_waits = total("futex_wait_count_1s")
    futex_wait_sum = total("futex_wait_sum_us_1s")
    futex_wait_max = max(
        (maps["futex_wait_max_us_1s"].get(bucket, 0) for bucket in selected),
        default=None,
    )
    futex_over_100us, futex_fraction_over_100us = threshold(
        "futex_wait_ge_100us_1s", futex_waits
    )
    futex_over_1ms, futex_fraction_over_1ms = threshold(
        "futex_wait_ge_1ms_1s", futex_waits
    )
    futex_over_10ms, futex_fraction_over_10ms = threshold(
        "futex_wait_ge_10ms_1s", futex_waits
    )
    return {
        "mode": "one-second-aggregates",
        "samples": samples,
        "selected_1s_buckets": len(selected),
        "mean_us": latency_sum / samples if samples else None,
        "max_us": maximum,
        "ge_100us_samples": over_100us,
        "ge_100us_fraction": fraction_over_100us,
        "ge_1ms_samples": over_1ms,
        "ge_1ms_fraction": fraction_over_1ms,
        "ge_10ms_samples": over_10ms,
        "ge_10ms_fraction": fraction_over_10ms,
        "schedule_in_cpu_change_samples": cpu_changes,
        "schedule_in_cpu_change_fraction": cpu_change_fraction,
        "cpu_migration_samples": cpu_changes,
        "cpu_migration_fraction": cpu_change_fraction,
        "cpu_migration_legacy_alias": (
            "cpu_migration_* aliases schedule-in CPU changes; use "
            "sched_migrate_task_samples for kernel migration events"
        ),
        "sched_migrate_task_samples": scheduler_migrations,
        "sched_migrate_task_per_schedule_in": (
            scheduler_migrations / samples if samples else None
        ),
        "futex_wait_us": {
            "samples": futex_waits,
            "mean_us": futex_wait_sum / futex_waits if futex_waits else None,
            "max_us": futex_wait_max,
            "ge_100us_samples": futex_over_100us,
            "ge_100us_fraction": futex_fraction_over_100us,
            "ge_1ms_samples": futex_over_1ms,
            "ge_1ms_fraction": futex_fraction_over_1ms,
            "ge_10ms_samples": futex_over_10ms,
            "ge_10ms_fraction": futex_fraction_over_10ms,
        },
        "futex_wake_calls": total("futex_wake_calls_1s"),
    }


def _summarize_legacy_histogram(
    output: str,
    start: float | None,
    end: float | None,
) -> dict[str, Any]:
    histogram: dict[int, int] = defaultdict(int)
    selected = False
    for line in output.splitlines():
        header = _LEGACY_HISTOGRAM_HEADER.search(line)
        if header:
            selected = _inside_window(int(header.group(1)), 0.25, start, end)
            continue
        row = _LEGACY_HISTOGRAM_ROW.match(line)
        if row and selected:
            lower = int(row.group(1))
            upper = int(row.group(2))
            count = int(row.group(3))
            histogram[(lower + upper) // 2] += count
    sample_count = sum(histogram.values())

    def percentile_upper_bound(percentile: float) -> int | None:
        if sample_count == 0:
            return None
        target = math.ceil(sample_count * percentile)
        cumulative = 0
        for midpoint, count in sorted(histogram.items()):
            cumulative += count
            if cumulative >= target:
                return 1 if midpoint == 0 else 1 << midpoint.bit_length()
        return None

    return {
        "mode": "legacy-250ms-histograms-partial",
        "samples": sample_count,
        "mean_bin_midpoint_us": (
            sum(value * count for value, count in histogram.items()) / sample_count
            if sample_count
            else None
        ),
        "p50_upper_bound_us": percentile_upper_bound(0.50),
        "p95_upper_bound_us": percentile_upper_bound(0.95),
        "p99_upper_bound_us": percentile_upper_bound(0.99),
        "warning": (
            "Legacy per-window histograms can hit the bpftrace map-entry limit; "
            "treat these latency estimates as partial."
        ),
    }
