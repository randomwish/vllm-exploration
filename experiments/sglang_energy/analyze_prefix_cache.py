#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any

from analyze_telemetry import aggregate_run, finite_number, write_csv


LABEL_RE = re.compile(
    r"^prefix-(?P<workload>PX0|PX50|PX87)-cache-(?P<cache>on|off)-seed-(?P<seed>\d+)$"
)
OUTCOMES = (
    "joules_per_request",
    "joules_per_1000_output_tokens",
    "throughput_requests_s",
    "throughput_output_tokens_s",
    "ttft_ms_p95",
    "tpot_ms_p95",
    "e2e_ms_p95",
    "total_average_power_w",
    "service_mean_sglang_cache_hit_rate",
    "service_mean_sglang_kv_used_tokens",
    "service_mean_sglang_kv_evictable_tokens",
    "gpu_mean_gpu_busy_pct",
    "gpu_mean_sm_activity_pct",
    "gpu_mean_tensor_activity_pct",
    "gpu_mean_dram_activity_pct",
    "gpu_mean_nvlink_rx_mib_s",
    "gpu_mean_nvlink_tx_mib_s",
    "gpu_power_imbalance_fraction",
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_median_ci(
    values: list[float], *, resamples: int = 10_000, seed: int = 20260834
) -> tuple[float, float]:
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def paired_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        match = LABEL_RE.match(str(run["label"]))
        if not match:
            continue
        key = (match.group("workload"), int(match.group("seed")))
        cells.setdefault(key, {})[match.group("cache")] = run

    output: list[dict[str, Any]] = []
    for (workload, seed), treatments in sorted(cells.items()):
        off = treatments.get("off")
        on = treatments.get("on")
        row: dict[str, Any] = {
            "workload": workload,
            "prefix_reuse_fraction": {"PX0": 0.0, "PX50": 0.5, "PX87": 0.875}[
                workload
            ],
            "seed": seed,
            "off_present": off is not None,
            "on_present": on is not None,
            "off_valid": bool(off and off["valid"]),
            "on_valid": bool(on and on["valid"]),
            "valid_pair": bool(off and on and off["valid"] and on["valid"]),
        }
        for outcome in OUTCOMES:
            off_value = finite_number(off.get(outcome)) if off else None
            on_value = finite_number(on.get(outcome)) if on else None
            row[f"off_{outcome}"] = off_value
            row[f"on_{outcome}"] = on_value
            if row["valid_pair"] and off_value is not None and on_value is not None:
                row[f"delta_{outcome}"] = on_value - off_value
                row[f"relative_delta_{outcome}"] = (
                    (on_value - off_value) / off_value if off_value else None
                )
        output.append(row)
    return output


def summary_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for workload in ("PX0", "PX50", "PX87"):
        workload_pairs = [row for row in pairs if row["workload"] == workload]
        valid_pairs = [row for row in workload_pairs if row["valid_pair"]]
        for outcome in OUTCOMES:
            absolute = [
                value
                for row in valid_pairs
                if (value := finite_number(row.get(f"delta_{outcome}"))) is not None
            ]
            relative = [
                value
                for row in valid_pairs
                if (
                    value := finite_number(row.get(f"relative_delta_{outcome}"))
                )
                is not None
            ]
            if not absolute:
                continue
            low, high = bootstrap_median_ci(absolute)
            if relative:
                relative_median = statistics.median(relative)
                relative_low, relative_high = bootstrap_median_ci(relative)
            else:
                relative_median = relative_low = relative_high = None
            output.append(
                {
                    "workload": workload,
                    "prefix_reuse_fraction": workload_pairs[0][
                        "prefix_reuse_fraction"
                    ],
                    "outcome": outcome,
                    "planned_pairs": len(workload_pairs),
                    "valid_pairs": len(absolute),
                    "median_cache_on_minus_off": statistics.median(absolute),
                    "bootstrap_median_ci_low": low,
                    "bootstrap_median_ci_high": high,
                    "median_relative_change": relative_median,
                    "bootstrap_relative_ci_low": relative_low,
                    "bootstrap_relative_ci_high": relative_high,
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze paired, prewarmed SGLang prefix-cache results"
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
    pairs = paired_rows(runs)
    summaries = summary_rows(pairs)
    write_csv(args.output / "prefix-cache-pairs.csv", pairs)
    write_csv(args.output / "prefix-cache-effects.csv", summaries)
    result = {
        "measured_cells": sum(
            int(row["off_present"]) + int(row["on_present"]) for row in pairs
        ),
        "planned_pairs_found": len(pairs),
        "valid_pairs": sum(row["valid_pair"] for row in pairs),
        "effect_rows": len(summaries),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
