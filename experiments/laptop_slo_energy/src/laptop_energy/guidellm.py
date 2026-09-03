from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .binaries import resolve_binary
from .config import CampaignConfig, Workload
from .plan import PlannedCell


def build_command(
    config: CampaignConfig,
    cell: PlannedCell,
    output_dir: Path,
) -> list[str]:
    workload = config.workload(cell.workload)
    endpoint = (
        f"http://{config.server['host']}:{config.server['port']},"
        "request_format=/v1/chat/completions"
    )
    data = (
        "kind=synthetic_text,"
        f"prompt_tokens={workload.prompt_tokens},prompt_tokens_stdev=1,"
        f"output_tokens={workload.output_tokens},output_tokens_stdev=1"
    )
    if cell.profile == "throughput":
        profile = (
            "kind=throughput,"
            f"max_concurrency={int(config.guidellm['capacity_max_concurrency'])},"
            f"rampup_duration={float(config.guidellm['rampup_seconds'])}"
        )
        duration = cell.duration_seconds
    elif cell.profile == "synchronous":
        profile = "kind=synchronous"
        duration = cell.duration_seconds
    elif (
        cell.profile in ("poisson", "constant")
        and cell.offered_rate_requests_s is not None
    ):
        profile = (
            f"kind={cell.profile},"
            f"rate={cell.offered_rate_requests_s:.8g},"
            f"max_concurrency={int(config.guidellm['policy_max_concurrency'])},"
            f"rampup_duration={float(config.guidellm['rampup_seconds'])}"
        )
        duration = cell.duration_seconds
    else:
        raise ValueError("policy cell requires a calibrated offered rate")

    configured_binary = str(config.guidellm["binary"])
    binary = resolve_binary(configured_binary) or configured_binary
    return [
        binary,
        "run",
        "--backend",
        "kind=openai_http,target=" + endpoint,
        "--data",
        data,
        "--tokenizer",
        "kind=huggingface_auto,model=" + str(config.guidellm["tokenizer"]),
        "--profile",
        profile,
        "--constraint",
        f"kind=max_duration,seconds={duration:g}",
        "--seed",
        f"kind=static,value={cell.seed}",
        "--output",
        f"kind=json,path={output_dir / 'guidellm.json'}",
        "--output",
        f"kind=csv,path={output_dir / 'guidellm.csv'}",
        "--disable-console-interactive",
    ]


def _number_at(value: Any, *path: str) -> float | None:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _distribution(metrics: dict[str, Any], name: str, statistic: str) -> float | None:
    distribution = metrics.get(name, {})
    value = _number_at(distribution, "successful", statistic)
    if value is not None:
        return value
    return _number_at(distribution, "successful", "percentiles", statistic)


def _status_count(metrics: dict[str, Any], name: str) -> int:
    value = _number_at(metrics, "request_totals", name)
    return int(value or 0)


def summarize_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("GuideLLM report has no benchmarks")
    benchmark = benchmarks[-1]
    metrics = benchmark.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("GuideLLM benchmark has no metrics object")

    successful = _status_count(metrics, "successful")
    incomplete = _status_count(metrics, "incomplete")
    errored = _status_count(metrics, "errored")
    total = _status_count(metrics, "total") or successful + incomplete + errored
    output_total = _distribution(metrics, "output_token_count", "total_sum")
    if output_total is None:
        output_total = _distribution(metrics, "output_token_count", "sum")
    if output_total is None:
        output_mean = _distribution(metrics, "output_token_count", "mean")
        if output_mean is not None:
            output_total = output_mean * successful
    request_rate = _distribution(metrics, "requests_per_second", "mean")
    p95_e2e_s = _distribution(metrics, "request_latency", "p95")
    p99_e2e_s = _distribution(metrics, "request_latency", "p99")
    return {
        "successful_requests": successful,
        "incomplete_requests": incomplete,
        "errored_requests": errored,
        "admitted_requests": total,
        "success_rate": successful / total if total else 0.0,
        "completed_requests_per_second": request_rate,
        "output_tokens_successful": output_total,
        "p50_ttft_ms": _distribution(metrics, "time_to_first_token_ms", "p50"),
        "p95_ttft_ms": _distribution(metrics, "time_to_first_token_ms", "p95"),
        "p99_ttft_ms": _distribution(metrics, "time_to_first_token_ms", "p99"),
        "p50_itl_ms": _distribution(metrics, "inter_token_latency_ms", "p50"),
        "p95_itl_ms": _distribution(metrics, "inter_token_latency_ms", "p95"),
        "p99_itl_ms": _distribution(metrics, "inter_token_latency_ms", "p99"),
        "p95_e2e_ms": p95_e2e_s * 1000 if p95_e2e_s is not None else None,
        "p99_e2e_ms": p99_e2e_s * 1000 if p99_e2e_s is not None else None,
        "measurement_start_unix_s": _number_at(benchmark, "start_time"),
        "measurement_end_unix_s": _number_at(benchmark, "end_time"),
        "measurement_duration_s": _number_at(benchmark, "duration"),
        "guidellm_schema_version": _number_at(report, "metadata", "version"),
        "guidellm_version": (
            report.get("metadata", {}).get("guidellm_version")
            if isinstance(report.get("metadata"), dict)
            else None
        ),
    }


def evaluate_slo(
    summary: dict[str, Any], workload: Workload, success_rate_min: float
) -> dict[str, Any]:
    checks = {
        "success_rate": summary.get("success_rate", 0) >= success_rate_min,
        "p95_ttft_ms": _at_most(summary.get("p95_ttft_ms"), workload.slo["p95_ttft_ms"]),
        "p95_itl_ms": _at_most(summary.get("p95_itl_ms"), workload.slo["p95_itl_ms"]),
        "p99_e2e_ms": _at_most(summary.get("p99_e2e_ms"), workload.slo["p99_e2e_ms"]),
    }
    return {
        "policy_label": workload.name,
        "thresholds": {"success_rate_min": success_rate_min, **workload.slo},
        "checks": checks,
        "cell_passes_slo": all(checks.values()),
    }


def _at_most(value: Any, maximum: float) -> bool:
    return isinstance(value, (int, float)) and value <= maximum
