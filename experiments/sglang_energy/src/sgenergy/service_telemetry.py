from __future__ import annotations

import argparse
import json
import signal
import time
import urllib.request
from pathlib import Path
from typing import Any


METRIC_PREFIXES = (
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:gen_throughput",
    "sglang:cache_hit_rate",
    "sglang:decode_sum_seq_lens",
    "sglang:token_usage",
    "sglang:full_token_usage",
    "sglang:kv_available_tokens",
    "sglang:kv_evictable_tokens",
    "sglang:kv_used_tokens",
)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cgroup_snapshot() -> dict[str, str | None]:
    if Path("/sys/fs/cgroup/cgroup.controllers").exists():
        return {
            "version": "2",
            "cpu_stat": _read("/sys/fs/cgroup/cpu.stat"),
            "cpu_max": _read("/sys/fs/cgroup/cpu.max"),
            "memory_current": _read("/sys/fs/cgroup/memory.current"),
            "memory_max": _read("/sys/fs/cgroup/memory.max"),
            "memory_events": _read("/sys/fs/cgroup/memory.events"),
        }
    return {
        "version": "1",
        "cpuacct_usage": _read("/sys/fs/cgroup/cpuacct/cpuacct.usage"),
        "cpu_stat": _read("/sys/fs/cgroup/cpu/cpu.stat"),
        "cpu_quota_us": _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "cpu_period_us": _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        "memory_usage": _read("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        "memory_limit": _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "memory_failcnt": _read("/sys/fs/cgroup/memory/memory.failcnt"),
        "memory_oom_control": _read("/sys/fs/cgroup/memory/memory.oom_control"),
    }


def _network_snapshot() -> dict[str, dict[str, int]]:
    raw = _read("/proc/net/dev") or ""
    output: dict[str, dict[str, int]] = {}
    for line in raw.splitlines()[2:]:
        if ":" not in line:
            continue
        interface, values = line.split(":", 1)
        columns = values.split()
        if len(columns) < 16:
            continue
        output[interface.strip()] = {
            "rx_bytes": int(columns[0]),
            "rx_packets": int(columns[1]),
            "rx_errors": int(columns[2]),
            "rx_drops": int(columns[3]),
            "tx_bytes": int(columns[8]),
            "tx_packets": int(columns[9]),
            "tx_errors": int(columns[10]),
            "tx_drops": int(columns[11]),
        }
    return output


def _metrics(url: str) -> tuple[list[str], str | None]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    selected = [
        line
        for line in body.splitlines()
        if line and not line.startswith("#") and line.startswith(METRIC_PREFIXES)
    ]
    return selected, None


def collect(
    output: Path, *, metrics_url: str, interval: float, duration: float | None = None
) -> None:
    stop = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = previous = time.monotonic()
    with output.open("w", encoding="utf-8", buffering=1) as stream:
        while not stop:
            deadline = previous + interval
            time.sleep(max(0.0, deadline - time.monotonic()))
            now = time.monotonic()
            metrics, error = _metrics(metrics_url)
            stream.write(
                json.dumps(
                    {
                        "unix_s": time.time(),
                        "monotonic_s": now,
                        "interval_s": now - previous,
                        "metrics": metrics,
                        "metrics_error": error,
                        "cgroup": _cgroup_snapshot(),
                        "network": _network_snapshot(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            previous = now
            if duration is not None and now - start >= duration:
                break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect SGLang and cgroup telemetry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:30000/metrics")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args(argv)
    collect(
        args.output,
        metrics_url=args.metrics_url,
        interval=args.interval,
        duration=args.duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
