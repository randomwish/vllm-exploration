from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import CampaignConfig, write_json_atomic
from .run_cell import measure_cell
from .replay import replay_trace
from .server import ServerProcess, default_remote_env, server_command
from .traces import build_trace, safe_token_ids, verify_trace


@dataclass(frozen=True)
class ServerSettings:
    max_running_requests: int
    chunked_prefill_size: int
    radix_cache: bool

    @property
    def key(self) -> str:
        cache = "cache-on" if self.radix_cache else "cache-off"
        return f"mrr-{self.max_running_requests}_chunk-{self.chunked_prefill_size}_{cache}"


class DeadlineReached(RuntimeError):
    pass


class CampaignRunner:
    def __init__(self, config: CampaignConfig, output_root: Path):
        self.config = config
        self.output_root = output_root
        self.traces_root = output_root / "traces"
        self.runs_root = output_root / "runs"
        self.start_unix = time.time()
        self.hard_deadline_unix = self.start_unix + float(config.runpod["hard_minutes"]) * 60
        self.finalization_reserve = float(
            config.measurement["finalization_reserve_seconds"]
        )
        self.base_url = f"http://127.0.0.1:{config.model['port']}"
        self.token_pool: list[int] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.capacities: dict[str, float] = {}
        self.selected: dict[str, Any] = {}
        self.run_counter = 0

    def event(self, event: str, **data: Any) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"unix_s": time.time(), "event": event, **data},
                    separators=(",", ":"),
                )
                + "\n"
            )

    def save_state(self, state: str) -> None:
        write_json_atomic(
            self.output_root / "state.json",
            {
                "state": state,
                "start_unix_s": self.start_unix,
                "hard_deadline_unix_s": self.hard_deadline_unix,
                "remaining_seconds": self.hard_deadline_unix - time.time(),
                "capacities": self.capacities,
                "selected": self.selected,
                "completed_runs": sorted(self.results),
            },
        )

    def require_time(self, watchdog_seconds: float, *, server_restart: bool = False) -> None:
        restart_reserve = 180 if server_restart else 0
        required = watchdog_seconds + self.finalization_reserve + restart_reserve
        remaining = self.hard_deadline_unix - time.time()
        if remaining < required:
            raise DeadlineReached(
                f"{remaining:.0f}s remain but next operation requires {required:.0f}s"
            )

    def prepare(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.traces_root.mkdir()
        self.runs_root.mkdir()
        config_copy = json.loads(json.dumps(self.config.raw))
        config_copy["runtime"] = {
            "hostname": socket.gethostname(),
            "start_unix_s": self.start_unix,
            "hard_deadline_unix_s": self.hard_deadline_unix,
            "pod_id": os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_POD_ID_ALT"),
        }
        write_json_atomic(self.output_root / "campaign.json", config_copy)
        shutil.copy2(self.config.path, self.output_root / "campaign.input.json")
        self.event("campaign-preparing")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model["path"],
            revision=self.config.model["revision"],
            trust_remote_code=True,
        )
        self.token_pool = safe_token_ids(tokenizer)
        write_json_atomic(
            self.output_root / "tokenizer.json",
            {
                "name_or_path": getattr(tokenizer, "name_or_path", None),
                "vocab_size": len(tokenizer),
                "safe_token_count": len(self.token_pool),
                "special_ids": list(getattr(tokenizer, "all_special_ids", [])),
            },
        )
        self.hardware_preflight()
        self.model_smoke_preflight()
        self.save_state("prepared")

    def hardware_preflight(self) -> None:
        preflight = self.output_root / "preflight"
        preflight.mkdir(exist_ok=True)
        commands = {
            "nvidia_smi_list": ["nvidia-smi", "-L"],
            "nvidia_smi_topology": ["nvidia-smi", "topo", "-m"],
            "nvidia_smi_query": ["nvidia-smi", "-q"],
            "python": ["python3", "--version"],
            "sglang": ["python3", "-c", "import sglang; print(sglang.__version__)"],
            "torch": [
                "python3",
                "-c",
                "import torch; print(torch.__version__, torch.version.cuda)",
            ],
        }
        output: dict[str, Any] = {}
        for name, command in commands.items():
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=120, check=False
            )
            output[name] = {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if result.returncode != 0:
                raise RuntimeError(f"hardware preflight command failed: {name}")
        if output["nvidia_smi_list"]["stdout"].count("GPU ") != 2:
            raise RuntimeError("hardware preflight requires exactly two visible GPUs")
        write_json_atomic(preflight / "environment.json", output)
        result = subprocess.run(
            [
                "python3",
                "-m",
                "sgenergy.gpu_telemetry",
                "--output",
                str(preflight / "gpu-smoke.jsonl"),
                "--interval",
                str(self.config.measurement["gpu_sample_seconds"]),
                "--duration",
                "0.6",
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"GPU telemetry preflight failed: {result.stderr}")
        rows = [
            json.loads(line)
            for line in (preflight / "gpu-smoke.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len({row["gpu_uuid"] for row in rows}) != 2:
            raise RuntimeError("GPU telemetry did not observe exactly two GPU UUIDs")
        if not all(row.get("gpm_supported") for row in rows):
            raise RuntimeError("Hopper GPM is not supported on every visible GPU")

    def model_smoke_preflight(self) -> None:
        m = self.config.model
        settings = ServerSettings(
            int(m["baseline_max_running_requests"]),
            int(m["baseline_chunked_prefill_size"]),
            False,
        )
        preflight = self.output_root / "preflight"
        command = server_command(
            self.config,
            max_running_requests=settings.max_running_requests,
            chunked_prefill_size=settings.chunked_prefill_size,
            radix_cache=False,
        )
        server = ServerProcess(
            command, preflight / "model-smoke-server.log", default_remote_env(self.config)
        )
        try:
            self.require_time(540, server_restart=True)
            server.start(base_url=self.base_url, timeout_seconds=1200)
            summaries: dict[str, Any] = {}
            for offset, workload in enumerate(("BAL", "PF", "DEC")):
                trace = self.trace(
                    workload,
                    seed=int(self.config.design["seeds"][0]) + offset,
                    kind="closed",
                    count=1,
                )
                summary = asyncio.run(
                    replay_trace(
                        trace,
                        base_url=self.base_url,
                        output_path=preflight / f"{workload}-requests.jsonl",
                        watchdog_seconds=540,
                        max_concurrency=1,
                    )
                )
                summaries[workload] = summary
                if summary["success_count"] != 1 or summary["watchdog_hit"]:
                    raise RuntimeError(f"model smoke failed for workload {workload}")
            write_json_atomic(preflight / "model-smoke.json", summaries)
        finally:
            server.stop()

    def trace(
        self,
        workload_name: str,
        *,
        seed: int,
        kind: str,
        rate: float | None = None,
        duration: float | None = None,
        count: int | None = None,
        prefix_seed: int | None = None,
    ) -> Path:
        workload = self.config.workloads[workload_name]
        if kind == "open":
            label = f"{workload_name}_poisson_{rate:.6f}_{duration:.0f}s_seed-{seed}"
        else:
            label = f"{workload_name}_closed_n-{count}_seed-{seed}"
        if prefix_seed is not None and prefix_seed != seed:
            label += f"_prefix-seed-{prefix_seed}"
        target = self.traces_root / label.replace(".", "p")
        if target.exists():
            verify_trace(target)
            return target
        build_trace(
            target,
            workload=workload,
            seed=seed,
            token_pool=self.token_pool,
            prefix_groups=int(self.config.design["prefix_groups"]),
            prefix_seed=prefix_seed,
            rate=rate,
            duration=duration,
            closed_loop_count=count,
        )
        return target

    def _run_id(self, label: str) -> str:
        self.run_counter += 1
        return f"{self.run_counter:03d}-{label}"

    def measure(
        self,
        *,
        label: str,
        trace_dir: Path,
        settings: ServerSettings,
        watchdog: float,
        max_concurrency: int | None = None,
        flush: bool = True,
    ) -> dict[str, Any]:
        self.require_time(watchdog)
        run_id = self._run_id(label)
        self.event("run-start", run_id=run_id, settings=asdict(settings))
        summary = measure_cell(
            self.config,
            trace_dir=trace_dir,
            run_dir=self.runs_root / run_id,
            base_url=self.base_url,
            watchdog_seconds=watchdog,
            max_concurrency=max_concurrency,
            flush_before_run=flush,
            server_settings=asdict(settings),
        )
        summary["run_id"] = run_id
        summary["label"] = label
        summary["settings"] = asdict(settings)
        self.results[run_id] = summary
        self.event("run-end", run_id=run_id, valid=summary["valid"])
        self.save_state("running")
        return summary

    def with_server(
        self,
        settings: ServerSettings,
        action: Callable[[], Any],
        *,
        label: str,
    ) -> Any:
        self.require_time(0, server_restart=True)
        command = server_command(
            self.config,
            max_running_requests=settings.max_running_requests,
            chunked_prefill_size=settings.chunked_prefill_size,
            radix_cache=settings.radix_cache,
        )
        log = self.output_root / "servers" / f"{label}-{settings.key}.log"
        self.event("server-start", settings=asdict(settings), log=str(log))
        server = ServerProcess(command, log, default_remote_env(self.config))
        try:
            server.start(base_url=self.base_url, timeout_seconds=1200)
            return action()
        finally:
            server.stop()
            self.event("server-stop", settings=asdict(settings))

    @staticmethod
    def capacity(summaries: Iterable[dict[str, Any]]) -> float:
        values = [
            float(item["throughput_requests_s"])
            for item in summaries
            if item.get("valid") and item.get("throughput_requests_s")
        ]
        if not values:
            raise RuntimeError("capacity scout produced no valid cells")
        maximum = max(values)
        plateau = [value for value in values if value >= maximum * 0.95]
        return statistics.median(plateau)

    def capacity_scout(
        self,
        workload: str,
        settings: ServerSettings,
        concurrencies: list[int],
        *,
        seed: int,
        label: str,
    ) -> float:
        summaries: list[dict[str, Any]] = []

        def action() -> None:
            for concurrency in concurrencies:
                if workload == "BAL":
                    request_count = max(256, concurrency * 2)
                elif workload == "PX0":
                    request_count = max(64, concurrency)
                else:
                    request_count = max(32, concurrency)
                trace = self.trace(
                    workload,
                    seed=seed,
                    kind="closed",
                    count=request_count,
                )
                summary = self.measure(
                    label=f"{label}-c{concurrency}",
                    trace_dir=trace,
                    settings=settings,
                    watchdog=float(self.config.measurement["screen_watchdog_seconds"]),
                    max_concurrency=concurrency,
                )
                summaries.append(summary)
                if not summary.get("valid"):
                    break

        self.with_server(settings, action, label=label)
        value = self.capacity(summaries)
        self.capacities[workload] = value
        self.event("capacity", workload=workload, requests_s=value)
        return value

    def open_trace(
        self, workload: str, rate: float, *, seed: int, confirm: bool
    ) -> Path:
        duration_key = "confirm_arrival_seconds" if confirm else "screen_arrival_seconds"
        return self.trace(
            workload,
            seed=seed,
            kind="open",
            rate=rate,
            duration=float(self.config.measurement[duration_key]),
        )

    def screen_settings(
        self,
        workload: str,
        rate: float,
        settings_values: list[ServerSettings],
        *,
        seed: int,
        label: str,
    ) -> dict[ServerSettings, dict[str, Any]]:
        trace = self.open_trace(workload, rate, seed=seed, confirm=False)
        output: dict[ServerSettings, dict[str, Any]] = {}
        for settings in settings_values:
            def action(settings: ServerSettings = settings) -> None:
                output[settings] = self.measure(
                    label=f"{label}-{settings.key}",
                    trace_dir=trace,
                    settings=settings,
                    watchdog=float(self.config.measurement["screen_watchdog_seconds"]),
                )

            self.with_server(settings, action, label=label)
        return output

    def candidate_from_screen(
        self,
        summaries: dict[ServerSettings, dict[str, Any]],
        baseline: ServerSettings,
    ) -> ServerSettings:
        base = summaries.get(baseline)
        if not base or not base.get("valid"):
            return baseline
        base_energy = float(base["joules_per_request"])
        base_tput = float(base["throughput_requests_s"])
        base_e2e = base["latency_ms"]["e2e_ms"]["p95"]
        candidates: list[tuple[float, ServerSettings]] = []
        for settings, summary in summaries.items():
            if settings == baseline or not summary.get("valid"):
                continue
            energy = float(summary["joules_per_request"])
            throughput = float(summary["throughput_requests_s"])
            e2e = summary["latency_ms"]["e2e_ms"]["p95"]
            if energy > base_energy * (
                1 - float(self.config.measurement["minimum_interesting_energy_fraction"])
            ):
                continue
            if throughput < base_tput * (
                1 - float(self.config.latency_guard["relative_throughput_regression"])
            ):
                continue
            if base_e2e and e2e and e2e > base_e2e * (
                1 + float(self.config.latency_guard["relative_p95_e2e_regression"])
            ):
                continue
            candidates.append((energy, settings))
        return min(candidates, default=(math.inf, baseline), key=lambda item: item[0])[1]

    def confirm_pair(
        self,
        workload: str,
        rate: float | list[float],
        baseline: ServerSettings,
        candidate: ServerSettings,
        *,
        seeds: list[int],
        label: str,
    ) -> ServerSettings:
        if candidate == baseline:
            return baseline
        paired: dict[ServerSettings, list[dict[str, Any]]] = {
            baseline: [],
            candidate: [],
        }
        rates = rate if isinstance(rate, list) else [rate]
        for settings in (baseline, candidate):
            def action(settings: ServerSettings = settings) -> None:
                for offered_rate in rates:
                    for seed in seeds:
                        trace = self.open_trace(
                            workload, offered_rate, seed=seed, confirm=True
                        )
                        paired[settings].append(
                            self.measure(
                                label=(
                                    f"{label}-{settings.key}-rate-{offered_rate:.4f}"
                                    f"-seed-{seed}"
                                ),
                                trace_dir=trace,
                                settings=settings,
                                watchdog=float(
                                    self.config.measurement["confirm_watchdog_seconds"]
                                ),
                            )
                        )

            self.with_server(settings, action, label=label)
        for base_result, candidate_result in zip(
            paired[baseline], paired[candidate], strict=True
        ):
            if not base_result.get("valid") or not candidate_result.get("valid"):
                return baseline
            if float(candidate_result["joules_per_request"]) >= float(
                base_result["joules_per_request"]
            ):
                return baseline
        savings = [
            1
            - float(candidate_result["joules_per_request"])
            / float(base_result["joules_per_request"])
            for base_result, candidate_result in zip(
                paired[baseline], paired[candidate], strict=True
            )
        ]
        if statistics.median(savings) < float(
            self.config.measurement["minimum_interesting_energy_fraction"]
        ):
            return baseline
        return candidate

    def stage1(self) -> ServerSettings:
        self.save_state("stage-1")
        m = self.config.model
        d = self.config.design
        seeds = list(d["seeds"])
        baseline = ServerSettings(
            int(m["baseline_max_running_requests"]),
            int(m["baseline_chunked_prefill_size"]),
            False,
        )
        c0 = self.capacity_scout(
            "BAL",
            baseline,
            list(d["capacity_concurrency"]),
            seed=seeds[0],
            label="s1-capacity",
        )
        knee_results: list[tuple[float, dict[str, Any]]] = []

        def knee_action() -> None:
            for fraction in d["load_fractions"]:
                trace = self.open_trace("BAL", c0 * float(fraction), seed=seeds[0], confirm=False)
                result = self.measure(
                    label=f"s1-knee-{float(fraction):.2f}",
                    trace_dir=trace,
                    settings=baseline,
                    watchdog=float(self.config.measurement["screen_watchdog_seconds"]),
                )
                knee_results.append((float(fraction), result))

        self.with_server(baseline, knee_action, label="s1-knee")
        saturated = [
            fraction
            for fraction, result in knee_results
            if result.get("queue", {}).get("median", 0) not in (None, 0)
            and (result.get("queue", {}).get("nonzero_fraction") or 0) >= 0.20
        ]
        knee_upper = min(saturated, default=1.05)
        self.selected["knee_upper_fraction"] = knee_upper
        load_values = [float(value) for value in d["load_fractions"]]
        upper_index = load_values.index(knee_upper)
        confirm_fractions = load_values[max(0, upper_index - 2) : upper_index + 1]

        def knee_confirm_action() -> None:
            for fraction in confirm_fractions:
                for seed in seeds:
                    trace = self.open_trace("BAL", c0 * fraction, seed=seed, confirm=True)
                    self.measure(
                        label=f"s1-knee-confirm-{fraction:.2f}-seed-{seed}",
                        trace_dir=trace,
                        settings=baseline,
                        watchdog=float(
                            self.config.measurement["confirm_watchdog_seconds"]
                        ),
                    )

        self.with_server(baseline, knee_confirm_action, label="s1-knee-confirm")

        settings_values = [
            ServerSettings(int(value), baseline.chunked_prefill_size, False)
            for value in d["max_running_requests"]
        ]
        screens = self.screen_settings(
            "BAL", c0 * 0.90, settings_values, seed=seeds[0], label="s1-mrr-screen"
        )
        candidate = self.candidate_from_screen(screens, baseline)
        selected = self.confirm_pair(
            "BAL",
            [c0 * 0.90, c0 * 1.05],
            baseline,
            candidate,
            seeds=seeds,
            label="s1-mrr-confirm",
        )
        self.selected["max_running_requests"] = selected.max_running_requests
        self.save_state("stage-1-complete")
        return selected

    def stage2(self, stage1: ServerSettings) -> ServerSettings:
        self.save_state("stage-2")
        d = self.config.design
        seeds = list(d["seeds"])
        baseline = ServerSettings(
            stage1.max_running_requests,
            int(self.config.model["baseline_chunked_prefill_size"]),
            False,
        )
        c_pf = self.capacity_scout(
            "PF",
            baseline,
            list(d["short_capacity_concurrency"]),
            seed=seeds[0],
            label="s2-capacity",
        )
        values = [
            ServerSettings(stage1.max_running_requests, int(chunk), False)
            for chunk in d["chunked_prefill_sizes"]
        ]
        screens = self.screen_settings(
            "PF", c_pf * 0.85, values, seed=seeds[0], label="s2-chunk-screen"
        )
        candidate = self.candidate_from_screen(screens, baseline)
        selected = self.confirm_pair(
            "PF",
            c_pf * 0.90,
            baseline,
            candidate,
            seeds=seeds,
            label="s2-chunk-confirm",
        )
        original_mrr = int(self.config.model["baseline_max_running_requests"])
        original_chunk = int(self.config.model["baseline_chunked_prefill_size"])
        interaction = list(
            dict.fromkeys(
                (
                    ServerSettings(original_mrr, original_chunk, False),
                    ServerSettings(stage1.max_running_requests, original_chunk, False),
                    ServerSettings(original_mrr, selected.chunked_prefill_size, False),
                    selected,
                )
            )
        )
        self.screen_settings(
            "PF",
            c_pf * 0.90,
            interaction,
            seed=seeds[0],
            label="s2-interaction",
        )
        self.selected["chunked_prefill_size"] = selected.chunked_prefill_size
        self.save_state("stage-2-complete")
        return selected

    def stage3(self, selected: ServerSettings) -> None:
        self.save_state("stage-3")
        d = self.config.design
        seeds = list(d["seeds"])
        cache_off = ServerSettings(
            selected.max_running_requests, selected.chunked_prefill_size, False
        )
        cache_on = ServerSettings(
            selected.max_running_requests, selected.chunked_prefill_size, True
        )
        c_px = self.capacity_scout(
            "PX0",
            cache_off,
            list(d["short_capacity_concurrency"]),
            seed=seeds[0],
            label="s3-capacity",
        )
        for settings in (cache_off, cache_on):
            def screen_action(settings: ServerSettings = settings) -> None:
                for workload in ("PX0", "PX50", "PX87"):
                    trace = self.open_trace(
                        workload, c_px * 0.85, seed=seeds[0], confirm=False
                    )
                    self.measure(
                        label=f"s3-cold-{workload}-{settings.key}",
                        trace_dir=trace,
                        settings=settings,
                        watchdog=float(self.config.measurement["screen_watchdog_seconds"]),
                    )

            self.with_server(settings, screen_action, label="s3-cold-screen")
        for settings in (cache_off, cache_on):
            def confirm_action(settings: ServerSettings = settings) -> None:
                for workload in ("PX0", "PX87"):
                    for seed in seeds:
                        trace = self.open_trace(workload, c_px * 0.90, seed=seed, confirm=True)
                        self.measure(
                            label=f"s3-confirm-{workload}-{settings.key}-seed-{seed}",
                            trace_dir=trace,
                            settings=settings,
                            watchdog=float(
                                self.config.measurement["confirm_watchdog_seconds"]
                            ),
                        )

            self.with_server(settings, confirm_action, label="s3-confirm")
        for settings in (cache_off, cache_on):
            def warm_action(settings: ServerSettings = settings) -> None:
                prime = self.trace(
                    "PX87",
                    seed=seeds[0] + 700,
                    kind="closed",
                    count=int(d["prefix_groups"]),
                )
                prime_output = (
                    self.output_root / "warmup" / f"{settings.key}-requests.jsonl"
                )
                prime_output.parent.mkdir(parents=True, exist_ok=True)
                prime_summary = asyncio.run(
                    replay_trace(
                        prime,
                        base_url=self.base_url,
                        output_path=prime_output,
                        watchdog_seconds=float(
                            self.config.measurement["screen_watchdog_seconds"]
                        ),
                        max_concurrency=1,
                    )
                )
                if prime_summary["success_count"] != prime_summary["request_count"]:
                    raise RuntimeError(f"warm-cache priming failed for {settings.key}")
                trace = self.open_trace("PX87", c_px * 0.85, seed=seeds[0], confirm=False)
                self.measure(
                    label=f"s3-warm-PX87-{settings.key}",
                    trace_dir=trace,
                    settings=settings,
                    watchdog=float(self.config.measurement["screen_watchdog_seconds"]),
                    flush=False,
                )

            self.with_server(settings, warm_action, label="s3-warm")
        self.save_state("stage-3-complete")

    def stage4(self, selected: ServerSettings) -> None:
        self.save_state("stage-4")
        d = self.config.design
        holdout = list(d["holdout_seeds"])
        baseline = ServerSettings(
            int(self.config.model["baseline_max_running_requests"]),
            int(self.config.model["baseline_chunked_prefill_size"]),
            False,
        )
        c_dec = self.capacity_scout(
            "DEC",
            baseline,
            list(d["short_capacity_concurrency"]),
            seed=holdout[0],
            label="s4-dec-capacity",
        )
        capacities = {
            "BAL": self.capacities["BAL"],
            "PF": self.capacities["PF"],
            "DEC": c_dec,
        }
        for settings in dict.fromkeys((baseline, selected)):
            def action(settings: ServerSettings = settings) -> None:
                for workload, capacity in capacities.items():
                    for seed in holdout:
                        trace = self.open_trace(
                            workload, capacity * 0.85, seed=seed, confirm=True
                        )
                        self.measure(
                            label=f"s4-{workload}-{settings.key}-seed-{seed}",
                            trace_dir=trace,
                            settings=settings,
                            watchdog=float(
                                self.config.measurement["confirm_watchdog_seconds"]
                            ),
                        )

            self.with_server(settings, action, label="s4-regimes")
        self.save_state("stage-4-complete")

    def write_summary(self) -> None:
        if not self.output_root.exists():
            return
        columns = [
            "run_id",
            "label",
            "valid",
            "max_running_requests",
            "chunked_prefill_size",
            "radix_cache",
            "request_count",
            "success_count",
            "throughput_requests_s",
            "throughput_output_tokens_s",
            "energy_j",
            "joules_per_request",
            "joules_per_1000_output_tokens",
            "p95_ttft_ms",
            "p95_tpot_ms",
            "p95_e2e_ms",
            "queue_median",
            "queue_nonzero_fraction",
        ]
        rows: list[dict[str, Any]] = []
        for run_id in sorted(self.results):
            summary = self.results[run_id]
            settings = summary.get("settings", {})
            latency = summary.get("latency_ms", {})
            queue = summary.get("queue", {})
            rows.append(
                {
                    "run_id": run_id,
                    "label": summary.get("label"),
                    "valid": summary.get("valid"),
                    "max_running_requests": settings.get("max_running_requests"),
                    "chunked_prefill_size": settings.get("chunked_prefill_size"),
                    "radix_cache": settings.get("radix_cache"),
                    "request_count": summary.get("request_count"),
                    "success_count": summary.get("success_count"),
                    "throughput_requests_s": summary.get("throughput_requests_s"),
                    "throughput_output_tokens_s": summary.get(
                        "throughput_output_tokens_s"
                    ),
                    "energy_j": summary.get("energy_j"),
                    "joules_per_request": summary.get("joules_per_request"),
                    "joules_per_1000_output_tokens": summary.get(
                        "joules_per_1000_output_tokens"
                    ),
                    "p95_ttft_ms": latency.get("ttft_ms", {}).get("p95"),
                    "p95_tpot_ms": latency.get("tpot_ms", {}).get("p95"),
                    "p95_e2e_ms": latency.get("e2e_ms", {}).get("p95"),
                    "queue_median": queue.get("median"),
                    "queue_nonzero_fraction": queue.get("nonzero_fraction"),
                }
            )
        summary_csv = self.output_root / "summary.csv"
        with summary_csv.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        summary_csv.with_suffix(".tmp").replace(summary_csv)
        valid = sum(bool(row["valid"]) for row in rows)
        markdown = [
            "# Campaign summary",
            "",
            f"- Completed cells: {len(rows)}",
            f"- Valid cells: {valid}",
            f"- Invalid cells: {len(rows) - valid}",
            f"- Capacities (requests/s): `{json.dumps(self.capacities, sort_keys=True)}`",
            f"- Selected settings: `{json.dumps(self.selected, sort_keys=True)}`",
            "",
            "See `summary.csv` for per-cell energy, latency, throughput, and queue results.",
            "Screening cells are descriptive; deployment requires the separately specified new-seed confirmation.",
            "",
        ]
        temporary = self.output_root / "summary.md.tmp"
        temporary.write_text("\n".join(markdown), encoding="utf-8")
        temporary.replace(self.output_root / "summary.md")

    def run(self) -> None:
        try:
            self.prepare()
            stage1 = self.stage1()
            selected = self.stage2(stage1)
            self.stage3(selected)
            self.stage4(selected)
            self.save_state("complete")
            self.event("campaign-complete")
        except DeadlineReached as exc:
            self.event("campaign-deadline", error=str(exc))
            self.save_state("deadline-stopped")
            raise
        except Exception as exc:
            self.event("campaign-failed", error=f"{type(exc).__name__}: {exc}")
            self.save_state("failed")
            raise
        finally:
            self.write_summary()
