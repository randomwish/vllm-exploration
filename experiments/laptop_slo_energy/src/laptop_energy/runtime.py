from __future__ import annotations

import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .binaries import resolve_binary
from .config import CampaignConfig, Treatment, write_json
from .energy import efficiency_metrics, parse_perf_stat
from .ebpf import summarize_phase_alignment, summarize_runqlat
from .guidellm import build_command, evaluate_slo, summarize_report
from .plan import PlannedCell, calibration_plan, policy_plan
from .preflight import inspect


def _progress(message: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}{suffix}",
        file=sys.stderr,
        flush=True,
    )


def _bpf_clock_s() -> float:
    """Return the suspend-aware clock used by bpftrace's nsecs builtin."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def _execution_validity(validity: dict[str, Any]) -> bool:
    explicit = validity.get("execution_valid")
    if isinstance(explicit, bool):
        return explicit
    checks = validity.get("checks", {})
    if not isinstance(checks, dict) or not checks:
        return bool(validity.get("valid"))
    required = [
        value
        for key, value in checks.items()
        if key != "minimum_successful_requests" and isinstance(value, bool)
    ]
    return bool(required) and all(required)


def _campaign_completion_status(summaries: list[dict[str, Any]]) -> str:
    if not all(_execution_validity(item.get("validity", {})) for item in summaries):
        return "complete_invalid"
    if not all(
        bool(
            item.get("validity", {})
            .get("checks", {})
            .get("minimum_successful_requests")
        )
        for item in summaries
    ):
        return "complete_evidence_limited"
    return "complete"


class ServerProcess(AbstractContextManager["ServerProcess"]):
    def __init__(
        self, config: CampaignConfig, treatment: Treatment, output_dir: Path
    ) -> None:
        self.config = config
        self.treatment = treatment
        self.output_dir = output_dir
        self.process: subprocess.Popen[str] | None = None
        self.log_stream: Any = None

    @property
    def command(self) -> list[str]:
        model_path = str(self.config.model.get("path", "")).strip()
        model_args = (
            ["-m", str(Path(model_path).expanduser())]
            if model_path
            else ["-hf", str(self.config.model["hf_ref"])]
        )
        configured_binary = str(self.config.server["binary"])
        binary = resolve_binary(configured_binary) or configured_binary
        return [
            binary,
            *model_args,
            "--host",
            str(self.config.server["host"]),
            "--port",
            str(self.config.server["port"]),
            "-c",
            str(self.config.server["context_size"]),
            "-np",
            str(self.config.server["parallel_slots"]),
            "-t",
            str(self.treatment.threads),
            "-tb",
            str(self.treatment.batch_threads),
            *self.treatment.server_args,
            *[str(value) for value in self.config.server.get("extra_args", [])],
        ]

    def __enter__(self) -> "ServerProcess":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _progress(
            "starting llama-server",
            treatment=self.treatment.name,
            threads=self.treatment.threads,
        )
        self.log_stream = (self.output_dir / "server.log").open(
            "w", encoding="utf-8", buffering=1
        )
        write_json(
            self.output_dir / "server-command.json",
            {"argv": self.command, "treatment": asdict(self.treatment)},
        )
        self.process = subprocess.Popen(
            self.command,
            text=True,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self._wait_ready()
        except Exception:
            self.stop()
            raise
        _progress(
            "llama-server ready",
            treatment=self.treatment.name,
            pid=self.process.pid,
        )
        return self

    def _wait_ready(self) -> None:
        timeout = float(self.config.server["ready_timeout_seconds"])
        deadline = time.monotonic() + timeout
        url = (
            f"http://{self.config.server['host']}:{self.config.server['port']}/health"
        )
        last_error = "server did not respond"
        next_heartbeat = time.monotonic() + 15.0
        started = time.monotonic()
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited before readiness with {self.process.returncode}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
            if time.monotonic() >= next_heartbeat:
                _progress(
                    "waiting for llama-server readiness",
                    treatment=self.treatment.name,
                    elapsed_s=round(time.monotonic() - started),
                )
                next_heartbeat += 15.0
            time.sleep(0.5)
        raise TimeoutError(f"llama-server readiness timed out: {last_error}")

    def warmup(self) -> None:
        _progress("starting server warm-up", treatment=self.treatment.name)
        url = (
            f"http://{self.config.server['host']}:{self.config.server['port']}"
            "/v1/chat/completions"
        )
        payload = json.dumps(
            {
                "model": self.config.model["id"],
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "temperature": 0,
                "max_tokens": 8,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"warm-up request returned HTTP {response.status}")
        _progress("server warm-up complete", treatment=self.treatment.name)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGINT)
                self.process.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


class HostSampler:
    def __init__(self, pid: int, output: Path, interval: float) -> None:
        self.pid = pid
        self.output = output
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.sample_count = 0
        self.sample_times: list[float] = []

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(
        self,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval * 3))
        window_sample_count = (
            sum(
                window_start <= sample_time <= window_end
                for sample_time in self.sample_times
            )
            if window_start is not None and window_end is not None
            else 0
        )
        return {
            "sample_count": self.sample_count,
            "window_sample_count": window_sample_count,
            "valid": window_sample_count > 0,
            "interval_seconds": self.interval,
        }

    def _run(self) -> None:
        previous = time.monotonic()
        with self.output.open("w", encoding="utf-8", buffering=1) as stream:
            while not self.stop_event.is_set():
                now = time.monotonic()
                row = {
                    "unix_s": time.time(),
                    "monotonic_s": now,
                    "interval_s": now - previous,
                    "process": _process_snapshot(self.pid),
                    "pressure": _pressure_snapshot(),
                    "cpu_frequency_khz": _cpu_frequencies(),
                    "thermal_millicelsius": _thermal_temperatures(),
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                self.sample_count += 1
                self.sample_times.append(float(row["unix_s"]))
                previous = now
                self.stop_event.wait(self.interval)


class EbpfCollector:
    def __init__(self, probe: Path, output_dir: Path) -> None:
        self.probe = probe
        self.output_dir = output_dir
        self.process: subprocess.Popen[str] | None = None
        self.stdout: Any = None
        self.stderr: Any = None
        self.started = False
        self.started_unix_s: float | None = None
        self.started_bpf_clock_s: float | None = None

    def start(self) -> None:
        self.stdout = (self.output_dir / "ebpf-runqlat.txt").open(
            "w", encoding="utf-8", buffering=1
        )
        self.stderr = (self.output_dir / "ebpf-stderr.log").open(
            "w", encoding="utf-8", buffering=1
        )
        self.process = subprocess.Popen(
            ["sudo", "-n", "bpftrace", str(self.probe)],
            text=True,
            stdout=self.stdout,
            stderr=self.stderr,
        )
        self.started = True
        time.sleep(0.5)
        self.started_unix_s = time.time()
        self.started_bpf_clock_s = _bpf_clock_s()

    def stop(
        self,
        measurement_start_bpf_clock: float | None = None,
        measurement_end_bpf_clock: float | None = None,
    ) -> dict[str, Any]:
        returncode: int | None = None
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.send_signal(signal.SIGINT)
                    self.process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
            returncode = self.process.returncode
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
        total_samples = 0
        measurement_samples = 0
        selected_buckets = 0
        try:
            output = (self.output_dir / "ebpf-runqlat.txt").read_text(
                encoding="utf-8"
            )
            for match in re.finditer(
                r"@samples_250ms\[(\d+)\]:\s+(\d+)", output
            ):
                bucket = int(match.group(1))
                count = int(match.group(2))
                total_samples += count
                bucket_center = (bucket + 0.5) * 0.25
                if (
                    measurement_start_bpf_clock is not None
                    and measurement_end_bpf_clock is not None
                    and measurement_start_bpf_clock
                    <= bucket_center
                    <= measurement_end_bpf_clock
                ):
                    measurement_samples += count
                    selected_buckets += 1
        except (OSError, ValueError):
            output = ""
        runqlat = summarize_runqlat(
            output,
            measurement_start_bpf_clock,
            measurement_end_bpf_clock,
        )
        if runqlat.get("mode") == "one-second-aggregates":
            measurement_samples = int(runqlat["samples"])
            selected_buckets = int(runqlat["selected_1s_buckets"])
            total_samples = sum(
                int(value)
                for value in re.findall(
                    r"@runqlat_count_1s\[\d+\]:\s+(\d+)", output
                )
            )
        result = {
            "started": self.started,
            "started_unix_s": self.started_unix_s,
            "started_bpf_clock_s": self.started_bpf_clock_s,
            "returncode": returncode,
            "samples": measurement_samples,
            "total_samples_including_prelude": total_samples,
            "selected_time_buckets": selected_buckets,
            "runqlat_us": runqlat,
            "valid": self.started and returncode == 0 and measurement_samples > 0,
            "probe": str(self.probe),
            "boundary": (
                "one-second monotonic buckets whose centers fall inside the measured "
                "policy window; raw output retains the pre-arrival attachment period"
            ),
            "prelude_seconds": (
                measurement_start_bpf_clock - self.started_bpf_clock_s
                if measurement_start_bpf_clock is not None
                and self.started_bpf_clock_s is not None
                else None
            ),
        }
        write_json(self.output_dir / "ebpf.json", result)
        return result


class PerfEnergyCollector:
    def __init__(self, config: CampaignConfig, output_dir: Path) -> None:
        energy = config.measurement["energy"]
        events = [
            str(energy["primary_event"]),
            *map(str, energy["diagnostic_events"]),
        ]
        perf = resolve_binary("perf") or "perf"
        self.command = [
            "sudo",
            "-n",
            perf,
            "stat",
            "-a",
            "-x",
            ",",
            "--no-big-num",
            "--delay=-1",
            "--control",
            "fd:0",
            "-e",
            ",".join(events),
            "--",
            "sleep",
            "86400",
        ]
        self.output = output_dir / "perf-stderr.log"
        self.process: subprocess.Popen[str] | None = None
        self.enabled = False

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.1)
        if self.process.poll() is not None:
            _, stderr = self.process.communicate()
            self.output.write_text(stderr, encoding="utf-8")
            raise RuntimeError(
                f"perf collector exited before measurement: {self.process.returncode}"
            )

    def enable(self) -> None:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("perf collector is unavailable at measurement start")
        if self.process.stdin is None:
            raise RuntimeError("perf control channel is unavailable")
        self.process.stdin.write("enable\n")
        self.process.stdin.flush()
        self.enabled = True

    def disable(self) -> None:
        if (
            self.enabled
            and self.process is not None
            and self.process.poll() is None
            and self.process.stdin is not None
        ):
            self.process.stdin.write("disable\n")
            self.process.stdin.flush()
            self.enabled = False

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"started": False, "returncode": None, "stderr": ""}
        if self.process.poll() is None:
            self.disable()
            self.process.send_signal(signal.SIGINT)
        try:
            _, stderr = self.process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                _, stderr = self.process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                _, stderr = self.process.communicate(timeout=5)
        self.output.write_text(stderr, encoding="utf-8")
        return {
            "started": True,
            "control_enabled": True,
            "returncode": self.process.returncode,
            "stderr": stderr,
            "argv": self.command,
        }


class SudoKeepalive:
    def __init__(self, enabled: bool, interval_seconds: float = 30.0) -> None:
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.valid = not enabled

    def start(self) -> None:
        if not self.enabled:
            return
        result = subprocess.run(
            ["sudo", "-n", "-v"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.valid = result.returncode == 0
        if not self.valid:
            raise RuntimeError("sudo keepalive could not validate cached credentials")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        _progress("sudo credential keepalive started", interval_s=self.interval_seconds)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            result = subprocess.run(
                ["sudo", "-n", "-v"],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                self.valid = False
                _progress("sudo credential keepalive failed")
                return

    def ensure_ready(self) -> None:
        if self.enabled and not self.valid:
            raise RuntimeError(
                "sudo credential keepalive failed; run sudo -v before resuming"
            )

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _process_snapshot(pid: int) -> dict[str, Any]:
    output: dict[str, Any] = {"pid": pid, "available": False}
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = stat.rfind(")")
        fields = stat[close + 2 :].split()
        output.update(
            {
                "available": True,
                "minor_faults": int(fields[7]),
                "major_faults": int(fields[9]),
                "user_ticks": int(fields[11]),
                "system_ticks": int(fields[12]),
                "resident_pages": int(fields[21]),
                "last_cpu": int(fields[36]),
            }
        )
    except (OSError, ValueError, IndexError):
        return output
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("Threads:"):
                output["threads"] = int(line.split()[1])
            elif line.startswith("voluntary_ctxt_switches:"):
                output["voluntary_context_switches"] = int(line.split()[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                output["involuntary_context_switches"] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return output


def _pressure_snapshot() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in ("cpu", "memory", "io"):
        try:
            values[name] = Path(f"/proc/pressure/{name}").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            values[name] = None
    return values


def _cpu_frequencies() -> dict[str, int]:
    values: dict[str, int] = {}
    for path in sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
    ):
        try:
            values[path.parts[-3]] = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return values


def _thermal_temperatures() -> dict[str, int]:
    values: dict[str, int] = {}
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            values[path.parent.name] = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return values


def _drain_stream(
    stream: Any,
    output: Path,
    marker: threading.Event | None = None,
) -> None:
    with output.open("w", encoding="utf-8", buffering=1) as target:
        for line in iter(stream.readline, ""):
            target.write(line)
            if marker is not None and "Setup complete, starting benchmarks" in line:
                marker.set()
    stream.close()


def _wait_for_file_marker(
    path: Path,
    offset: int,
    marker: str,
    process: subprocess.Popen[str],
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    next_heartbeat = started + 15.0
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with path.open("r", encoding="utf-8") as stream:
                stream.seek(offset)
                if marker in stream.read():
                    return True
        except OSError:
            pass
        if time.monotonic() >= next_heartbeat:
            _progress(
                "waiting for first benchmark request",
                elapsed_s=round(time.monotonic() - started),
            )
            next_heartbeat += 15.0
        time.sleep(0.02)
    return False


class CampaignRunner:
    def __init__(
        self,
        config: CampaignConfig,
        output_root: Path,
        *,
        use_energy: bool = True,
        use_ebpf: bool = True,
        resume_root: Path | None = None,
    ) -> None:
        self.config = config
        self.output_root = output_root.resolve()
        self.use_energy = use_energy and bool(config.measurement["energy"]["enabled"])
        self.use_ebpf = use_ebpf and bool(config.measurement["ebpf"]["enabled"])
        self.resume_root = resume_root.resolve() if resume_root is not None else None
        self.sudo_keepalive = SudoKeepalive(self.use_energy or self.use_ebpf)

    def run(self) -> Path:
        resumed = self.resume_root is not None
        if resumed:
            assert self.resume_root is not None
            root = self.resume_root
            if not root.is_dir():
                raise RuntimeError(f"resume directory does not exist: {root}")
            stored_config = json.loads(
                (root / "campaign.json").read_text(encoding="utf-8")
            )
            if stored_config != self.config.raw:
                raise RuntimeError("resume configuration does not match campaign.json")
        else:
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            root = self.output_root / f"{self.config.raw['campaign_name']}-{timestamp}"
            root.mkdir(parents=True, exist_ok=False)
            write_json(root / "campaign.json", self.config.raw)
        _progress(
            "campaign resumed" if resumed else "campaign started",
            campaign=self.config.raw["campaign_name"],
            output=root,
        )
        try:
            preflight = inspect(
                self.config,
                privileged=self.use_energy or self.use_ebpf,
                require_energy=self.use_energy,
                require_ebpf=self.use_ebpf,
            )
            write_json(root / "preflight.json", preflight)
            if not preflight["ready_to_execute"]:
                raise RuntimeError(
                    "preflight failed: " + "; ".join(preflight["errors"])
                )
            self.sudo_keepalive.start()
            if resumed:
                capacities = self._load_resume_capacities(root)
                _progress("reusing completed calibration", capacities=capacities)
            else:
                capacities = self._run_calibration(root)
                write_json(root / "capacities.json", capacities)
            summaries: list[dict[str, Any]] = []
            policy_cells = policy_plan(self.config, capacities)
            _progress(
                "calibration phase complete",
                policy_cells=len(policy_cells),
            )
            for index, cell in enumerate(policy_cells, start=1):
                existing = self._load_resumable_summary(root, cell) if resumed else None
                if existing is not None:
                    summaries.append(existing)
                    _progress(
                        "reusing completed policy cell",
                        cell=f"{index}/{len(policy_cells)}",
                        workload=cell.workload,
                        treatment=cell.treatment,
                    )
                    continue
                cell_dir = root / "policy" / cell.cell_id
                if resumed and cell_dir.exists():
                    _progress(
                        "replacing incomplete or contaminated policy cell",
                        cell=f"{index}/{len(policy_cells)}",
                        cell_id=cell.cell_id,
                    )
                    shutil.rmtree(cell_dir)
                self.sudo_keepalive.ensure_ready()
                summaries.append(
                    self._run_policy_cell(root, cell, index, len(policy_cells))
                )
            self._write_campaign_summary(root, summaries)
        except KeyboardInterrupt:
            write_json(
                root / "FINAL_STATUS.json",
                {
                    "status": "interrupted",
                    "finished_unix_s": time.time(),
                    "resumable": True,
                },
            )
            _progress("campaign interrupted; completed cells can be resumed", output=root)
            raise
        except Exception as exc:
            write_json(
                root / "FINAL_STATUS.json",
                {
                    "status": "failed",
                    "finished_unix_s": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "resumable": True,
                },
            )
            raise
        finally:
            self.sudo_keepalive.stop()
        valid_cells = sum(item["validity"]["valid"] for item in summaries)
        execution_valid_cells = sum(
            _execution_validity(item["validity"]) for item in summaries
        )
        status = _campaign_completion_status(summaries)
        write_json(
            root / "FINAL_STATUS.json",
            {
                "status": status,
                "finished_unix_s": time.time(),
                "valid_policy_cells": valid_cells,
                "execution_valid_policy_cells": execution_valid_cells,
                "total_policy_cells": len(summaries),
                "resumed": resumed,
            },
        )
        _progress(
            "campaign finished",
            status=status,
            execution_valid_cells=f"{execution_valid_cells}/{len(summaries)}",
            sample_sufficient_cells=f"{valid_cells}/{len(summaries)}",
            output=root,
        )
        return root

    def _load_resume_capacities(self, root: Path) -> dict[str, float]:
        path = root / "capacities.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            capacities = {
                workload.name: float(value[workload.name])
                for workload in self.config.workloads
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "resume requires a complete capacities.json; start a new run"
            ) from exc
        if any(rate <= 0 for rate in capacities.values()):
            raise RuntimeError("resume capacities must all be positive")
        return capacities

    def _load_resumable_summary(
        self, root: Path, cell: PlannedCell
    ) -> dict[str, Any] | None:
        path = root / "policy" / cell.cell_id / "summary.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if summary.get("cell_id") != cell.cell_id:
            return None
        if not _execution_validity(summary.get("validity", {})):
            return None
        started_monotonic = summary.get("started_monotonic_s")
        finished_monotonic = summary.get("finished_monotonic_s")
        started_unix = summary.get("started_unix_s")
        finished_unix = summary.get("finished_unix_s")
        if not all(
            isinstance(value, (int, float))
            for value in (
                started_monotonic,
                finished_monotonic,
                started_unix,
                finished_unix,
            )
        ):
            return None
        active_duration = float(finished_monotonic) - float(started_monotonic)
        wall_duration = float(finished_unix) - float(started_unix)
        suspend_gap = max(0.0, wall_duration - active_duration)
        maximum_suspend_gap = float(
            self.config.measurement.get("max_suspend_gap_seconds", 5.0)
        )
        if active_duration < cell.duration_seconds * 0.95:
            return None
        if suspend_gap > maximum_suspend_gap:
            return None
        return summary

    def _run_calibration(self, root: Path) -> dict[str, float]:
        baseline = self.config.baseline_treatment
        capacities: dict[str, float] = {}
        server_root = root / "calibration" / "server"
        cells = calibration_plan(self.config)
        with ServerProcess(self.config, baseline, server_root) as server:
            server.warmup()
            for index, cell in enumerate(cells, start=1):
                _progress(
                    "calibration cell started",
                    cell=f"{index}/{len(cells)}",
                    workload=cell.workload,
                    duration_s=cell.duration_seconds,
                )
                cell_dir = root / "calibration" / cell.cell_id
                cell_dir.mkdir(parents=True, exist_ok=False)
                write_json(cell_dir / "cell.json", cell.as_dict())
                command = build_command(self.config, cell, cell_dir)
                write_json(cell_dir / "guidellm-command.json", {"argv": command})
                calibration_started = time.monotonic()
                with (cell_dir / "stdout.log").open(
                    "w", encoding="utf-8"
                ) as stdout, (cell_dir / "stderr.log").open(
                    "w", encoding="utf-8"
                ) as stderr:
                    process = subprocess.Popen(
                        command,
                        text=True,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    try:
                        next_heartbeat = time.monotonic() + 30.0
                        while process.poll() is None:
                            if time.monotonic() >= next_heartbeat:
                                _progress(
                                    "calibration cell running",
                                    cell=f"{index}/{len(cells)}",
                                    workload=cell.workload,
                                    elapsed_s=round(
                                        time.monotonic() - calibration_started
                                    ),
                                )
                                next_heartbeat += 30.0
                            time.sleep(0.5)
                    except BaseException:
                        if process.poll() is None:
                            process.send_signal(signal.SIGINT)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                        raise
                    returncode = process.returncode
                if returncode != 0:
                    raise RuntimeError(
                        f"calibration failed for {cell.cell_id}: {returncode}"
                    )
                summary = summarize_report(cell_dir / "guidellm.json")
                rate = summary.get("completed_requests_per_second")
                if not isinstance(rate, (int, float)) or rate <= 0:
                    raise RuntimeError(
                        f"calibration produced no capacity for {cell.workload}"
                    )
                capacities[cell.workload] = float(rate)
                write_json(cell_dir / "summary.json", summary)
                _progress(
                    "calibration cell complete",
                    cell=f"{index}/{len(cells)}",
                    workload=cell.workload,
                    capacity_rps=round(float(rate), 6),
                    elapsed_s=round(time.monotonic() - calibration_started),
                )
        return capacities

    def _run_policy_cell(
        self,
        root: Path,
        cell: PlannedCell,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        _progress(
            "policy cell started",
            cell=f"{index}/{total}",
            workload=cell.workload,
            treatment=cell.treatment,
            rate_rps=round(float(cell.offered_rate_requests_s or 0), 6),
            duration_s=cell.duration_seconds,
        )
        cell_dir = root / "policy" / cell.cell_id
        cell_dir.mkdir(parents=True, exist_ok=False)
        write_json(cell_dir / "cell.json", cell.as_dict())
        treatment = next(
            item for item in self.config.treatments if item.name == cell.treatment
        )
        with ServerProcess(self.config, treatment, cell_dir) as server:
            server.warmup()
            assert server.process is not None
            assert server.log_stream is not None
            server.log_stream.flush()
            server_log = cell_dir / "server.log"
            server_log_offset = server_log.stat().st_size
            host = HostSampler(
                server.process.pid,
                cell_dir / "host.jsonl",
                float(self.config.measurement["host_sample_seconds"]),
            )
            ebpf: EbpfCollector | None = None
            perf = PerfEnergyCollector(self.config, cell_dir) if self.use_energy else None
            command = build_command(self.config, cell, cell_dir)
            write_json(
                cell_dir / "guidellm-command.json",
                {
                    "argv": command,
                    "executed_argv": command,
                    "perf_argv": perf.command if perf is not None else None,
                },
            )
            marker = threading.Event()
            process = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_thread = threading.Thread(
                target=_drain_stream,
                args=(process.stdout, cell_dir / "stdout.log", marker),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_drain_stream,
                args=(process.stderr, cell_dir / "stderr.log"),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            host_started = False
            collectors_prepared = False
            marker_detected = False
            request_marker_detected = False
            collectors_stopped = False
            started = time.time()
            finished = started
            started_monotonic: float | None = None
            finished_monotonic: float | None = None
            started_bpf_clock: float | None = None
            finished_bpf_clock: float | None = None
            host_status: dict[str, Any] = {"valid": False, "sample_count": 0}
            ebpf_status: dict[str, Any] = {"valid": None}
            perf_status: dict[str, Any] = {"started": False, "stderr": ""}
            try:
                deadline = time.monotonic() + float(
                    self.config.server["ready_timeout_seconds"]
                )
                while process.poll() is None and not marker.wait(0.1):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "GuideLLM did not emit its benchmark-start marker"
                        )
                marker_detected = marker.is_set()
                if marker_detected and process.poll() is None:
                    _progress(
                        "GuideLLM setup complete; preparing collectors",
                        cell=f"{index}/{total}",
                    )
                    if self.use_ebpf:
                        probe = (
                            self.config.path.parent
                            / str(self.config.measurement["ebpf"]["probe"])
                        ).resolve()
                        ebpf = EbpfCollector(probe, cell_dir)
                        ebpf.start()
                    host.start()
                    host_started = True
                    if perf is not None:
                        perf.start()
                    collectors_prepared = True
                    _progress(
                        "collectors ready; waiting for first request",
                        cell=f"{index}/{total}",
                    )
                    request_marker_detected = _wait_for_file_marker(
                        server_log,
                        server_log_offset,
                        "processing task",
                        process,
                        float(self.config.server["ready_timeout_seconds"]),
                    )
                if request_marker_detected and process.poll() is None:
                    if perf is not None:
                        perf.enable()
                    started = time.time()
                    started_monotonic = time.monotonic()
                    started_bpf_clock = _bpf_clock_s()
                    _progress(
                        "measurement started",
                        cell=f"{index}/{total}",
                        duration_s=cell.duration_seconds,
                    )
                    window_deadline = started_monotonic + cell.duration_seconds
                    progress_interval = max(
                        15.0, min(60.0, cell.duration_seconds / 4.0)
                    )
                    next_progress = started_monotonic + progress_interval
                    while process.poll() is None:
                        now = time.monotonic()
                        remaining = window_deadline - now
                        if remaining <= 0:
                            break
                        if now >= next_progress:
                            elapsed = now - started_monotonic
                            _progress(
                                "measurement running",
                                cell=f"{index}/{total}",
                                progress=f"{min(100, round(elapsed / cell.duration_seconds * 100))}%",
                                elapsed_s=round(elapsed),
                                remaining_s=round(remaining),
                            )
                            next_progress += progress_interval
                        time.sleep(min(0.05, remaining))
                    finished = time.time()
                    finished_monotonic = time.monotonic()
                    finished_bpf_clock = _bpf_clock_s()
                    if perf is not None:
                        perf.disable()
                    if perf is not None and perf.process is not None:
                        perf_status = perf.stop()
                    if host_started:
                        host_status = host.stop(started, finished)
                    if ebpf is not None:
                        ebpf_status = ebpf.stop(
                            started_bpf_clock, finished_bpf_clock
                        )
                    collectors_stopped = True
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                        process.wait(timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                if not collectors_stopped:
                    finished = time.time()
                    if perf is not None and perf.process is not None:
                        perf_status = perf.stop()
                    if host_started:
                        host_status = host.stop(
                            started if request_marker_detected else None,
                            finished if request_marker_detected else None,
                        )
                    if ebpf is not None:
                        ebpf_status = ebpf.stop(
                            started_bpf_clock if request_marker_detected else None,
                            finished_bpf_clock if request_marker_detected else None,
                        )
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)

        measurement_window = {
            "marker": "GuideLLM Setup complete, starting benchmarks",
            "marker_detected": marker_detected,
            "collectors_prepared": collectors_prepared,
            "request_marker": "llama-server processing task",
            "request_marker_detected": request_marker_detected,
            "target_duration_s": cell.duration_seconds,
            "started_unix_s": started,
            "finished_unix_s": finished,
            "started_monotonic_s": started_monotonic,
            "finished_monotonic_s": finished_monotonic,
            "started_bpf_clock_s": started_bpf_clock,
            "finished_bpf_clock_s": finished_bpf_clock,
            "wall_duration_s": (
                finished - started if request_marker_detected else None
            ),
            "duration_s": (
                finished_monotonic - started_monotonic
                if request_marker_detected
                and started_monotonic is not None
                and finished_monotonic is not None
                else None
            ),
        }
        wall_duration = measurement_window["wall_duration_s"]
        active_duration = measurement_window["duration_s"]
        measurement_window["suspend_gap_s"] = (
            max(0.0, float(wall_duration) - float(active_duration))
            if isinstance(wall_duration, (int, float))
            and isinstance(active_duration, (int, float))
            else None
        )

        guide: dict[str, Any] = {}
        guide_error: str | None = None
        try:
            guide = summarize_report(cell_dir / "guidellm.json")
        except Exception as exc:
            guide_error = f"{type(exc).__name__}: {exc}"
        guide_started = guide.get("measurement_start_unix_s")
        guide_finished = guide.get("measurement_end_unix_s")
        measurement_window.update(
            {
                "guidellm_started_unix_s": guide_started,
                "guidellm_finished_unix_s": guide_finished,
                "server_trigger_lag_from_guidellm_s": (
                    started - guide_started
                    if request_marker_detected
                    and isinstance(guide_started, (int, float))
                    else None
                ),
                "collector_end_lag_from_guidellm_s": (
                    finished - guide_finished
                    if request_marker_detected
                    and isinstance(guide_finished, (int, float))
                    else None
                ),
            }
        )
        write_json(cell_dir / "measurement-window.json", measurement_window)

        if self.use_ebpf and bool(ebpf_status.get("valid")):
            try:
                ebpf_output = (cell_dir / "ebpf-runqlat.txt").read_text(
                    encoding="utf-8"
                )
                ebpf_status["phase_alignment"] = summarize_phase_alignment(
                    ebpf_output,
                    cell_dir / "guidellm.json",
                    started_bpf_clock,
                    finished_bpf_clock,
                    started,
                )
            except OSError as exc:
                ebpf_status["phase_alignment"] = {
                    "valid": False,
                    "error": f"eBPF phase alignment unavailable: {exc}",
                }
            write_json(cell_dir / "ebpf.json", ebpf_status)

        if self.use_energy:
            energy = parse_perf_stat(
                str(perf_status.get("stderr", "")),
                str(self.config.measurement["energy"]["primary_event"]),
            )
            energy["window_seconds"] = measurement_window["duration_s"]
        else:
            energy = {
                "source": None,
                "total_energy_j": None,
                "valid": None,
                "disabled": True,
            }
        write_json(cell_dir / "energy.json", energy)

        slo = evaluate_slo(
            guide,
            self.config.workload(cell.workload),
            float(self.config.evaluation["success_rate_min"]),
        )
        efficiency = efficiency_metrics(
            guide, energy, bool(slo["cell_passes_slo"])
        )
        energy_required = bool(self.config.evaluation.get("require_energy", False))
        ebpf_required = bool(self.config.evaluation.get("require_ebpf", False))
        validity_checks = {
            "benchmark_marker_detected": marker_detected,
            "server_request_marker_detected": request_marker_detected,
            "measurement_window_duration_valid": (
                isinstance(measurement_window["duration_s"], (int, float))
                and float(measurement_window["duration_s"])
                >= cell.duration_seconds * 0.95
            ),
            "suspend_gap_within_limit": (
                isinstance(measurement_window["suspend_gap_s"], (int, float))
                and float(measurement_window["suspend_gap_s"])
                <= float(
                    self.config.measurement.get(
                        "max_suspend_gap_seconds", 5.0
                    )
                )
            ),
            "collectors_prepared_before_request": collectors_prepared,
            "guidellm_exit_zero": returncode == 0,
            "guidellm_report_parsed": guide_error is None,
            "minimum_successful_requests": (
                isinstance(guide.get("successful_requests"), (int, float))
                and guide["successful_requests"]
                >= int(
                    self.config.evaluation.get(
                        "minimum_successful_requests", 1
                    )
                )
            ),
            "host_samples_present": bool(host_status["valid"]),
            "energy_valid": (
                bool(energy["valid"])
                if self.use_energy
                else (False if energy_required else None)
            ),
            "ebpf_valid": (
                bool(ebpf_status["valid"])
                if self.use_ebpf
                else (False if ebpf_required else None)
            ),
        }
        required = [
            value
            for value in validity_checks.values()
            if isinstance(value, bool)
        ]
        validity = {
            "valid": all(required),
            "execution_valid": all(
                value
                for key, value in validity_checks.items()
                if key != "minimum_successful_requests"
                and isinstance(value, bool)
            ),
            "sample_sufficient": validity_checks[
                "minimum_successful_requests"
            ],
            "checks": validity_checks,
            "guidellm_error": guide_error,
        }
        write_json(cell_dir / "validity.json", validity)
        summary = {
            **cell.as_dict(),
            **measurement_window,
            "guide": guide,
            "energy": energy,
            "slo": slo,
            "efficiency": efficiency,
            "host": host_status,
            "ebpf": ebpf_status,
            "validity": validity,
        }
        write_json(cell_dir / "summary.json", summary)
        _progress(
            "policy cell complete",
            cell=f"{index}/{total}",
            successful=guide.get("successful_requests"),
            slo_pass=slo["cell_passes_slo"],
            energy_j=energy.get("total_energy_j"),
            ebpf_samples=ebpf_status.get("samples"),
            execution_valid=validity["execution_valid"],
            sample_sufficient=validity["sample_sufficient"],
        )
        return summary

    def _write_campaign_summary(
        self, root: Path, summaries: list[dict[str, Any]]
    ) -> None:
        winners: list[dict[str, Any]] = []
        grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
        for item in summaries:
            key = (str(item["workload"]), float(item["load_fraction"]))
            grouped.setdefault(key, []).append(item)
        for (workload, fraction), group in grouped.items():
            eligible = [
                item
                for item in group
                if item["validity"]["valid"]
                and item["slo"]["cell_passes_slo"]
                and isinstance(
                    item["efficiency"]["slo_good_output_tokens_per_joule"],
                    (int, float),
                )
            ]
            if eligible:
                winner = max(
                    eligible,
                    key=lambda item: item["efficiency"][
                        "slo_good_output_tokens_per_joule"
                    ],
                )
                winners.append(
                    {
                        "workload": workload,
                        "load_fraction": fraction,
                        "cell_id": winner["cell_id"],
                        "treatment": winner["treatment"],
                        "slo_good_output_tokens_per_joule": winner["efficiency"][
                            "slo_good_output_tokens_per_joule"
                        ],
                    }
                )
        write_json(
            root / "summary.json",
            {
                "policy_status": self.config.evaluation["policy_status"],
                "single_seed_exploratory": True,
                "evidence_valid": all(
                    item["validity"]["valid"] for item in summaries
                ),
                "execution_evidence_valid": all(
                    item["validity"].get(
                        "execution_valid", item["validity"]["valid"]
                    )
                    for item in summaries
                ),
                "cells": summaries,
                "slo_qualified_winners": winners,
            },
        )
        columns = [
            "cell_id",
            "workload",
            "treatment",
            "threads",
            "load_fraction",
            "offered_rate_requests_s",
            "target_duration_s",
            "successful_requests",
            "success_rate",
            "p95_ttft_ms",
            "p95_itl_ms",
            "p99_e2e_ms",
            "total_energy_j",
            "slo_pass",
            "slo_good_output_tokens_per_joule",
            "valid",
            "execution_valid",
            "sample_sufficient",
        ]
        with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for item in summaries:
                writer.writerow(
                    {
                        "cell_id": item["cell_id"],
                        "workload": item["workload"],
                        "treatment": item["treatment"],
                        "threads": item["threads"],
                        "load_fraction": item["load_fraction"],
                        "offered_rate_requests_s": item[
                            "offered_rate_requests_s"
                        ],
                        "target_duration_s": item["target_duration_s"],
                        "successful_requests": item["guide"].get(
                            "successful_requests"
                        ),
                        "success_rate": item["guide"].get("success_rate"),
                        "p95_ttft_ms": item["guide"].get("p95_ttft_ms"),
                        "p95_itl_ms": item["guide"].get("p95_itl_ms"),
                        "p99_e2e_ms": item["guide"].get("p99_e2e_ms"),
                        "total_energy_j": item["energy"].get("total_energy_j"),
                        "slo_pass": item["slo"]["cell_passes_slo"],
                        "slo_good_output_tokens_per_joule": item["efficiency"][
                            "slo_good_output_tokens_per_joule"
                        ],
                        "valid": item["validity"]["valid"],
                        "execution_valid": item["validity"].get(
                            "execution_valid", item["validity"]["valid"]
                        ),
                        "sample_sufficient": item["validity"].get(
                            "sample_sufficient",
                            item["validity"]["checks"].get(
                                "minimum_successful_requests"
                            ),
                        ),
                    }
                )
