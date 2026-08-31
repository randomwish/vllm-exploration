from __future__ import annotations

import asyncio
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import CampaignConfig, write_json_atomic
from .replay import replay_trace
from .server import ServerProcess, default_remote_env, flush_cache, server_command
from .validate import validate_run


def _stop_collector(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _gpu_health_snapshot(path: Path) -> None:
    with path.open("wb") as stream:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            stdout=stream,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi health snapshot failed: {result.stderr.decode(errors='replace')}"
        )


def measure_cell(
    config: CampaignConfig,
    *,
    trace_dir: Path,
    run_dir: Path,
    base_url: str,
    watchdog_seconds: float,
    max_concurrency: int | None = None,
    flush_before_run: bool = True,
    server_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "config.json",
        {
            "trace_dir": str(trace_dir.resolve()),
            "server_settings": server_settings or {},
            "watchdog_seconds": watchdog_seconds,
            "max_concurrency": max_concurrency,
        },
    )
    gpu_collector: subprocess.Popen[bytes] | None = None
    gpm_collector: subprocess.Popen[bytes] | None = None
    hbm_collector: subprocess.Popen[bytes] | None = None
    service_collector: subprocess.Popen[bytes] | None = None
    env = default_remote_env(config)
    status: dict[str, Any] = {"state": "measuring", "error": None}
    write_json_atomic(run_dir / "status.json", status)
    try:
        if flush_before_run:
            flush_cache(base_url)
        _gpu_health_snapshot(run_dir / "gpu-health-before.xml")
        gpu_log = (run_dir / "gpu-collector.log").open("wb")
        hbm_log = (run_dir / "hbm-collector.log").open("wb")
        service_log = (run_dir / "service-collector.log").open("wb")
        gpu_command = [
                "python3",
                "-m",
                "sgenergy.gpu_telemetry",
                "--output",
                str(run_dir / "gpu.jsonl"),
                "--interval",
                str(config.measurement["gpu_sample_seconds"]),
            ]
        if config.measurement.get("separate_gpm_collector"):
            gpu_command.append("--no-gpm")
        gpu_collector = subprocess.Popen(
            gpu_command,
            stdout=gpu_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        gpm_sample_seconds = config.measurement.get("gpm_sample_seconds")
        if config.measurement.get("separate_gpm_collector") and gpm_sample_seconds:
            gpm_log = (run_dir / "gpm-collector.log").open("wb")
            gpm_collector = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "sgenergy.gpu_telemetry",
                    "--output",
                    str(run_dir / "gpm.jsonl"),
                    "--interval",
                    str(gpm_sample_seconds),
                ],
                stdout=gpm_log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        service_collector = subprocess.Popen(
            [
                "python3",
                "-m",
                "sgenergy.service_telemetry",
                "--output",
                str(run_dir / "service.jsonl"),
                "--metrics-url",
                f"{base_url}/metrics",
                "--interval",
                str(config.measurement["service_sample_seconds"]),
            ],
            stdout=service_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        if config.measurement.get("hbm_enabled", True):
            hbm_collector = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "sgenergy.hbm_telemetry",
                    "--output",
                    str(run_dir / "hbm.jsonl"),
                    "--interval",
                    "1.0",
                ],
                stdout=hbm_log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        time.sleep(0.5)
        replay_summary = asyncio.run(
            replay_trace(
                trace_dir,
                base_url=base_url,
                output_path=run_dir / "requests.jsonl",
                watchdog_seconds=watchdog_seconds,
                max_concurrency=max_concurrency,
            )
        )
        write_json_atomic(run_dir / "replay_summary.json", replay_summary)
        _stop_collector(service_collector)
        _stop_collector(hbm_collector)
        _stop_collector(gpm_collector)
        _stop_collector(gpu_collector)
        service_collector = hbm_collector = gpm_collector = gpu_collector = None
        _gpu_health_snapshot(run_dir / "gpu-health-after.xml")
        status["state"] = "validating"
        write_json_atomic(run_dir / "status.json", status)
        validity = validate_run(run_dir, config)
        status["state"] = "complete" if validity["valid"] else "invalid"
        status["valid"] = validity["valid"]
        write_json_atomic(run_dir / "status.json", status)
        return validity
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(run_dir / "status.json", status)
        raise
    finally:
        _stop_collector(service_collector)
        _stop_collector(hbm_collector)
        _stop_collector(gpm_collector)
        _stop_collector(gpu_collector)
        try:
            gpu_log.close()
        except (NameError, UnboundLocalError):
            pass
        try:
            hbm_log.close()
        except (NameError, UnboundLocalError):
            pass
        try:
            gpm_log.close()
        except (NameError, UnboundLocalError):
            pass
        try:
            service_log.close()
        except (NameError, UnboundLocalError):
            pass


def run_cell(
    config: CampaignConfig,
    *,
    trace_dir: Path,
    run_dir: Path,
    max_running_requests: int,
    chunked_prefill_size: int,
    radix_cache: bool,
    watchdog_seconds: float,
    max_concurrency: int | None = None,
    flush_before_run: bool = True,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{config.model['port']}"
    command = server_command(
        config,
        max_running_requests=max_running_requests,
        chunked_prefill_size=chunked_prefill_size,
        radix_cache=radix_cache,
    )
    env = default_remote_env(config)
    server = ServerProcess(command, run_dir.parent / f"{run_dir.name}-server.log", env)
    try:
        server.start(base_url=base_url)
        return measure_cell(
            config,
            trace_dir=trace_dir,
            run_dir=run_dir,
            base_url=base_url,
            watchdog_seconds=watchdog_seconds,
            max_concurrency=max_concurrency,
            flush_before_run=flush_before_run,
            server_settings={
                "command": command,
                "max_running_requests": max_running_requests,
                "chunked_prefill_size": chunked_prefill_size,
                "radix_cache": radix_cache,
            },
        )
    finally:
        server.stop()
