from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import CampaignConfig


def server_command(
    config: CampaignConfig,
    *,
    max_running_requests: int,
    chunked_prefill_size: int,
    radix_cache: bool,
) -> list[str]:
    model = config.model
    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model["path"]),
        "--revision",
        str(model["revision"]),
        "--served-model-name",
        str(model["served_name"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(model["port"]),
        "--dtype",
        str(model["dtype"]),
        "--tp-size",
        str(model["tensor_parallel_size"]),
        "--context-length",
        str(model["context_length"]),
        "--mem-fraction-static",
        str(model["mem_fraction_static"]),
        "--schedule-policy",
        str(model["schedule_policy"]),
        "--max-running-requests",
        str(max_running_requests),
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--stream-interval",
        "1",
        "--enable-metrics",
    ]
    if model.get("attention_backend"):
        command.extend(["--attention-backend", str(model["attention_backend"])])
    if not radix_cache:
        command.append("--disable-radix-cache")
    return command


def _url_ok(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> bool:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


class ServerProcess:
    def __init__(self, command: list[str], log_path: Path, env: dict[str, str] | None = None):
        self.command = command
        self.log_path = log_path
        self.env = env
        self.process: subprocess.Popen[bytes] | None = None
        self._log = None

    def start(self, *, base_url: str, timeout_seconds: float = 900) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab", buffering=0)
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"SGLang exited with {self.process.returncode}; see {self.log_path}"
                )
            if _url_ok(base_url.rstrip("/") + "/health"):
                return
            if _url_ok(
                base_url.rstrip("/") + "/health_generate",
                method="POST",
                payload={},
            ):
                return
            time.sleep(2)
        raise TimeoutError(f"SGLang was not ready within {timeout_seconds} seconds")

    def stop(self, *, grace_seconds: float = 30) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self._log is not None:
            self._log.close()
        self.process = None
        self._log = None

    def __enter__(self) -> "ServerProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


def flush_cache(base_url: str) -> None:
    url = base_url.rstrip("/") + "/flush_cache"
    request = urllib.request.Request(url, data=b"{}", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"cache flush failed with HTTP {response.status}")


def default_remote_env(config: CampaignConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("RUNPOD_API_KEY", None)
    workspace = Path(config.runpod["volume_mount_path"])
    environment.setdefault("HF_HOME", str(workspace / "huggingface-cache"))
    environment.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    return environment
