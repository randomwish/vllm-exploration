from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

from .binaries import resolve_binary
from .config import CampaignConfig


def _version(command: str, args: list[str]) -> dict[str, Any]:
    resolved = resolve_binary(command)
    if not resolved:
        return {"found": False, "path": None, "version": None}
    try:
        result = subprocess.run(
            [resolved, *args], text=True, capture_output=True, timeout=10, check=False
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        version = output[0] if output else None
        return {
            "found": True,
            "path": resolved,
            "version": version,
            "returncode": result.returncode,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"found": True, "path": resolved, "version": None, "error": str(exc)}


def inspect(
    config: CampaignConfig,
    *,
    privileged: bool = False,
    require_energy: bool = True,
    require_ebpf: bool = True,
) -> dict[str, Any]:
    commands = {
        "llama_server": _version(str(config.server["binary"]), ["--version"]),
        "guidellm": _version(str(config.guidellm["binary"]), ["--version"]),
        "perf": _version("perf", ["--version"]),
        "bpftrace": _version("bpftrace", ["--version"]),
    }
    required_commands = {"llama_server", "guidellm"}
    if require_energy:
        required_commands.add("perf")
    if require_ebpf:
        required_commands.add("bpftrace")
    errors = [
        f"{name} is not installed"
        for name, item in commands.items()
        if name in required_commands and not item["found"]
    ]
    warnings: list[str] = []
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        errors.append(
            "run the harness as an unprivileged user after sudo -v; "
            "do not run it from sudo su"
        )

    server_port = _port_available(
        str(config.server["host"]), int(config.server["port"])
    )
    if server_port["available"] is False:
        errors.append(
            f"server endpoint is unavailable: {config.server['host']}:"
            f"{config.server['port']} ({server_port['error']})"
        )
    elif server_port["available"] is None:
        warnings.append(
            "could not check server port availability: "
            f"{server_port['error']}"
        )

    btf = Path("/sys/kernel/btf/vmlinux")
    if not btf.exists():
        warnings.append("kernel BTF is unavailable; CO-RE probes may not load")
    paranoid = _read_text(Path("/proc/sys/kernel/perf_event_paranoid"))
    unprivileged_bpf = _read_text(Path("/proc/sys/kernel/unprivileged_bpf_disabled"))
    energy_events = _perf_energy_events(commands["perf"].get("path"))
    primary_event = str(config.measurement["energy"]["primary_event"])
    if require_energy and primary_event not in energy_events:
        errors.append(f"configured energy event is unavailable: {primary_event}")

    required_guidellm = str(config.guidellm.get("required_version", "")).strip()
    detected_guidellm = commands["guidellm"].get("version") or ""
    if (
        commands["guidellm"]["found"]
        and required_guidellm
        and required_guidellm not in detected_guidellm
    ):
        errors.append(
            "GuideLLM version mismatch: expected "
            f"{required_guidellm}, detected {detected_guidellm or 'unknown'}"
        )

    probe = (config.path.parent / str(config.measurement["ebpf"]["probe"])).resolve()
    if require_ebpf and not probe.exists():
        errors.append(f"configured eBPF probe does not exist: {probe}")

    sudo_ready: bool | None = None
    ebpf_compile: dict[str, Any] = {
        "checked": False,
        "returncode": None,
        "valid": None,
        "stderr": None,
    }
    if privileged and (require_energy or require_ebpf):
        sudo_ready = subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, check=False
        ).returncode == 0
        if not sudo_ready:
            errors.append("passwordless or pre-authorized sudo is unavailable; run sudo -v")
        elif require_ebpf and probe.exists() and commands["bpftrace"]["found"]:
            result = subprocess.run(
                [
                    "sudo",
                    "-n",
                    str(commands["bpftrace"]["path"]),
                    "-d",
                    str(probe),
                ],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            ebpf_compile = {
                "checked": True,
                "returncode": result.returncode,
                "valid": result.returncode == 0,
                "stderr": result.stderr.strip()[-2000:] or None,
            }
            if result.returncode != 0:
                errors.append("configured eBPF probe failed its compile-only check")
    elif paranoid is not None and int(paranoid) >= 2:
        warnings.append("perf/eBPF need sudo or CAP_PERFMON on this host")

    model_path = str(config.model.get("path", "")).strip()
    model_manifest: dict[str, Any]
    if model_path and not Path(model_path).expanduser().exists():
        errors.append(f"configured model file does not exist: {model_path}")
        model_manifest = {"path": model_path, "exists": False, "sha256": None}
    elif model_path:
        resolved_model = Path(model_path).expanduser().resolve()
        try:
            model_manifest = {
                "path": str(resolved_model),
                "exists": True,
                "size_bytes": resolved_model.stat().st_size,
                "sha256": _sha256(resolved_model),
            }
        except OSError as exc:
            errors.append(f"cannot read configured model file: {exc}")
            model_manifest = {
                "path": str(resolved_model),
                "exists": True,
                "sha256": None,
            }
    else:
        model_manifest = {
            "hf_ref": config.model.get("hf_ref"),
            "source_is_immutable": bool(config.model.get("source_is_immutable", False)),
            "sha256": None,
        }
    if not model_path:
        warnings.append("model uses a Hugging Face ref; first execution may download it")
        if not config.model.get("source_is_immutable", False):
            warnings.append(
                "model ref is mutable; use a local hashed model for audit-quality runs"
            )

    return {
        "ready_to_execute": not errors,
        "errors": errors,
        "warnings": warnings,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "kernel_btf": str(btf) if btf.exists() else None,
            "perf_event_paranoid": int(paranoid) if paranoid is not None else None,
            "unprivileged_bpf_disabled": (
                int(unprivileged_bpf) if unprivileged_bpf is not None else None
            ),
        },
        "commands": commands,
        "model": model_manifest,
        "energy_events": energy_events,
        "ebpf_probe": str(probe),
        "ebpf_compile": ebpf_compile,
        "sudo_noninteractive_ready": sudo_ready,
        "server_port": server_port,
        "mutates_system": False,
    }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _port_available(host: str, port: int) -> dict[str, Any]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # Match the normal server bind behavior. Without SO_REUSEADDR,
            # recently closed benchmark connections can make this preflight
            # probe report EADDRINUSE even though no process is listening.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
        return {"host": host, "port": port, "available": True, "error": None}
    except PermissionError as exc:
        return {"host": host, "port": port, "available": None, "error": str(exc)}
    except OSError as exc:
        return {"host": host, "port": port, "available": False, "error": str(exc)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perf_energy_events(perf_path: str | None) -> list[str]:
    if not perf_path:
        return []
    try:
        result = subprocess.run(
            [perf_path, "list"], text=True, capture_output=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return []
    events: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "energy" not in stripped or "/" not in stripped:
            continue
        events.append(stripped.split()[0])
    return sorted(set(events))
