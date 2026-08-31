from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_results(output_root: Path, *, status: str, error: str | None = None) -> None:
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "FINAL_STATUS.json"}
    )
    checksums = output_root / "SHA256SUMS"
    temporary = checksums.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(output_root)}\n")
    temporary.replace(checksums)
    write_json_atomic(
        output_root / "FINAL_STATUS.json",
        {
            "status": status,
            "error": error,
            "finished_unix_s": time.time(),
            "file_count": len(files),
            "checksums": "SHA256SUMS",
        },
    )
    os.sync()


def delete_current_pod(runpodctl: Path) -> dict[str, Any]:
    pod_id = os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_POD_ID_ALT")
    if not pod_id:
        raise RuntimeError("RUNPOD_POD_ID is not set; refusing an unscoped delete")
    if not os.environ.get("RUNPOD_API_KEY"):
        raise RuntimeError("RUNPOD_API_KEY is not available for self-deletion")
    result = subprocess.run(
        [str(runpodctl), "pod", "delete", pod_id],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pod deletion failed ({result.returncode}): {result.stderr.strip()}"
        )
    return {"pod_id": pod_id, "stdout": result.stdout.strip()}
