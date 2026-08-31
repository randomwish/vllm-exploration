from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


NUMBER = re.compile(r"[-+]?[0-9]*\.?[0-9]+")


def _number(text: str | None) -> float | None:
    match = NUMBER.search(text or "")
    return float(match.group()) if match else None


def snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "-q", "-d", "POWER", "-x"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    root = ET.fromstring(result.stdout)
    output: list[dict[str, Any]] = []
    for index, gpu in enumerate(root.findall("gpu")):
        uuid = gpu.findtext("uuid") or f"gpu-{index}"
        memory_power = gpu.find("gpu_memory_power_readings")
        values: dict[str, float | None] = {
            "hbm_power_draw_w": None,
            "hbm_power_average_w": None,
        }
        if memory_power is not None:
            values["hbm_power_draw_w"] = _number(
                memory_power.findtext("power_draw")
                or memory_power.findtext("instant_power_draw")
            )
            values["hbm_power_average_w"] = _number(
                memory_power.findtext("average_power_draw")
            )
        output.append({"gpu_index": index, "gpu_uuid": uuid, **values})
    return output


def collect(output: Path, *, interval: float, duration: float | None = None) -> None:
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
            unix = time.time()
            try:
                rows = snapshot()
                error = None
            except Exception as exc:
                rows = []
                error = f"{type(exc).__name__}: {exc}"
            stream.write(
                json.dumps(
                    {
                        "unix_s": unix,
                        "monotonic_s": now,
                        "interval_s": now - previous,
                        "gpus": rows,
                        "error": error,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            previous = now
            if duration is not None and now - start >= duration:
                break


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample H100 GPU-memory power")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args(argv)
    collect(args.output, interval=args.interval, duration=args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
