from __future__ import annotations

import csv
import io
import math
from typing import Any


def parse_perf_stat(text: str, primary_event: str) -> dict[str, Any]:
    events: dict[str, dict[str, Any]] = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 3:
            continue
        raw_value, unit, event_name = (item.strip() for item in row[:3])
        if "energy" not in event_name:
            continue
        value: float | None
        try:
            value = float(raw_value)
            if not math.isfinite(value):
                value = None
        except ValueError:
            value = None
        events[event_name] = {
            "joules": value if unit.lower().startswith("joule") else None,
            "unit": unit or None,
            "raw_value": raw_value,
        }
    primary = events.get(primary_event, {}).get("joules")
    return {
        "source": "linux-perf-system-wide",
        "boundary": (
            "local package for the configured benchmark window, aligned to the first "
            "post-setup llama-server request using pre-armed, disabled perf counters; "
            "includes model server and active load generator, excludes tokenizer and "
            "request-loader setup"
        ),
        "primary_event": primary_event,
        "total_energy_j": primary,
        "events": events,
        "valid": isinstance(primary, (int, float)) and primary > 0,
    }


def efficiency_metrics(
    guide: dict[str, Any], energy: dict[str, Any], slo_pass: bool
) -> dict[str, float | None]:
    joules = energy.get("total_energy_j")
    successful = guide.get("successful_requests")
    output_tokens = guide.get("output_tokens_successful")
    if not isinstance(joules, (int, float)) or joules <= 0:
        return {
            "joules_per_successful_request": None,
            "output_tokens_per_joule": None,
            "slo_good_output_tokens_per_joule": None,
        }
    return {
        "joules_per_successful_request": (
            joules / successful if isinstance(successful, (int, float)) and successful > 0 else None
        ),
        "output_tokens_per_joule": (
            output_tokens / joules
            if isinstance(output_tokens, (int, float)) and output_tokens >= 0
            else None
        ),
        "slo_good_output_tokens_per_joule": (
            output_tokens / joules
            if slo_pass and isinstance(output_tokens, (int, float)) and output_tokens >= 0
            else 0.0
        ),
    }
