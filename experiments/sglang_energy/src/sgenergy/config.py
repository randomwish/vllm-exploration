from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Workload:
    name: str
    input_tokens: int
    output_tokens: int
    prefix_tokens: int

    @property
    def suffix_tokens(self) -> int:
        return self.input_tokens - self.prefix_tokens


@dataclass(frozen=True)
class CampaignConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def campaign_id(self) -> str:
        return str(self.raw["campaign_id"])

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def runpod(self) -> dict[str, Any]:
        return self.raw["runpod"]

    @property
    def measurement(self) -> dict[str, Any]:
        return self.raw["measurement"]

    @property
    def design(self) -> dict[str, Any]:
        return self.raw["design"]

    @property
    def latency_guard(self) -> dict[str, Any]:
        return self.raw["latency_guard"]

    @property
    def workloads(self) -> dict[str, Workload]:
        return {
            name: Workload(name=name, **values)
            for name, values in self.raw["workloads"].items()
        }

    @property
    def hard_minutes_expected(self) -> int:
        planned = Decimal(str(self.runpod["planned_minutes"]))
        return int((planned * Decimal("1.10")).to_integral_value(rounding=ROUND_CEILING))

    @property
    def calculated_max_gpu_cost_usd(self) -> float:
        return (
            float(self.runpod["observed_two_gpu_hourly_usd"])
            * float(self.runpod["hard_minutes"])
            / 60.0
        )

    def validate(self, *, launch: bool = False) -> list[str]:
        errors: list[str] = []
        if self.raw.get("schema_version") != 1:
            errors.append("schema_version must equal 1")

        model = self.model
        if model.get("tensor_parallel_size") != 2:
            errors.append("this pilot requires tensor_parallel_size=2")
        if model.get("dtype") != "bfloat16":
            errors.append("this pilot requires dtype=bfloat16")
        if int(model.get("context_length", 0)) < max(
            w.input_tokens + w.output_tokens for w in self.workloads.values()
        ):
            errors.append("context_length is smaller than a workload's total tokens")
        if not 0 < float(model.get("mem_fraction_static", 0)) < 1:
            errors.append("mem_fraction_static must be between zero and one")

        runpod = self.runpod
        if int(runpod.get("gpu_count", 0)) != 2:
            errors.append("Runpod gpu_count must equal the TP size of two")
        if not str(runpod.get("data_center_ids", "")).strip():
            errors.append("data_center_ids must be explicit for volume co-location")
        if int(runpod.get("network_volume_size_gb", 0)) < 100:
            errors.append("network_volume_size_gb must be at least 100 GB")
        if int(runpod.get("hard_minutes", 0)) != self.hard_minutes_expected:
            errors.append(
                "hard_minutes must equal ceil(planned_minutes * 1.10): "
                f"expected {self.hard_minutes_expected}"
            )
        configured_cost = float(runpod.get("max_gpu_cost_usd", 0))
        if abs(configured_cost - self.calculated_max_gpu_cost_usd) > 0.02:
            errors.append(
                "max_gpu_cost_usd does not match hourly price and hard deadline"
            )

        measurement = self.measurement
        if float(measurement.get("gpu_sample_seconds", 0)) < 0.1:
            errors.append("GPU sampling interval must be at least 0.1 seconds")
        if measurement.get("separate_gpm_collector") and float(
            measurement.get("gpm_sample_seconds", 0)
        ) < 0.1:
            errors.append("separate GPM sampling interval must be at least 0.1 seconds")
        if int(measurement.get("finalization_reserve_seconds", 0)) < 300:
            errors.append("finalization reserve must be at least five minutes")

        design = self.design
        seeds = list(design.get("seeds", []))
        holdout = list(design.get("holdout_seeds", []))
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            errors.append("at least two unique confirmation seeds are required")
        if len(holdout) < 2 or set(seeds) & set(holdout):
            errors.append("holdout_seeds must be unique and disjoint from tuning seeds")
        if int(design.get("prefix_groups", 0)) < 2:
            errors.append("prefix_groups must be at least two")

        for workload in self.workloads.values():
            if workload.input_tokens <= 0 or workload.output_tokens <= 0:
                errors.append(f"{workload.name}: token lengths must be positive")
            if not 0 <= workload.prefix_tokens < workload.input_tokens:
                errors.append(f"{workload.name}: invalid prefix_tokens")

        guard = self.latency_guard
        if guard.get("mode") not in {"absolute", "exploratory_relative"}:
            errors.append("latency_guard.mode must be absolute or exploratory_relative")
        if guard.get("mode") == "absolute":
            for key in ("p95_ttft_ms", "p95_tpot_ms", "p95_e2e_ms"):
                if not isinstance(guard.get(key), (int, float)) or guard[key] <= 0:
                    errors.append(f"absolute latency guard requires positive {key}")

        if launch:
            if runpod.get("network_volume_id") in {None, "", "REQUIRED_BEFORE_LAUNCH"}:
                errors.append("network_volume_id must be set before launch")
            if model.get("revision") in {None, "", "main"}:
                errors.append("model revision must be an immutable commit before launch")
            image = str(runpod.get("image", ""))
            if image.endswith(":latest") or image.endswith(":dev"):
                errors.append("Runpod image must not use a mutable latest/dev tag")
            if "@sha256:" not in image:
                errors.append("Runpod image must be pinned by registry sha256 digest")
        return errors


def load_config(path: str | Path) -> CampaignConfig:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("campaign config must contain a JSON object")
    config = CampaignConfig(path=config_path, raw=raw)
    errors = config.validate()
    if errors:
        raise ConfigError("invalid campaign config:\n- " + "\n- ".join(errors))
    return config


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
