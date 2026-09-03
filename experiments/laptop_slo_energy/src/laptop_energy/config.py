from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the campaign configuration is invalid."""


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_tokens: int
    output_tokens: int
    slo: dict[str, float]


@dataclass(frozen=True)
class Treatment:
    name: str
    threads: int
    batch_threads: int
    server_args: tuple[str, ...]


@dataclass(frozen=True)
class CampaignConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def server(self) -> dict[str, Any]:
        return self.raw["server"]

    @property
    def guidellm(self) -> dict[str, Any]:
        return self.raw["guidellm"]

    @property
    def measurement(self) -> dict[str, Any]:
        return self.raw["measurement"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def workloads(self) -> list[Workload]:
        return [
            Workload(
                name=str(item["name"]),
                prompt_tokens=int(item["prompt_tokens"]),
                output_tokens=int(item["output_tokens"]),
                slo={key: float(value) for key, value in item["slo"].items()},
            )
            for item in self.raw["workloads"]
        ]

    @property
    def treatments(self) -> list[Treatment]:
        return [
            Treatment(
                name=str(item["name"]),
                threads=int(item["threads"]),
                batch_threads=int(item.get("batch_threads", item["threads"])),
                server_args=tuple(str(value) for value in item.get("server_args", [])),
            )
            for item in self.raw["treatments"]
        ]

    @property
    def baseline_treatment(self) -> Treatment:
        name = str(self.raw["design"]["capacity_baseline_treatment"])
        return next(item for item in self.treatments if item.name == name)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.raw.get("schema_version") != 1:
            errors.append("schema_version must equal 1")

        required_sections = {
            "model",
            "server",
            "guidellm",
            "measurement",
            "evaluation",
            "design",
            "workloads",
            "treatments",
        }
        missing = sorted(required_sections - self.raw.keys())
        if missing:
            return [f"missing configuration section: {name}" for name in missing]

        try:
            workloads = self.workloads
            treatments = self.treatments
        except (KeyError, TypeError, ValueError) as exc:
            return [f"invalid workload or treatment entry: {exc}"]

        workload_names = [item.name for item in workloads]
        if not workloads or len(set(workload_names)) != len(workload_names):
            errors.append("workload names must be non-empty and unique")
        treatment_names = [item.name for item in treatments]
        if len(treatments) < 2 or len(set(treatment_names)) != len(treatment_names):
            errors.append("at least two uniquely named treatments are required")

        context_size = int(self.server.get("context_size", 0))
        for workload in workloads:
            if workload.prompt_tokens <= 0 or workload.output_tokens < 2:
                errors.append(
                    f"{workload.name}: prompt_tokens must be positive and "
                    "output_tokens must be at least two"
                )
            if workload.prompt_tokens + workload.output_tokens > context_size:
                errors.append(f"{workload.name}: workload exceeds server context_size")
            for key in ("p95_ttft_ms", "p95_itl_ms", "p99_e2e_ms"):
                if workload.slo.get(key, 0) <= 0:
                    errors.append(f"{workload.name}: {key} must be positive")

        if any(item.threads <= 0 or item.batch_threads <= 0 for item in treatments):
            errors.append("treatment thread counts must be positive")
        for item in self.raw["treatments"]:
            server_args = item.get("server_args", [])
            if not isinstance(server_args, list) or any(
                not isinstance(value, (str, int, float)) for value in server_args
            ):
                errors.append("treatment server_args must be a list of scalars")
        baseline = str(self.raw["design"].get("capacity_baseline_treatment", ""))
        if baseline not in treatment_names:
            errors.append("capacity_baseline_treatment must name a treatment")

        fractions = self.raw["design"].get("load_fractions", [])
        if (
            not isinstance(fractions, list)
            or not fractions
            or any(not isinstance(value, (int, float)) or value <= 0 for value in fractions)
        ):
            errors.append("load_fractions must contain positive numbers")

        success_rate = self.evaluation.get("success_rate_min")
        if not isinstance(success_rate, (int, float)) or not 0 < success_rate <= 1:
            errors.append("evaluation.success_rate_min must be in (0, 1]")

        for key in ("capacity_seconds", "policy_seconds"):
            if float(self.guidellm.get(key, 0)) <= 0:
                errors.append(f"guidellm.{key} must be positive")
        if float(self.guidellm.get("rampup_seconds", -1)) < 0:
            errors.append("guidellm.rampup_seconds must be nonnegative")
        capacity_profile = self.guidellm.get("capacity_profile", "throughput")
        if capacity_profile not in ("throughput", "synchronous"):
            errors.append(
                "guidellm.capacity_profile must be throughput or synchronous"
            )
        policy_profile = self.guidellm.get("policy_profile", "poisson")
        if policy_profile not in ("poisson", "constant"):
            errors.append("guidellm.policy_profile must be poisson or constant")
        target_arrivals = self.guidellm.get("policy_target_arrivals")
        if target_arrivals is not None and (
            not isinstance(target_arrivals, (int, float)) or target_arrivals <= 0
        ):
            errors.append("guidellm.policy_target_arrivals must be positive")
        policy_max_seconds = self.guidellm.get("policy_max_seconds")
        if policy_max_seconds is not None and (
            not isinstance(policy_max_seconds, (int, float))
            or policy_max_seconds < float(self.guidellm.get("policy_seconds", 0))
        ):
            errors.append(
                "guidellm.policy_max_seconds must be at least policy_seconds"
            )
        for key in ("capacity_max_concurrency", "policy_max_concurrency"):
            if int(self.guidellm.get(key, 0)) <= 0:
                errors.append(f"guidellm.{key} must be positive")
        if float(self.server.get("ready_timeout_seconds", 0)) <= 0:
            errors.append("server.ready_timeout_seconds must be positive")
        if not self.raw["design"].get("paired_absolute_rates", False):
            errors.append("this MVP requires design.paired_absolute_rates=true")
        if not self.raw["design"].get("restart_server_between_policy_cells", False):
            errors.append(
                "this MVP requires design.restart_server_between_policy_cells=true"
            )

        if not str(self.model.get("hf_ref", "")).strip() and not str(
            self.model.get("path", "")
        ).strip():
            errors.append("model.path or model.hf_ref must be set")
        if not str(self.server.get("binary", "")).strip():
            errors.append("server.binary must be set")
        if not str(self.guidellm.get("binary", "")).strip():
            errors.append("guidellm.binary must be set")
        if not str(self.guidellm.get("tokenizer", "")).strip():
            errors.append("guidellm.tokenizer must name a Hugging Face tokenizer")
        minimum_successful = self.evaluation.get("minimum_successful_requests", 1)
        if not isinstance(minimum_successful, int) or minimum_successful <= 0:
            errors.append(
                "evaluation.minimum_successful_requests must be a positive integer"
            )
        return errors

    def workload(self, name: str) -> Workload:
        try:
            return next(item for item in self.workloads if item.name == name)
        except StopIteration as exc:
            raise KeyError(f"unknown workload: {name}") from exc


def load_config(path: str | Path) -> CampaignConfig:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("campaign configuration must contain a JSON object")
    config = CampaignConfig(config_path, raw)
    errors = config.validate()
    if errors:
        raise ConfigError("invalid campaign configuration:\n- " + "\n- ".join(errors))
    return config


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
