from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .config import CampaignConfig


@dataclass(frozen=True)
class PlannedCell:
    cell_id: str
    phase: str
    workload: str
    treatment: str
    threads: int
    profile: str
    load_fraction: float | None
    offered_rate_requests_s: float | None
    duration_seconds: float
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibration_plan(config: CampaignConfig) -> list[PlannedCell]:
    baseline = config.baseline_treatment
    seed = int(config.guidellm["seed"])
    profile = str(config.guidellm.get("capacity_profile", "throughput"))
    return [
        PlannedCell(
            cell_id=f"cal-{workload.name}-{baseline.name}",
            phase="calibration",
            workload=workload.name,
            treatment=baseline.name,
            threads=baseline.threads,
            profile=profile,
            load_fraction=None,
            offered_rate_requests_s=None,
            duration_seconds=float(config.guidellm["capacity_seconds"]),
            seed=seed,
        )
        for workload in config.workloads
    ]


def policy_plan(
    config: CampaignConfig, capacities: dict[str, float] | None = None
) -> list[PlannedCell]:
    seed = int(config.guidellm["seed"])
    fractions = [float(value) for value in config.raw["design"]["load_fractions"]]
    profile = str(config.guidellm.get("policy_profile", "poisson"))
    cells: list[PlannedCell] = []
    pair_index = 0
    for workload in config.workloads:
        capacity = capacities.get(workload.name) if capacities else None
        for fraction in fractions:
            order = list(config.treatments)
            if pair_index % 2:
                order.reverse()
            for treatment in order:
                rate = capacity * fraction if capacity is not None else None
                duration = _policy_duration_seconds(config, rate)
                fraction_label = str(fraction).replace(".", "p")
                cells.append(
                    PlannedCell(
                        cell_id=(
                            f"policy-{workload.name}-load-{fraction_label}-"
                            f"{treatment.name}"
                        ),
                        phase="policy",
                        workload=workload.name,
                        treatment=treatment.name,
                        threads=treatment.threads,
                        profile=profile,
                        load_fraction=fraction,
                        offered_rate_requests_s=rate,
                        duration_seconds=duration,
                        seed=seed,
                    )
                )
            pair_index += 1
    return cells


def _policy_duration_seconds(
    config: CampaignConfig, offered_rate_requests_s: float | None
) -> float:
    minimum = float(config.guidellm["policy_seconds"])
    target = config.guidellm.get("policy_target_arrivals")
    if (
        offered_rate_requests_s is None
        or not isinstance(target, (int, float))
        or target <= 0
    ):
        return minimum
    duration = max(minimum, float(target) / offered_rate_requests_s)
    maximum = config.guidellm.get("policy_max_seconds")
    if isinstance(maximum, (int, float)) and maximum > 0:
        duration = min(duration, float(maximum))
    return float(math.ceil(duration))


def full_symbolic_plan(config: CampaignConfig) -> dict[str, Any]:
    return {
        "calibration": [item.as_dict() for item in calibration_plan(config)],
        "policy": [item.as_dict() for item in policy_plan(config)],
        "counts": {
            "calibration": len(calibration_plan(config)),
            "policy": len(policy_plan(config)),
        },
        "note": (
            "Policy rates are capacity-relative until calibration reports have "
            "been produced. Capacity-aware durations use policy_seconds in this "
            "symbolic plan and are resolved after calibration."
        ),
    }
