from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import CampaignConfig


@dataclass(frozen=True)
class PlannedBlock:
    stage: str
    block: str
    workload: str
    kind: str
    settings: dict[str, Any]
    load: str
    seeds: list[int]
    cache_mode: str
    note: str = ""


def build_symbolic_plan(config: CampaignConfig) -> list[PlannedBlock]:
    if config.raw.get("experiment_kind", "full") == "prefix_cache":
        return build_prefix_cache_plan(config)
    d = config.design
    m = config.model
    seeds = list(d["seeds"])
    holdout = list(d["holdout_seeds"])
    baseline = {
        "max_running_requests": m["baseline_max_running_requests"],
        "chunked_prefill_size": m["baseline_chunked_prefill_size"],
    }
    blocks: list[PlannedBlock] = []

    for concurrency in d["capacity_concurrency"]:
        blocks.append(
            PlannedBlock(
                "1",
                "capacity-scout",
                "BAL",
                "screen",
                {**baseline, "client_max_concurrency": concurrency},
                "closed-loop",
                [seeds[0]],
                "disabled",
            )
        )
    for load in d["load_fractions"]:
        blocks.append(
            PlannedBlock(
                "1",
                "baseline-knee-screen",
                "BAL",
                "screen",
                baseline,
                f"{load:.2f}*C0",
                [seeds[0]],
                "disabled",
                "Confirm at most three adjacent points after screening.",
            )
        )
    for value in d["max_running_requests"]:
        blocks.append(
            PlannedBlock(
                "1",
                "max-running-screen",
                "BAL",
                "screen",
                {**baseline, "max_running_requests": value},
                "0.90*C0",
                [seeds[0]],
                "disabled",
            )
        )
    blocks.append(
        PlannedBlock(
            "1",
            "max-running-confirm",
            "BAL",
            "confirm",
            {"candidate": "baseline-and-selected"},
            "0.90*C0 and 1.05*C0",
            seeds,
            "disabled",
        )
    )

    for concurrency in d["short_capacity_concurrency"]:
        blocks.append(
            PlannedBlock(
                "2",
                "prefill-capacity-scout",
                "PF",
                "screen",
                {**baseline, "client_max_concurrency": concurrency},
                "closed-loop",
                [seeds[0]],
                "disabled",
            )
        )
    for value in d["chunked_prefill_sizes"]:
        blocks.append(
            PlannedBlock(
                "2",
                "chunk-screen",
                "PF",
                "screen",
                {"max_running_requests": "selected-stage-1", "chunked_prefill_size": value},
                "0.85*C_pf",
                [seeds[0]],
                "disabled",
            )
        )
    blocks.append(
        PlannedBlock(
            "2",
            "chunk-confirm-and-interaction",
            "PF",
            "confirm",
            {"factorial": "2x2 baseline/selected running-requests and chunk-size"},
            "0.90*C_pf",
            seeds,
            "disabled",
        )
    )

    for concurrency in d["short_capacity_concurrency"]:
        blocks.append(
            PlannedBlock(
                "3",
                "prefix-capacity-scout",
                "PX0",
                "screen",
                {"client_max_concurrency": concurrency, "scheduler": "selected-stage-2"},
                "closed-loop",
                [seeds[0]],
                "disabled",
            )
        )
    for cache_mode in ("disabled", "enabled"):
        for workload in ("PX0", "PX50", "PX87"):
            blocks.append(
                PlannedBlock(
                    "3",
                    "cold-prefix-screen",
                    workload,
                    "screen",
                    {"scheduler": "selected-stage-2"},
                    "0.85*C_px",
                    [seeds[0]],
                    cache_mode,
                )
            )
    blocks.append(
        PlannedBlock(
            "3",
            "cold-prefix-confirm",
            "PX0/PX87",
            "confirm",
            {"scheduler": "selected-stage-2"},
            "0.90*C_px",
            seeds,
            "disabled-and-enabled",
        )
    )
    blocks.append(
        PlannedBlock(
            "3",
            "warm-prefix-mechanism",
            "PX87",
            "screen",
            {"scheduler": "selected-stage-2", "prime_each_group": True},
            "0.85*C_px",
            [seeds[0]],
            "disabled-and-enabled",
        )
    )

    for concurrency in d["short_capacity_concurrency"]:
        blocks.append(
            PlannedBlock(
                "4",
                "decode-capacity-scout",
                "DEC",
                "screen",
                {**baseline, "client_max_concurrency": concurrency},
                "closed-loop",
                [holdout[0]],
                "disabled",
            )
        )
    for workload, capacity in (("BAL", "C0"), ("PF", "C_pf"), ("DEC", "C_dec")):
        blocks.append(
            PlannedBlock(
                "4",
                "regime-holdout",
                workload,
                "confirm",
                {"compare": "explicit-baseline-vs-selected-stage-2"},
                f"0.85*{capacity}",
                holdout,
                "disabled",
            )
        )
    return blocks


def build_prefix_cache_plan(config: CampaignConfig) -> list[PlannedBlock]:
    design = config.design
    settings = {
        "max_running_requests": config.model["baseline_max_running_requests"],
        "chunked_prefill_size": config.model["baseline_chunked_prefill_size"],
        "prime_each_group": True,
        "same_prefix_seed_different_suffix_seed": True,
    }
    blocks: list[PlannedBlock] = []
    for seed_index, seed in enumerate(design["seeds"]):
        cache_modes = ["disabled", "enabled"]
        if seed_index % 2:
            cache_modes.reverse()
        workloads = list(design["prefix_cache_workloads"])
        shift = seed_index % len(workloads)
        workloads = workloads[shift:] + workloads[:shift]
        for cache_mode in cache_modes:
            for workload in workloads:
                blocks.append(
                    PlannedBlock(
                        "prefix-cache",
                        "paired-prewarmed-confirm",
                        workload,
                        "confirm",
                        settings,
                        f"{design['prefix_cache_rate_requests_s']:.6f} requests/s",
                        [seed],
                        cache_mode,
                        "Cache order alternates AB/BA by seed; workload order rotates by seed.",
                    )
                )
    return blocks


def plan_as_dicts(config: CampaignConfig) -> list[dict[str, Any]]:
    return [asdict(block) for block in build_symbolic_plan(config)]
