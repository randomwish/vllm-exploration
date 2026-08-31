from __future__ import annotations

import asyncio

from .campaign import CampaignRunner, DeadlineReached, ServerSettings
from .config import write_json_atomic
from .replay import replay_trace
from .server import flush_cache


class PrefixCacheRunner(CampaignRunner):
    """Run a paired, prewarmed radix-cache confirmation campaign."""

    def _prime(
        self,
        *,
        workload: str,
        seed: int,
        settings: ServerSettings,
        label: str,
    ) -> None:
        prefix_groups = int(self.config.design["prefix_groups"])
        prime_seed = seed + int(self.config.design["prefix_prime_seed_offset"])
        trace = self.trace(
            workload,
            seed=prime_seed,
            prefix_seed=seed,
            kind="closed",
            count=prefix_groups,
        )
        warmup_root = self.output_root / "warmup" / label
        warmup_root.mkdir(parents=True, exist_ok=False)
        flush_cache(self.base_url)
        summary = asyncio.run(
            replay_trace(
                trace,
                base_url=self.base_url,
                output_path=warmup_root / "requests.jsonl",
                watchdog_seconds=float(
                    self.config.measurement["screen_watchdog_seconds"]
                ),
                max_concurrency=1,
            )
        )
        write_json_atomic(
            warmup_root / "summary.json",
            {
                **summary,
                "workload": workload,
                "measurement_seed": seed,
                "prime_seed": prime_seed,
                "prefix_seed": seed,
                "settings": {
                    "max_running_requests": settings.max_running_requests,
                    "chunked_prefill_size": settings.chunked_prefill_size,
                    "radix_cache": settings.radix_cache,
                },
            },
        )
        if summary["success_count"] != summary["request_count"]:
            raise RuntimeError(f"prefix priming failed for {label}")

    def prefix_cache_experiment(self) -> None:
        self.save_state("prefix-cache-confirm")
        design = self.config.design
        seeds = [int(seed) for seed in design["seeds"]]
        workloads = [str(name) for name in design["prefix_cache_workloads"]]
        rate = float(design["prefix_cache_rate_requests_s"])
        duration = float(self.config.measurement["confirm_arrival_seconds"])
        watchdog = float(self.config.measurement["confirm_watchdog_seconds"])
        settings_by_cache = {
            False: ServerSettings(
                int(self.config.model["baseline_max_running_requests"]),
                int(self.config.model["baseline_chunked_prefill_size"]),
                False,
            ),
            True: ServerSettings(
                int(self.config.model["baseline_max_running_requests"]),
                int(self.config.model["baseline_chunked_prefill_size"]),
                True,
            ),
        }
        self.selected = {
            "max_running_requests": settings_by_cache[False].max_running_requests,
            "chunked_prefill_size": settings_by_cache[False].chunked_prefill_size,
            "paired_seeds": seeds,
            "workloads": workloads,
            "offered_rate_requests_s": rate,
            "cache_order": "AB/BA alternating by seed",
            "state": "prewarmed",
        }

        for seed_index, seed in enumerate(seeds):
            cache_order = [False, True] if seed_index % 2 == 0 else [True, False]
            shift = seed_index % len(workloads)
            workload_order = workloads[shift:] + workloads[:shift]
            for cache_enabled in cache_order:
                settings = settings_by_cache[cache_enabled]

                def action(
                    settings: ServerSettings = settings,
                    workload_order: list[str] = workload_order,
                    seed: int = seed,
                ) -> None:
                    for workload in workload_order:
                        cache = "on" if settings.radix_cache else "off"
                        label = f"prefix-{workload}-cache-{cache}-seed-{seed}"
                        self._prime(
                            workload=workload,
                            seed=seed,
                            settings=settings,
                            label=label,
                        )
                        trace = self.trace(
                            workload,
                            seed=seed,
                            prefix_seed=seed,
                            kind="open",
                            rate=rate,
                            duration=duration,
                        )
                        self.measure(
                            label=label,
                            trace_dir=trace,
                            settings=settings,
                            watchdog=watchdog,
                            flush=False,
                        )

                self.with_server(settings, action, label=f"prefix-seed-{seed}")
        self.save_state("prefix-cache-complete")

    def run(self) -> None:
        try:
            self.prepare()
            self.prefix_cache_experiment()
            self.save_state("complete")
            self.event("campaign-complete")
        except DeadlineReached as exc:
            self.event("campaign-deadline", error=str(exc))
            self.save_state("deadline-stopped")
            raise
        except Exception as exc:
            self.event("campaign-failed", error=f"{type(exc).__name__}: {exc}")
            self.save_state("failed")
            raise
        finally:
            self.write_summary()
