from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sgenergy.config import Workload, load_config
from sgenergy.plan import build_symbolic_plan
from sgenergy.traces import build_trace, iter_trace_requests, load_prefixes, verify_trace


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_config_and_ten_percent_buffer(self) -> None:
        config = load_config(ROOT / "campaign.json")
        self.assertEqual(config.hard_minutes_expected, 198)
        self.assertAlmostEqual(config.calculated_max_gpu_cost_usd, 21.714)
        self.assertEqual(config.validate(), [])

    def test_launch_configuration_is_complete(self) -> None:
        config = load_config(ROOT / "campaign.json")
        errors = config.validate(launch=True)
        self.assertEqual(errors, [])

    def test_symbolic_plan_covers_all_stages(self) -> None:
        config = load_config(ROOT / "campaign.json")
        plan = build_symbolic_plan(config)
        self.assertEqual({block.stage for block in plan}, {"1", "2", "3", "4"})
        self.assertTrue(any(block.block == "baseline-knee-screen" for block in plan))
        self.assertTrue(any(block.block == "chunk-screen" for block in plan))
        self.assertTrue(any(block.block == "cold-prefix-confirm" for block in plan))
        self.assertTrue(any(block.block == "regime-holdout" for block in plan))

    def test_prefix_cache_config_and_plan(self) -> None:
        config = load_config(ROOT / "prefix_cache_campaign.json")
        self.assertEqual(config.hard_minutes_expected, 99)
        self.assertAlmostEqual(config.calculated_max_gpu_cost_usd, 10.857)
        self.assertEqual(config.validate(launch=True), [])
        plan = build_symbolic_plan(config)
        self.assertEqual(len(plan), 36)
        self.assertEqual({block.workload for block in plan}, {"PX0", "PX50", "PX87"})
        self.assertEqual({block.cache_mode for block in plan}, {"disabled", "enabled"})
        self.assertTrue(all(block.block == "paired-prewarmed-confirm" for block in plan))


class FrozenTraceTests(unittest.TestCase):
    def test_trace_is_deterministic_and_verifiable(self) -> None:
        workload = Workload("TEST", input_tokens=32, output_tokens=4, prefix_tokens=16)
        token_pool = list(range(256, 4096))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build_trace(
                root / "first",
                workload=workload,
                seed=42,
                token_pool=token_pool,
                prefix_groups=4,
                rate=4,
                duration=5,
            )
            second = build_trace(
                root / "second",
                workload=workload,
                seed=42,
                token_pool=token_pool,
                prefix_groups=4,
                rate=4,
                duration=5,
            )
            self.assertEqual(first["trace_id"], second["trace_id"])
            self.assertEqual(first["files"], second["files"])
            self.assertEqual(verify_trace(root / "first")["trace_id"], first["trace_id"])
            requests = list(iter_trace_requests(root / "first"))
            self.assertEqual(len(requests) % 4, 0)
            self.assertTrue(all(request.input_sha256 for request in requests))

    def test_closed_loop_arrivals_are_frozen_at_zero(self) -> None:
        workload = Workload("TEST", input_tokens=16, output_tokens=2, prefix_tokens=0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trace"
            build_trace(
                root,
                workload=workload,
                seed=7,
                token_pool=list(range(256, 4096)),
                prefix_groups=4,
                closed_loop_count=8,
            )
            self.assertEqual([r.arrival_s for r in iter_trace_requests(root)], [0.0] * 8)

    def test_prefix_seed_can_be_shared_without_reusing_suffixes(self) -> None:
        workload = Workload("TEST", input_tokens=32, output_tokens=4, prefix_tokens=24)
        token_pool = list(range(256, 4096))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_trace(
                root / "prime",
                workload=workload,
                seed=1007,
                prefix_seed=7,
                token_pool=token_pool,
                prefix_groups=4,
                closed_loop_count=4,
            )
            build_trace(
                root / "measure",
                workload=workload,
                seed=7,
                prefix_seed=7,
                token_pool=token_pool,
                prefix_groups=4,
                closed_loop_count=4,
            )
            self.assertEqual(load_prefixes(root / "prime"), load_prefixes(root / "measure"))
            prime = list(iter_trace_requests(root / "prime"))
            measure = list(iter_trace_requests(root / "measure"))
            self.assertNotEqual(
                [request.suffix_ids for request in prime],
                [request.suffix_ids for request in measure],
            )


if __name__ == "__main__":
    unittest.main()
