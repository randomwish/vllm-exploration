from __future__ import annotations

import json
import signal
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from laptop_energy.binaries import resolve_binary
from laptop_energy.config import Workload, load_config
from laptop_energy.energy import efficiency_metrics, parse_perf_stat
from laptop_energy.ebpf import summarize_phase_alignment, summarize_runqlat
from laptop_energy.guidellm import build_command, evaluate_slo, summarize_report
from laptop_energy.plan import calibration_plan, policy_plan
from laptop_energy.preflight import _port_available
from laptop_energy.runtime import (
    CampaignRunner,
    EbpfCollector,
    PerfEnergyCollector,
    ServerProcess,
    SudoKeepalive,
    _bpf_clock_s,
    _campaign_completion_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigAndPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "campaign.json")

    def test_default_plan_has_three_calibration_and_eighteen_policy_cells(self) -> None:
        self.assertEqual(len(calibration_plan(self.config)), 3)
        cells = policy_plan(self.config)
        self.assertEqual(len(cells), 18)
        self.assertIsNone(cells[0].offered_rate_requests_s)

    def test_active_environment_binary_is_resolved(self) -> None:
        resolved = resolve_binary("python")
        self.assertIsNotNone(resolved)
        self.assertEqual(
            Path(resolved).parent, Path(__import__("sys").executable).parent
        )

    def test_policy_uses_equal_absolute_rate_for_treatment_pair(self) -> None:
        cells = policy_plan(
            self.config, {"short": 2.0, "medium": 1.0, "long": 0.5}
        )
        first_pair = cells[:2]
        self.assertEqual(
            {cell.offered_rate_requests_s for cell in first_pair}, {1.0}
        )
        self.assertEqual(
            {cell.treatment for cell in first_pair}, {"threads-4", "threads-8"}
        )

    def test_guidellm_command_has_nominal_synthetic_lengths_and_poisson_rate(self) -> None:
        cell = policy_plan(
            self.config, {"short": 2.0, "medium": 1.0, "long": 0.5}
        )[0]
        command = build_command(self.config, cell, Path("/tmp/cell"))
        joined = " ".join(command)
        self.assertIn("prompt_tokens=512,prompt_tokens_stdev=1", joined)
        self.assertIn("output_tokens=128,output_tokens_stdev=1", joined)
        self.assertIn("kind=poisson,rate=1", joined)
        self.assertIn("rampup_duration=5.0", joined)
        self.assertIn("/v1/chat/completions", joined)
        self.assertIn("kind=huggingface_auto,model=Qwen/Qwen3.5-4B", joined)

    def test_pilot_uses_capacity_aware_paired_durations(self) -> None:
        pilot = load_config(PROJECT_ROOT / "campaign.pilot.json")
        cells = policy_plan(
            pilot, {"short": 0.2, "medium": 0.1, "long": 0.05}
        )
        self.assertEqual(len(cells), 6)
        pairs = [cells[index : index + 2] for index in range(0, len(cells), 2)]
        self.assertEqual(
            [{cell.duration_seconds for cell in pair} for pair in pairs],
            [{120.0}, {236.0}, {471.0}],
        )
        for pair in pairs:
            self.assertEqual(
                len({cell.offered_rate_requests_s for cell in pair}), 1
            )

    def test_frontier_uses_synchronous_calibration_and_constant_policy(self) -> None:
        frontier = load_config(PROJECT_ROOT / "campaign.frontier.json")
        calibration = calibration_plan(frontier)
        self.assertEqual(len(calibration), 1)
        self.assertEqual(calibration[0].profile, "synchronous")
        calibration_command = " ".join(
            build_command(frontier, calibration[0], Path("/tmp/calibration"))
        )
        self.assertIn("--profile kind=synchronous", calibration_command)

        cells = policy_plan(frontier, {"medium": 0.05})
        self.assertEqual(len(cells), 2)
        self.assertEqual({cell.profile for cell in cells}, {"constant"})
        self.assertEqual(
            {cell.offered_rate_requests_s for cell in cells}, {0.0125}
        )
        self.assertEqual({cell.duration_seconds for cell in cells}, {1600.0})
        policy_command = " ".join(
            build_command(frontier, cells[0], Path("/tmp/policy"))
        )
        self.assertIn("kind=constant,rate=0.0125", policy_command)
        self.assertIn("rampup_duration=0.0", policy_command)

    def test_affinity_treatment_pins_generation_and_prefill_threads(self) -> None:
        affinity = load_config(PROJECT_ROOT / "campaign.affinity.json")
        pinned = next(
            treatment
            for treatment in affinity.treatments
            if treatment.name == "threads-4-fast-cores"
        )
        command = ServerProcess(affinity, pinned, Path("/tmp/server")).command
        self.assertEqual(pinned.batch_threads, 4)
        self.assertIn("0x55", command)
        self.assertIn("--cpu-strict", command)
        self.assertIn("--cpu-strict-batch", command)

    def test_coordination_campaign_changes_only_batch_thread_count(self) -> None:
        coordination = load_config(PROJECT_ROOT / "campaign.coordination.json")
        commands = {
            treatment.name: ServerProcess(
                coordination, treatment, Path("/tmp/server")
            ).command
            for treatment in coordination.treatments
        }
        self.assertIn("-t", commands["threads-4-batch-2"])
        self.assertEqual(
            commands["threads-4-batch-2"][
                commands["threads-4-batch-2"].index("-t") + 1
            ],
            "4",
        )
        self.assertEqual(
            commands["threads-4-batch-2"][
                commands["threads-4-batch-2"].index("-tb") + 1
            ],
            "2",
        )
        self.assertEqual(
            commands["threads-4-batch-4"][
                commands["threads-4-batch-4"].index("-tb") + 1
            ],
            "4",
        )


class MetricsTests(unittest.TestCase):
    def _report_path(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "guidellm.json"
        report = {
            "metadata": {"version": 2, "guidellm_version": "0.7.2"},
            "benchmarks": [
                {
                    "metrics": {
                        "request_totals": {
                            "successful": 99,
                            "incomplete": 0,
                            "errored": 1,
                            "total": 100,
                        },
                        "requests_per_second": {"successful": {"mean": 2.5}},
                        "output_token_count": {
                            "successful": {"mean": 128, "total_sum": 12672}
                        },
                        "time_to_first_token_ms": {
                            "successful": {
                                "percentiles": {"p50": 300, "p95": 900, "p99": 1200}
                            }
                        },
                        "inter_token_latency_ms": {
                            "successful": {
                                "percentiles": {"p50": 20, "p95": 40, "p99": 50}
                            }
                        },
                        "request_latency": {
                            "successful": {
                                "percentiles": {"p95": 4, "p99": 5}
                            }
                        },
                    }
                }
            ],
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        return temporary, path

    def test_guidellm_report_summary_and_slo(self) -> None:
        temporary, path = self._report_path()
        self.addCleanup(temporary.cleanup)
        summary = summarize_report(path)
        self.assertEqual(summary["successful_requests"], 99)
        self.assertEqual(summary["output_tokens_successful"], 12672)
        self.assertAlmostEqual(summary["success_rate"], 0.99)
        self.assertEqual(summary["p95_itl_ms"], 40)
        self.assertEqual(summary["p99_e2e_ms"], 5000)
        self.assertIsNone(summary["measurement_start_unix_s"])

        workload = Workload(
            "fixture",
            512,
            128,
            {"p95_ttft_ms": 1000, "p95_itl_ms": 50, "p99_e2e_ms": 6000},
        )
        self.assertTrue(evaluate_slo(summary, workload, 0.99)["cell_passes_slo"])
        workload.slo["p95_itl_ms"] = 30
        self.assertFalse(evaluate_slo(summary, workload, 0.99)["cell_passes_slo"])

    def test_perf_energy_and_efficiency(self) -> None:
        perf = "123.45,Joules,power/energy-pkg/,1000000000,100.00\n"
        energy = parse_perf_stat(perf, "power/energy-pkg/")
        self.assertTrue(energy["valid"])
        self.assertEqual(energy["total_energy_j"], 123.45)

        metrics = efficiency_metrics(
            {"successful_requests": 10, "output_tokens_successful": 1000},
            energy,
            True,
        )
        self.assertAlmostEqual(metrics["joules_per_successful_request"], 12.345)
        self.assertAlmostEqual(
            metrics["slo_good_output_tokens_per_joule"], 1000 / 123.45
        )

    def test_non_slo_cell_gets_zero_slo_good_efficiency(self) -> None:
        metrics = efficiency_metrics(
            {"successful_requests": 10, "output_tokens_successful": 1000},
            {"total_energy_j": 100},
            False,
        )
        self.assertEqual(metrics["slo_good_output_tokens_per_joule"], 0.0)


class CollectorLifecycleTests(unittest.TestCase):
    def test_sample_shortfall_is_evidence_limited_not_execution_failure(self) -> None:
        summaries = [
            {
                "validity": {
                    "valid": False,
                    "checks": {
                        "guidellm_exit_zero": True,
                        "energy_valid": True,
                        "ebpf_valid": True,
                        "minimum_successful_requests": False,
                    },
                }
            }
        ]
        self.assertEqual(
            _campaign_completion_status(summaries),
            "complete_evidence_limited",
        )

    def test_ebpf_histogram_reports_windowed_scheduler_delay(self) -> None:
        output = """
@runqlat_us_250ms[400]:
[2, 4)                 3 |@@|
[8, 16)                1 |@|
@runqlat_us_250ms[404]:
[1024, 2048)            8 |@@@@|
"""
        summary = summarize_runqlat(output, 100.0, 100.25)
        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["p50_upper_bound_us"], 4)
        self.assertEqual(summary["p95_upper_bound_us"], 16)

    def test_ebpf_one_second_aggregates_report_tail_fractions(self) -> None:
        output = """
@runqlat_count_1s[100]: 10
@runqlat_sum_us_1s[100]: 2500
@runqlat_max_us_1s[100]: 1200
@runqlat_ge_100us_1s[100]: 4
@runqlat_ge_1ms_1s[100]: 1
@cpu_changes_1s[100]: 2
@sched_migrate_task_1s[100]: 1
@futex_wait_count_1s[100]: 5
@futex_wait_sum_us_1s[100]: 10000
@futex_wait_max_us_1s[100]: 5000
@futex_wait_ge_1ms_1s[100]: 2
@futex_wake_calls_1s[100]: 7
@runqlat_count_1s[102]: 99
"""
        summary = summarize_runqlat(output, 100.0, 101.0)
        self.assertEqual(summary["samples"], 10)
        self.assertEqual(summary["mean_us"], 250)
        self.assertEqual(summary["max_us"], 1200)
        self.assertEqual(summary["ge_100us_fraction"], 0.4)
        self.assertEqual(summary["ge_1ms_fraction"], 0.1)
        self.assertEqual(summary["cpu_migration_samples"], 2)
        self.assertEqual(summary["cpu_migration_fraction"], 0.2)
        self.assertEqual(summary["sched_migrate_task_samples"], 1)
        self.assertEqual(summary["futex_wait_us"]["mean_us"], 2000)
        self.assertEqual(summary["futex_wait_us"]["ge_1ms_fraction"], 0.4)
        self.assertEqual(summary["futex_wake_calls"], 7)

    def test_ebpf_aggregates_align_with_prefill_decode_and_idle(self) -> None:
        output = """
@runqlat_count_1s[100]: 10
@runqlat_sum_us_1s[100]: 100
@futex_wait_count_1s[100]: 2
@futex_wait_sum_us_1s[100]: 2000
@runqlat_count_1s[101]: 20
@runqlat_sum_us_1s[101]: 400
@sched_migrate_task_1s[101]: 3
@runqlat_count_1s[102]: 5
@runqlat_sum_us_1s[102]: 25
"""
        report = {
            "benchmarks": [
                {
                    "requests": {
                        "successful": [
                            {
                                "request_start_time": 1000.0,
                                "request_end_time": 1002.0,
                                "time_to_first_token_ms": 1000.0,
                            }
                        ]
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "guidellm.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            aligned = summarize_phase_alignment(
                output,
                report_path,
                measurement_start_bpf_clock=100.0,
                measurement_end_bpf_clock=103.0,
                measurement_start_unix=1000.0,
            )

        self.assertTrue(aligned["valid"])
        phases = aligned["phases"]
        self.assertEqual(phases["prefill"]["samples"], 10)
        self.assertEqual(phases["prefill"]["futex_wait_us"]["mean_us"], 1000)
        self.assertEqual(phases["decode"]["samples"], 20)
        self.assertEqual(phases["decode"]["sched_migrate_task_samples"], 3)
        self.assertEqual(phases["idle"]["samples"], 5)

    def test_sudo_keepalive_validates_cached_credentials(self) -> None:
        result = mock.Mock(returncode=0)
        with mock.patch(
            "laptop_energy.runtime.subprocess.run", return_value=result
        ) as run:
            keepalive = SudoKeepalive(True, interval_seconds=60)
            keepalive.start()
            keepalive.ensure_ready()
            keepalive.stop()
        self.assertEqual(run.call_args_list[0].args[0], ["sudo", "-n", "-v"])

    def test_resume_rejects_a_cell_that_crossed_suspend(self) -> None:
        pilot = load_config(PROJECT_ROOT / "campaign.pilot.json")
        cell = policy_plan(
            pilot, {"short": 0.2, "medium": 0.1, "long": 0.05}
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "policy" / cell.cell_id / "summary.json"
            summary_path.parent.mkdir(parents=True)
            clean = {
                "cell_id": cell.cell_id,
                "validity": {"valid": True},
                "started_monotonic_s": 100.0,
                "finished_monotonic_s": 100.0 + cell.duration_seconds,
                "started_unix_s": 1000.0,
                "finished_unix_s": 1000.0 + cell.duration_seconds,
            }
            summary_path.write_text(json.dumps(clean), encoding="utf-8")
            runner = CampaignRunner(
                pilot,
                root,
                use_energy=False,
                use_ebpf=False,
            )
            self.assertIsNotNone(runner._load_resumable_summary(root, cell))

            clean["finished_unix_s"] += 60.0
            summary_path.write_text(json.dumps(clean), encoding="utf-8")
            self.assertIsNone(runner._load_resumable_summary(root, cell))

    def test_ebpf_boundaries_use_the_suspend_aware_boot_clock(self) -> None:
        with mock.patch(
            "laptop_energy.runtime.time.clock_gettime", return_value=123.5
        ) as clock_gettime:
            self.assertEqual(_bpf_clock_s(), 123.5)
        clock_gettime.assert_called_once_with(time.CLOCK_BOOTTIME)

    def test_port_probe_allows_reuse_after_recent_connections(self) -> None:
        probe = mock.MagicMock()
        probe.__enter__.return_value = probe
        with mock.patch("laptop_energy.preflight.socket.socket", return_value=probe):
            result = _port_available("127.0.0.1", 18081)

        probe.setsockopt.assert_called_once_with(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        probe.bind.assert_called_once_with(("127.0.0.1", 18081))
        self.assertTrue(result["available"])

    def test_ebpf_preserves_terminal_session_and_flushes_with_sigint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = mock.Mock()
            process.poll.return_value = None
            process.wait.return_value = None
            process.returncode = 0
            with mock.patch(
                "laptop_energy.runtime.subprocess.Popen", return_value=process
            ) as popen, mock.patch(
                "laptop_energy.runtime.time.sleep"
            ), mock.patch(
                "laptop_energy.runtime._bpf_clock_s", return_value=100.0
            ):
                collector = EbpfCollector(
                    Path(directory) / "probe.bt", Path(directory)
                )
                collector.start()
                assert collector.stdout is not None
                collector.stdout.write("@samples_250ms[401]: 7\n")
                collector.stdout.flush()
                status = collector.stop(100.0, 101.0)

            self.assertNotIn("start_new_session", popen.call_args.kwargs)
            process.send_signal.assert_called_once_with(signal.SIGINT)
            self.assertTrue(status["valid"])
            self.assertEqual(status["samples"], 7)
            self.assertEqual(status["total_samples_including_prelude"], 7)

    def test_perf_is_prearmed_disabled_and_controlled_over_stdin(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        process.stdin = mock.Mock()
        process.communicate.return_value = (
            "",
            "1.0,Joules,power/energy-pkg/\n",
        )
        process.returncode = 0
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "laptop_energy.runtime.subprocess.Popen", return_value=process
        ) as popen, mock.patch("laptop_energy.runtime.time.sleep"):
            collector = PerfEnergyCollector(
                load_config(PROJECT_ROOT / "campaign.smoke.json"), Path(directory)
            )
            collector.start()
            collector.enable()
            collector.disable()
            status = collector.stop()

        command = popen.call_args.args[0]
        self.assertIn("--delay=-1", command)
        self.assertIn("fd:0", command)
        self.assertEqual(
            [call.args[0] for call in process.stdin.write.call_args_list],
            ["enable\n", "disable\n"],
        )
        self.assertTrue(status["control_enabled"])


if __name__ == "__main__":
    unittest.main()
