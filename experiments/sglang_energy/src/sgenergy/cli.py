from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config, write_json_atomic
from .finalize import delete_current_pod, finalize_results
from .plan import plan_as_dicts
from .preflight import local_prelaunch
from .traces import build_trace, safe_token_ids, synthetic_token_ids, verify_trace
from .validate import validate_run


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _config_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("campaign.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgenergy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config = subparsers.add_parser("validate-config")
    _config_parser(validate_config)
    validate_config.add_argument("--launch", action="store_true")

    plan = subparsers.add_parser("plan")
    _config_parser(plan)
    plan.add_argument("--output", type=Path)

    prelaunch = subparsers.add_parser("prelaunch")
    _config_parser(prelaunch)

    generate = subparsers.add_parser("generate-trace")
    _config_parser(generate)
    generate.add_argument("--workload", required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--output", type=Path, required=True)
    traffic = generate.add_mutually_exclusive_group(required=True)
    traffic.add_argument("--rate", type=float)
    traffic.add_argument("--closed-loop-count", type=int)
    generate.add_argument("--duration", type=float)
    generate.add_argument("--tokenizer")
    generate.add_argument("--revision")
    generate.add_argument("--vocab-size", type=int)

    verify = subparsers.add_parser("verify-trace")
    verify.add_argument("trace_dir", type=Path)

    replay = subparsers.add_parser("replay")
    replay.add_argument("trace_dir", type=Path)
    replay.add_argument("--base-url", default="http://127.0.0.1:30000")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--watchdog", type=float, required=True)
    replay.add_argument("--max-concurrency", type=int)

    cell = subparsers.add_parser("run-cell")
    _config_parser(cell)
    cell.add_argument("trace_dir", type=Path)
    cell.add_argument("--run-dir", type=Path, required=True)
    cell.add_argument("--max-running-requests", type=int, required=True)
    cell.add_argument("--chunked-prefill-size", type=int, required=True)
    cell.add_argument("--radix-cache", choices=("on", "off"), required=True)
    cell.add_argument("--watchdog", type=float, required=True)
    cell.add_argument("--max-concurrency", type=int)

    validate = subparsers.add_parser("validate-run")
    _config_parser(validate)
    validate.add_argument("run_dir", type=Path)

    campaign = subparsers.add_parser("campaign")
    _config_parser(campaign)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--execute", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--status", required=True)
    finalize.add_argument("--error")
    finalize.add_argument("--delete-pod", action="store_true")
    finalize.add_argument("--runpodctl", type=Path, default=Path("runpodctl"))
    delete_pod = subparsers.add_parser("delete-pod")
    delete_pod.add_argument("--runpodctl", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify-trace":
            _print(verify_trace(args.trace_dir))
            return 0
        if args.command == "replay":
            import asyncio

            from .replay import replay_trace

            summary = asyncio.run(
                replay_trace(
                    args.trace_dir,
                    base_url=args.base_url,
                    output_path=args.output,
                    watchdog_seconds=args.watchdog,
                    max_concurrency=args.max_concurrency,
                )
            )
            _print(summary)
            return 0
        if args.command == "finalize":
            finalize_results(args.output, status=args.status, error=args.error)
            result = {"finalized": True, "deleted": False}
            if args.delete_pod:
                result["delete_result"] = delete_current_pod(args.runpodctl)
                result["deleted"] = True
            _print(result)
            return 0
        if args.command == "delete-pod":
            _print(delete_current_pod(args.runpodctl))
            return 0

        config = load_config(args.config)
        if args.command == "validate-config":
            errors = config.validate(launch=args.launch)
            _print({"valid": not errors, "errors": errors})
            return 0 if not errors else 2
        if args.command == "plan":
            value = plan_as_dicts(config)
            if args.output:
                write_json_atomic(args.output, value)
            _print(value)
            return 0
        if args.command == "prelaunch":
            value = local_prelaunch(config)
            _print(value)
            return 0 if value["ready"] else 2
        if args.command == "generate-trace":
            if args.tokenizer and args.vocab_size:
                raise ValueError("use only one of --tokenizer and --vocab-size")
            if args.tokenizer:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer,
                    revision=args.revision,
                    trust_remote_code=True,
                )
                token_pool = safe_token_ids(tokenizer)
            elif args.vocab_size:
                token_pool = synthetic_token_ids(args.vocab_size)
            else:
                raise ValueError("provide --tokenizer or --vocab-size")
            metadata = build_trace(
                args.output,
                workload=config.workloads[args.workload],
                seed=args.seed,
                token_pool=token_pool,
                prefix_groups=int(config.design["prefix_groups"]),
                rate=args.rate,
                duration=args.duration,
                closed_loop_count=args.closed_loop_count,
            )
            _print(metadata)
            return 0
        if args.command == "run-cell":
            from .run_cell import run_cell

            summary = run_cell(
                config,
                trace_dir=args.trace_dir,
                run_dir=args.run_dir,
                max_running_requests=args.max_running_requests,
                chunked_prefill_size=args.chunked_prefill_size,
                radix_cache=args.radix_cache == "on",
                watchdog_seconds=args.watchdog,
                max_concurrency=args.max_concurrency,
            )
            _print(summary)
            return 0 if summary["valid"] else 3
        if args.command == "validate-run":
            summary = validate_run(args.run_dir, config)
            _print(summary)
            return 0 if summary["valid"] else 3
        if args.command == "campaign":
            if not args.execute:
                _print(
                    {
                        "executed": False,
                        "reason": "campaign execution requires the explicit --execute flag",
                        "output": str(args.output),
                    }
                )
                return 2
            launch_errors = config.validate(launch=True)
            if launch_errors:
                _print({"executed": False, "errors": launch_errors})
                return 2
            experiment_kind = config.raw.get("experiment_kind", "full")
            if experiment_kind == "full":
                from .campaign import CampaignRunner

                runner = CampaignRunner(config, args.output)
            elif experiment_kind == "prefix_cache":
                from .prefix_cache import PrefixCacheRunner

                runner = PrefixCacheRunner(config, args.output)
            else:
                raise ValueError(f"unsupported experiment_kind: {experiment_kind}")
            runner.run()
            return 0
        raise AssertionError(f"unhandled command {args.command}")
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
