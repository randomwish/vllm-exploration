from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ConfigError, load_config, write_json
from .plan import full_symbolic_plan
from .preflight import inspect
from .runtime import CampaignRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the laptop SLO-qualified energy MVP"
    )
    parser.add_argument("--config", type=Path, default=Path("campaign.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-config", help="Validate campaign.json")
    preflight = subparsers.add_parser("preflight", help="Inspect local prerequisites")
    preflight.add_argument(
        "--privileged",
        action="store_true",
        help="also require non-interactive sudo to be ready",
    )
    plan = subparsers.add_parser("plan", help="Print or save the symbolic plan")
    plan.add_argument("--output", type=Path)
    run = subparsers.add_parser("run", help="Print the plan or execute the campaign")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--output-root", type=Path, default=Path("results"))
    run.add_argument(
        "--resume",
        type=Path,
        help="resume a partial run directory and replace invalid cells",
    )
    run.add_argument("--skip-energy", action="store_true")
    run.add_argument("--skip-ebpf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(str(exc))
        return 2

    if args.command == "validate-config":
        print(json.dumps({"valid": True, "config": str(config.path)}, indent=2))
        return 0
    if args.command == "preflight":
        result = inspect(config, privileged=args.privileged)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready_to_execute"] else 1
    if args.command == "plan":
        result = full_symbolic_plan(config)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        if not args.execute:
            print(json.dumps(full_symbolic_plan(config), indent=2, sort_keys=True))
            print("\nDry run only. Add --execute to launch the local campaign.")
            return 0
        runner = CampaignRunner(
            config,
            args.output_root,
            use_energy=not args.skip_energy,
            use_ebpf=not args.skip_ebpf,
            resume_root=args.resume,
        )
        try:
            root = runner.run()
        except KeyboardInterrupt:
            print(
                "Campaign interrupted. Resume the same output directory with --resume.",
            )
            return 130
        terminal = json.loads((root / "FINAL_STATUS.json").read_text(encoding="utf-8"))
        status = str(terminal["status"])
        print(json.dumps({"status": status, "output": str(root)}, indent=2))
        return 0 if status in ("complete", "complete_evidence_limited") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
