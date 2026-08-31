from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .config import CampaignConfig


def local_prelaunch(config: CampaignConfig) -> dict[str, Any]:
    errors = config.validate(launch=True)
    cli = (config.path.parent.parent.parent / config.runpod["cli_path"]).resolve()
    if not cli.exists():
        errors.append(f"runpodctl not found at {cli}")
        version = None
    elif not os.access(cli, os.X_OK):
        errors.append(f"runpodctl is not executable: {cli}")
        version = None
    else:
        result = subprocess.run(
            [str(cli), "version"], text=True, capture_output=True, check=False
        )
        version = result.stdout.strip() or result.stderr.strip()
        if result.returncode != 0:
            errors.append(f"runpodctl version failed: {version}")
    return {
        "ready": not errors,
        "errors": errors,
        "runpodctl": str(cli),
        "runpodctl_version": version,
        "planned_minutes": config.runpod["planned_minutes"],
        "hard_minutes": config.runpod["hard_minutes"],
        "maximum_gpu_cost_usd_at_observed_rate": config.calculated_max_gpu_cost_usd,
        "contacts_runpod": False,
        "creates_resources": False,
    }
