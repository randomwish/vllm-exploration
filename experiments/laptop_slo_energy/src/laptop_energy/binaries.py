from __future__ import annotations

import shutil
import sys
from pathlib import Path


def resolve_binary(command: str) -> str | None:
    """Resolve a configured binary, including a sibling in the active venv."""
    configured = Path(command).expanduser()
    if "/" in command:
        return str(configured.resolve()) if configured.exists() else None

    sibling = Path(sys.executable).parent / command
    if sibling.exists():
        return str(sibling.absolute())

    resolved = shutil.which(command)
    return str(Path(resolved).resolve()) if resolved else None
