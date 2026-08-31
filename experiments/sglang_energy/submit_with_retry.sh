#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ATTEMPTS="${SGLANG_SUBMIT_ATTEMPTS:-10}"
DELAY_SECONDS="${SGLANG_SUBMIT_DELAY_SECONDS:-45}"

for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
  printf 'Submission attempt %d/%d at %s\n' "$attempt" "$ATTEMPTS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if "$SCRIPT_DIR/launch_runpod.sh" --execute; then
    exit 0
  fi
  if ((attempt < ATTEMPTS)); then
    printf 'No usable allocation; retrying in %s seconds.\n' "$DELAY_SECONDS"
    sleep "$DELAY_SECONDS"
  fi
done

printf 'No usable two-H100 allocation after %d attempts.\n' "$ATTEMPTS" >&2
exit 1
