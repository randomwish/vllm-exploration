#!/usr/bin/env bash
set -uo pipefail

CONFIG="${1:?campaign config is required}"
RESULTS_PARENT="${2:?results parent is required}"
HARNESS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ID="$(python3 - "$CONFIG" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["campaign_id"])
PY
)"
OUTPUT="$RESULTS_PARENT/$CAMPAIGN_ID-$(date -u +%Y%m%dT%H%M%SZ)"
STATUS="failed"
ERROR="campaign exited before final status"

if [[ -r /run/sgenergy-delete.env ]]; then
  set -a
  source /run/sgenergy-delete.env
  set +a
fi

export HF_HOME="${HF_HOME:-/workspace/huggingface-cache}"
export TOKENIZERS_PARALLELISM=false

finalize() {
  local exit_code=$?
  local -a finalize_args
  trap - EXIT INT TERM
  if [[ $exit_code -eq 0 ]]; then
    STATUS="complete"
    ERROR=""
  else
    ERROR="campaign exited with code $exit_code"
  fi
  finalize_args=(--output "$OUTPUT" --status "$STATUS")
  if [[ -n "$ERROR" ]]; then
    finalize_args+=(--error "$ERROR")
  fi
  if [[ -d "$OUTPUT" ]]; then
    python3 -m sgenergy.cli finalize "${finalize_args[@]}" || true
  fi
  sync
  if [[ -d "$OUTPUT" && -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
    python3 -m sgenergy.cli delete-pod \
      --runpodctl "$HARNESS_DIR/runpodctl" || true
  fi
  exit "$exit_code"
}
trap finalize EXIT INT TERM

python3 -m sgenergy.cli campaign --config "$CONFIG" --output "$OUTPUT" --execute
