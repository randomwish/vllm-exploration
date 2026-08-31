#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/campaign.json"
MODE="check"

usage() {
  echo "Usage: $0 [--config PATH] [--check|--execute]"
}

while (($#)); do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --check)
      MODE="check"
      shift
      ;;
    --execute)
      MODE="execute"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m sgenergy.cli prelaunch --config "$CONFIG"

if [[ "$MODE" == "check" ]]; then
  echo "Local checks only. No Runpod API call was made."
  exit 0
fi

RUNPODCTL="$(python3 - "$CONFIG" "$REPO_ROOT" <<'PY'
import json
import pathlib
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
print((pathlib.Path(sys.argv[2]) / config["runpod"]["cli_path"]).resolve())
PY
)"

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "RUNPOD_API_KEY must be exported for provisioning and self-deletion." >&2
  exit 2
fi
if [[ "${RUNPOD_SELF_DELETE_ACK:-}" != "YES" ]]; then
  echo "Set RUNPOD_SELF_DELETE_ACK=YES to acknowledge ephemeral on-pod use of the API key for early self-deletion." >&2
  exit 2
fi

readarray -t VALUES < <(python3 - "$CONFIG" <<'PY'
import json
import sys

c = json.load(open(sys.argv[1], encoding="utf-8"))
r = c["runpod"]
for value in (
    c["campaign_id"], r["image"], r["cloud_type"], r["gpu_id"],
    r["gpu_count"], r["min_cuda_version"], r["container_disk_gb"],
    r["data_center_ids"], r["network_volume_id"], r["volume_mount_path"],
    r["ports"], r["hard_minutes"], r["docker_args"],
):
    print(value)
PY
)

CAMPAIGN_ID="${VALUES[0]}"
IMAGE="${VALUES[1]}"
CLOUD_TYPE="${VALUES[2]}"
GPU_ID="${VALUES[3]}"
GPU_COUNT="${VALUES[4]}"
MIN_CUDA="${VALUES[5]}"
CONTAINER_DISK="${VALUES[6]}"
DATA_CENTER_IDS="${VALUES[7]}"
VOLUME_ID="${VALUES[8]}"
VOLUME_MOUNT="${VALUES[9]}"
PORTS="${VALUES[10]}"
HARD_MINUTES="${VALUES[11]}"
DOCKER_ARGS="${VALUES[12]}"
POD_NAME="$CAMPAIGN_ID-$(date -u +%Y%m%dT%H%M%SZ)"
TERMINATE_AT="$(date -u -d "+${HARD_MINUTES} minutes" +%Y-%m-%dT%H:%M:%SZ)"
LAUNCH_DIR="$REPO_ROOT/.launch/$POD_NAME"
mkdir -p "$LAUNCH_DIR"

"$RUNPODCTL" user >"$LAUNCH_DIR/user.json"
"$RUNPODCTL" gpu list --include-unavailable >"$LAUNCH_DIR/gpus.json"
python3 - "$CONFIG" "$LAUNCH_DIR/user.json" "$LAUNCH_DIR/gpus.json" "$LAUNCH_DIR/pricing.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
user = json.load(open(sys.argv[2], encoding="utf-8"))
gpus = json.load(open(sys.argv[3], encoding="utf-8"))
runpod = config["runpod"]
matches = [gpu for gpu in gpus if gpu["gpuId"] == runpod["gpu_id"]]
if len(matches) != 1:
    raise SystemExit(f"expected one GPU catalog match, found {len(matches)}")
gpu = matches[0]
if not gpu.get("available"):
    raise SystemExit(f"{runpod['gpu_id']} is currently unavailable")
per_gpu = gpu.get("securePricePerHr")
if per_gpu is None:
    raise SystemExit(f"{runpod['gpu_id']} has no Secure Cloud price")
hourly = float(per_gpu) * int(runpod["gpu_count"])
maximum = hourly * float(runpod["hard_minutes"]) / 60.0
if hourly > float(runpod["observed_two_gpu_hourly_usd"]) + 0.01:
    raise SystemExit(
        f"live two-GPU price ${hourly:.2f}/h exceeds configured ${runpod['observed_two_gpu_hourly_usd']:.2f}/h"
    )
reserve = float(runpod["minimum_balance_reserve_usd"])
balance = float(user["clientBalance"])
if balance - maximum < reserve:
    raise SystemExit(
        f"balance ${balance:.2f} minus hard exposure ${maximum:.2f} leaves less than ${reserve:.2f} reserve"
    )
json.dump(
    {
        "gpu_id": runpod["gpu_id"],
        "per_gpu_hourly_usd": per_gpu,
        "two_gpu_hourly_usd": hourly,
        "hard_limit_gpu_usd": maximum,
        "balance_usd": balance,
        "balance_reserve_usd": balance - maximum,
    },
    open(sys.argv[4], "w", encoding="utf-8"),
    indent=2,
)
PY

POD_ID=""
cleanup_failed_launch() {
  if [[ -n "$POD_ID" ]]; then
    "$RUNPODCTL" pod delete "$POD_ID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_failed_launch ERR INT TERM

CREATE_STDERR="$LAUNCH_DIR/pod-create.stderr"
set +e
"$RUNPODCTL" pod create \
  --name "$POD_NAME" \
  --cloud-type "$CLOUD_TYPE" \
  --gpu-id "$GPU_ID" \
  --gpu-count "$GPU_COUNT" \
  --image "$IMAGE" \
  --docker-args "$DOCKER_ARGS" \
  --min-cuda-version "$MIN_CUDA" \
  --container-disk-in-gb "$CONTAINER_DISK" \
  --data-center-ids "$DATA_CENTER_IDS" \
  --network-volume-id "$VOLUME_ID" \
  --volume-mount-path "$VOLUME_MOUNT" \
  --ports "$PORTS" \
  --terminate-after "$TERMINATE_AT" \
  --wait \
  --wait-timeout 12m >"$LAUNCH_DIR/pod.json" 2>"$CREATE_STDERR"
CREATE_RC=$?
set -e

if [[ $CREATE_RC -ne 0 ]]; then
  POD_ID="$(python3 - "$CREATE_STDERR" <<'PY'
import json
import sys

pod_id = ""
for line in reversed(open(sys.argv[1], encoding="utf-8").read().splitlines()):
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and value.get("id"):
        pod_id = str(value["id"])
        break
print(pod_id)
PY
)"
  cat "$CREATE_STDERR" >&2
  cleanup_failed_launch
  trap - ERR INT TERM
  exit "$CREATE_RC"
fi

POD_ID="$(python3 - "$LAUNCH_DIR/pod.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["id"])
PY
)"
"$RUNPODCTL" ssh info "$POD_ID" >"$LAUNCH_DIR/ssh.json"

readarray -t SSH_VALUES < <(python3 - "$LAUNCH_DIR/ssh.json" <<'PY'
import json
import sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(s["ip"])
print(s["port"])
print(s["ssh_key"]["path"])
PY
)
SSH_IP="${SSH_VALUES[0]}"
SSH_PORT="${SSH_VALUES[1]}"
SSH_KEY="${SSH_VALUES[2]}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$SSH_PORT" "root@$SSH_IP")
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$SSH_PORT")

"${SSH[@]}" "mkdir -p '$VOLUME_MOUNT/sglang-energy-harness' '$VOLUME_MOUNT/sglang-energy-results'"
"${SCP[@]}" -r "$SCRIPT_DIR"/* "root@$SSH_IP:$VOLUME_MOUNT/sglang-energy-harness/"
"${SCP[@]}" "$CONFIG" "root@$SSH_IP:$VOLUME_MOUNT/sglang-energy-harness/selected-campaign.json"
"${SCP[@]}" "$RUNPODCTL" "root@$SSH_IP:$VOLUME_MOUNT/sglang-energy-harness/runpodctl"

printf '%s\n' "$RUNPOD_API_KEY" | "${SSH[@]}" "umask 077; IFS= read -r key; printf 'RUNPOD_API_KEY=%q\\nRUNPOD_POD_ID=%q\\n' \"\$key\" '$POD_ID' > /run/sgenergy-delete.env"

"${SSH[@]}" "chmod 700 '$VOLUME_MOUNT/sglang-energy-harness/runpodctl' '$VOLUME_MOUNT/sglang-energy-harness/pod_entrypoint.sh'; python3 -m pip install -r '$VOLUME_MOUNT/sglang-energy-harness/requirements-lock.txt'; python3 -m pip install --no-deps -e '$VOLUME_MOUNT/sglang-energy-harness'; python3 -c 'import aiohttp, pynvml, transformers'; setsid '$VOLUME_MOUNT/sglang-energy-harness/pod_entrypoint.sh' '$VOLUME_MOUNT/sglang-energy-harness/selected-campaign.json' '$VOLUME_MOUNT/sglang-energy-results' > '$VOLUME_MOUNT/sglang-energy-results/launcher.log' 2>&1 < /dev/null &"

"${SSH[@]}" "pgrep -af pod_entrypoint.sh; test -s '$VOLUME_MOUNT/sglang-energy-results/launcher.log' || true"
trap - ERR INT TERM

python3 - "$LAUNCH_DIR/launch.json" "$POD_ID" "$POD_NAME" "$TERMINATE_AT" <<'PY'
import json
import sys
json.dump(
    {"pod_id": sys.argv[2], "pod_name": sys.argv[3], "terminate_at": sys.argv[4]},
    open(sys.argv[1], "w", encoding="utf-8"),
    indent=2,
)
PY

echo "Campaign launched on pod $POD_ID. Hard termination: $TERMINATE_AT"
echo "Results: $VOLUME_MOUNT/sglang-energy-results"
