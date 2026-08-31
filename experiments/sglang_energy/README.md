# SGLang H100 energy campaign harness

This directory contains the executable form of [`SGLANG-ENERGY-EXPERIMENT-DESIGN.md`](../../SGLANG-ENERGY-EXPERIMENT-DESIGN.md). The default command performs local checks only. Provisioning requires an explicit `--execute` flag.

## What is ready

- SGLang v0.5.16 CUDA 12.9 image pinned to registry digest `sha256:b688781f…`.
- Qwen model pinned to Hugging Face commit `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- 2× H100 SXM, BF16, TP=2 configuration.
- Four-stage symbolic and adaptive experiment matrix.
- Deterministic, checksummed, gzip-compressed token-ID traces.
- Native `/generate` streaming replay with exact output lengths.
- 100 ms multi-GPU board-energy and Hopper GPM collection.
- 500 ms SGLang, cgroup, and network collection.
- Per-run energy, latency, queue, telemetry-coverage, and error validation.
- New holdout seeds for the prefill/decode regime comparison.
- Detached execution, atomic state files, result checksums, immediate self-deletion, and a 198-minute Runpod termination backstop.

The frozen trace format uses SGLang's native `input_ids` request field and streaming endpoint, which are documented in SGLang's current [sampling-parameter reference](https://github.com/sgl-project/sglang/blob/main/docs/docs/basic_usage/sampling_params.mdx). The image tag follows the official [SGLang installation guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/install.mdx).

## Local verification

These commands do not contact Runpod:

```bash
cd /home/randomwish/dev/vllm-exploration/experiments/sglang_energy

PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sgenergy.cli validate-config --config campaign.json
PYTHONPATH=src python3 -m sgenergy.cli plan --config campaign.json --output /tmp/sgenergy-plan.json
./launch_runpod.sh --check
```

`--check` exits non-zero while a launch blocker remains. It never provisions anything.

## Prefix-cache follow-up

[`PREFIX-CACHE-EXPERIMENT.md`](PREFIX-CACHE-EXPERIMENT.md) contains the full pilot telemetry table, scoped relationship analysis, and the paired prewarmed follow-up design. Validate its 36-cell plan locally with:

```bash
cd /home/randomwish/dev/vllm-exploration/experiments/sglang_energy

PYTHONPATH=src python3 -m sgenergy.cli validate-config \
  --config prefix_cache_campaign.json --launch
PYTHONPATH=src python3 -m sgenergy.cli plan \
  --config prefix_cache_campaign.json \
  --output /tmp/sgenergy-prefix-plan.json
./launch_runpod.sh --config prefix_cache_campaign.json --check
```

The follow-up records board energy/power at 100 ms without GPM calls, while a separate 1 s process records Hopper GPM. It disables the nonfunctional direct-HBM probe. No pod is created unless `--execute` is supplied explicitly.

## Runpod placement

The campaign uses the existing 150 GB network volume `fheewtfzvw` in `EUR-IS-3`. The pod and volume must remain co-located. The volume holds approximately 72 GB of model weights plus traces and results.

The volume was created with:

```bash
./runpodctl network-volume get fheewtfzvw
```

Its ID is pinned in `runpod.network_volume_id` in `campaign.json`. Verify the complete launch configuration with:

```bash
./launch_runpod.sh --check
```

## Launch command—do not run until approved

The live read-only check on 2026-08-25 found H100 SXM Secure Cloud pricing of $3.29/GPU-hour. The launcher rechecks availability, price, account balance, and a $5 post-campaign balance reserve immediately before creating anything. It aborts if the live two-GPU price exceeds $6.58/hour.

```bash
export RUNPOD_API_KEY='temporary-key'
export RUNPOD_SELF_DELETE_ACK=YES

./launch_runpod.sh --execute
```

`--execute` performs these operations:

1. validates the immutable model/image, volume, live price, balance, CLI, and 10% deadline;
2. creates one 2×H100 pod in the volume's data center with `--terminate-after` set to 198 minutes;
3. copies this harness and the pinned local `runpodctl` 2.11 binary to the persistent workspace;
4. passes the API key over SSH into a mode-600 file under ephemeral `/run`, never the persistent volume;
5. launches `pod_entrypoint.sh` with `setsid`, so closing the laptop does not stop it;
6. runs hardware/GPM and three-workload model smoke gates before the experiment matrix;
7. writes and syncs final checksums, then deletes the pod immediately on success or failure;
8. relies on Runpod's independent 198-minute termination if the runner or self-delete path fails.

The campaign process needs the API key only for self-deletion. SGLang and telemetry subprocesses explicitly have `RUNPOD_API_KEY` removed from their environments. The current Runpod key is account-wide, so use a temporary key and revoke it after the pod disappears.

## Runtime layout

The volume-backed result directory is:

```text
/workspace/sglang-energy-results/
  launcher.log
  qwen35-sglang-energy-h100-pilot-<timestamp>/
    campaign.json
    events.jsonl
    state.json
    preflight/
    traces/
    runs/
    FINAL_STATUS.json
    SHA256SUMS
```

Each run contains exact settings, request outcomes, GPU telemetry, service telemetry, replay boundaries, server logs, and `validity.json`. An invalid cell is retained with its exclusion reason and is not silently turned into a result.

## Cost behavior

- Planned campaign: 180 minutes.
- Hard termination: `ceil(180 × 1.10) = 198` minutes.
- Current two-H100 rate: $6.58/hour.
- Maximum GPU exposure at that rate: $21.714.
- The pod self-deletes as soon as results are synced, so early completion does not wait for the deadline.
- The network volume persists and continues storage billing until it is explicitly deleted.

After retrieving the results and confirming their checksums, remove the volume separately:

```bash
./runpodctl network-volume delete <volume-id>
```

The pod must already be deleted before Runpod allows deletion of its attached volume.
