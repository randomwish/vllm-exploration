# Instrumented vLLM exploration

This repository runs a small inference program against an instrumented vLLM
checkout. The goal is to observe scheduler, model-runner, attention, and sampler
markers on a real NVIDIA GPU.

The entry point is [`intro.py`](intro.py). It loads
`Qwen/Qwen2.5-1.5B-Instruct`, generates four short responses, and uses a small
context window so the first GPU run is practical on a 24 GB card.

## Target environment

Use a Runpod pod with the following characteristics:

- one NVIDIA GeForce RTX 4090;
- a CUDA 13.0 host;
- the PyTorch image `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-cluster`;
- Python 3.12 on Ubuntu 24.04;
- at least 30 GB of disk space.

The host CUDA filter and the container image must both target CUDA 13.0. The
host's reported CUDA version comes from the NVIDIA driver, so verify it with
`nvidia-smi` after SSH is ready.

The current vLLM checkout pins `torch==2.13.0`. Check the actual PyTorch version
inside the image instead of relying on its image tag:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
PY
```

The CUDA major and minor versions, the PyTorch build, and the vLLM checkout need
to be compatible. If the base image has a different PyTorch version, let the
normal dependency install replace it before trying the `--no-deps` shortcut.

## Create the pod

This command uses the image and GPU combination from the captured Runpod
preflight. `POD_ID` is read from the JSON returned by `runpodctl`.

```bash
POD_NAME="vllm-exploration-$(date +%s)"
POD_JSON="$(
  runpodctl pod create \
    --name "$POD_NAME" \
    --cloud-type COMMUNITY \
    --gpu-id "NVIDIA GeForce RTX 4090" \
    --image "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-cluster" \
    --min-cuda-version 13.0 \
    --container-disk-in-gb 30 \
    --volume-in-gb 30 \
    --ports 22/tcp \
    --public-ip \
    --wait
)"
POD_ID="$(jq -r '.id' <<<"$POD_JSON")"
printf 'POD_ID=%s\n' "$POD_ID"
```

Use `--wait` so the command returns after the public SSH port accepts a
connection. Keep `POD_ID` for the rest of the session.

## Verify SSH and network access

Retrieve the current SSH endpoint before every new connection:

```bash
SSH_INFO="$(runpodctl ssh info "$POD_ID")"

SSH_IP="$(jq -r '.ip' <<<"$SSH_INFO")"
SSH_PORT="$(jq -r '.port' <<<"$SSH_INFO")"
SSH_KEY="$(jq -r '.ssh_key.path' <<<"$SSH_INFO")"

SSH=(ssh -i "$SSH_KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -p "$SSH_PORT" \
  "root@$SSH_IP")

"${SSH[@]}" 'nvidia-smi'
```

The captured preflight reported an RTX 4090 with 24,564 MiB of memory, driver
580.95.05, and CUDA 13.0. The shallow clone below completed in about 13 seconds
on that host:

```bash
"${SSH[@]}" 'bash -se' <<'REMOTE'
set -euo pipefail
cd /workspace
time git clone --depth 1 --branch instrumented https://github.com/randomwish/vllm.git network-check-vllm
rm -rf -- /workspace/network-check-vllm
REMOTE
```

## Set up the repository

Run these commands after connecting to the pod as `root`:

```bash
cd /workspace
git clone https://github.com/randomwish/vllm-exploration.git
cd vllm-exploration

# Keep the uv cache on the workspace so a rerun can reuse downloaded metadata
# and wheels. Omit this line if the workspace is not persistent.
export UV_CACHE_DIR=/workspace/uv-cache

bash setup.sh
source vllm_source/.venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/workspace/huggingface-cache

python intro.py
```

`setup.sh` clones the instrumented vLLM branch, creates
`vllm_source/.venv`, and installs vLLM with `VLLM_USE_PRECOMPILED=1`.

In the captured run, the repository clones succeeded, but the setup script
stopped at `pip install -q uv` with `externally-managed-environment` before it
created the virtual environment. The `source` and `python intro.py` commands
therefore did not run. If a failed run already created `vllm_source`, use the
recovery procedure below instead of cloning it again.

## Connect to the pod again

`runpodctl ssh info` returns connection details. It does not open a shell. From
the same local terminal, reuse the `SSH` array from the original session:

```bash
"${SSH[@]}"
```

From a new terminal, retrieve the current host, port, and key:

```bash
POD_ID="your-pod-id"

SSH_INFO="$(runpodctl ssh info "$POD_ID")"
SSH_IP="$(jq -r '.ip' <<<"$SSH_INFO")"
SSH_PORT="$(jq -r '.port' <<<"$SSH_INFO")"
SSH_KEY="$(jq -r '.ssh_key.path' <<<"$SSH_INFO")"

ssh -i "$SSH_KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -p "$SSH_PORT" \
  "root@$SSH_IP"
```

Read `ssh info` again after a pod stop or restart. Runpod can assign a new
external SSH port, and the first response after startup can be stale while
`sshd` is still coming up.

## Recover from the PEP 668 failure

Ubuntu 24.04 marks its system Python as externally managed. If `bash setup.sh`
reports `externally-managed-environment`, it has stopped before creating
`vllm_source/.venv`.

After the failure, recover the existing checkout without cloning again:

```bash
cd /workspace/vllm-exploration
export UV_CACHE_DIR=/workspace/uv-cache
export PATH="$HOME/.local/bin:$PATH"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd vllm_source
uv venv .venv
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

Run the PyTorch verification in the next section before running `intro.py`.
The recovery commands are a rerun path for the observed failure; the captured
log did not reach this step.

Installing with `python3 -m pip install --break-system-packages uv` can work on
a disposable pod, but it changes the system interpreter and is not the default
path in this repository.

## Reduce repeat install time

The first full install must resolve vLLM's runtime dependencies. On later source
changes, skip dependency resolution only when those dependencies are already
installed:

```bash
cd /workspace/vllm-exploration/vllm_source
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

VLLM_USE_PRECOMPILED=1 uv pip install --no-deps -e .
uv pip check
```

`--no-deps` does not install missing packages. If `uv pip check` reports missing
dependencies, rerun the normal command:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

`VLLM_USE_PRECOMPILED=1` skips local native-extension compilation when matching
precompiled artifacts are available. It does not skip dependency resolution.

## Verify the instrumented run

Before running the model, verify the GPU and PyTorch build:

```bash
nvidia-smi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
assert torch.__version__.startswith("2.13."), torch.__version__
assert torch.version.cuda.startswith("13.0"), torch.version.cuda
PY
```

A successful `python intro.py` run should include:

- `GPU KV cache size: N tokens`;
- GPU `block_size` 16 (CPU uses 128);
- the selected attention backend;
- `[add_request]`;
- `[schedule] decided`;
- `[model_runner]`;
- `[attn_meta]`; and
- `[sampler]`.

The captured session did not reach this section. These are acceptance checks
for a completed run, not results from the failed setup attempt.

Capture the first run from the local machine with `tee`. Set `pipefail` before
the pipeline so a failed SSH command is not hidden by `tee`:

```bash
set -o pipefail

"${SSH[@]}" 'bash -se' <<'REMOTE' 2>&1 | tee first-gpu-run.log
set -euo pipefail
cd /workspace/vllm-exploration
source vllm_source/.venv/bin/activate
export HF_HOME=/workspace/huggingface-cache
python intro.py
REMOTE
```

## Troubleshooting

### The clone is slow

If cloning or dependency resolution stays near 1 MB/s during the first two
minutes, the pod host may have poor network performance. Terminate that pod and
choose another host rather than spending the session waiting on downloads.

### SSH reports connection refused

Run `runpodctl ssh info "$POD_ID"` again, wait briefly for `sshd` to start, and
retry with the new port. Do not assume the port from the previous pod session is
still valid.

### `torch` has the wrong version

The source checkout declares `torch==2.13.0`. Use the normal install command so
`uv` can install the compatible build. Do not use `--no-deps` until the PyTorch
version and CUDA version pass the verification commands above.

### The model downloads again

Set `HF_HOME=/workspace/huggingface-cache` before running `intro.py`. Keep that
directory on a persistent workspace or network volume if you want it to survive
pod replacement.

## Clean up

When the experiment is complete, delete the pod. Do not leave it running or
stopped:

```bash
runpodctl pod delete "$POD_ID"
runpodctl pod list
```

The final command should return `[]`. Record the approximate session cost before
deleting the pod if you need it for the experiment log.

For the shorter operational checklist, see
[`DEPLOY-CHECKLIST.md`](DEPLOY-CHECKLIST.md).
