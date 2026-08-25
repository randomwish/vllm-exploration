# GPU deployment checklist

## RunPod selection

- RunPod → Community Cloud → 1× RTX 4090 → On-Demand.
- CUDA filter: `13.0` (the current checkout's CUDA target). Do not use a 12.8 host;
  the development checkout requires the newer host driver/runtime.
- Check the host card's network-speed rating. Skip bottom-tier hosts. If cloning or
  resolving is crawling at roughly 1 MB/s during the first two minutes, terminate
  the pod and draw another host.
- Use a PyTorch CUDA 13.0 image such as
  `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-cluster` with a 30 GB volume
  disk. The host CUDA filter and the container image must both be CUDA 13.0.

## Bootstrap and verify

On the pod:

```bash
cd /workspace
git clone https://github.com/randomwish/vllm-exploration.git
cd vllm-exploration
bash setup.sh
source vllm_source/.venv/bin/activate
export HF_HOME=/workspace/huggingface-cache

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
assert torch.__version__.startswith("2.13."), torch.__version__
assert torch.version.cuda.startswith("13.0"), torch.version.cuda
PY

python intro.py
```

Success requires the startup log line `GPU KV cache size: N tokens`. Record `N`,
the GPU `block_size` (expect 16 on GPU versus 128 on CPU), and the attention backend
name. Also confirm the instrumented markers `[add_request]`, `[schedule] decided`,
`[model_runner]`, `[attn_meta]`, and `[sampler]`.

When finished, terminate/delete the pod; never leave it stopped. Verify that no pod
is still running and record the approximate session cost.
