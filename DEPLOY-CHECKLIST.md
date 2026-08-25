# GPU deployment checklist

## RunPod selection

- RunPod → Community Cloud → 1× RTX 4090 → On-Demand.
- CUDA filter: select the newest version listed. Do not use the 12.8 minimum when a
  newer version is available; the development-checkout torch may require a newer
  host driver.
- Check the host card's network-speed rating. Skip bottom-tier hosts. If cloning or
  resolving is crawling at roughly 1 MB/s during the first two minutes, terminate
  the pod and draw another host.
- Use a PyTorch cu12.8.1-or-newer template with a 30 GB volume disk.

## Bootstrap and verify

On the pod:

```bash
cd /workspace
git clone https://github.com/randomwish/vllm-exploration.git
cd vllm-exploration
bash setup.sh
source vllm_source/.venv/bin/activate
python intro.py
```

Success requires the startup log line `GPU KV cache size: N tokens`. Record `N`,
the GPU `block_size` (expect 16 on GPU versus 128 on CPU), and the attention backend
name. Also confirm the instrumented markers `[add_request]`, `[schedule] decided`,
`[model_runner]`, `[attn_meta]`, and `[sampler]`.

When finished, terminate/delete the pod; never leave it stopped. Verify that no pod
is still running and record the approximate session cost.
