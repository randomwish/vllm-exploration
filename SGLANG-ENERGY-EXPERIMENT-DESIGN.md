# SGLang energy experiment: H100 pilot protocol

Status: design only; no Runpod resources have been created.

## Decision this pilot should support

For a fixed model, hardware allocation, request trace, and latency requirement, determine whether an SGLang scheduler or cache setting reduces total GPU energy-to-finish without reducing completed work or hiding overload in a longer queue.

This is a one-model pilot using `Qwen/Qwen3.5-35B-A3B`, BF16, tensor parallelism 2, and two H100 SXM GPUs. A second model should be a separate confirmation campaign after this protocol passes its operational and measurement gates. The SGLang image digest and model revision remain launch blockers until the live compatibility preflight succeeds.

The independent variables are:

1. `--max-running-requests` near the scheduler saturation knee;
2. `--chunked-prefill-size` for a prompt-heavy workload;
3. RadixAttention prefix-cache reuse, compared with `--disable-radix-cache`;
4. balanced, prefill-heavy, and decode-heavy request regimes.

Power is observed, not controlled. The earlier Runpod preflight showed that power-limit reads work but writes are blocked on a normal pod.

## Experimental contract

| Item | Fixed decision |
|---|---|
| Hardware | One pod with 2× H100 SXM; verify GPU UUIDs and the TP link topology before loading the model |
| Software | One immutable container digest, SGLang commit/version, CUDA/driver combination, tokenizer revision, and model revision for the full campaign |
| Model | `Qwen/Qwen3.5-35B-A3B`, BF16, TP=2; no quantization or speculative decoding |
| Generation | Deterministic decoding, exact requested output length, EOS ignored if the preflight verifies that behavior, and identical sampling arguments across cells |
| Request traces | Generated once, saved, checksummed, then replayed byte-for-byte for every paired condition |
| Traffic | Open-loop Poisson arrival times for load curves; offered request rate is fixed in absolute requests/s within each comparison block |
| Cache control | Radix cache disabled for stages 1, 2, and 4. Stage 3 is the only cache experiment |
| Warm-up | Model warm-up is outside measurement. Cache warm-up is outside measurement only in the explicitly labeled warm-cache subtest |
| Telemetry | 100 ms GPU/energy/GPM samples, 500 ms SGLang and cgroup samples, and one record per request |
| Measurement windows | Report both arrival-active energy and finite-cohort energy through the last completed request |
| Order | Randomize candidate order inside each block; run an anchor baseline at the start and end of each stage to detect drift |

SGLang exposes `--max-running-requests`, `--chunked-prefill-size`, and `--disable-radix-cache` as server launch arguments, so changing any of them requires a controlled server restart. The benchmark client supports request rate, maximum concurrency, fixed random lengths, shared-prefix generation, seeds, warm-up requests, cache flushing, and JSONL output. See the current [server arguments](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py) and [benchmark guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx). Exact command lines will be generated only after the pod preflight records the installed version's `--help` output.

## Workload definitions

| Workload ID | Input tokens | Output tokens | Prefix pattern | Purpose |
|---|---:|---:|---|---|
| `BAL` | 512 | 128 | Unique | Baseline and scheduler knee |
| `PF` | 8,192 | 128 | Unique | Prefill-heavy and chunk-size study |
| `DEC` | 512 | 1,024 | Unique | Decode-heavy regime |
| `PX0` | 4,096 | 128 | Unique from the first non-BOS token | Prefix-cache near-zero-reuse control |
| `PX50` | 4,096 | 128 | 2,048 shared + 2,048 unique | Moderate reuse |
| `PX87` | 4,096 | 128 | 3,584 shared + 512 unique | High reuse without a nearly empty unique suffix |

Lengths must be validated using the server tokenizer, not assumed from source strings. The replay manifest stores token IDs or tokenizer-verified prompts, arrival offsets, group ID, requested output length, and seed. Runs are invalid if token counts differ between paired conditions.

For shared-prefix traces, use 16 equally popular groups and round the request count to a multiple of 16. This avoids confusing cache reuse with a Zipf popularity effect. A later experiment can deliberately vary popularity.

## Stage 0 — compatibility and measurement preflight

Do not proceed to the benchmark matrix until all gates pass.

1. Record the pod offer price, GPU UUIDs, driver, CUDA, topology, NVLink state, container digest, SGLang version, model revision, and tokenizer revision.
2. Verify the model starts in BF16 with TP=2 and produces valid output for all three maximum length pairs.
3. Capture the effective server configuration. Resolve the explicit baseline chunk size from the running SGLang version rather than accepting an undocumented automatic value.
4. Run one 60-second balanced smoke trace and verify request-level JSONL, `/metrics`, cumulative energy, and Hopper GPM collection.
5. Cross-check the summed cumulative-energy delta against trapezoidal integration of instantaneous board power. Difference must be at most 2%.
6. Run a short NCCL/TP transfer and confirm the expected NVLink path and no increase in link, ECC, Xid, or PCIe replay errors.
7. Verify the experiment runner can flush/sync results to the persistent volume and delete only its own pod.

The SGLang metrics needed here include running and queued requests, generation throughput, cache-hit rate, token usage, and KV available/used/evictable tokens. Their current names and definitions are in the official [metrics collector source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py) and [observability guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/observability.mdx).

## Stage 1 — locate the saturation knee and tune running requests

### 1A. Baseline capacity scout

Use `BAL`, Radix cache off, and an explicit baseline chunk size. Sweep client maximum concurrency:

`32, 64, 128, 256, 512`

Each screen cell has 45 seconds of measured steady work. Stop the ladder early after two consecutive doublings improve completed request throughput by less than 5%, or if a validity/safety gate fails. Let `C0` be the median completed requests/s over the plateau cells. The screen is descriptive; it does not select a winner.

### 1B. Baseline open-loop curve

Run one 45-second screen at each of these fixed absolute rates:

`0.50 C0, 0.70 C0, 0.85 C0, 0.95 C0, 1.05 C0`

Use the screen to identify a provisional knee bracket. Then replay two paired 90-second seeds at no more than three points: the provisional lower and upper bounds and the immediately preceding load point. Each confirmation trace has 90 seconds of arrivals followed by a bounded drain. The saturation knee is reported as an interval:

- lower bound: the highest rate with no saturation signal;
- upper bound: the lowest rate with at least one saturation signal.

A cell has a saturation signal when any of these occurs:

- the request queue is non-zero for at least 20% of arrival-active samples and its median is greater than zero;
- the queue remains non-zero when arrivals stop and drain time exceeds 10% of the 90-second arrival window;
- p95 end-to-end latency breaches the declared SLO or is more than twice the preceding load point;
- offered load rises by at least 10%, completed throughput rises by less than 5%, and queueing increases.

This follows the USE interpretation of saturation as work waiting to be serviced, rather than treating high utilization alone as proof of a bottleneck. See Brendan Gregg's [USE Method](https://www.brendangregg.com/usemethod.html).

### 1C. `max-running-requests` screen and confirmation

Screen `32, 64, 128, 256` at a fixed `0.90 C0` with one 45-second paired trace. Prune a value if the preflight shows insufficient KV capacity or a request failure.

Retain the explicit baseline and at most one candidate. Confirm both at `0.90 C0` and `1.05 C0` with two 90-second paired seeds. The loads remain fractions of baseline `C0`; they are not renormalized to each candidate's capacity.

Selection rule: retain a candidate only if every paired seed completes the same work, passes the latency gate, and does not increase median joules per successful request. Screening results alone cannot declare a winner.

## Stage 2 — chunked-prefill trade-off

Use `PF`, unique prompts, Radix cache off, and the stage-1 `max-running-requests` winner. If stage 1 has no winner, use its explicit baseline.

First estimate `C_pf` with a shortened three-point closed-loop concurrency scout for `PF`. Then screen these explicit chunk sizes at `0.85 C_pf`:

`-1 (disabled), 2,048, 4,096, 8,192`

The 8,192-token prompt therefore produces approximately four, two, or one chunk at the three enabled settings. Confirm the baseline and at most one candidate at `0.90 C_pf` using two 90-second paired seeds.

Finally, complete a four-cell interaction screen at `0.90 C_pf`, reusing any cells already measured in the confirmation block:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Running requests | Stage-1 baseline | Stage-1 retained value |
| Chunk size | Explicit chunk baseline | Stage-2 retained value |

This 2×2 check prevents a main-effect choice from silently depending on the other scheduler setting. If there is no retained value for a factor, omit the duplicate cells.

Hypothesis: smaller chunks can reduce TTFT and queue blocking near saturation, but may add scheduling work and reduce tensor efficiency. Board energy is the outcome; H100 SM occupancy, HMMA activity, DRAM activity, and HBM power are the mechanism evidence.

## Stage 3 — prefix-cache reuse

Fix the selected scheduler combination from stage 2. Run the same request bytes with Radix caching enabled and with `--disable-radix-cache`.

### 3A. Cold-cohort primary comparison

Before each measured trace, flush the cache. The finite cohort includes the first fill of each prefix group, so this measures practical energy-to-finish rather than only best-case steady state.

First estimate `C_px` with a shortened three-point closed-loop concurrency scout using cache-disabled `PX0`. Then screen the 2×3 matrix once at `0.85 C_px`:

| Cache state | `PX0` | `PX50` | `PX87` |
|---|---:|---:|---:|
| Disabled | Run | Run | Run |
| Enabled | Run | Run | Run |

Confirm `PX0` and `PX87`, cache on and off, at `0.90 C_px` with two paired 90-second seeds. Offered load remains anchored to the cache-disabled `PX0` baseline.

### 3B. Warm-cache mechanism check

For `PX87` only, issue one unmeasured priming request per prefix group, verify the expected rise in cache-hit and evictable-token metrics, then run one 45-second paired screen at `0.85 C_px`. Label this result warm-cache; do not combine it with cold-cohort energy.

The analysis relates joules saved to observed cache-hit rate and KV pressure. In particular, inspect `sglang:cache_hit_rate`, `sglang:kv_available_tokens`, `sglang:kv_used_tokens`, and `sglang:kv_evictable_tokens`. Shared-prefix generation controls the workload's potential reuse, but the observed cache metrics determine whether reuse actually occurred.

## Stage 4 — separate prefill-heavy and decode-heavy regimes

This stage tests whether the selected scheduler combination generalizes rather than averaging prefill and decode into one workload.

For each of `BAL`, `PF`, and `DEC`:

1. reuse `C0` for `BAL` and `C_pf` for `PF`; estimate only `DEC` capacity with a shortened concurrency scout;
2. compare the original explicit scheduler baseline with the selected stage-2 combination;
3. use unique prompts and disable Radix cache;
4. replay two paired 90-second seeds at `0.85 C_w`.

Within a workload, both settings use the same absolute request rate. Do not interpret energy differences between `BAL`, `PF`, and `DEC` as a scheduler treatment effect: their useful work differs. Report them as separate regimes.

## Measurements and USE interpretation

| Layer | Utilization | Saturation | Errors / validity |
|---|---|---|---|
| GPU compute | SM activity, occupancy, HMMA/other activity, GPU busy | High activity plus growing scheduler queue; power/thermal clock-event time | Xid, clock events, unexpected GPU reset |
| HBM/KV | DRAM activity, HBM power, framebuffer and KV-token use | Low KV headroom, evictable growth, retractions/pauses, allocation pressure | OOM, ECC, row-remap change |
| SGLang scheduler | Running requests, output-token throughput, batch/token use | Queue depth and time, tail-latency growth, long drain | Request errors, timeouts, wrong token counts |
| TP fabric | NVLink bytes/rates and NCCL timing in selected traces | Communication time dominates or link rate plateaus | Link down, CRC/replay/recovery counter delta |
| CPU/container | CPU quota use, memory use | CFS throttling or memory pressure while GPU has gaps | OOM, fail-count or process-exit delta |
| Service | Completed requests and tokens | TTFT, TPOT, end-to-end tails and unfinished queue | Non-2xx response, invalid output, disconnect |

Primary outcomes, in order:

1. summed two-GPU board energy for the finite cohort;
2. joules per successful request and joules per 1,000 output tokens;
3. completed request and output-token throughput;
4. p50/p95/p99 TTFT, TPOT, and end-to-end latency;
5. arrival-active energy and drain duration.

Do not subtract idle power from the headline energy. Energy-to-finish is the direct cumulative-energy increase across both GPUs from the chosen start boundary through the chosen end boundary. Idle power may be reported separately to explain differences in drain time.

Mechanism outcomes include 100 ms power, clocks, temperature, SM occupancy/activity, HMMA, DRAM, PCIe/NVLink, and sampled HBM power. Estimated MFU metrics may be retained as annotations but are not substitutes for board-energy or hardware-activity measurements.

## Run validity and analysis rules

A run is valid only when all applicable checks pass:

- all intended requests finish successfully within the drain watchdog;
- paired cells have identical request bytes, arrival offsets, input tokens, and requested output tokens;
- output lengths are exact, with no unexpected EOS truncation;
- at least 98% of expected 100 ms samples are present and no telemetry gap exceeds 250 ms;
- cumulative energy and integrated power agree within 2%;
- GPU UUIDs, topology, software hashes, and effective launch arguments match the manifest;
- no new Xid, ECC, row-remap, PCIe replay, NVLink error, OOM, or process-restart event occurs;
- neither GPU reports thermal, power-brake, or reliability clock throttling. Normal software-power-cap activity at the provider's unchanged enforced limit is recorded rather than automatically rejected;
- the end-of-stage anchor baseline has energy/request within 3% of the opening anchor. Otherwise, investigate drift and repeat the block.

The full request trace is the replication unit. Screening cells are descriptive. Confirmation reports each paired seed and the median paired percentage difference. Two seeds are enough for a budgeted pilot consistency check, not a narrow confidence interval; any setting proposed for deployment requires a later confirmation with at least three new paired seeds. Stage 4 uses trace seeds not seen during tuning. A setting is not called better if one seed saves energy but another increases it, if the median saving is less than 5%, or if it violates the latency/work gate. Smaller effects are reported as indistinguishable in this pilot.

Before launch, declare absolute p95 TTFT, TPOT, and end-to-end SLOs. If no production SLO exists, label the campaign exploratory and use these predeclared non-inferiority guards: completed throughput no worse than 2%, p95 end-to-end no worse than 5%, and p95 TTFT/TPOT no worse than 10% relative to the paired baseline.

## Runtime, cost, and unattended cleanup

The runner uses stage budgets rather than a long idle timeout.

| Block | Maximum screen / confirm cells | Planned upper bound |
|---|---:|---:|
| Pod, software, model, telemetry, and deletion preflight | — | 20 min |
| Stage 1: capacity, knee, running requests, and closing anchor | 15 / 14 | 58 min |
| Stage 2: prefill scout, chunking, interaction, and closing anchor | 10 / 4 | 27 min |
| Stage 3: prefix scout, cold/warm reuse, and closing anchor | 12 / 8 | 36 min |
| Stage 4: decode scout and regime split | 3 / 12 | 34 min |
| Result sync, checksums, summary, and cleanup | — | 5 min |
| **Planned total** | **40 / 38** | **180 min** |
| **Hard pod lifetime: `ceil(180 × 1.10)`** | — | **198 min (3 h 18 min)** |

The cell counts are ceilings: adaptive pruning should normally finish earlier. At the previously observed offer price of $3.29 per H100-hour, the two-GPU hard-limit exposure is about `$6.58 × 3.30 = $21.71`, excluding persistent-volume charges. Re-read the live price before provisioning; if it is higher, recompute the limit and require an explicit budget decision.

The 10% buffer is a backstop, not normal runtime:

- after a successful or failed campaign, write a final status manifest, checksum the artifacts, `sync` the persistent volume, and immediately delete the pod;
- independently configure the Runpod hard termination for 198 minutes;
- the controller stops starting new cells when the remaining lifetime is less than the next cell's watchdog plus the five-minute finalization reserve;
- every screen cell has a 75-second watchdog; every confirmation cell has a 135-second watchdog;
- an invalid cell is recorded and skipped. It is not retried automatically if doing so would consume finalization reserve;
- use a temporary credential capable of deleting only the experiment pod if Runpod supports that scope. Never store a primary API key in the repository, result bundle, shell history, or logs;
- pod deletion does not delete the persistent volume. Review and delete the volume separately after retrieving the results if it is no longer needed.

If model download or startup exhausts the 20-minute preflight budget, abort, save diagnostics, and delete the pod. Do not silently consume the measurement budget or extend the hard deadline.

## Required artifacts

The persistent result directory should contain:

- `campaign.json`: immutable software, hardware, price, time budget, SLO, and parameter manifest;
- `traces/`: frozen request/arrival manifests and checksums;
- `runs/<run-id>/requests.jsonl`: request-level timestamps, token counts, and result status;
- `runs/<run-id>/gpu.parquet`: 100 ms GPU board-energy, power, state, and GPM samples;
- `runs/<run-id>/service.parquet`: 500 ms SGLang and cgroup samples;
- `runs/<run-id>/config.json`: exact server/client commands and effective settings;
- `runs/<run-id>/validity.json`: every gate with observed value and pass/fail result;
- `summary.csv` and `summary.md`: paired outcomes, saturation-knee brackets, and excluded-run reasons;
- `FINAL_STATUS.json` and `SHA256SUMS`: written before self-deletion.

## Launch blockers

Do not provision the H100 pod until these values are filled or procedures are verified:

- immutable SGLang image/commit compatible with the model;
- immutable model and tokenizer revision;
- explicit baseline `max-running-requests`, chunk size, memory fraction, attention backend, scheduling policy, and context length;
- absolute SLOs, or an explicit decision to use the exploratory relative guards;
- live two-H100 offer price and persistent-volume choice;
- deletion credential/endpoint and a safe test proving it deletes only the current pod;
- a campaign runner that implements deadlines, result sync, and final status atomically enough to survive failure.

Passing these blockers authorizes the later provisioning step; this document does not provision anything.
