# H100 pilot telemetry and prefix-cache follow-up

## Bottom line

The pilot contains 56 measured cells: 38 valid and 18 invalid. The strongest prefix-cache signal is the PX87 treatment. Across the two cold confirmation seeds, enabling radix cache reduced the descriptive mean from 105.98 to 70.44 J/request (33.5%), p95 TTFT from 553.6 to 115.7 ms, and p95 E2E from 8.77 to 2.56 s. Both cache-on confirmation cells were invalid because of telemetry timing gaps, so this is a strong hypothesis, not yet a confirmed effect.

The follow-up campaign fixes the comparison by prewarming every prefix group, pairing cache-off and cache-on runs by seed, alternating treatment order, rotating workload order, and separating the 100 ms energy collector from slower GPM calls.

## Prefix results from the pilot

Every row below is retained. `Valid` shows how many cells passed every request, watchdog, energy cross-check, GPU sampling, clock-event, and health gate. Means that include invalid cells are descriptive only.

| State | Reuse | Cache | Cells | Valid | req/s | J/req | p95 TTFT ms | p95 TPOT ms | p95 E2E ms | Mean cache hit | W/GPU | GPU busy % | Tensor % | DRAM % |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold screen | 0% | Off | 1 | 1 | 11.84 | 106.95 | 798.1 | 68.59 | 9,007.9 | 0.000 | 629.2 | 96.4 | 16.33 | 47.82 |
| Cold screen | 0% | On | 1 | 1 | 11.81 | 105.58 | 1,265.5 | 56.95 | 7,682.5 | 0.000 | 619.6 | 92.9 | 15.73 | 46.12 |
| Cold screen | 50% | Off | 1 | 0 | 11.23 | 102.22 | 7,570.7 | 130.47 | 16,778.1 | 0.000 | 649.5 | 95.7 | 18.24 | 47.43 |
| Cold screen | 50% | On | 1 | 1 | 11.92 | 88.95 | 221.2 | 25.40 | 3,400.5 | 0.386 | 527.1 | 92.8 | 10.87 | 43.83 |
| Cold screen | 87.5% | Off | 1 | 0 | 11.83 | 108.04 | 463.4 | 67.38 | 8,745.4 | 0.000 | 635.8 | 96.3 | 16.39 | 47.30 |
| Cold screen | 87.5% | On | 1 | 0 | 11.94 | 72.39 | 210.1 | 20.11 | 2,685.4 | 0.596 | 428.7 | 84.6 | 5.89 | 37.91 |
| Cold confirm | 0% | Off | 2 | 1 | 12.27 | 105.86 | 603.0 | 65.93 | 8,687.0 | 0.000 | 646.7 | 96.6 | 17.02 | 48.15 |
| Cold confirm | 0% | On | 2 | 0 | 12.26 | 103.44 | 3,787.8 | 73.01 | 10,844.3 | 0.000 | 637.4 | 92.2 | 16.69 | 46.02 |
| Cold confirm | 87.5% | Off | 2 | 1 | 12.28 | 105.98 | 553.6 | 66.93 | 8,767.3 | 0.000 | 648.9 | 96.8 | 16.86 | 48.19 |
| Cold confirm | 87.5% | On | 2 | 0 | 12.32 | 70.44 | 115.7 | 19.25 | 2,557.4 | 0.642 | 432.6 | 87.2 | 5.95 | 39.51 |
| Prewarmed screen | 87.5% | Off | 1 | 0 | 11.87 | 105.82 | 850.1 | 65.37 | 8,668.6 | 0.000 | 621.6 | 95.5 | 16.59 | 46.68 |
| Prewarmed screen | 87.5% | On | 1 | 1 | 11.93 | 70.62 | 151.5 | 18.75 | 2,491.7 | 0.615 | 419.0 | 85.9 | 6.24 | 38.76 |

`W/GPU` and activity columns are time-and-device means over the replay window. The near-identical offered throughput makes the PX87 energy difference especially interesting: the cache-on GPU used less board power and much less tensor/DRAM activity while completing approximately the same number of requests per second.

## Recorded telemetry inventory

The aggregate table contains 204 fields per run. Raw JSONL and XML remain the source of truth; the analyzer retains per-run means, maxima, deltas, rates, error counts, sample counts, and availability where appropriate.

| Resource or outcome | Utilization recorded | Saturation recorded | Errors and validity recorded | Pilot availability |
|---|---|---|---|---:|
| GPU board and clocks | Instant/average power, cumulative energy, GPU and memory busy, used/free/total VRAM, temperature, SM/memory clocks, p-state, enforced power limit | Per-GPU mean power/busy imbalance, sample interval and maximum gap | HW slowdown, SW/HW thermal, power-brake and SW power-cap flags; before/after `nvidia-smi -q -x` health counters | 56/56 |
| Hopper GPM compute | Graphics, SM activity, SM occupancy, integer, FP16/32/64, tensor, DFMA/HMMA/IMMA activity | Mean and maximum activity for every counter | GPM support flag and collector logs | 56/56 |
| Hopper GPM data movement | DRAM activity, PCIe RX/TX, NVLink RX/TX | Mean and peak bandwidth/activity | Collector logs and sampling gaps | 56/56 |
| SGLang scheduler | Running requests, generation throughput, decode sum sequence lengths | Queued requests; queue median, maximum and nonzero fraction | Metrics HTTP/parse error count and maximum scrape gap | 56/56 |
| SGLang KV/radix cache | Cache-hit rate, token usage, full-token usage, KV used/available/evictable tokens | KV usage and evictable-token maxima | Cache mode, configured reuse fraction, service errors | 56/56; reuse defined for 19/56 |
| Requests | Input/output token counts, requested length, dispatch/first-token/completion timestamps, TTFT, TPOT, E2E, output hash | req/s, output-token/s, p50/p95/p99 latency | HTTP status, finish reason, success/error, watchdog, exact-length and all-success gates | 56/56 |
| CPU cgroup v2 | User/system/total CPU time and per-second rates; configured 34-core limit | Period, throttled-period/time and burst counters | CPU throttling deltas | 56/56 |
| Host memory cgroup v2 | Current/max memory (pilot peak 83.00 GB of 377.00 GB) | `low`, `high`, and `max` memory-event deltas | OOM, OOM-kill and group-kill deltas | 56/56 |
| Network | Loopback and `eth0` RX/TX bytes and packets, plus rates | RX/TX drops | RX/TX errors and drops | 56/56 |
| Energy outcomes | Counter energy, integrated power, average total power, J/request and J/1k output tokens | Sampling coverage and per-GPU gaps | Counter-vs-integral cross-check | 56/56 |
| Direct HBM/DCGM probe | Collector timestamps only | None | Error text and success/error row counts | 56/56 files; 0 successful samples |

No cell recorded CPU throttling, cgroup OOM, network errors/drops, forbidden thermal/slowdown/power-brake events, or changed GPU health counters. Mean two-GPU power imbalance was 0.85% (maximum 2.78%), so gross tensor-parallel power skew was not evident. Direct HBM/DCGM telemetry was unavailable and should not be treated as a measured counter.

## Relationships worth following

Spearman rank correlations use valid cells only. They show monotonic association, not causality. The scoped comparisons are more informative than pooling unlike workloads.

| Scope | Relationship | n | Spearman rho | Reading |
|---|---|---:|---:|---|
| Balanced-load cells | Throughput → J/request | 19 | -0.986 | Static/model residency energy is amortized sharply as useful throughput rises, until latency saturation becomes unacceptable. |
| Balanced-load cells | Mean SGLang queue → p95 TTFT | 19 | +0.947 | The queue is the clearest saturation-knee signal. |
| Balanced-load cells | Maximum queue → p95 TTFT | 19 | +0.936 | Transient scheduler backlog also predicts TTFT inflation. |
| Balanced-load cells | Queue nonzero fraction → p95 TTFT | 19 | +0.877 | Persistent queueing is an early warning before outright failure. |
| Balanced-load cells | Mean DRAM activity → p95 TTFT | 19 | +0.867 | Memory activity rises with congestion, but load is a common cause; test at fixed offered load. |
| Prefix-related cells | Cache-hit rate → J/request | 9 | -0.730 | Consistent with prefix reuse avoiding prefill work; requires paired confirmation. |
| Prefix-related cells | Cache-hit rate → p95 TTFT | 9 | -0.730 | Higher reuse coincides with much faster first token. |
| Prefix-related cells | Cache-hit rate → p95 E2E | 9 | -0.730 | The gain persists through request completion. |
| Prefix-related cells | Cache-hit rate → mean tensor activity | 9 | -0.730 | Mechanistically consistent with skipping repeated-prefix compute. |
| Prefix-related cells | Cache-hit rate → mean DRAM activity | 9 | -0.730 | Reuse also coincides with less HBM traffic pressure. |
| Prefix-related cells | Configured reuse → observed cache hit | 9 | +0.696 | The server metric responds in the expected direction, but not one-for-one. |
| All valid cells | Total power → request throughput | 38 | -0.250 | Pooling BAL/PF/DEC/prefix regimes reverses intuition; do not use this aggregate for policy decisions. |

The USE-method direction is therefore: locate saturation using queue/TTFT, characterize whether each regime is compute- or memory-heavy using GPM, then optimize energy only within a fixed workload and offered-load stratum. Error counters remain exclusion gates rather than explanatory features.

## Focused prefix-cache experiment

| Factor | Values |
|---|---|
| Repeated-prefix fraction | 0%, 50%, 87.5% of a fixed 4,096-token prompt |
| Radix cache | Off, on |
| Replication | 6 paired seeds; 36 measured cells |
| Cache state | Flush, then prime one distinct request for each of 16 prefix groups before every cell |
| Trace control | Prime and measurement traces share prefix tokens but use different suffix seeds |
| Offered load | 12.4454 requests/s, fixed across treatments |
| Measurement | 90 s open-loop Poisson arrivals; 135 s watchdog |
| SGLang settings | `max_running_requests=128`, `chunked_prefill_size=8192`, TP=2 |
| Order control | Cache AB/BA alternates by seed; workload order rotates by seed |
| Primary outcomes | Paired difference in J/request, p95 TTFT, p95 TPOT, p95 E2E and achieved throughput |
| Mechanism checks | SGLang cache-hit/KV counters; GPM tensor, SM, DRAM, PCIe and NVLink activity |
| Telemetry fix | Critical NVML power/energy at 100 ms without GPM; independent GPM sampler at 1 s |
| Time/cost bound | 90 min planned, 99 min hard stop (+10%); at $6.58/h, $10.86 maximum GPU exposure |

Primary analysis should compute cache-on minus cache-off within each `(reuse, seed)` pair, report the paired median and bootstrap confidence interval, and fit a model with cache, reuse, their interaction, seed block, and order. PX0 is the negative control: cache-on should not materially improve energy or latency when no prefix is reusable.

## Files

- `analysis/telemetry-run-aggregates.csv`: one row per run and every aggregated telemetry field.
- `analysis/telemetry-availability.csv`: availability of every field across all 56 runs.
- `analysis/telemetry-correlations.csv`: all 1,232 valid-scope telemetry correlations.
- `analysis/selected-relationships.csv`: the preselected mechanism and saturation relationships.
- `prefix_cache_campaign.json`: ready-to-validate campaign configuration.
- `analyze_telemetry.py`: reproducible standard-library aggregation and correlation script.
- `analyze_prefix_cache.py`: paired effects and bootstrap confidence intervals for the follow-up results.
