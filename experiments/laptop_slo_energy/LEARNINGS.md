# Laptop MVP learnings

This file records implementation lessons from bringing up the collector smoke test. The retained smoke result is `smoke-results/laptop-qwen35-4b-collector-smoke-20260901T063122Z`.

## What worked

- Run the harness as the login user after `sudo -v`. Elevate only `perf` and bpftrace. Running the parent process as root creates output-permission problems for GuideLLM.
- Use the local GGUF for inference and a separate Hugging Face tokenizer reference for GuideLLM request generation.
- Set synthetic token-length standard deviations to at least one. GuideLLM 0.7.2 rejects zero.
- Convert GuideLLM request latency from seconds to milliseconds before evaluating end-to-end SLO thresholds.
- Set `SO_REUSEADDR` in the preflight port probe. Otherwise, recently closed connections can look like an active listener.

## Measurement alignment

- GuideLLM's setup-complete message does not mark the first request. A low-rate Poisson profile can wait several seconds before its first arrival.
- Do not pause GuideLLM after its benchmark clock starts. Wall-clock time continues while the process is stopped, which shortens the active workload inside a fixed-duration benchmark.
- Start `perf stat` with counters disabled. Enable it when the first post-setup request reaches `llama-server`, and disable it at the policy deadline.
- Start host sampling before the first request, but count only samples whose timestamps fall inside the measured window.
- Record the server-trigger offset from GuideLLM's client-side timestamp. The retained smoke run observed about 0.3 seconds of request dispatch and server-entry delay.
- Keep cached sudo credentials alive during long campaigns. A successful preflight does not prevent the timestamp from expiring before a later `perf` or bpftrace launch.
- Compare wall-clock and monotonic durations. A laptop suspend can consume GuideLLM's wall-clock deadline while providing much less active measurement time.
- Resume only completed cells with valid evidence, sufficient active duration, and no material suspend gap. Replace partial or contaminated cells.

## eBPF alignment

- Scheduler tracepoint string fields such as `args->comm` are fixed strings in bpftrace 0.20.2. Compare them directly instead of wrapping them in `str()`.
- Attach the scheduler probe before inference so startup does not hide early scheduler delay.
- Store run-queue latency as one-second timestamped aggregates. Select only buckets centered inside the measured policy window, while retaining pre-arrival buckets in the raw output.
- bpftrace's `nsecs` uses Linux `CLOCK_BOOTTIME`, which includes suspend time. Python's `time.monotonic()` does not. Use `time.clock_gettime(time.CLOCK_BOOTTIME)` for eBPF window selection.
- Do not create one histogram for every short time bucket. The pilot's 250 ms histogram map could exhaust bpftrace's map-entry limit during long cells. One-second scalar maps for count, sum, maximum, and threshold exceedances preserve the scheduler signal without unbounded histogram cardinality.

## Interpretation

The five-second smoke test verifies orchestration and evidence collection. It completed only one request per treatment, so its latency percentiles and energy difference must not be used to choose a thread policy. `campaign.pilot.json` adds capacity-aware durations and requires at least 12 successful requests per cell. That pilot supports a directional comparison; repeated runs and larger samples are still needed for stable tail-latency estimates.

## Context pilot result

The resumed pilot finished normally at `pilot-results/laptop-qwen35-4b-context-pilot-20260901T085040Z`. The original harness labeled it `complete_invalid` because the two 8,192-token cells completed 9 requests instead of the configured sample target of 12. That label describes sample sufficiency, not a workload or execution failure. Both cells collected complete GuideLLM, energy, host, and eBPF evidence. Their lower completion count is itself a service-capacity result: this observation does not support the hypothesis that the 8,192-token workload can meet the provisional SLO at the tested rate.

| Context | Threads | Successful / admitted | Success rate | p95 TTFT | p95 ITL | p99 E2E | Package energy | J / success | Sample support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 4 | 23 / 31 | 74.2% | 104.0 s | 96.8 ms | 116.3 s | 7,966.5 J | 346.4 | sufficient for pilot |
| 512 | 8 | 24 / 31 | 77.4% | 98.2 s | 88.4 ms | 109.6 s | 7,777.9 J | 324.1 | sufficient for pilot |
| 2,048 | 4 | 22 / 29 | 75.9% | 150.5 s | 103.7 ms | 161.9 s | 7,012.8 J | 318.8 | sufficient for pilot |
| 2,048 | 8 | 22 / 30 | 73.3% | 148.4 s | 89.0 ms | 159.7 s | 9,512.2 J | 432.4 | sufficient for pilot |
| 8,192 | 4 | 9 / 16 | 56.3% | 84.3 s | 93.3 ms | 96.0 s | 7,160.7 J | 795.6 | limited: 9 observations |
| 8,192 | 8 | 9 / 16 | 56.3% | 83.6 s | 93.7 ms | 95.5 s | 7,112.6 J | 790.3 | limited: 9 observations |

No observation supported the hypothesis that the tested 85%-of-calibrated-capacity rate could meet all provisional SLOs. Every cell passed its p95 ITL objective, but every cell missed success rate, p95 TTFT, and p99 end-to-end latency. The large TTFT relative to ITL points to admission or prefill queueing as the dominant problem rather than slow token-to-token decoding. Because no cell met the SLO, the pilot does not identify an SLO-qualified energy winner.

The energy comparisons are diagnostic because the cells missed their SLOs. Directionally, 8 threads used about 6% fewer joules per success for the 512-token workload. Four threads used about 26% fewer joules per success for the 2,048-token workload. The two 8,192-token results were within 1%, but their samples are insufficient.

The pilot also exposed a design problem in its capacity calibration. The 60-second throughput calibration returned only 5, 3, and 1 successful requests for the short, medium, and long workloads, respectively, while already showing tens of seconds of TTFT. Treating those saturated, small-sample rates as service capacity and applying an 85% load fraction did not produce an SLO-safe arrival rate. For the long workload, the nominal rate predicted only 8.5 arrivals in the 600-second cap, so the run was not sized to satisfy a 12-success minimum reliably.

## eBPF contribution

The eBPF probe is the novel attribution layer in this experiment. GuideLLM tells us that a request missed an objective. RAPL tells us how much package energy the measurement consumed. The scheduler tracepoints show whether `llama-server` was runnable but waiting for CPU time while those outcomes occurred. The pilot recorded about 46,700 runnable-to-running transitions in each short cell, 48,500 in each medium cell, and 13,400 in each long cell, all aligned to the same measurement windows as the service and energy metrics.

The pilot's raw eBPF event counts are usable, but its per-250-ms latency histograms are incomplete because their map cardinality can exceed bpftrace's default capacity. Do not use percentiles reconstructed from those partial histograms. The revised probe uses one-second aggregates and reports mean and maximum scheduler wait plus the fraction of samples at or above 100 microseconds, 1 millisecond, and 10 milliseconds. This makes the next experiment able to distinguish service-level queueing from host CPU scheduler contention without relying on kernel structure offsets.

## Focused next experiment

`campaign.frontier.json` implements the smallest useful follow-up. It runs one 180-second GuideLLM synchronous calibration for the 2,048-token workload with 8 threads. It then sends the same deterministic constant arrival rate, 25% of that sequential capacity, to the 4-thread and 8-thread treatments. The harness sizes each policy cell for 20 expected arrivals, up to one hour, and treats 12 completed observations as enough to retain an exploratory cell.

The harness now reports a cleanly executed cell with fewer than 12 completions as `complete_evidence_limited`, not as an execution failure. The lower completion count remains part of the SLO result, while the status warns that its tail estimates have limited support. Once the constant-rate observation supports the SLO hypothesis, use Poisson traffic with multiple seeds to measure burst sensitivity. Tail percentiles from the present 9-to-24-request samples remain exploratory and must not be treated as SLA evidence.

## Constant-rate frontier result

The focused run at `frontier-results/laptop-qwen35-4b-medium-constant-frontier-20260901T110450Z` completed with valid execution evidence and 20 successful requests in each treatment. Synchronous calibration measured 0.04444 requests per second with 8 threads. The policy therefore used a constant 0.01111 requests per second, or one arrival every 90 seconds, for 1,800 seconds per treatment. No request was incomplete or errored.

| Metric | 4 threads | 8 threads | Interpretation |
| --- | ---: | ---: | --- |
| Successful / admitted | 20 / 20 | 20 / 20 | Both support the success-rate objective at this load. |
| p95 TTFT | 9.94 s | 11.09 s | Both miss the provisional 8 s objective. |
| p95 ITL | 94.1 ms | 105.6 ms | Both meet the 120 ms objective. |
| p99 end-to-end | 22.02 s | 24.65 s | Both meet the 30 s objective. |
| Package energy | 11,115 J | 12,872 J | Equal 1,800-second measurement windows. |
| Average package power | 6.18 W | 7.15 W | Includes idle time between arrivals. |
| Joules per success | 555.8 J | 643.6 J | Four threads use about 14% less. |
| Output tokens per joule | 0.231 | 0.200 | Four threads produce about 16% more. |

This observation supports 4 threads over 8 threads for the tested laptop, model, context, and constant arrival rate. Four threads have lower TTFT, ITL, end-to-end latency, package energy, and joules per successful request. The observation does not support the full provisional SLO because p95 TTFT remains 1.94 seconds above its 8-second objective. Since requests arrive 90 seconds apart and complete in about 22 seconds, reducing the offered rate further is unlikely to remove that isolated-request TTFT gap.

### Scheduler attribution from eBPF

The revised eBPF probe produced complete one-second scheduler aggregates across both 1,800-second windows:

| Scheduler signal | 4 threads | 8 threads |
| --- | ---: | ---: |
| Runnable-to-running samples | 21,904 | 23,682 |
| Mean run-queue wait | 17.6 microseconds | 18.3 microseconds |
| Maximum wait | 1.69 ms | 3.00 ms |
| Waits at least 100 microseconds | 1.46% | 1.80% |
| Waits at least 1 millisecond | 5 (0.023%) | 18 (0.076%) |
| Waits at least 10 milliseconds | 0 | 0 |

These kernel measurements do not support CPU scheduler delay as the cause of the 9.94-to-11.09-second TTFT. Scheduler waits are several orders of magnitude smaller, and neither treatment records a wait of 10 milliseconds or more. Eight threads create 8% more runnable-to-running events and about 38% more involuntary process context switches, but that overhead is still too small to explain TTFT directly. Together with 14% higher energy per request and slightly higher temperatures, the result is more consistent with extra parallelism overhead or shared-resource contention. The experiment does not measure memory bandwidth or cache misses, so it cannot distinguish those mechanisms yet.

For an 8-second TTFT objective, this laptop configuration does not establish a supportable service point. The next decision is therefore about the service definition or the implementation, not offered load: adopt a roughly 10-second laptop TTFT objective for this exploratory tier, or optimize the inference path while keeping the 8-second objective. Before treating the energy difference as dynamic inference energy, add an idle RAPL baseline because the 90-second arrival interval makes idle package energy a material part of joules per request.

## Fast-core affinity screen

The Ryzen AI 7 PRO 350 exposes four physical cores with a 5.09 GHz maximum and four with a 3.51 GHz maximum. Each physical core has two logical CPUs. The 4-thread frontier treatment did not constrain worker placement, so an easy implementation target is strict placement on logical CPUs 0, 2, 4, and 6: one hardware thread on each faster physical core, represented by llama.cpp CPU mask `0x55`.

A prompt-only `llama-bench` screen at 2,048 tokens improved from 41.44 tokens per second without affinity to 43.70 tokens per second with `--cpu-mask 0x55 --cpu-strict 1`, a 5.5% directional gain. Keeping the pinned placement, increasing physical micro-batch size from the default 512 reduced prompt throughput: 43.77 tokens per second at 512, 41.23 at 1,024, and 40.72 at 2,048. The focused follow-up therefore keeps `--ubatch-size 512` and compares unpinned and fast-core-pinned 4-thread treatments in `campaign.affinity.json`.

The eBPF probe now counts CPU migrations alongside run-queue latency. The affinity experiment can therefore test whether the candidate improves TTFT and energy while changing actual scheduler placement behavior. The `llama-bench` results are a screening signal, not a replacement for the GuideLLM, eBPF, and RAPL comparison.

## Fast-core affinity result

The focused run at `affinity-results/laptop-qwen35-4b-medium-fast-core-affinity-20260901T130842Z` completed with valid GuideLLM, RAPL, host, and eBPF evidence. Both treatments completed 12 requests at the same constant rate of 0.01111 requests per second. The unpinned treatment admitted a thirteenth request at the 1,080-second measurement boundary, which could not finish before the cutoff. Its reported 92.3% success rate is therefore a boundary artifact, not evidence that unpinned execution is less reliable.

| Metric | 4 threads, unpinned | 4 threads, fast-core affinity | Interpretation |
| --- | ---: | ---: | --- |
| Completed requests | 12 | 12 | Equal useful work. |
| Median TTFT | 10.07 s | 9.92 s | Affinity improved the center by 1.5%. |
| Mean TTFT | 10.04 s | 9.94 s | Affinity improved the mean by 1.1%. |
| p95 TTFT | 10.16 s | 12.02 s | One slow pinned observation made the exploratory tail 18% worse. |
| p95 ITL | 96.2 ms | 101.0 ms | Both meet the 120 ms objective; affinity is 5% worse. |
| p95 end-to-end | 22.36 s | 24.75 s | Both meet the 30 s objective; affinity is 11% worse. |
| Package energy | 9,962 J | 8,837 J | Affinity is 11% lower over equal windows. |
| Average package power | 9.22 W | 8.18 W | Includes the long idle periods between requests. |
| Joules per completed request | 830.2 J | 736.4 J | Affinity is 11% lower. |
| Output tokens per joule | 0.155 | 0.174 | Affinity is 13% higher. |

This single-seed observation does not support strict fast-core affinity as a TTFT-tail optimization. It supports only a small central-latency improvement, while p95 TTFT remains above the provisional 8-second objective and is less stable than the unpinned result. The package-energy result is promising but not yet attributable to inference: package energy fell by 11%, while the separate core-energy event rose by 43%, and most of each 18-minute window was idle. Repeat the treatments in alternating order and subtract an idle baseline before treating the package-energy difference as an affinity effect.

### eBPF explanation

| Scheduler signal | Unpinned | Fast-core affinity |
| --- | ---: | ---: |
| Runnable-to-running samples | 12,056 | 13,595 |
| Mean run-queue wait | 8.74 microseconds | 16.30 microseconds |
| Maximum run-queue wait | 2.98 ms | 1.99 ms |
| Waits at least 100 microseconds | 34 (0.282%) | 131 (0.964%) |
| Waits at least 1 millisecond | 5 (0.041%) | 10 (0.074%) |
| Waits at least 10 milliseconds | 0 | 0 |
| CPU changed since the task's preceding observed schedule-in | 1,793 (14.9%) | 2,659 (19.6%) |

The kernel evidence rejects the simple mechanism proposed by the screen: strict affinity did not improve latency by reducing process-wide scheduler waiting or CPU changes. Mean runnable wait almost doubled, waits of at least 100 microseconds became more frequent, and host sampling recorded 83% more involuntary context switches. The lower maximum wait does not change the attribution because every observed scheduler delay remained below 3 milliseconds. These delays are still orders of magnitude below the roughly 10-to-12-second TTFT, so neither the TTFT level nor the pinned tail outlier can be explained directly by Linux run-queue latency.

The CPU-change counter needs careful naming. It compares the CPU used by consecutive observed schedule-ins for every task whose `comm` is `llama-server`; it is not the kernel's `sched_migrate_task` event and does not isolate the four llama.cpp compute workers. Support, HTTP, and coordination threads can therefore contribute even when the compute-worker mask is strict. For the next thread-coordination experiment, record true `sched_migrate_task` events and futex wait duration by thread ID, then align their one-second aggregates with GuideLLM's prefill and decode intervals.

The practical conclusion is to keep the simpler unpinned 4-thread policy for now. Fast-core pinning is not the easy TTFT win suggested by `llama-bench`. The next inexpensive target is thread coordination during prefill: add eBPF futex/off-CPU wait accounting before changing another llama.cpp knob. This preserves the experiment's main method—use eBPF to choose the mechanism to optimize instead of inferring a cause from TTFT alone.

## Thread-coordination experiment

`campaign.coordination.json` implements the next mechanism test. It keeps four unpinned decode threads, the 2,048-token prompt, the 128-token response, and `--ubatch-size 512`. It compares `--threads-batch 4` with `--threads-batch 2` at the same constant offered rate, calibrated synchronously from the four-batch-thread baseline and reduced to 25% of that capacity.

The eBPF probe now adds two direct coordination signals. `sched:sched_migrate_task` counts migrations chosen by the scheduler. `syscalls:sys_enter_futex` and `syscalls:sys_exit_futex` measure the duration of futex operations that can block a `llama-server` thread, while a separate counter records futex wake and requeue calls. The futex duration includes syscall execution and any blocked time, so it is an off-CPU coordination proxy rather than proof of a particular llama.cpp lock or barrier.

The harness maps each one-second eBPF bucket from `CLOCK_BOOTTIME` to GuideLLM request time. It classifies the bucket center as prefill from request start to first token, decode from first token to request end, or idle. A blocking futex call is assigned to the bucket in which the syscall exits. This resolution is sufficient for the 10-second prefill and roughly 12-second decode intervals observed on the laptop, but it cannot place a wait that crosses a phase boundary with subsecond precision.

The hypothesis is directional: if four batch threads lose substantial prefill time in futex waits or migrations, two batch threads might reduce coordination overhead, TTFT, and energy. If two batch threads instead reduce prefill parallelism without materially reducing those eBPF signals, the observation does not support that optimization. In either case, compare the kernel mechanism alongside TTFT and joules rather than selecting a setting from latency alone.
