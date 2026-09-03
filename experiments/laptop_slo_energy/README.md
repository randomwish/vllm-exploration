# Laptop SLO-qualified energy MVP

This experiment answers one narrow question:

> At a nominal, realistic prompt length and fixed offered load, which CPU thread setting uses less energy while still meeting the latency and success-rate objectives?

It runs a small Qwen model with `llama-server`, generates OpenAI-compatible traffic with GuideLLM, reads package energy through Linux `perf`, and records `llama-server` scheduler delay with a small bpftrace probe. The same result schema can later be used on an Orange Pi, although its energy sensor will probably be a board power monitor rather than RAPL/perf.

The configured thresholds are **provisional experiment objectives, not an SLA or a regulatory compliance claim**.

## What the campaign measures

The default campaign compares 4 and 8 CPU threads at three prompt lengths:

| Workload | Prompt | Output | Offered load |
| --- | ---: | ---: | --- |
| short | 512 tokens | 128 tokens | 50%, 85%, and 105% of calibrated capacity |
| medium | 2,048 tokens | 128 tokens | 50%, 85%, and 105% of calibrated capacity |
| long | 8,192 tokens | 128 tokens | 50%, 85%, and 105% of calibrated capacity |

GuideLLM varies each synthetic prompt and output around these nominal lengths with a one-token standard deviation, the smallest variance accepted by version 0.7.2.

The harness first measures maximum request throughput with the baseline treatment. It then sends the **same absolute Poisson arrival rate** to both treatments. Treatment order alternates within each workload/load pair to reduce simple time-order bias.

The decision metrics are:

- successful-request rate;
- p95 time to first token (TTFT);
- p95 inter-token latency (ITL);
- p99 end-to-end request latency;
- package joules;
- joules per successful request;
- successful output tokens per joule; and
- SLO-qualified output tokens per joule.

The last metric is output tokens per joule when the entire test cell passes every objective, and zero otherwise. This MVP evaluates aggregate GuideLLM percentiles for a cell; it does not reconstruct a per-request “good token” count.

Diagnostic signals include CPU frequency, pressure stall information, process CPU time, faults, context switches, memory residency, temperature when exposed by sysfs, and eBPF measurements of time that `llama-server` threads wait runnable on a CPU. The eBPF probe reports one-second scheduler-wait counts, actual `sched_migrate_task` events, schedule-in CPU changes, blocking-capable futex durations, and futex wake activity. The harness aligns those buckets with GuideLLM's prefill, decode, and idle intervals. These measurements explain misses; they are not themselves service objectives.

## Measurement boundary

`perf stat -a` measures the laptop package for the configured policy duration. After GuideLLM setup, the harness pre-arms perf with its counters disabled and starts eBPF and host sampling. When the first benchmark request reaches `llama-server`, it enables perf without pausing the server or GuideLLM, then disables perf at the policy deadline. This excludes tokenizer and request-loader initialization while retaining active load-generator overhead. Host samples are counted as valid only inside that same window. The eBPF probe records scheduler latency in one-second suspend-aware boot-clock buckets; the harness uses the same Linux clock to select buckets centered inside the measurement window while preserving the pre-arrival buckets in the raw artifact. The package boundary still includes background system activity and any other work represented by the counter; it does not include fans or display energy that the package counter cannot see.

For comparisons:

- connect AC power and keep the battery charge state stable;
- stop unrelated heavy workloads;
- keep the same power profile and cooling conditions;
- do not compare runs that use different model files; and
- repeat promising cells with multiple seeds before drawing a policy conclusion.

On an Orange Pi, run GuideLLM from another host and replace the energy adapter with an external DC power sensor. Keep the GuideLLM JSON and summary fields unchanged.

## Prerequisites

The harness targets Linux and Python 3.11 or newer. It requires:

- [`llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), available on `PATH` or configured with an absolute path;
- GuideLLM 0.7.2;
- `perf` with a joule-valued package energy event; and
- `bpftrace`, kernel BTF, and sudo permission for the energy and eBPF collectors.

The laptop campaign uses the existing local `Qwen3.5-4B-Q4_K_M.gguf` and llama.cpp build, both referenced by absolute path in `campaign.json`. GuideLLM separately loads the matching Hugging Face tokenizer named by `guidellm.tokenizer`. Preflight hashes the GGUF into `preflight.json`, so a changed model cannot silently share the same evidence identity. On another host, replace those two absolute paths before running.

Create a local environment without changing the system Python:

```bash
cd /home/randomwish/dev/vllm-exploration/experiments/laptop_slo_energy
uv venv
uv pip install -e '.[benchmark,test]'
```

The package pins GuideLLM to 0.7.2 because the report parser depends on its version-2 JSON metrics schema.

## Validate before running

These commands do not launch a model or require sudo:

```bash
.venv/bin/laptop-energy --config campaign.json validate-config
.venv/bin/laptop-energy --config campaign.json plan
.venv/bin/laptop-energy --config campaign.json run
```

The last command is a dry run. It prints 3 calibration cells and 18 policy cells. Policy rates remain symbolic until calibration completes.

Check the actual machine and privilege path:

```bash
sudo -v
.venv/bin/laptop-energy --config campaign.json preflight --privileged
```

Preflight fails if a required binary, configured energy event, model file, probe, GuideLLM version, or pre-authorized sudo session is missing. With `--privileged`, it also asks bpftrace to compile the probe without attaching it. It does not change the machine.

## Execute

### Run the context pilot

The laptop pilot compares 4 and 8 threads at 512-, 2,048-, and 8,192-token prompts. It uses one offered load at 85% of the capacity measured with eight threads. Each paired treatment receives the same absolute request rate and duration.

Each policy cell runs for at least 120 seconds. After calibration, the harness extends the cell toward 20 expected arrivals, up to 600 seconds. A cell needs at least 12 successful requests, complete GuideLLM metrics, energy, host samples, and in-window eBPF samples to be valid. These sample counts support an exploratory comparison, not a production tail-latency claim.

The measured policy time is between 12 and 60 minutes for all six cells, plus three minutes of calibration, model restarts, warm-ups, and the first Poisson arrival in each cell. Run the pilot with:

```bash
sudo -v
.venv/bin/laptop-energy --config campaign.pilot.json preflight --privileged
.venv/bin/laptop-energy --config campaign.pilot.json run --execute \
  --output-root pilot-results
```

The command remains in the foreground. Progress messages on stderr report the calibration or policy cell number, server and collector state, resolved request rate and duration, measurement percentage and remaining seconds, and each completed cell's evidence summary. The final JSON result remains on stdout. Avoid suspending the laptop or starting other heavy workloads during the run.

For a long laptop run, `systemd-inhibit` can block automatic sleep while the command is active:

```bash
sudo -v
systemd-inhibit --what=sleep --mode=block --why="LLM energy pilot" \
  .venv/bin/laptop-energy --config campaign.pilot.json run --execute \
  --output-root pilot-results
```

The harness refreshes the cached sudo credential every 30 seconds so `perf` and bpftrace can start in later cells.

### Resume an interrupted pilot

To resume a partial result, pass its run directory:

```bash
sudo -v
systemd-inhibit --what=sleep --mode=block --why="LLM energy pilot" \
  .venv/bin/laptop-energy --config campaign.pilot.json run --execute \
  --resume pilot-results/laptop-qwen35-4b-context-pilot-20260901T085040Z
```

Resume reuses calibration and completed cells only when their evidence is valid, their active measurement reached the target duration, and wall time does not indicate a suspend gap longer than five seconds. It replaces partial or contaminated cell directories and rebuilds the campaign summary. A Ctrl-C marks the run as resumable and stops active child processes.

### Run the broader exploratory campaign

Run the complete exploratory campaign:

```bash
.venv/bin/laptop-energy --config campaign.json run --execute --output-root results
```

The default durations account for about 15 minutes of measured traffic, plus model starts, 8K-token prefill, downloads, and warm-ups. A laptop run can therefore take considerably longer.

### Run the focused constant-rate frontier

The focused campaign calibrates isolated 2,048-token requests synchronously with 8 threads. It then compares 4 and 8 threads at a constant arrival rate equal to 25% of that sequential capacity:

```bash
sudo -v
.venv/bin/laptop-energy --config campaign.frontier.json preflight --privileged
systemd-inhibit --what=sleep --mode=block --why="LLM energy frontier" \
  .venv/bin/laptop-energy --config campaign.frontier.json run --execute \
  --output-root frontier-results
```

After calibration, the harness resolves each policy duration to target 20 arrivals, with a maximum of one hour per treatment. Constant traffic reduces arrival-process variance so the first comparison isolates thread count more clearly. A later Poisson campaign can test burst sensitivity at a rate that this experiment supports.

### Run the thread-coordination experiment

The coordination campaign keeps four decode threads and compares four prefill batch threads with two. It uses the same 2,048-token workload, synchronous calibration, 25%-of-capacity constant traffic, and 512-token physical micro-batch as the affinity experiment. Its eBPF output separates runnable delay, scheduler migrations, and futex coordination by prefill, decode, and idle phase:

```bash
sudo -v
.venv/bin/laptop-energy --config campaign.coordination.json preflight --privileged
systemd-inhibit --what=sleep --mode=block --why="LLM coordination experiment" \
  .venv/bin/laptop-energy --config campaign.coordination.json run --execute \
  --output-root coordination-results
```

After calibration, each treatment targets 12 arrivals and can run for up to 40 minutes. On the capacity observed in the earlier frontier experiment, expect about 18 minutes per treatment plus three minutes of calibration and server restarts. The privileged preflight is required because syscall tracepoint field names can vary across kernels, and bpftrace must compile the probe against the laptop's kernel before the run.

This laptop also has a short checked-in collector smoke configuration that uses the existing local Qwen3.5 4B GGUF:

```bash
sudo -v
.venv/bin/laptop-energy --config campaign.smoke.json preflight --privileged
.venv/bin/laptop-energy --config campaign.smoke.json run --execute \
  --output-root smoke-results
```

It runs one 5-second capacity calibration and two 5-second policy cells. Its thresholds only verify plumbing and must not be interpreted as service objectives.

Run the harness as the normal login user, not from `sudo su`. It elevates only `perf` and bpftrace; running the parent process as root creates output that the deliberately de-privileged GuideLLM process cannot write.

For a functional plumbing test without privileged collectors:

```bash
.venv/bin/laptop-energy --config campaign.json run --execute \
  --skip-energy --skip-ebpf --output-root smoke-results
```

Those smoke-test policy cells are marked invalid because `campaign.json` requires both collectors. Change the policy flags only if the evidence requirement itself is intentionally changing.

## Outputs and interpretation

Every run gets a timestamped directory. Important files are:

| File | Purpose |
| --- | --- |
| `campaign.json` | exact campaign policy copied into the run |
| `preflight.json` | host, tool versions, energy events, and model identity |
| `capacities.json` | calibrated request rate for each context length |
| `policy/*/guidellm.json` | raw GuideLLM result |
| `policy/*/energy.json` | raw parsed joules and measurement boundary |
| `policy/*/ebpf-runqlat.txt` | raw timestamped scheduler, migration, and futex aggregates |
| `policy/*/ebpf.json` | windowed eBPF summary with prefill, decode, and idle alignment |
| `policy/*/host.jsonl` | time-series host diagnostics |
| `policy/*/validity.json` | evidence-completeness checks |
| `summary.csv` | compact cell comparison table |
| `summary.json` | complete results and SLO-qualified winners |
| `FINAL_STATUS.json` | complete, complete-invalid, or failed terminal state |

A “winner” must have valid evidence, pass every configured SLO check, and have the highest SLO-qualified output tokens per joule within the same workload and offered-load pair. Do not compare energy efficiency across prompt lengths as though they were the same service.

`FINAL_STATUS.json` separates execution validity from sample sufficiency. `complete_evidence_limited` means every required collector and measurement boundary worked, but at least one cell did not reach the requested sample count. That status exits successfully because the observed completion shortfall is a result, not an execution failure. `complete_invalid` and a nonzero exit code indicate missing or contaminated execution evidence. `complete` means every policy cell also reached its sample target; it does not mean every cell passed its SLO.

## MVP limitations

- One seed and one pass are exploratory, not statistically defensible evidence.
- SLO thresholds are placeholders that should be replaced by requirements derived from a real service tier.
- The eBPF probe uses Linux scheduler tracepoints and the `llama-server` task name. Tracepoints are more portable than kernel-structure offsets, but availability and field names still need validation on each kernel.
- Package counters do not measure wall-plug energy and may be absent on ARM boards.
- The harness does not yet capture per-request traces, queue depth inside llama.cpp, idle-energy subtraction, confidence intervals, or an external power meter.
- A Hugging Face tag is mutable. Use a local, hashed GGUF for evidence intended to survive review.

## Suggested next increment

After one clean laptop campaign, add three or more repetitions per cell, randomized pair order, an idle-power phase, and bootstrap confidence intervals. The Orange Pi port should preserve the workload, GuideLLM, SLO, validity, and summary layers and replace only server launch details and the energy collector.
