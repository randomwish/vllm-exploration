# Runpod LLM Inference Telemetry: USE Method Matrix

Date: 2026-08-25

This document records what was directly verified on normal Runpod Secure Cloud pods with one NVIDIA A40 and one NVIDIA H100 SXM. It distinguishes portable telemetry, Hopper-only telemetry, model estimates, and privileged counters that remain unavailable in the standard container.

## Practical collection matrix

| Resource or layer | Collectible metrics | Interface and useful cadence | Verified status | Main use |
|---|---|---|---|---|
| GPU board energy | Cumulative energy in mJ; instantaneous and one-second averaged board power in W | Direct NVML field IDs 83, 186, and 185 at 100 ms | **Works.** Over a steady 4.90 s load, integrated instantaneous power agreed with the energy-counter delta to within 0.04% | Energy-to-finish, J/request, J/1k tokens, power time series |
| GPU operating state | GPU and memory clocks, P-state, temperature, requested/default/min/max/enforced power limits, clock-event reasons | NVML or DCGM at 100 ms to 1 s | **Works read-only.** The tested A40 exposed a 100–300 W range and a 300 W enforced limit | Explain power and throughput changes; detect thermal or power throttling |
| GPU coarse activity | GPU busy percentage and memory-copy-engine busy percentage | NVML/DCGM at 100 ms; `nvidia-smi dmon` at 1 s | **Works.** Loaded samples reached 100% GPU busy | GPU utilization and coarse memory activity; burst distributions |
| GPU memory capacity | Framebuffer total, used, free, and reserved bytes; per-process framebuffer use | NVML/DCGM at 100 ms–1 s; `nvidia-smi pmon` at 1 s | **Works** | HBM/KV-capacity utilization and OOM headroom |
| GPU process activity | PID, command, SM busy, memory activity, framebuffer allocation | `nvidia-smi pmon` at 1 s; NVML process APIs | **Works** | Attribute load to the SGLang worker and identify interference |
| GPU health/errors | ECC mode and error counts, SRAM ECC, row-remap status, PCIe replay count, shutdown/slowdown temperatures | DCGM/NVML at about 1 s or as event counters | **Works for supported fields** | Detect integrity, bus, and thermal faults |
| PCIe | Current generation and width; sampled RX/TX throughput; replay count | NVML or `nvidia-smi dmon` at 1 s | **Works, but sampled only.** DCGM cumulative PCIe byte fields were unsupported on this A40 | Detect host-transfer bottlenecks and link degradation |
| NVLink | Link state and error counters; topology | NVML/`nvidia-smi` during multi-GPU preflight | **Conditional.** All links were down on the single-A40 pod; throughput was not established | Tensor-parallel communication bottlenecks and topology validation |
| Privileged hardware counters | Detailed Nsight Compute metrics and DCGM's legacy profiling path | Nsight Compute or DCGM on pre-Hopper GPUs | **Blocked.** Nsight Compute failed on both GPUs; A40 DCGM profiling also failed. Hopper GPM is the unprivileged exception in the next row | Fine-grained mechanism analysis beyond the portable telemetry set |
| Hopper GPU Performance Metrics | Graphics/SM activity, occupancy, tensor/HMMA/IMMA/DFMA activity, DRAM activity, integer/FP activity, PCIe, NVLink, cache and context-switch metrics | Direct NVML GPM at 100 ms or `nvidia-smi dmon` at 1 s | **Works unprivileged on the tested H100 SXM.** Unsupported by A40 | Distinguish compute-, tensor-, and bandwidth-bound phases and explain parameter effects |
| H100 HBM power | Averaged GPU-memory power in W | `nvidia-smi -q -d POWER`; snapshot/low-rate parsing | **Works.** Approximately 48 W idle, 116 W under tensor matmul, and 178 W under the memory-bound control | Separate HBM-power behavior from total board power; no cumulative HBM-energy counter was exposed |
| CUDA/NCCL execution | CUDA API duration, kernel duration/name/count, synchronization, NCCL trace where enabled | Nsight Systems for short targeted traces | **Works.** CUDA kernel and API summaries were captured | Explain why a parameter changes energy or latency; not a continuous energy meter |
| Container CPU | CPU time, quota, throttled periods/time | cgroup v1 `cpuacct` and `cpu.stat` at 100 ms–1 s | **Works.** The pod exposed a 7.65-CPU CFS quota | CPU utilization, saturation, and scheduler/tokenizer bottlenecks |
| Container memory | Usage, limit, fail count, OOM state | cgroup v1 memory files at 100 ms–1 s | **Works.** The pod exposed an approximately 50 GB limit | Memory-capacity utilization and allocation failures |
| Container/process I/O | Per-process CPU, RSS, disk I/O; network bytes, packets, errors, and drops | `psutil`, `/proc/<pid>`, and network-namespace `/proc/net/dev` | **Works.** Host-wide `/proc/stat` and `/proc/meminfo` are misleading for tenant capacity; use cgroups | Detect host or request-ingress bottlenecks |
| SGLang scheduler and cache | Running/queued requests, generation throughput, cache hit rate, token usage, KV available/evictable/used tokens | Prometheus `/metrics`, normally 0.5–1 s; enable with `--enable-metrics` | **Available when SGLang is launched with metrics** | Directly measure server saturation, cache behavior, and useful work |
| SGLang request outcomes | TTFT, inter-token/TPOT, end-to-end latency, input/output tokens, status/error/timeout | Client request log plus SGLang metrics/tracing | **Collectible** | Service-level performance, tail latency, and errors |
| SGLang model estimates | Estimated cumulative per-GPU FLOPs and read/write bytes | `--enable-mfu-metrics` | **Available but estimated, not hardware counters** | Compare modeled compute/data movement across parameter settings |
| Runpod control plane | Coarse CPU, memory, GPU, and GPU-memory utilization | Runpod pod API | **Works but is too coarse for primary measurement** | Liveness and sanity checking only |

Power-limit **reads** work, but power-limit **writes do not**: both `nvidia-smi --power-limit` and DCGM configuration returned insufficient-permission errors. Therefore a conventional Runpod pod is suitable for observational SGLang experiments, but not for a controlled GPU power-cap sweep.

## H100 preflight results

The H100 SXM probe used driver 580.126.09, CUDA 12.8 user-space libraries, DCGM 4.6.1, an 80 GB HBM3 GPU, and a normal Secure Cloud container without `CAP_SYS_ADMIN` or `CAP_PERFMON`.

| Controlled state | Board power | HBM power | SM activity | SM occupancy | HMMA tensor activity | DRAM activity | Coarse GPU / memory busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Idle | ~74 W | ~48 W | 0% | 0% | 0% | ~0% | 0% / 0% |
| FP16 8192×8192 matmul loop | ~700 W average | ~116 W | ~96.7% | ~14.3% | ~93.8% | ~25.7% | 100% / ~34% |
| FP16 vector-add memory control | ~410 W average | ~178 W | ~99.2% | ~61.4% | 0% | ~91.1% | 100% / 100% |

This demonstrates why Hopper GPM materially improves the study: ordinary GPU utilization was 100% for both workloads, but the hardware-activity vector correctly separated tensor-compute saturation from HBM-bandwidth saturation.

Additional findings:

- Direct NVML GPM sampling worked at 100 ms without enabling streaming or adding container capabilities.
- DCGM profiling group A/B fields also worked without privileged access, including SM/tensor/DRAM activity and sampled PCIe/NVLink byte rates. DCGM's lower-level cycle-counter group was listed but returned `N/A`.
- The current GPM catalogue exposes total and per-link NVLink rates for 18 H100 links. The links were active, but traffic was zero because only one GPU was allocated.
- The GPU reported PCIe Gen5 x16, separate GPU/HBM temperatures, fabric state, row-remap headroom, and NVLink CRC/replay/recovery counters. Record counter deltas because non-zero values such as the PCIe replay baseline may predate the experiment.
- MIG mode was disabled, while the read-only catalogue exposed 1g through 7g instance profiles. MIG-level GPM remains untested because the pod configuration was not changed.
- Nsight Compute still failed with `ERR_NVGPUCTRPERM`, so GPM does not grant access to arbitrary detailed performance counters.
- Power-cap writes still failed through both NVML/`nvidia-smi` and DCGM.
- `dcgmi health --set a` reported an NVLink failure solely because the provider's IMEX daemon was not ready, even though direct link error counters were zero and fabric initialization succeeded. Do not treat aggregate DCGM health as a benchmark failure without inspecting the subsystem reason.
- Short instantaneous-power samples can exceed the 700 W enforced limit: the 200 ms series reached about 785 W, while NVIDIA's rolling sample buffer reported a shorter peak near 993 W. Treat the cap as a control-loop average ceiling, not an instantaneous maximum.

## USE-method checklist

Brendan Gregg's USE method asks, for every resource, about **utilization**, **saturation**, and **errors**. Saturation means excess work that cannot immediately be serviced, usually visible as queueing or throttling. For inference, the application queue is often a better saturation signal than GPU utilization alone.

| Resource | Utilization | Saturation | Errors | Collection status and interpretation |
|---|---|---|---|---|
| GPU compute engines | GPU busy%; optionally SM/tensor activity | SGLang queued requests while GPU busy is high; power/thermal clock-event duration; latency growth at fixed offered load | Xid/driver failures, unhealthy DCGM state | Coarse utilization works. Privileged SM/tensor counters are blocked on A40. Use 100 ms distributions rather than only averages |
| GPU board-power envelope | Instantaneous/average W and energy delta | Enforced-limit proximity and power-cap clock-event duration | Power-brake event or unexpected power/energy discontinuity | Fully observable, read-only. This is the primary energy measurement resource |
| HBM/framebuffer capacity | Used/total framebuffer; SGLang KV-token usage | Low KV headroom, evictable-token growth, request retractions/queue growth, allocation pressure | OOM, ECC, row-remap events | Capacity is directly observable. NVML “memory utilization” is engine busy time, not percent of bandwidth consumed |
| SGLang scheduler/batch capacity | Running requests, batch occupancy, throughput | Queue depth/age, paused or retracted requests, TTFT/TPOT tail growth | Request failures, timeouts, invalid outputs | Strongest direct saturation view; correlate every GPU energy sample with this layer |
| Prefix/KV cache | Cache-hit rate and used/available/evictable tokens | Evictions, falling hit rate under pressure, recomputation and queue growth | Cache allocation or consistency errors | Directly exposes energy saved or spent through reuse versus recomputation |
| PCIe | RX/TX rate, generation, width | Sustained throughput near an empirically measured ceiling, transfer-related CUDA stalls | Replay-count growth, link downgrade | Sampled throughput works; cumulative byte accounting was unavailable on the tested A40 |
| NVLink / tensor-parallel fabric | Link traffic where supported; topology/link state | Communication queues or NCCL duration dominating iteration time | Link/NVLink CRC or replay errors, link down | Requires a multi-GPU preflight. Short Nsight Systems traces can supply timing when continuous counters are unavailable |
| Container CPU quota | CPU-time delta divided by CFS quota | CFS throttled time/periods, runnable pressure, rising request queue with GPU gaps | Process exits, tokenizer/runtime exceptions | Use cgroup capacity, not host-wide `/proc` percentages |
| Container memory | Usage/limit and working set | Near-limit operation, allocation stalls, rising fail count | OOM and memory fail events | Direct cgroup accounting works |
| Network/request ingress | Interface byte/packet rates and connection/request rate | Socket/request backlog, client-side wait, server queue growth | Drops, retransmits, request disconnects/timeouts | Namespace counters plus client timestamps; separate loopback benchmark traffic from external traffic |
| Model/storage loading | Read rate and load duration | I/O wait or model-start delay | Read failures, corrupt/missing weights | Mostly relevant to cold start, not steady-state inference |

## Most useful experimental directions

### 1. Find the scheduler saturation knee first

Sweep one concurrency/batching control at a time, beginning with `--max-running-requests`, and repeat across fixed offered-load levels. A second experiment can sweep `--chunked-prefill-size`, especially for mixed prompt lengths.

Record:

- cumulative GPU energy around the exact workload window;
- completed requests and input/output tokens;
- p50/p95/p99 TTFT, TPOT, and end-to-end latency;
- running/queued requests and KV-token state;
- 100 ms GPU busy, power, clocks, and clock-event reasons;
- cgroup CPU use/throttling and memory use.

The interesting region is the **USE saturation knee**: the point where throughput flattens, queues and tail latency start rising, but energy per useful token may still improve because batches become more efficient. This directly tests whether an SGLang parameter reduces energy-to-finish without disguising overload as efficiency.

### 2. Study prefix-cache reuse versus KV pressure

Compare a controlled repeated-prefix workload with a unique-prefix workload, then vary cache-related capacity or policy. Relate cache-hit rate and KV evictions to joules per output token and TTFT. This can reveal whether saved prefill work outweighs memory-capacity pressure and eviction/recomputation costs.

### 3. Separate prefill and decode regimes

Use at least a prompt-heavy workload and a decode-heavy workload. A setting may reduce energy in prefill by improving batching while worsening decode latency or queue saturation. Report energy by phase if client timing or targeted traces permit it; otherwise report the two workload classes separately.

### 4. Use Hopper GPM as the mechanism layer

On H100 runs, add 100 ms SM activity, occupancy, tensor/HMMA activity, DRAM activity, and PCIe/NVLink rates. Use them to explain changes in joules and latency, while keeping cumulative board energy and completed tokens as the primary outcome. Retain the A40-compatible schema as the portable baseline.

### 5. Preflight a two-GPU H100 topology before tensor-parallel experiments

The single-GPU H100 exposed 18 active NVLinks and per-link telemetry, but could not generate peer traffic. Before a tensor-parallel campaign, allocate two H100s on one pod, verify the `nvidia-smi topo -m` path, and run one NCCL transfer while sampling total and per-link NVLink rates and error/recovery counters.

### 6. Use short traces only to explain mechanisms

For a baseline and one surprising parameter setting, capture a short Nsight Systems trace. Use kernel/API/NCCL timing to explain differences seen in power, queueing, and latency. Do not run tracing throughout the benchmark campaign because profiler overhead can perturb the quantities being measured.

### 7. Move the power-cap experiment to a privileged environment

If changing the GPU cap remains a central independent variable, use bare metal or a provider mode that explicitly grants NVML/DCGM management permission. Keep the same telemetry and workload protocol so the SGLang-only and power-cap experiments remain comparable.

## Recommended minimum data schema

| Cadence | Data |
|---|---|
| 100 ms | Monotonic and native timestamps; GPU UUID; instantaneous/average power; cumulative energy; GPU/memory busy; clocks; P-state; temperature; enforced limit; clock-event reasons; framebuffer used/free; sampled PCIe RX/TX; on Hopper, GPM SM/occupancy/tensor/DRAM and link rates |
| 0.5–1 s | SGLang running/queued requests, throughput, cache hit, token/KV state; cgroup CPU utilization/throttling, memory/fail/OOM state; interface bytes/errors/drops |
| Per request | Request ID/workload class; arrival/start/first-token/end times; input/output tokens; result/error/timeout |
| Per run | Full command/config; model and revision; SGLang/container/driver/GPU versions; warm-up policy; offered load; random seed; GPU UUID; topology |
| Selected runs | Nsight Systems CUDA/NCCL trace and summary |

Prefer cumulative energy-counter deltas for the headline energy number. Use instantaneous power for plots and cross-checking. Warm the workload before opening the measured window: a one-second averaged-power field lags across ramp-up and ramp-down boundaries.

## Sources

- [Brendan Gregg: The USE Method](https://www.brendangregg.com/usemethod.html)
- [NVIDIA NVML field-value identifiers](https://docs.nvidia.com/deploy/nvml-api/group__nvmlFieldValueEnums.html)
- [NVIDIA NVML GPU Performance Metrics](https://docs.nvidia.com/deploy/nvml-api/group__GPM.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [SGLang observability](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/observability.mdx)
- [SGLang production metrics](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/references/production_metrics.mdx)
- [SGLang metrics collector source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py)
