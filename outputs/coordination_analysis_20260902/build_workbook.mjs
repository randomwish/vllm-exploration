import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const runRoot = "/home/randomwish/dev/vllm-exploration/experiments/laptop_slo_energy/coordination-results/laptop-qwen35-4b-medium-batch-thread-coordination-20260902T071754Z";
const outputDir = "/home/randomwish/dev/vllm-exploration/outputs/coordination_analysis_20260902";
const outputPath = path.join(outputDir, "coordination-ebpf-analysis.xlsx");

const COLORS = {
  navy: "#15324B",
  teal: "#147D7E",
  orange: "#D97706",
  paleBlue: "#E8F1F8",
  paleTeal: "#E8F6F3",
  paleOrange: "#FFF3E0",
  paleGreen: "#E7F5EA",
  paleRed: "#FDECEC",
  gray: "#5B6573",
  lightGray: "#E3E8ED",
  white: "#FFFFFF",
};

function colName(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

function pctChange(from, to) {
  return from ? to / from - 1 : null;
}

function pearson(rows, xKey, yKey) {
  const pairs = rows
    .map((row) => [row[xKey], row[yKey]])
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pairs.length < 3) return null;
  const xMean = pairs.reduce((sum, [x]) => sum + x, 0) / pairs.length;
  const yMean = pairs.reduce((sum, [, y]) => sum + y, 0) / pairs.length;
  let numerator = 0;
  let xSquares = 0;
  let ySquares = 0;
  for (const [x, y] of pairs) {
    const xd = x - xMean;
    const yd = y - yMean;
    numerator += xd * yd;
    xSquares += xd * xd;
    ySquares += yd * yd;
  }
  return xSquares && ySquares ? numerator / Math.sqrt(xSquares * ySquares) : null;
}

function parsePsi(value, kind = "some") {
  if (typeof value !== "string") return null;
  const match = value.match(new RegExp(`${kind} avg10=([0-9.]+)`));
  return match ? Number(match[1]) : null;
}

function parseEbpfMaps(text) {
  const maps = new Map();
  const pattern = /@(\w+_1s)\[(\d+)\]:\s+(\d+)/g;
  for (const match of text.matchAll(pattern)) {
    const [, name, bucketText, valueText] = match;
    if (!maps.has(name)) maps.set(name, new Map());
    maps.get(name).set(Number(bucketText), Number(valueText));
  }
  return maps;
}

function mapValue(maps, name, bucket) {
  return maps.get(name)?.get(bucket) ?? 0;
}

function successfulIntervals(report) {
  const requests = report.benchmarks.at(-1).requests.successful ?? [];
  return requests
    .filter((request) => Number.isFinite(request.request_start_time) && Number.isFinite(request.request_end_time) && Number.isFinite(request.time_to_first_token_ms))
    .sort((a, b) => a.request_start_time - b.request_start_time)
    .map((request, index) => ({
      index: index + 1,
      start: request.request_start_time,
      first: request.request_start_time + request.time_to_first_token_ms / 1000,
      end: request.request_end_time,
      request,
    }));
}

function classifyBucket(bucket, summary, intervals) {
  const centerBpf = bucket + 0.5;
  const centerUnix = summary.started_unix_s + (centerBpf - summary.started_bpf_clock_s);
  for (const interval of intervals) {
    if (centerUnix >= interval.start && centerUnix < interval.first) {
      return { phase: "prefill", requestIndex: interval.index, centerUnix };
    }
    if (centerUnix >= interval.first && centerUnix <= interval.end) {
      return { phase: "decode", requestIndex: interval.index, centerUnix };
    }
  }
  return { phase: "idle", requestIndex: null, centerUnix };
}

function bucketRange(summary) {
  const first = Math.ceil(summary.started_bpf_clock_s - 0.5);
  const last = Math.floor(summary.finished_bpf_clock_s - 0.5);
  const buckets = [];
  for (let bucket = first; bucket <= last; bucket += 1) buckets.push(bucket);
  return buckets;
}

function aggregateBuckets(bucketRows, phase) {
  const selected = bucketRows.filter((row) => row.phase === phase);
  const sum = (key) => selected.reduce((total, row) => total + (row[key] ?? 0), 0);
  const max = (key) => selected.reduce((value, row) => Math.max(value, row[key] ?? 0), 0);
  const runqlatSamples = sum("runqlat_samples");
  const futexSamples = sum("futex_wait_samples");
  const migrations = sum("sched_migrate_task");
  return {
    phase,
    selected_1s_buckets: selected.length,
    runqlat_samples: runqlatSamples,
    runqlat_mean_us: runqlatSamples ? sum("runqlat_sum_us") / runqlatSamples : null,
    runqlat_max_us: max("runqlat_max_us"),
    runqlat_ge_100us_samples: sum("runqlat_ge_100us"),
    runqlat_ge_100us_fraction: runqlatSamples ? sum("runqlat_ge_100us") / runqlatSamples : null,
    runqlat_ge_1ms_samples: sum("runqlat_ge_1ms"),
    runqlat_ge_10ms_samples: sum("runqlat_ge_10ms"),
    sched_migrate_task_samples: migrations,
    sched_migrate_task_per_schedule_in: runqlatSamples ? migrations / runqlatSamples : null,
    sched_migrations_per_phase_second: selected.length ? migrations / selected.length : null,
    schedule_in_cpu_change_samples: sum("cpu_changes"),
    schedule_in_cpu_change_fraction: runqlatSamples ? sum("cpu_changes") / runqlatSamples : null,
    futex_wait_samples: futexSamples,
    futex_wait_mean_us: futexSamples ? sum("futex_wait_sum_us") / futexSamples : null,
    futex_wait_max_us: max("futex_wait_max_us"),
    futex_wait_ge_10ms_samples: sum("futex_wait_ge_10ms"),
    futex_waits_per_phase_second: selected.length ? futexSamples / selected.length : null,
    futex_wake_calls: sum("futex_wake_calls"),
    futex_wakes_per_phase_second: selected.length ? sum("futex_wake_calls") / selected.length : null,
  };
}

function aggregateRequestPhase(bucketRows, requestIndex, phase) {
  const selected = bucketRows.filter((row) => row.request_index === requestIndex && row.phase === phase);
  const sum = (key) => selected.reduce((total, row) => total + (row[key] ?? 0), 0);
  const max = (key) => selected.reduce((value, row) => Math.max(value, row[key] ?? 0), 0);
  const runqlatSamples = sum("runqlat_samples");
  const futexSamples = sum("futex_wait_samples");
  return {
    buckets: selected.length,
    runqlat_samples: runqlatSamples,
    runqlat_mean_us: runqlatSamples ? sum("runqlat_sum_us") / runqlatSamples : null,
    runqlat_max_us: max("runqlat_max_us"),
    runqlat_ge_100us: sum("runqlat_ge_100us"),
    migrations: sum("sched_migrate_task"),
    cpu_changes: sum("cpu_changes"),
    futex_waits: futexSamples,
    futex_wait_mean_us: futexSamples ? sum("futex_wait_sum_us") / futexSamples : null,
    futex_wait_max_us: max("futex_wait_max_us"),
    futex_wakes: sum("futex_wake_calls"),
  };
}

function styleTitle(sheet, range, text) {
  range.merge();
  range.values = [[text]];
  range.format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 18 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 34;
}

function styleHeader(range, fill = COLORS.teal) {
  range.format = {
    fill,
    font: { bold: true, color: COLORS.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "inside", style: "thin", color: COLORS.lightGray },
  };
  range.format.rowHeight = 32;
}

function setWidths(sheet, widths, lastRow) {
  widths.forEach((width, index) => {
    sheet.getRange(`${colName(index)}1:${colName(index)}${lastRow}`).format.columnWidth = width;
  });
}

const campaign = JSON.parse(await fs.readFile(path.join(runRoot, "campaign.json"), "utf8"));
const campaignSummary = JSON.parse(await fs.readFile(path.join(runRoot, "summary.json"), "utf8"));
const treatmentConfig = new Map(campaign.treatments.map((item) => [item.name, item]));
const treatments = ["threads-4-batch-4", "threads-4-batch-2"];
const treatmentData = [];

for (const treatment of treatments) {
  const cell = campaignSummary.cells.find((item) => item.treatment === treatment);
  const cellDir = path.join(runRoot, "policy", cell.cell_id);
  const report = JSON.parse(await fs.readFile(path.join(cellDir, "guidellm.json"), "utf8"));
  const ebpfText = await fs.readFile(path.join(cellDir, "ebpf-runqlat.txt"), "utf8");
  const maps = parseEbpfMaps(ebpfText);
  const intervals = successfulIntervals(report);
  const buckets = bucketRange(cell);
  const bucketRows = buckets.map((bucket) => {
    const classified = classifyBucket(bucket, cell, intervals);
    return {
      treatment,
      batch_threads: treatmentConfig.get(treatment).batch_threads,
      bucket_boottime_s: bucket,
      bucket_center_unix_s: classified.centerUnix,
      bucket_center_iso_utc: new Date(classified.centerUnix * 1000).toISOString(),
      phase: classified.phase,
      request_index: classified.requestIndex,
      runqlat_samples: mapValue(maps, "runqlat_count_1s", bucket),
      runqlat_sum_us: mapValue(maps, "runqlat_sum_us_1s", bucket),
      runqlat_max_us: mapValue(maps, "runqlat_max_us_1s", bucket),
      runqlat_ge_100us: mapValue(maps, "runqlat_ge_100us_1s", bucket),
      runqlat_ge_1ms: mapValue(maps, "runqlat_ge_1ms_1s", bucket),
      runqlat_ge_10ms: mapValue(maps, "runqlat_ge_10ms_1s", bucket),
      cpu_changes: mapValue(maps, "cpu_changes_1s", bucket),
      sched_migrate_task: mapValue(maps, "sched_migrate_task_1s", bucket),
      futex_wait_samples: mapValue(maps, "futex_wait_count_1s", bucket),
      futex_wait_sum_us: mapValue(maps, "futex_wait_sum_us_1s", bucket),
      futex_wait_max_us: mapValue(maps, "futex_wait_max_us_1s", bucket),
      futex_wait_ge_100us: mapValue(maps, "futex_wait_ge_100us_1s", bucket),
      futex_wait_ge_1ms: mapValue(maps, "futex_wait_ge_1ms_1s", bucket),
      futex_wait_ge_10ms: mapValue(maps, "futex_wait_ge_10ms_1s", bucket),
      futex_wake_calls: mapValue(maps, "futex_wake_calls_1s", bucket),
    };
  });

  const rawHost = (await fs.readFile(path.join(cellDir, "host.jsonl"), "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line))
    .filter((row) => row.unix_s >= cell.started_unix_s && row.unix_s <= cell.finished_unix_s);
  const firstHost = rawHost[0];
  const lastHost = rawHost.at(-1);
  const hostSummary = {
    host_samples: rawHost.length,
    user_ticks_delta: lastHost.process.user_ticks - firstHost.process.user_ticks,
    system_ticks_delta: lastHost.process.system_ticks - firstHost.process.system_ticks,
    voluntary_context_switches_delta: lastHost.process.voluntary_context_switches - firstHost.process.voluntary_context_switches,
    involuntary_context_switches_delta: lastHost.process.involuntary_context_switches - firstHost.process.involuntary_context_switches,
    minor_faults_delta: lastHost.process.minor_faults - firstHost.process.minor_faults,
    major_faults_delta: lastHost.process.major_faults - firstHost.process.major_faults,
    avg_rss_mib: rawHost.reduce((sum, row) => sum + row.process.resident_pages * 4096 / 1048576, 0) / rawHost.length,
    avg_temp_c: rawHost.reduce((sum, row) => sum + (row.thermal_millicelsius.thermal_zone0 ?? 0) / 1000, 0) / rawHost.length,
    max_temp_c: Math.max(...rawHost.map((row) => (row.thermal_millicelsius.thermal_zone0 ?? 0) / 1000)),
    avg_cpu_psi_some_avg10: rawHost.reduce((sum, row) => sum + (parsePsi(row.pressure.cpu) ?? 0), 0) / rawHost.length,
  };

  const rawRequests = [];
  for (const status of ["successful", "incomplete", "errored"]) {
    for (const request of report.benchmarks.at(-1).requests[status] ?? []) {
      const start = request.request_start_time;
      rawRequests.push({
        treatment,
        batch_threads: treatmentConfig.get(treatment).batch_threads,
        status,
        measured_request: Number.isFinite(start) && start >= cell.guide.measurement_start_unix_s && start <= cell.guide.measurement_end_unix_s,
        request_id: request.request_id ?? request.info?.request_id ?? null,
        targeted_start_unix_s: request.info?.timings?.targeted_start ?? null,
        request_start_unix_s: start ?? null,
        first_token_unix_s: Number.isFinite(start) && Number.isFinite(request.time_to_first_token_ms) ? start + request.time_to_first_token_ms / 1000 : null,
        request_end_unix_s: request.request_end_time ?? null,
        prompt_tokens: request.prompt_tokens ?? null,
        output_tokens: request.output_tokens ?? null,
        ttft_ms: request.time_to_first_token_ms ?? null,
        itl_ms: request.inter_token_latency_ms ?? null,
        e2e_ms: Number.isFinite(request.request_latency) ? request.request_latency * 1000 : null,
        output_tokens_per_second: request.output_tokens_per_second ?? null,
        error: request.info?.error ?? null,
      });
    }
  }
  rawRequests.sort((a, b) => (a.targeted_start_unix_s ?? Infinity) - (b.targeted_start_unix_s ?? Infinity));

  const requestEbpf = intervals.map((interval) => {
    const prefill = aggregateRequestPhase(bucketRows, interval.index, "prefill");
    const decode = aggregateRequestPhase(bucketRows, interval.index, "decode");
    return {
      treatment,
      batch_threads: treatmentConfig.get(treatment).batch_threads,
      request_index: interval.index,
      request_start_iso_utc: new Date(interval.start * 1000).toISOString(),
      prompt_tokens: interval.request.prompt_tokens,
      output_tokens: interval.request.output_tokens,
      ttft_ms: interval.request.time_to_first_token_ms,
      itl_ms: interval.request.inter_token_latency_ms,
      e2e_ms: interval.request.request_latency * 1000,
      prefill,
      decode,
    };
  });

  const phaseRows = ["prefill", "decode", "idle"].map((phase) => ({
    treatment,
    batch_threads: treatmentConfig.get(treatment).batch_threads,
    ...aggregateBuckets(bucketRows, phase),
  }));

  treatmentData.push({ treatment, cell, report, bucketRows, rawHost, hostSummary, rawRequests, requestEbpf, phaseRows });
}

const rawCellRows = treatmentData.map(({ treatment, cell, hostSummary }) => {
  const runq = cell.ebpf.runqlat_us;
  return {
    cell_id: cell.cell_id,
    treatment,
    threads: cell.threads,
    batch_threads: treatmentConfig.get(treatment).batch_threads,
    offered_rate_rps: cell.offered_rate_requests_s,
    duration_s: cell.duration_s,
    successful_requests: cell.guide.successful_requests,
    incomplete_requests: cell.guide.incomplete_requests,
    admitted_requests: cell.guide.admitted_requests,
    success_rate: cell.guide.success_rate,
    p50_ttft_ms: cell.guide.p50_ttft_ms,
    p95_ttft_ms: cell.guide.p95_ttft_ms,
    p99_ttft_ms: cell.guide.p99_ttft_ms,
    p50_itl_ms: cell.guide.p50_itl_ms,
    p95_itl_ms: cell.guide.p95_itl_ms,
    p99_itl_ms: cell.guide.p99_itl_ms,
    p95_e2e_ms: cell.guide.p95_e2e_ms,
    p99_e2e_ms: cell.guide.p99_e2e_ms,
    output_tokens_successful: cell.guide.output_tokens_successful,
    package_energy_j: cell.energy.total_energy_j,
    core_energy_j: cell.energy.events["power_core/energy-core/"].joules,
    average_package_power_w: cell.energy.total_energy_j / cell.duration_s,
    average_core_power_w: cell.energy.events["power_core/energy-core/"].joules / cell.duration_s,
    joules_per_success: cell.efficiency.joules_per_successful_request,
    output_tokens_per_joule: cell.efficiency.output_tokens_per_joule,
    cell_passes_slo: cell.slo.cell_passes_slo,
    success_rate_pass: cell.slo.checks.success_rate,
    ttft_pass: cell.slo.checks.p95_ttft_ms,
    itl_pass: cell.slo.checks.p95_itl_ms,
    e2e_pass: cell.slo.checks.p99_e2e_ms,
    runqlat_samples: runq.samples,
    runqlat_mean_us: runq.mean_us,
    runqlat_max_us: runq.max_us,
    runqlat_ge_100us_samples: runq.ge_100us_samples,
    runqlat_ge_100us_fraction: runq.ge_100us_fraction,
    runqlat_ge_1ms_samples: runq.ge_1ms_samples,
    runqlat_ge_10ms_samples: runq.ge_10ms_samples,
    sched_migrate_task_samples: runq.sched_migrate_task_samples,
    sched_migrate_task_per_schedule_in: runq.sched_migrate_task_per_schedule_in,
    schedule_in_cpu_change_samples: runq.schedule_in_cpu_change_samples,
    schedule_in_cpu_change_fraction: runq.schedule_in_cpu_change_fraction,
    futex_wait_samples: runq.futex_wait_us.samples,
    futex_wait_mean_us: runq.futex_wait_us.mean_us,
    futex_wait_max_us: runq.futex_wait_us.max_us,
    futex_wait_ge_10ms_samples: runq.futex_wait_us.ge_10ms_samples,
    futex_wake_calls: runq.futex_wake_calls,
    ...hostSummary,
  };
});

const rawCellHeaders = Object.keys(rawCellRows[0]);
const rawCellColumn = new Map(rawCellHeaders.map((name, index) => [name, colName(index)]));
const rawCellRow = new Map(rawCellRows.map((row, index) => [row.treatment, index + 2]));

const phaseRows = treatmentData.flatMap((item) => item.phaseRows);
const phaseHeaders = Object.keys(phaseRows[0]);
const phaseColumn = new Map(phaseHeaders.map((name, index) => [name, colName(index)]));
const phaseRow = new Map(phaseRows.map((row, index) => [`${row.treatment}|${row.phase}`, index + 2]));

const requestRows = treatmentData.flatMap((item) => item.requestEbpf);
const requestFlatRows = requestRows.map((row) => ({
  treatment: row.treatment,
  batch_threads: row.batch_threads,
  request_index: row.request_index,
  request_start_iso_utc: row.request_start_iso_utc,
  prompt_tokens: row.prompt_tokens,
  output_tokens: row.output_tokens,
  ttft_ms: row.ttft_ms,
  itl_ms: row.itl_ms,
  e2e_ms: row.e2e_ms,
  prefill_buckets: row.prefill.buckets,
  prefill_runqlat_samples: row.prefill.runqlat_samples,
  prefill_runqlat_mean_us: row.prefill.runqlat_mean_us,
  prefill_runqlat_max_us: row.prefill.runqlat_max_us,
  prefill_runqlat_ge_100us: row.prefill.runqlat_ge_100us,
  prefill_sched_migrations: row.prefill.migrations,
  prefill_migrations_per_bucket: row.prefill.buckets ? row.prefill.migrations / row.prefill.buckets : null,
  prefill_cpu_changes: row.prefill.cpu_changes,
  prefill_futex_waits: row.prefill.futex_waits,
  prefill_futex_waits_per_bucket: row.prefill.buckets ? row.prefill.futex_waits / row.prefill.buckets : null,
  prefill_futex_wait_mean_us: row.prefill.futex_wait_mean_us,
  prefill_futex_wait_max_us: row.prefill.futex_wait_max_us,
  prefill_futex_wakes: row.prefill.futex_wakes,
  prefill_futex_wakes_per_bucket: row.prefill.buckets ? row.prefill.futex_wakes / row.prefill.buckets : null,
  decode_buckets: row.decode.buckets,
  decode_runqlat_samples: row.decode.runqlat_samples,
  decode_runqlat_mean_us: row.decode.runqlat_mean_us,
  decode_runqlat_max_us: row.decode.runqlat_max_us,
  decode_runqlat_ge_100us: row.decode.runqlat_ge_100us,
  decode_sched_migrations: row.decode.migrations,
  decode_cpu_changes: row.decode.cpu_changes,
  decode_futex_waits: row.decode.futex_waits,
  decode_futex_wait_mean_us: row.decode.futex_wait_mean_us,
  decode_futex_wait_max_us: row.decode.futex_wait_max_us,
  decode_futex_wakes: row.decode.futex_wakes,
}));

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");
const comparison = workbook.worksheets.add("Comparison");
const phaseSheet = workbook.worksheets.add("eBPF by phase");
const requestSheet = workbook.worksheets.add("Request eBPF");
const rawCellsSheet = workbook.worksheets.add("Raw cells");
const rawRequestsSheet = workbook.worksheets.add("Raw requests");
const rawEbpfSheet = workbook.worksheets.add("Raw eBPF 1s");
const rawHostSheet = workbook.worksheets.add("Raw host");
const notes = workbook.worksheets.add("Notes");
for (const sheet of [overview, comparison, phaseSheet, requestSheet, rawCellsSheet, rawRequestsSheet, rawEbpfSheet, rawHostSheet, notes]) {
  sheet.showGridLines = false;
}

// Raw cells.
rawCellsSheet.getRange(`A1:${colName(rawCellHeaders.length - 1)}${rawCellRows.length + 1}`).values = [
  rawCellHeaders,
  ...rawCellRows.map((row) => rawCellHeaders.map((header) => row[header] ?? null)),
];
styleHeader(rawCellsSheet.getRange(`A1:${colName(rawCellHeaders.length - 1)}1`));
rawCellsSheet.freezePanes.freezeRows(1);
rawCellsSheet.freezePanes.freezeColumns(2);
rawCellsSheet.tables.add(`A1:${colName(rawCellHeaders.length - 1)}${rawCellRows.length + 1}`, true, "RawCellsTable");
setWidths(rawCellsSheet, rawCellHeaders.map((header) => header.includes("cell_id") ? 38 : header === "treatment" ? 24 : 16), rawCellRows.length + 1);
rawCellsSheet.getRange(`E2:F${rawCellRows.length + 1}`).format.numberFormat = "0.0000";
rawCellsSheet.getRange(`J2:J${rawCellRows.length + 1}`).format.numberFormat = "0.0%";

// eBPF phase summary.
phaseSheet.getRange(`A1:${colName(phaseHeaders.length - 1)}${phaseRows.length + 1}`).values = [
  phaseHeaders,
  ...phaseRows.map((row) => phaseHeaders.map((header) => row[header] ?? null)),
];
styleHeader(phaseSheet.getRange(`A1:${colName(phaseHeaders.length - 1)}1`), COLORS.orange);
phaseSheet.freezePanes.freezeRows(1);
phaseSheet.freezePanes.freezeColumns(3);
phaseSheet.tables.add(`A1:${colName(phaseHeaders.length - 1)}${phaseRows.length + 1}`, true, "EbpfPhaseTable");
setWidths(phaseSheet, phaseHeaders.map((header) => header === "treatment" ? 24 : header === "phase" ? 12 : 17), phaseRows.length + 1);

// Request + eBPF table.
const requestHeaders = Object.keys(requestFlatRows[0]);
requestSheet.getRange(`A1:${colName(requestHeaders.length - 1)}${requestFlatRows.length + 1}`).values = [
  requestHeaders,
  ...requestFlatRows.map((row) => requestHeaders.map((header) => row[header] ?? null)),
];
styleHeader(requestSheet.getRange(`A1:${colName(requestHeaders.length - 1)}1`), COLORS.orange);
requestSheet.freezePanes.freezeRows(1);
requestSheet.freezePanes.freezeColumns(3);
requestSheet.tables.add(`A1:${colName(requestHeaders.length - 1)}${requestFlatRows.length + 1}`, true, "RequestEbpfTable");
setWidths(requestSheet, requestHeaders.map((header) => header === "treatment" ? 24 : header.includes("iso") ? 25 : 16), requestFlatRows.length + 1);

// Raw requests, including records outside the measured window with an explicit flag.
const rawRequestRows = treatmentData.flatMap((item) => item.rawRequests);
const rawRequestHeaders = Object.keys(rawRequestRows[0]);
rawRequestsSheet.getRange(`A1:${colName(rawRequestHeaders.length - 1)}${rawRequestRows.length + 1}`).values = [
  rawRequestHeaders,
  ...rawRequestRows.map((row) => rawRequestHeaders.map((header) => row[header] ?? null)),
];
styleHeader(rawRequestsSheet.getRange(`A1:${colName(rawRequestHeaders.length - 1)}1`));
rawRequestsSheet.freezePanes.freezeRows(1);
rawRequestsSheet.freezePanes.freezeColumns(3);
rawRequestsSheet.tables.add(`A1:${colName(rawRequestHeaders.length - 1)}${rawRequestRows.length + 1}`, true, "RawRequestsTable");
setWidths(rawRequestsSheet, rawRequestHeaders.map((header) => header === "treatment" ? 24 : header.includes("request_id") ? 38 : header === "error" ? 24 : 18), rawRequestRows.length + 1);

// Raw one-second eBPF buckets. Two derived columns remain formulas for auditability.
const rawBucketRows = treatmentData.flatMap((item) => item.bucketRows);
const rawBucketHeaders = [
  "treatment", "batch_threads", "bucket_boottime_s", "bucket_center_unix_s", "bucket_center_iso_utc", "phase", "request_index",
  "runqlat_samples", "runqlat_sum_us", "runqlat_mean_us", "runqlat_max_us", "runqlat_ge_100us", "runqlat_ge_1ms", "runqlat_ge_10ms",
  "cpu_changes", "cpu_change_fraction", "sched_migrate_task", "migrations_per_schedule_in",
  "futex_wait_samples", "futex_wait_sum_us", "futex_wait_mean_us", "futex_wait_max_us", "futex_wait_ge_100us", "futex_wait_ge_1ms", "futex_wait_ge_10ms", "futex_wake_calls",
];
const rawBucketStaticHeaders = rawBucketHeaders.filter((header) => !["runqlat_mean_us", "cpu_change_fraction", "migrations_per_schedule_in", "futex_wait_mean_us"].includes(header));
const bucketMatrix = rawBucketRows.map((row) => rawBucketHeaders.map((header) => {
  if (!rawBucketStaticHeaders.includes(header)) return null;
  return row[header] ?? null;
}));
rawEbpfSheet.getRange(`A1:${colName(rawBucketHeaders.length - 1)}${rawBucketRows.length + 1}`).values = [rawBucketHeaders, ...bucketMatrix];
const rb = Object.fromEntries(rawBucketHeaders.map((header, index) => [header, colName(index)]));
rawEbpfSheet.getRange(`${rb.runqlat_mean_us}2`).formulas = [[`=IFERROR(${rb.runqlat_sum_us}2/${rb.runqlat_samples}2,"")`]];
rawEbpfSheet.getRange(`${rb.runqlat_mean_us}2:${rb.runqlat_mean_us}${rawBucketRows.length + 1}`).fillDown();
rawEbpfSheet.getRange(`${rb.cpu_change_fraction}2`).formulas = [[`=IFERROR(${rb.cpu_changes}2/${rb.runqlat_samples}2,"")`]];
rawEbpfSheet.getRange(`${rb.cpu_change_fraction}2:${rb.cpu_change_fraction}${rawBucketRows.length + 1}`).fillDown();
rawEbpfSheet.getRange(`${rb.migrations_per_schedule_in}2`).formulas = [[`=IFERROR(${rb.sched_migrate_task}2/${rb.runqlat_samples}2,"")`]];
rawEbpfSheet.getRange(`${rb.migrations_per_schedule_in}2:${rb.migrations_per_schedule_in}${rawBucketRows.length + 1}`).fillDown();
rawEbpfSheet.getRange(`${rb.futex_wait_mean_us}2`).formulas = [[`=IFERROR(${rb.futex_wait_sum_us}2/${rb.futex_wait_samples}2,"")`]];
rawEbpfSheet.getRange(`${rb.futex_wait_mean_us}2:${rb.futex_wait_mean_us}${rawBucketRows.length + 1}`).fillDown();
styleHeader(rawEbpfSheet.getRange(`A1:${colName(rawBucketHeaders.length - 1)}1`), COLORS.orange);
rawEbpfSheet.freezePanes.freezeRows(1);
rawEbpfSheet.freezePanes.freezeColumns(7);
rawEbpfSheet.tables.add(`A1:${colName(rawBucketHeaders.length - 1)}${rawBucketRows.length + 1}`, true, "RawEbpfTable");
setWidths(rawEbpfSheet, rawBucketHeaders.map((header) => header === "treatment" ? 24 : header.includes("iso") ? 25 : header === "phase" ? 10 : 16), rawBucketRows.length + 1);
rawEbpfSheet.getRange(`${rb.cpu_change_fraction}2:${rb.cpu_change_fraction}${rawBucketRows.length + 1}`).format.numberFormat = "0.00%";
rawEbpfSheet.getRange(`${rb.migrations_per_schedule_in}2:${rb.migrations_per_schedule_in}${rawBucketRows.length + 1}`).format.numberFormat = "0.00%";

// Raw host rows.
const rawHostRows = treatmentData.flatMap(({ treatment, rawHost }) => rawHost.map((row) => {
  const frequencies = Object.values(row.cpu_frequency_khz ?? {}).filter(Number.isFinite);
  const fastFrequencies = [0, 2, 4, 6].map((cpu) => row.cpu_frequency_khz?.[`cpu${cpu}`]).filter(Number.isFinite);
  return {
    treatment,
    unix_s: row.unix_s,
    iso_utc: new Date(row.unix_s * 1000).toISOString(),
    interval_s: row.interval_s,
    user_ticks: row.process.user_ticks,
    system_ticks: row.process.system_ticks,
    minor_faults: row.process.minor_faults,
    major_faults: row.process.major_faults,
    resident_mib: row.process.resident_pages * 4096 / 1048576,
    threads: row.process.threads,
    last_cpu: row.process.last_cpu,
    voluntary_context_switches: row.process.voluntary_context_switches,
    involuntary_context_switches: row.process.involuntary_context_switches,
    cpu_psi_some_avg10: parsePsi(row.pressure.cpu),
    memory_psi_some_avg10: parsePsi(row.pressure.memory),
    io_psi_some_avg10: parsePsi(row.pressure.io),
    avg_cpu_frequency_mhz: frequencies.length ? frequencies.reduce((sum, value) => sum + value, 0) / frequencies.length / 1000 : null,
    avg_fast_core_frequency_mhz: fastFrequencies.length ? fastFrequencies.reduce((sum, value) => sum + value, 0) / fastFrequencies.length / 1000 : null,
    temperature_c: (row.thermal_millicelsius.thermal_zone0 ?? 0) / 1000,
  };
}));
const rawHostHeaders = Object.keys(rawHostRows[0]);
rawHostSheet.getRange(`A1:${colName(rawHostHeaders.length - 1)}${rawHostRows.length + 1}`).values = [
  rawHostHeaders,
  ...rawHostRows.map((row) => rawHostHeaders.map((header) => row[header] ?? null)),
];
styleHeader(rawHostSheet.getRange(`A1:${colName(rawHostHeaders.length - 1)}1`));
rawHostSheet.freezePanes.freezeRows(1);
rawHostSheet.freezePanes.freezeColumns(3);
rawHostSheet.tables.add(`A1:${colName(rawHostHeaders.length - 1)}${rawHostRows.length + 1}`, true, "RawHostTable");
setWidths(rawHostSheet, rawHostHeaders.map((header) => header === "treatment" ? 24 : header === "iso_utc" ? 25 : 17), rawHostRows.length + 1);

// Formula-driven comparison sheet.
const cellRef = (treatment, field) => `'Raw cells'!${rawCellColumn.get(field)}${rawCellRow.get(treatment)}`;
const phaseRef = (treatment, phase, field) => `'eBPF by phase'!${phaseColumn.get(field)}${phaseRow.get(`${treatment}|${phase}`)}`;
const comparisonMetrics = [
  ["Service", "Completed requests", "count", "same", "successful_requests", null, "Equal useful work"],
  ["Service", "Admitted requests", "count", "context", "admitted_requests", null, "Batch-4's extra request was admitted at the deadline"],
  ["Service", "p50 TTFT", "ms", "lower", "p50_ttft_ms", null, "Central prefill latency"],
  ["Service", "p95 TTFT", "ms", "lower", "p95_ttft_ms", null, "Primary missed objective; 8,000 ms target"],
  ["Service", "p95 ITL", "ms", "lower", "p95_itl_ms", null, "Both meet the 120 ms objective"],
  ["Service", "p99 end-to-end", "ms", "lower", "p99_e2e_ms", null, "Both meet the 30,000 ms objective"],
  ["Energy", "Package energy", "J", "lower", "package_energy_j", null, "Equal 1,080-second windows; idle energy included"],
  ["Energy", "Average package power", "W", "lower", "average_package_power_w", null, "Package energy divided by measured duration"],
  ["Energy", "Joules per completed request", "J/request", "lower", "joules_per_success", null, "Equal completed request count"],
  ["Energy", "Output tokens per joule", "tokens/J", "higher", "output_tokens_per_joule", null, "Package-domain efficiency"],
  ["Energy", "Core energy event", "J", "lower", "core_energy_j", null, "Moves opposite package energy; treat attribution cautiously"],
  ["eBPF overall", "Runnable-to-running samples", "count", "context", "runqlat_samples", null, "Process-wide schedule-ins after wakeup or preemption"],
  ["eBPF overall", "Mean run-queue wait", "µs", "lower", "runqlat_mean_us", null, "Still microseconds, not the seconds-scale TTFT cause"],
  ["eBPF overall", "Maximum run-queue wait", "µs", "lower", "runqlat_max_us", null, "No observed wait reached 10 ms"],
  ["eBPF overall", "Waits ≥100 µs", "%", "lower", "runqlat_ge_100us_fraction", null, "Tail fraction increased with two batch threads"],
  ["eBPF overall", "Actual scheduler migrations", "count", "lower", "sched_migrate_task_samples", null, "Direct sched_migrate_task events"],
  ["eBPF overall", "Migrations per schedule-in", "%", "lower", "sched_migrate_task_per_schedule_in", null, "Normalizes migration count for activity"],
  ["eBPF overall", "Schedule-in CPU changes", "count", "lower", "schedule_in_cpu_change_samples", null, "Consecutive observed CPU changed"],
  ["eBPF overall", "Futex wait completions", "count", "lower", "futex_wait_samples", null, "Overall count stayed almost unchanged"],
  ["eBPF overall", "Futex wake calls", "count", "lower", "futex_wake_calls", null, "Overall wake activity stayed almost unchanged"],
  ["eBPF prefill", "Prefill phase seconds", "1 s buckets", "context", "selected_1s_buckets", "prefill", "Bucket-center classification"],
  ["eBPF prefill", "Prefill schedule-ins", "count", "lower", "runqlat_samples", "prefill", "Less runnable activity with two batch threads"],
  ["eBPF prefill", "Prefill actual migrations", "count", "lower", "sched_migrate_task_samples", "prefill", "Direct migration events fell substantially"],
  ["eBPF prefill", "Prefill migrations per schedule-in", "%", "lower", "sched_migrate_task_per_schedule_in", "prefill", "Activity-normalized migration rate"],
  ["eBPF prefill", "Prefill migrations per phase second", "count/s", "lower", "sched_migrations_per_phase_second", "prefill", "Normalizes for shorter prefill"],
  ["eBPF prefill", "Prefill CPU changes", "count", "lower", "schedule_in_cpu_change_samples", "prefill", "Supports reduced cross-CPU movement"],
  ["eBPF prefill", "Prefill futex wait completions", "count", "lower", "futex_wait_samples", "prefill", "Do not sum durations as wall time"],
  ["eBPF prefill", "Prefill futex waits per phase second", "count/s", "lower", "futex_waits_per_phase_second", "prefill", "Normalizes for shorter prefill"],
  ["eBPF prefill", "Prefill futex wake calls", "count", "lower", "futex_wake_calls", "prefill", "Coordination wake activity"],
  ["eBPF prefill", "Prefill wakes per phase second", "count/s", "lower", "futex_wakes_per_phase_second", "prefill", "Normalizes for shorter prefill"],
  ["eBPF decode", "Decode actual migrations", "count", "lower", "sched_migrate_task_samples", "decode", "Migration reduction also persists in decode"],
  ["eBPF decode", "Decode migrations per schedule-in", "%", "lower", "sched_migrate_task_per_schedule_in", "decode", "Activity-normalized migration rate"],
  ["Host", "Involuntary context switches", "count", "lower", "involuntary_context_switches_delta", null, "Process /proc delta"],
  ["Host", "Average temperature", "°C", "lower", "avg_temp_c", null, "Single exposed thermal zone"],
  ["Host", "Maximum temperature", "°C", "lower", "max_temp_c", null, "Single exposed thermal zone"],
];

const comparisonRows = comparisonMetrics.map(([category, metric, unit, preference, field, phase, note], index) => {
  const row = index + 2;
  const batch4Ref = phase ? phaseRef(treatments[0], phase, field) : cellRef(treatments[0], field);
  const batch2Ref = phase ? phaseRef(treatments[1], phase, field) : cellRef(treatments[1], field);
  return { category, metric, unit, preference, batch4Ref, batch2Ref, row, note };
});
comparison.getRange(`A1:I${comparisonRows.length + 1}`).values = [
  ["Category", "Metric", "Unit", "Preferred direction", "4 batch threads", "2 batch threads", "Absolute change", "Relative change", "Interpretation"],
  ...comparisonRows.map((row) => [row.category, row.metric, row.unit, row.preference, null, null, null, null, row.note]),
];
for (const row of comparisonRows) {
  comparison.getRange(`E${row.row}:H${row.row}`).formulas = [[
    `=${row.batch4Ref}`,
    `=${row.batch2Ref}`,
    `=F${row.row}-E${row.row}`,
    `=IFERROR(F${row.row}/E${row.row}-1,"")`,
  ]];
}
styleHeader(comparison.getRange("A1:I1"));
comparison.freezePanes.freezeRows(1);
comparison.freezePanes.freezeColumns(2);
comparison.tables.add(`A1:I${comparisonRows.length + 1}`, true, "ComparisonTable");
setWidths(comparison, [18, 34, 13, 18, 17, 17, 17, 17, 52], comparisonRows.length + 1);
comparison.getRange(`H2:H${comparisonRows.length + 1}`).format.numberFormat = "0.0%";
comparison.getRange(`I2:I${comparisonRows.length + 1}`).format.wrapText = true;
for (const row of comparisonRows) {
  if (["count", "1 s buckets"].includes(row.unit)) {
    comparison.getRange(`E${row.row}:G${row.row}`).format.numberFormat = "#,##0";
  } else if (row.unit === "%") {
    comparison.getRange(`E${row.row}:G${row.row}`).format.numberFormat = "0.00%";
  } else {
    comparison.getRange(`E${row.row}:G${row.row}`).format.numberFormat = "#,##0.000";
  }
  if (row.category.startsWith("eBPF")) comparison.getRange(`A${row.row}:I${row.row}`).format.fill = COLORS.paleOrange;
}

// Overview and charts.
styleTitle(overview, overview.getRange("A1:H1"), "Batch-thread coordination: SLO, energy, and eBPF evidence");
overview.getRange("A2:H2").merge();
overview.getRange("A2:H2").values = [["Qwen3.5-4B Q4_K_M · 2,048-token prompt · 128-token response · constant 25%-capacity traffic · 12 completed requests per treatment"]];
overview.getRange("A2:H2").format = { fill: COLORS.paleBlue, font: { color: COLORS.navy, italic: true }, wrapText: true };

const metricRow = new Map(comparisonRows.map((row) => [row.metric, row.row]));
const kpis = [
  ["p95 TTFT", "ms", metricRow.get("p95 TTFT"), COLORS.paleOrange],
  ["Package J/request", "J/request", metricRow.get("Joules per completed request"), COLORS.paleGreen],
  ["Actual migrations", "count", metricRow.get("Actual scheduler migrations"), COLORS.paleGreen],
  ["Prefill futex wakes", "count", metricRow.get("Prefill futex wake calls"), COLORS.paleGreen],
];
for (let index = 0; index < kpis.length; index += 1) {
  const startCol = index * 2 + 1;
  const left = colName(startCol - 1);
  const right = colName(startCol);
  const [label, unit, row, fill] = kpis[index];
  overview.getRange(`${left}4:${right}4`).merge();
  overview.getRange(`${left}4:${right}4`).values = [[`${label} change`]];
  overview.getRange(`${left}5:${right}5`).merge();
  overview.getRange(`${left}5:${right}5`).formulas = [[`='Comparison'!H${row}`]];
  overview.getRange(`${left}6:${right}6`).merge();
  overview.getRange(`${left}6:${right}6`).values = [[`2 batch threads vs 4 · ${unit}`]];
  overview.getRange(`${left}4:${right}6`).format = {
    fill,
    font: { color: COLORS.navy },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: COLORS.lightGray },
  };
  overview.getRange(`${left}4:${right}4`).format.font = { bold: true, color: COLORS.navy };
  overview.getRange(`${left}5:${right}5`).format.font = { bold: true, color: COLORS.navy, size: 18 };
  overview.getRange(`${left}5:${right}5`).format.numberFormat = "0.0%";
}

overview.getRange("A8:H10").merge();
overview.getRange("A8:H10").values = [["Result: two batch threads directionally dominate four for this run. They reduced p95 TTFT by 6.8%, package joules per completed request by 3.1%, and actual scheduler migrations by 30.2%. The TTFT objective still was not met: 10.67 s versus the provisional 8 s threshold."]];
overview.getRange("A8:H10").format = { fill: COLORS.paleTeal, font: { bold: true, color: COLORS.navy, size: 12 }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: COLORS.teal } };
overview.getRange("A12:H14").merge();
overview.getRange("A12:H14").values = [["eBPF attribution: the improvement coincides with fewer process-wide migrations and fewer prefill futex wake/wait completions, even after normalizing by phase length. Run-queue wait became slightly worse but remained below 2.2 ms, so CPU scheduling delay cannot explain the seconds-scale TTFT. Futex duration is not additive wall time: overlapping threads and prelude-started waits produce values as large as 1,166 s."]];
overview.getRange("A12:H14").format = { fill: COLORS.paleOrange, font: { color: COLORS.navy }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: COLORS.orange } };
setWidths(overview, [18, 18, 18, 18, 18, 18, 18, 18, 3, 16, 16, 16, 16, 16, 16, 16, 16, 16], 55);

// Formula-backed chart helper ranges.
overview.getRange("J35:L35").values = [["Request", "4 batch threads TTFT (ms)", "2 batch threads TTFT (ms)"]];
styleHeader(overview.getRange("J35:L35"), COLORS.gray);
const requestRowsByTreatment = new Map();
for (const treatment of treatments) {
  requestRowsByTreatment.set(treatment, requestFlatRows.map((row, index) => ({ row, sheetRow: index + 2 })).filter((item) => item.row.treatment === treatment));
}
const requestHeaderIndex = new Map(requestHeaders.map((header, index) => [header, colName(index)]));
for (let index = 0; index < 12; index += 1) {
  const row = 36 + index;
  const batch4SheetRow = requestRowsByTreatment.get(treatments[0])[index].sheetRow;
  const batch2SheetRow = requestRowsByTreatment.get(treatments[1])[index].sheetRow;
  overview.getRange(`J${row}:L${row}`).formulas = [[
    `='Request eBPF'!${requestHeaderIndex.get("request_index")}${batch4SheetRow}`,
    `='Request eBPF'!${requestHeaderIndex.get("ttft_ms")}${batch4SheetRow}`,
    `='Request eBPF'!${requestHeaderIndex.get("ttft_ms")}${batch2SheetRow}`,
  ]];
}
overview.getRange("J50:L50").values = [["Phase", "4 batch threads migrations", "2 batch threads migrations"]];
styleHeader(overview.getRange("J50:L50"), COLORS.gray);
for (const [index, phase] of ["prefill", "decode", "idle"].entries()) {
  const row = 51 + index;
  overview.getRange(`J${row}:L${row}`).formulas = [[
    `="${phase}"`,
    `=${phaseRef(treatments[0], phase, "sched_migrate_task_samples")}`,
    `=${phaseRef(treatments[1], phase, "sched_migrate_task_samples")}`,
  ]];
}
const ttftChart = overview.charts.add("line", overview.getRange("J35:L47"));
ttftChart.title = "TTFT sequence reveals an order/warm-state effect";
ttftChart.hasLegend = true;
ttftChart.xAxis = { axisType: "textAxis" };
ttftChart.yAxis = { numberFormatCode: "#,##0", min: 8000 };
ttftChart.setPosition("J2", "R16");
const migrationChart = overview.charts.add("bar", overview.getRange("J50:L53"));
migrationChart.title = "Actual scheduler migrations by phase";
migrationChart.hasLegend = true;
migrationChart.yAxis = { numberFormatCode: "#,##0" };
migrationChart.setPosition("J18", "R32");

// Notes and provenance.
styleTitle(notes, notes.getRange("A1:F1"), "Definitions, evidence boundaries, and source files");
const noteRows = [
  ["Result status", "complete; 2/2 execution-valid and 2/2 sample-sufficient cells"],
  ["Study scope", "Single seed and sequential treatment order. Exploratory mechanism evidence, not an SLA or regulatory compliance claim."],
  ["Primary conclusion", "Two batch threads improved all reported service-latency percentiles and package energy efficiency, but p95 TTFT remained above 8 seconds."],
  ["Boundary request", "Both treatments completed 12 requests. Four batch threads admitted a thirteenth request at the 1,080-second cutoff, so its 92.3% success figure is a deadline artifact."],
  ["eBPF run-queue latency", "Time from a llama-server task becoming runnable to being scheduled. One-second aggregates use bucket centers inside the measurement window."],
  ["Actual migration", "sched:sched_migrate_task events whose tracepoint comm is llama-server. This is process-wide and includes support threads, not only compute workers."],
  ["Schedule-in CPU change", "CPU differs from the task's previous observed schedule-in. Retained as a complementary inferred movement metric."],
  ["Futex wait duration", "Duration of blocking-capable futex syscalls. Multiple threads can wait simultaneously, so durations are not additive wall time or CPU time."],
  ["Futex phase assignment", "The entire duration is assigned to the one-second bucket in which the syscall exits. Waits can begin during the 91-second collector prelude."],
  ["Phase definitions", "Prefill: request start to first token. Decode: first token to request end. Idle: no successful request phase at the bucket center."],
  ["Raw eBPF formulas", "runqlat_mean_us, cpu_change_fraction, migrations_per_schedule_in, and futex_wait_mean_us are formulas derived from raw one-second counters."],
  ["Energy boundary", "RAPL package-domain energy for equal 1,080-second windows. It includes idle intervals and background package activity; it is not wall-plug energy."],
  ["Energy caveat", "Package energy fell 3.1%, while the separate core-energy event rose 22.0%. Repeat with alternating order and idle subtraction before causal attribution."],
  ["Order effect", "Four-batch-thread requests 1–5 were the slow cluster; two-batch-thread requests 11–12 were slow. Sequence effects can influence p95 with only 12 samples."],
  ["Source: campaign summary", path.join(runRoot, "summary.json")],
  ["Source: raw GuideLLM", path.join(runRoot, "policy/*/guidellm.json")],
  ["Source: raw eBPF", path.join(runRoot, "policy/*/ebpf-runqlat.txt")],
  ["Source: raw host", path.join(runRoot, "policy/*/host.jsonl")],
  ["Generated UTC", new Date().toISOString()],
];
notes.getRange(`A3:B${noteRows.length + 3}`).values = [["Item", "Definition / evidence note"], ...noteRows];
styleHeader(notes.getRange("A3:B3"));
notes.getRange(`A4:A${noteRows.length + 3}`).format.font = { bold: true, color: COLORS.navy };
notes.getRange(`B4:B${noteRows.length + 3}`).format.wrapText = true;
setWidths(notes, [28, 110], noteRows.length + 3);
notes.freezePanes.freezeRows(3);

// Descriptive per-request correlations, clearly marked exploratory.
const correlationStart = noteRows.length + 6;
notes.getRange(`A${correlationStart}:D${correlationStart}`).values = [["Exploratory request-level correlation", "4 batch threads", "2 batch threads", "Caution"]];
styleHeader(notes.getRange(`A${correlationStart}:D${correlationStart}`), COLORS.orange);
const correlationDefs = [
  ["TTFT vs prefill migrations per phase second", "ttft_ms", "prefill_migrations_per_bucket"],
  ["TTFT vs prefill futex wakes per phase second", "ttft_ms", "prefill_futex_wakes_per_bucket"],
  ["TTFT vs prefill run-queue mean", "ttft_ms", "prefill_runqlat_mean_us"],
  ["TTFT vs prefill futex waits per phase second", "ttft_ms", "prefill_futex_waits_per_bucket"],
];
const corrRows = correlationDefs.map(([label, x, y]) => [
  label,
  pearson(requestFlatRows.filter((row) => row.treatment === treatments[0]), x, y),
  pearson(requestFlatRows.filter((row) => row.treatment === treatments[1]), x, y),
  "n=12 per treatment; descriptive only, autocorrelation and order effects are not controlled",
]);
notes.getRange(`A${correlationStart + 1}:D${correlationStart + corrRows.length}`).values = corrRows;
notes.getRange(`B${correlationStart + 1}:C${correlationStart + corrRows.length}`).format.numberFormat = "0.000";
notes.getRange(`D${correlationStart + 1}:D${correlationStart + corrRows.length}`).format.wrapText = true;
setWidths(notes, [38, 70, 18, 66], correlationStart + corrRows.length);
notes.getRange(`B${noteRows.length + 3}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";

// Number formats and compact borders.
for (const sheet of [comparison, phaseSheet, requestSheet, rawCellsSheet, rawRequestsSheet, rawEbpfSheet, rawHostSheet]) {
  const used = sheet.getUsedRange();
  used.format.borders = { insideHorizontal: { style: "thin", color: "#EEF1F4" } };
}
comparison.getRange(`E2:H${comparisonRows.length + 1}`).format.horizontalAlignment = "right";
phaseSheet.getRange(`D2:${colName(phaseHeaders.length - 1)}${phaseRows.length + 1}`).format.numberFormat = "#,##0.000";
for (const [index, header] of phaseHeaders.entries()) {
  if (header.includes("fraction") || header.includes("per_schedule_in")) {
    phaseSheet.getRange(`${colName(index)}2:${colName(index)}${phaseRows.length + 1}`).format.numberFormat = "0.00%";
  } else if (header.includes("samples") || header.includes("calls") || header.includes("buckets")) {
    phaseSheet.getRange(`${colName(index)}2:${colName(index)}${phaseRows.length + 1}`).format.numberFormat = "#,##0";
  }
}
requestSheet.getRange(`D2:D${requestFlatRows.length + 1}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
requestSheet.getRange(`G2:I${requestFlatRows.length + 1}`).format.numberFormat = "#,##0.0";
rawRequestsSheet.getRange(`K2:M${rawRequestRows.length + 1}`).format.numberFormat = "#,##0.0";
rawEbpfSheet.getRange(`E2:E${rawBucketRows.length + 1}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
rawHostSheet.getRange(`C2:C${rawHostRows.length + 1}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
rawHostSheet.getRange(`Q2:S${rawHostRows.length + 1}`).format.numberFormat = "#,##0.0";

// Compact inspections and preview renders.
const overviewInspect = await workbook.inspect({ kind: "table", range: "Overview!A1:L14", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 12 });
console.log("OVERVIEW_INSPECT\n" + overviewInspect.ndjson);
const comparisonInspect = await workbook.inspect({ kind: "table", range: "Comparison!A1:I36", include: "values,formulas", tableMaxRows: 40, tableMaxCols: 9 });
console.log("COMPARISON_INSPECT\n" + comparisonInspect.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log("FORMULA_ERRORS\n" + errors.ndjson);

const previewRanges = {
  Overview: "A1:R53",
  Comparison: "A1:I22",
  "eBPF by phase": `A1:${colName(phaseHeaders.length - 1)}7`,
  "Request eBPF": `A1:${colName(requestHeaders.length - 1)}15`,
  "Raw cells": `A1:${colName(Math.min(rawCellHeaders.length - 1, 15))}3`,
  "Raw requests": `A1:${colName(rawRequestHeaders.length - 1)}15`,
  "Raw eBPF 1s": `A1:${colName(rawBucketHeaders.length - 1)}15`,
  "Raw host": `A1:${colName(rawHostHeaders.length - 1)}15`,
  Notes: `A1:F${correlationStart + corrRows.length}`,
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "-");
  await fs.writeFile(path.join(outputDir, `preview-${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: { rawCells: rawCellRows.length, phases: phaseRows.length, requests: rawRequestRows.length, requestEbpf: requestFlatRows.length, rawEbpf: rawBucketRows.length, rawHost: rawHostRows.length } }, null, 2));
