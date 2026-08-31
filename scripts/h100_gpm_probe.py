#!/usr/bin/env python3
"""Sample Hopper NVML GPM metrics together with power and energy fields."""

from __future__ import annotations

import argparse
import csv
import sys
import time

import pynvml as nvml


GPM_METRICS = {
    1: "graphics_activity_pct",
    2: "sm_activity_pct",
    3: "sm_occupancy_pct",
    4: "integer_activity_pct",
    5: "tensor_activity_pct",
    6: "dfma_tensor_activity_pct",
    7: "hmma_tensor_activity_pct",
    9: "imma_tensor_activity_pct",
    10: "dram_activity_pct",
    11: "fp64_activity_pct",
    12: "fp32_activity_pct",
    13: "fp16_activity_pct",
    20: "pcie_tx_mib_s",
    21: "pcie_rx_mib_s",
    60: "nvlink_rx_mib_s",
    61: "nvlink_tx_mib_s",
}

POWER_FIELDS = {
    "power_instant_w": getattr(nvml, "NVML_FI_DEV_POWER_INSTANT", 186),
    "power_average_w": getattr(nvml, "NVML_FI_DEV_POWER_AVERAGE", 185),
    "energy_j": getattr(nvml, "NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION", 83),
}


def field_value(handle, field_id: int, scale: float) -> float | None:
    value = nvml.nvmlDeviceGetFieldValues(handle, [field_id])[0]
    if value.nvmlReturn != nvml.NVML_SUCCESS:
        return None
    accessors = {
        nvml.NVML_VALUE_TYPE_DOUBLE: "dVal",
        nvml.NVML_VALUE_TYPE_UNSIGNED_INT: "uiVal",
        nvml.NVML_VALUE_TYPE_UNSIGNED_LONG: "ulVal",
        nvml.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: "ullVal",
        nvml.NVML_VALUE_TYPE_SIGNED_LONG_LONG: "sllVal",
        nvml.NVML_VALUE_TYPE_SIGNED_INT: "siVal",
        nvml.NVML_VALUE_TYPE_UNSIGNED_SHORT: "usVal",
    }
    accessor = accessors.get(value.valueType)
    if accessor is None:
        return None
    return float(getattr(value.value, accessor)) / scale


def gpm_values(sample1, sample2) -> dict[str, float | None]:
    request = nvml.c_nvmlGpmMetricsGet_t()
    request.version = nvml.NVML_GPM_METRICS_GET_VERSION
    request.numMetrics = len(GPM_METRICS)
    request.sample1 = sample1
    request.sample2 = sample2
    for index, metric_id in enumerate(GPM_METRICS):
        request.metrics[index].metricId = metric_id

    nvml.nvmlGpmMetricsGet(request)
    output: dict[str, float | None] = {}
    for index, name in enumerate(GPM_METRICS.values()):
        metric = request.metrics[index]
        output[name] = (
            float(metric.value) if metric.nvmlReturn == nvml.NVML_SUCCESS else None
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    if args.interval < 0.1:
        parser.error("NVML GPM requires intervals of at least 0.1 seconds")

    nvml.nvmlInit()
    sample1 = sample2 = None
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(args.gpu)
        support = nvml.nvmlGpmQueryDeviceSupport(handle)
        if not support.isSupportedDevice:
            print("GPM is not supported on this device", file=sys.stderr)
            return 2

        sample1 = nvml.nvmlGpmSampleAlloc()
        sample2 = nvml.nvmlGpmSampleAlloc()
        nvml.nvmlGpmSampleGet(handle, sample1)

        columns = [
            "monotonic_s",
            "interval_s",
            *POWER_FIELDS,
            "gpu_busy_pct",
            "memory_busy_pct",
            *GPM_METRICS.values(),
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        start = previous_time = time.monotonic()
        while time.monotonic() - start < args.duration:
            deadline = previous_time + args.interval
            time.sleep(max(0.0, deadline - time.monotonic()))
            current_time = time.monotonic()
            nvml.nvmlGpmSampleGet(handle, sample2)
            row: dict[str, float | None] = {
                "monotonic_s": current_time,
                "interval_s": current_time - previous_time,
                "power_instant_w": field_value(
                    handle, POWER_FIELDS["power_instant_w"], 1000.0
                ),
                "power_average_w": field_value(
                    handle, POWER_FIELDS["power_average_w"], 1000.0
                ),
                "energy_j": field_value(handle, POWER_FIELDS["energy_j"], 1000.0),
            }
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            row["gpu_busy_pct"] = utilization.gpu
            row["memory_busy_pct"] = utilization.memory
            row.update(gpm_values(sample1, sample2))
            writer.writerow(row)
            sys.stdout.flush()
            sample1, sample2 = sample2, sample1
            previous_time = current_time
    finally:
        if sample1 is not None:
            nvml.nvmlGpmSampleFree(sample1)
        if sample2 is not None:
            nvml.nvmlGpmSampleFree(sample2)
        nvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
