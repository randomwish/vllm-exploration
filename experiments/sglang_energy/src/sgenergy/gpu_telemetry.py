from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

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


def _field_value(handle: Any, field_id: int, scale: float) -> float | None:
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
    return None if accessor is None else float(getattr(value.value, accessor)) / scale


def _try(call: Any, default: Any = None) -> Any:
    try:
        return call()
    except (nvml.NVMLError, AttributeError):
        return default


class DeviceSampler:
    def __init__(self, index: int, *, include_gpm: bool = True):
        self.index = index
        self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        uuid = nvml.nvmlDeviceGetUUID(self.handle)
        self.uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
        self.sample1 = self.sample2 = None
        support = _try(lambda: nvml.nvmlGpmQueryDeviceSupport(self.handle))
        self.gpm_supported = bool(support and support.isSupportedDevice)
        self.gpm_enabled = self.gpm_supported and include_gpm
        if self.gpm_enabled:
            self.sample1 = nvml.nvmlGpmSampleAlloc()
            self.sample2 = nvml.nvmlGpmSampleAlloc()
            nvml.nvmlGpmSampleGet(self.handle, self.sample1)

    def close(self) -> None:
        if self.sample1 is not None:
            nvml.nvmlGpmSampleFree(self.sample1)
        if self.sample2 is not None:
            nvml.nvmlGpmSampleFree(self.sample2)
        self.sample1 = self.sample2 = None

    def _gpm(self) -> dict[str, float | None]:
        output = {name: None for name in GPM_METRICS.values()}
        if not self.gpm_supported or self.sample1 is None or self.sample2 is None:
            return output
        nvml.nvmlGpmSampleGet(self.handle, self.sample2)
        request = nvml.c_nvmlGpmMetricsGet_t()
        request.version = nvml.NVML_GPM_METRICS_GET_VERSION
        request.numMetrics = len(GPM_METRICS)
        request.sample1 = self.sample1
        request.sample2 = self.sample2
        for index, metric_id in enumerate(GPM_METRICS):
            request.metrics[index].metricId = metric_id
        nvml.nvmlGpmMetricsGet(request)
        for index, name in enumerate(GPM_METRICS.values()):
            metric = request.metrics[index]
            if metric.nvmlReturn == nvml.NVML_SUCCESS:
                output[name] = float(metric.value)
        self.sample1, self.sample2 = self.sample2, self.sample1
        return output

    def sample(self, monotonic_s: float, unix_s: float, interval_s: float) -> dict[str, Any]:
        utilization = _try(lambda: nvml.nvmlDeviceGetUtilizationRates(self.handle))
        memory = _try(lambda: nvml.nvmlDeviceGetMemoryInfo(self.handle))
        clock_reasons = _try(
            lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle), 0
        )
        reason_bits = {
            "clock_event_sw_power_cap": getattr(
                nvml, "nvmlClocksEventReasonSwPowerCap", 0x0000000000000004
            ),
            "clock_event_hw_slowdown": getattr(
                nvml, "nvmlClocksEventReasonHwSlowdown", 0x0000000000000008
            ),
            "clock_event_sw_thermal": getattr(
                nvml, "nvmlClocksEventReasonSwThermalSlowdown", 0x0000000000000020
            ),
            "clock_event_hw_thermal": getattr(
                nvml, "nvmlClocksEventReasonHwThermalSlowdown", 0x0000000000000040
            ),
            "clock_event_power_brake": getattr(
                nvml, "nvmlClocksEventReasonHwPowerBrakeSlowdown", 0x0000000000000080
            ),
        }
        row: dict[str, Any] = {
            "unix_s": unix_s,
            "monotonic_s": monotonic_s,
            "interval_s": interval_s,
            "gpu_index": self.index,
            "gpu_uuid": self.uuid,
            "power_instant_w": _field_value(
                self.handle, POWER_FIELDS["power_instant_w"], 1000.0
            ),
            "power_average_w": _field_value(
                self.handle, POWER_FIELDS["power_average_w"], 1000.0
            ),
            "energy_j": _field_value(self.handle, POWER_FIELDS["energy_j"], 1000.0),
            "gpu_busy_pct": getattr(utilization, "gpu", None),
            "memory_busy_pct": getattr(utilization, "memory", None),
            "memory_total_bytes": getattr(memory, "total", None),
            "memory_used_bytes": getattr(memory, "used", None),
            "memory_free_bytes": getattr(memory, "free", None),
            "temperature_c": _try(
                lambda: nvml.nvmlDeviceGetTemperature(
                    self.handle, nvml.NVML_TEMPERATURE_GPU
                )
            ),
            "sm_clock_mhz": _try(
                lambda: nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_SM)
            ),
            "memory_clock_mhz": _try(
                lambda: nvml.nvmlDeviceGetClockInfo(self.handle, nvml.NVML_CLOCK_MEM)
            ),
            "pstate": _try(lambda: nvml.nvmlDeviceGetPowerState(self.handle)),
            "enforced_power_limit_w": (
                _try(lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(self.handle), 0) / 1000.0
            ),
            "clock_event_reasons": clock_reasons,
            "gpm_supported": self.gpm_supported,
            "gpm_enabled": self.gpm_enabled,
        }
        row.update(
            {name: bool(clock_reasons & bit) for name, bit in reason_bits.items()}
        )
        row.update(self._gpm())
        return row


def collect(
    output: Path,
    *,
    interval: float,
    duration: float | None = None,
    include_gpm: bool = True,
) -> None:
    if interval < 0.1:
        raise ValueError("NVML GPM requires an interval of at least 0.1 seconds")
    stop = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    nvml.nvmlInit()
    samplers: list[DeviceSampler] = []
    try:
        samplers = [
            DeviceSampler(index, include_gpm=include_gpm)
            for index in range(nvml.nvmlDeviceGetCount())
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        start = previous = time.monotonic()
        with output.open("w", encoding="utf-8", buffering=1) as stream:
            while not stop:
                deadline = previous + interval
                time.sleep(max(0.0, deadline - time.monotonic()))
                now = time.monotonic()
                unix = time.time()
                for sampler in samplers:
                    stream.write(
                        json.dumps(
                            sampler.sample(now, unix, now - previous), separators=(",", ":")
                        )
                        + "\n"
                    )
                previous = now
                if duration is not None and now - start >= duration:
                    break
    finally:
        for sampler in samplers:
            sampler.close()
        nvml.nvmlShutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect multi-GPU NVML/GPM telemetry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--no-gpm",
        action="store_true",
        help="collect critical NVML energy/power fields without blocking GPM queries",
    )
    args = parser.parse_args(argv)
    collect(
        args.output,
        interval=args.interval,
        duration=args.duration,
        include_gpm=not args.no_gpm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
