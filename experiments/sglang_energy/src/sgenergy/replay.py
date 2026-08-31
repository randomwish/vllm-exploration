from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .traces import (
    TraceRequest,
    iter_trace_requests,
    load_prefixes,
    materialize_input_ids,
    verify_trace,
)


@dataclass
class RequestResult:
    request_id: str
    input_sha256: str
    input_tokens: int
    requested_output_tokens: int
    group_id: int | None
    scheduled_s: float
    dispatched_s: float | None = None
    first_token_s: float | None = None
    completed_s: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int | None = None
    output_sha256: str | None = None
    finish_reason: Any = None
    http_status: int | None = None
    success: bool = False
    error: str | None = None


def _completion_tokens(meta: dict[str, Any]) -> int | None:
    for key in ("completion_tokens", "output_tokens"):
        value = meta.get(key)
        if isinstance(value, int):
            return value
    return None


async def _run_request(
    *,
    session: aiohttp.ClientSession,
    endpoint: str,
    request: TraceRequest,
    input_ids: list[int],
    run_start: float,
    semaphore: asyncio.Semaphore | None,
) -> RequestResult:
    result = RequestResult(
        request_id=request.request_id,
        input_sha256=request.input_sha256,
        input_tokens=len(input_ids),
        requested_output_tokens=request.output_tokens,
        group_id=request.group_id,
        scheduled_s=request.arrival_s,
    )
    await asyncio.sleep(max(0.0, run_start + request.arrival_s - time.monotonic()))
    acquired = False
    try:
        if semaphore is not None:
            await semaphore.acquire()
            acquired = True
        dispatched = time.monotonic()
        result.dispatched_s = dispatched - run_start
        payload = {
            "rid": request.request_id,
            "input_ids": input_ids,
            "stream": True,
            "log_metrics": True,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": request.output_tokens,
                "min_new_tokens": request.output_tokens,
                "ignore_eos": True,
                "skip_special_tokens": True,
            },
        }
        text_seen = ""
        final_meta: dict[str, Any] = {}
        async with session.post(endpoint, json=payload) as response:
            result.http_status = response.status
            if response.status != 200:
                result.error = (await response.text())[:2000]
                return result
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                content = line[5:].strip()
                if content == "[DONE]":
                    break
                event = json.loads(content)
                text_value = event.get("text", "")
                if isinstance(text_value, list):
                    text_value = text_value[0]
                text_seen = str(text_value)
                meta = event.get("meta_info") or {}
                if isinstance(meta, dict):
                    final_meta = meta
                tokens_so_far = _completion_tokens(final_meta)
                if result.first_token_s is None and (text_seen or (tokens_so_far or 0) > 0):
                    result.first_token_s = time.monotonic() - run_start

        completed = time.monotonic()
        result.completed_s = completed - run_start
        result.e2e_ms = (completed - dispatched) * 1000.0
        if result.first_token_s is not None:
            result.ttft_ms = (result.first_token_s - result.dispatched_s) * 1000.0
        result.output_tokens = _completion_tokens(final_meta)
        result.finish_reason = final_meta.get("finish_reason")
        result.output_sha256 = hashlib.sha256(text_seen.encode()).hexdigest()
        if result.output_tokens and result.output_tokens > 1 and result.ttft_ms is not None:
            result.tpot_ms = max(0.0, result.e2e_ms - result.ttft_ms) / (
                result.output_tokens - 1
            )
        result.success = (
            result.output_tokens == request.output_tokens
            and result.first_token_s is not None
        )
        if not result.success:
            result.error = (
                f"expected {request.output_tokens} output tokens, "
                f"observed {result.output_tokens}"
            )
        return result
    except asyncio.CancelledError:
        result.error = "campaign watchdog cancelled request"
        raise
    except Exception as exc:  # request failure is data, not a harness crash
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if acquired and semaphore is not None:
            semaphore.release()


async def replay_trace(
    trace_dir: Path,
    *,
    base_url: str,
    output_path: Path,
    watchdog_seconds: float,
    max_concurrency: int | None = None,
) -> dict[str, Any]:
    metadata = verify_trace(trace_dir)
    requests = list(iter_trace_requests(trace_dir))
    prefixes = load_prefixes(trace_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_start_wall = time.time()
    run_start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=watchdog_seconds)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    tasks: list[asyncio.Task[RequestResult]] = []
    results: list[RequestResult] = []
    endpoint = base_url.rstrip("/") + "/generate"
    watchdog_hit = False

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for request in requests:
            input_ids = materialize_input_ids(request, prefixes)
            tasks.append(
                asyncio.create_task(
                    _run_request(
                        session=session,
                        endpoint=endpoint,
                        request=request,
                        input_ids=input_ids,
                        run_start=run_start,
                        semaphore=semaphore,
                    )
                )
            )
        try:
            async with asyncio.timeout(watchdog_seconds):
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
        except TimeoutError:
            watchdog_hit = True
            for task in tasks:
                task.cancel()
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

    for request, item in zip(requests, gathered, strict=True):
        if isinstance(item, RequestResult):
            results.append(item)
        else:
            results.append(
                RequestResult(
                    request_id=request.request_id,
                    input_sha256=request.input_sha256,
                    input_tokens=0,
                    requested_output_tokens=request.output_tokens,
                    group_id=request.group_id,
                    scheduled_s=request.arrival_s,
                    error=f"{type(item).__name__}: {item}",
                )
            )

    with output_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(asdict(result), separators=(",", ":")) + "\n")

    completed_times = [r.completed_s for r in results if r.completed_s is not None]
    summary = {
        "trace_id": metadata["trace_id"],
        "run_start_unix_s": run_start_wall,
        "elapsed_s": time.monotonic() - run_start,
        "cohort_end_s": max(completed_times) if completed_times else None,
        "request_count": len(results),
        "success_count": sum(r.success for r in results),
        "watchdog_hit": watchdog_hit,
        "max_concurrency": max_concurrency,
    }
    return summary
