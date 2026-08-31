from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from .config import Workload, write_json_atomic


TRACE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    arrival_s: float
    output_tokens: int
    group_id: int | None
    input_ids: list[int] | None
    suffix_ids: list[int] | None
    input_sha256: str


def sha256_json(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_token_ids(tokenizer: object) -> list[int]:
    special = set(getattr(tokenizer, "all_special_ids", []))
    pool: list[int] = []
    for token_id in range(len(tokenizer)):  # type: ignore[arg-type]
        if token_id in special:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)  # type: ignore[attr-defined]
        if token is None or str(token).startswith("<|"):
            continue
        pool.append(token_id)
    if len(pool) < 1024:
        raise ValueError(f"tokenizer exposes only {len(pool)} safe content token IDs")
    return pool


def synthetic_token_ids(vocab_size: int) -> list[int]:
    if vocab_size < 2048:
        raise ValueError("synthetic vocab_size must be at least 2048")
    return list(range(256, vocab_size))


def _index_marker(index: int, pool: list[int]) -> list[int]:
    base = len(pool)
    if index < 0:
        raise ValueError("marker index cannot be negative")
    marker = [pool[index % base]]
    index //= base
    while index:
        marker.append(pool[index % base])
        index //= base
    marker.append(pool[-1])
    return marker


def _random_tokens(
    length: int, *, pool: list[int], seed: int, marker_index: int
) -> list[int]:
    marker = _index_marker(marker_index, pool)
    if len(marker) > length:
        raise ValueError("token sequence is too short for its uniqueness marker")
    rng = random.Random(seed)
    return marker + [pool[rng.randrange(len(pool))] for _ in range(length - len(marker))]


def poisson_arrivals(rate: float, duration: float, seed: int) -> list[float]:
    if rate <= 0 or duration <= 0:
        raise ValueError("rate and duration must be positive")
    rng = random.Random(seed)
    arrivals: list[float] = []
    current = 0.0
    while True:
        current += rng.expovariate(rate)
        if current > duration:
            break
        arrivals.append(current)
    if not arrivals:
        arrivals.append(min(duration, 1.0 / rate))
    return arrivals


def closed_loop_arrivals(count: int) -> list[float]:
    if count <= 0:
        raise ValueError("closed-loop request count must be positive")
    return [0.0] * count


def build_trace(
    output_dir: Path,
    *,
    workload: Workload,
    seed: int,
    token_pool: list[int],
    prefix_groups: int,
    prefix_seed: int | None = None,
    rate: float | None = None,
    duration: float | None = None,
    closed_loop_count: int | None = None,
) -> dict[str, object]:
    effective_prefix_seed = seed if prefix_seed is None else prefix_seed
    if (rate is None) == (closed_loop_count is None):
        raise ValueError("provide exactly one of rate or closed_loop_count")
    if rate is not None:
        if duration is None:
            raise ValueError("open-loop traces require duration")
        arrivals = poisson_arrivals(rate, duration, seed ^ 0xA771A1)
        traffic = {"mode": "poisson", "rate": rate, "duration": duration}
    else:
        arrivals = closed_loop_arrivals(int(closed_loop_count))
        traffic = {"mode": "closed-loop", "count": int(closed_loop_count)}

    if workload.prefix_tokens:
        usable = len(arrivals) - (len(arrivals) % prefix_groups)
        if usable < prefix_groups:
            raise ValueError("shared-prefix trace has fewer requests than prefix groups")
        arrivals = arrivals[:usable]

    output_dir.mkdir(parents=True, exist_ok=False)
    prefixes: dict[str, list[int]] = {}
    if workload.prefix_tokens:
        for group_id in range(prefix_groups):
            prefixes[str(group_id)] = _random_tokens(
                workload.prefix_tokens,
                pool=token_pool,
                seed=effective_prefix_seed ^ (group_id * 0x9E3779B1),
                marker_index=group_id,
            )

    requests_path = output_dir / "requests.jsonl.gz"
    with requests_path.open("wb") as raw_stream:
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for index, arrival in enumerate(arrivals):
                    group_id = index % prefix_groups if workload.prefix_tokens else None
                    suffix_length = workload.suffix_tokens
                    suffix = _random_tokens(
                        suffix_length,
                        pool=token_pool,
                        seed=seed ^ ((index + 1) * 0x85EBCA6B),
                        marker_index=index + prefix_groups,
                    )
                    if group_id is None:
                        full_input = suffix
                        input_ids = suffix
                        suffix_ids = None
                    else:
                        full_input = prefixes[str(group_id)] + suffix
                        input_ids = None
                        suffix_ids = suffix
                    request = TraceRequest(
                        request_id=f"{workload.name}-{seed}-{index:06d}",
                        arrival_s=round(arrival, 9),
                        output_tokens=workload.output_tokens,
                        group_id=group_id,
                        input_ids=input_ids,
                        suffix_ids=suffix_ids,
                        input_sha256=sha256_json(full_input),
                    )
                    stream.write(
                        json.dumps(asdict(request), separators=(",", ":")) + "\n"
                    )

    prefixes_path = output_dir / "prefixes.json.gz"
    with prefixes_path.open("wb") as raw_stream:
        with gzip.GzipFile(
            fileobj=raw_stream, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                json.dump(prefixes, stream, separators=(",", ":"), sort_keys=True)

    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "workload": asdict(workload),
        "seed": seed,
        "prefix_seed": effective_prefix_seed,
        "request_count": len(arrivals),
        "prefix_groups": prefix_groups if workload.prefix_tokens else 0,
        "traffic": traffic,
        "token_pool_sha256": sha256_json(token_pool),
        "files": {
            requests_path.name: file_sha256(requests_path),
            prefixes_path.name: file_sha256(prefixes_path),
        },
    }
    metadata["trace_id"] = sha256_json(metadata)
    write_json_atomic(output_dir / "metadata.json", metadata)
    return metadata


def load_prefixes(trace_dir: Path) -> dict[int, list[int]]:
    with gzip.open(trace_dir / "prefixes.json.gz", "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    return {int(key): value for key, value in raw.items()}


def iter_trace_requests(trace_dir: Path) -> Iterator[TraceRequest]:
    prefixes = load_prefixes(trace_dir)
    with gzip.open(trace_dir / "requests.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            request = TraceRequest(**raw)
            if request.input_ids is None:
                if request.group_id is None or request.suffix_ids is None:
                    raise ValueError(f"malformed trace request {request.request_id}")
                full_input = prefixes[request.group_id] + request.suffix_ids
            else:
                full_input = request.input_ids
            if sha256_json(full_input) != request.input_sha256:
                raise ValueError(f"input checksum mismatch for {request.request_id}")
            yield request


def materialize_input_ids(
    request: TraceRequest, prefixes: dict[int, list[int]]
) -> list[int]:
    if request.input_ids is not None:
        return request.input_ids
    if request.group_id is None or request.suffix_ids is None:
        raise ValueError(f"malformed trace request {request.request_id}")
    return prefixes[request.group_id] + request.suffix_ids


def verify_trace(trace_dir: Path) -> dict[str, object]:
    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in metadata["files"].items():
        observed = file_sha256(trace_dir / name)
        if observed != expected:
            raise ValueError(f"trace file checksum mismatch: {name}")
    count = sum(1 for _ in iter_trace_requests(trace_dir))
    if count != metadata["request_count"]:
        raise ValueError("trace request count does not match metadata")
    return metadata
