"""Timeline overlap probe: did KV transfer overlap DeepEP dispatch/combine?

Answers the bounded question behind the Prefill + KV Transfer regression:
not "was end-to-end latency worse" but "did the Python-side KV transfer
actually run concurrently with DeepEP dispatch/combine, or was it serialized".

The serialized case has a specific trace signature. When a DeepEP CPU-side
dispatch/combine op waits on the GPU without releasing the Python GIL, the
launcher thread is busy inside one long cpu_op while the transfer thread sits
idle -- a gap that contains no transfer events even though GPU dispatch work
is in flight. A bare latency number cannot show this; the timeline can.

Input is a Chrome trace JSON export (torch profiler / kineto ``ph == "X"``
complete events), either as ``{"traceEvents": [...]}`` or a bare list. The
probe only observes; it never modifies the environment. Lane patterns are
configurable because kernel and transfer names differ across stacks.

Emits one JSON object on the last stdout line:

    {"probe": ..., "events": {...}, "lanes": {"overlap_ratio": ...},
     "gil": {"blocked_transfer_gap_us": ..., "windows": [...]},
     "verdict": "OVERLAPPED" | "GIL_SUSPECT" | "SERIALIZED" | "INSUFFICIENT_DATA"}

Exit code is 0 whenever the analysis ran; 2 only when the input cannot be
read or parsed. A probe reports -- the caller's Validator judges.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Iterable

# kineto/torch-profiler categories that live on the device timeline.
_GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

_DEFAULT_TRANSFER_PATTERN = r"(?i)(mooncake|kv[_-]?transfer|kv[_-]?(send|recv|fetch))"
_DEFAULT_DISPATCH_PATTERN = r"(?i)(deep[_-]?ep|dispatch|combine)"

Interval = tuple[float, float]


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _union_us(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in _merge(intervals))


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Merged pairwise intersection of two merged interval lists."""
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _side(event: dict) -> str:
    return "gpu" if str(event.get("cat", "")).lower() in _GPU_CATEGORIES else "cpu"


def _complete_events(payload) -> list[dict]:
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("trace payload is neither a Chrome trace object nor an event list")
    return [
        event for event in events
        if isinstance(event, dict)
        and event.get("ph", "X") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]


def _interval(event: dict) -> Interval:
    return (float(event["ts"]), float(event["ts"]) + float(event["dur"]))


def _thread_gaps(events: list[dict]) -> dict[str, list[Interval]]:
    """Idle gaps per thread: the space between merged busy intervals."""
    per_thread: dict[str, list[Interval]] = {}
    for event in events:
        key = f"{event.get('pid')}/{event.get('tid')}"
        per_thread.setdefault(key, []).append(_interval(event))
    gaps: dict[str, list[Interval]] = {}
    for thread, intervals in per_thread.items():
        merged = _merge(intervals)
        gaps[thread] = [
            (merged[i][1], merged[i + 1][0])
            for i in range(len(merged) - 1)
            if merged[i + 1][0] > merged[i][1]
        ]
    return gaps


def analyze(events: list[dict], *, transfer_pattern: str, dispatch_pattern: str,
            min_overlap: float, max_windows: int = 20) -> dict:
    transfer_re = re.compile(transfer_pattern)
    dispatch_re = re.compile(dispatch_pattern)

    transfer = [e for e in events if transfer_re.search(str(e.get("name", "")))]
    dispatch = [e for e in events if dispatch_re.search(str(e.get("name", "")))]

    result: dict = {
        "events": {
            "total": len(events),
            "transfer": len(transfer),
            "dispatch_combine": len(dispatch),
            "transfer_names": sorted({str(e.get("name", "")) for e in transfer})[:10],
            "dispatch_combine_names": sorted({str(e.get("name", "")) for e in dispatch})[:10],
        },
        "patterns": {"transfer": transfer_pattern, "dispatch_combine": dispatch_pattern},
        "thresholds": {"min_overlap": min_overlap},
    }

    if not transfer or not dispatch:
        result["verdict"] = "INSUFFICIENT_DATA"
        result["verdict_reason"] = (
            "trace has no transfer or no dispatch/combine events; "
            "check the lane patterns against the captured stack"
        )
        return result

    transfer_intervals = [_interval(e) for e in transfer]
    dispatch_intervals = [_interval(e) for e in dispatch]
    transfer_merged = _merge(transfer_intervals)
    dispatch_merged = _merge(dispatch_intervals)

    transfer_busy = _union_us(transfer_merged)
    dispatch_busy = _union_us(dispatch_merged)
    overlap = _union_us(_intersect(transfer_merged, dispatch_merged))
    overlap_ratio = overlap / transfer_busy if transfer_busy > 0 else 0.0

    window_start = min(s for s, _ in transfer_merged + dispatch_merged)
    window_end = max(e for _, e in transfer_merged + dispatch_merged)

    # GIL signature: a gap on a transfer thread that falls inside a CPU-side
    # dispatch/combine op. The transfer thread could have progressed but the
    # launcher thread held the interpreter while waiting on the device.
    dispatch_cpu = _merge([_interval(e) for e in dispatch if _side(e) == "cpu"])
    dispatch_gpu = _merge([_interval(e) for e in dispatch if _side(e) == "gpu"])
    windows: list[dict] = []
    blocked_total = 0.0
    for thread, gaps in _thread_gaps([e for e in transfer if _side(e) == "cpu"]).items():
        for gap_start, gap_end in gaps:
            blocked = _intersect([(gap_start, gap_end)], dispatch_cpu)
            if not blocked:
                continue
            blocked_us = _union_us(blocked)
            gpu_during = _union_us(_intersect(blocked, dispatch_gpu))
            blocked_total += blocked_us
            windows.append({
                "thread": thread,
                "start_us": gap_start,
                "end_us": gap_end,
                "blocked_us": blocked_us,
                "gpu_dispatch_during_blocked_us": gpu_during,
            })
    windows.sort(key=lambda w: w["blocked_us"], reverse=True)

    if overlap_ratio >= min_overlap:
        verdict = "OVERLAPPED"
        reason = f"transfer overlaps dispatch/combine for {overlap_ratio:.1%} of its busy time"
    elif blocked_total > 0:
        verdict = "GIL_SUSPECT"
        reason = (
            f"transfer overlap is {overlap_ratio:.1%} and transfer threads show "
            f"{blocked_total:.0f}us of idle gaps inside CPU dispatch/combine ops"
        )
    else:
        verdict = "SERIALIZED"
        reason = (
            f"transfer overlap is {overlap_ratio:.1%} with no GIL signature; "
            "the transfer path is serialized for another reason"
        )

    result.update({
        "window": {"start_us": window_start, "end_us": window_end,
                   "span_us": window_end - window_start},
        "lanes": {
            "transfer_busy_us": transfer_busy,
            "dispatch_combine_busy_us": dispatch_busy,
            "overlap_us": overlap,
            "overlap_ratio": overlap_ratio,
            "serialized_transfer_us": transfer_busy - overlap,
        },
        "gil": {
            "blocked_transfer_gap_us": blocked_total,
            "windows": windows[:max_windows],
            "windows_truncated": len(windows) > max_windows,
        },
        "verdict": verdict,
        "verdict_reason": reason,
    })
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Measure KV-transfer / dispatch-combine timeline overlap from a Chrome trace"
    )
    parser.add_argument("trace", help="Chrome trace JSON (torch profiler export) or event list")
    parser.add_argument("--transfer-pattern", default=_DEFAULT_TRANSFER_PATTERN,
                        help="regex for transfer-lane event names (Mooncake / KV transfer)")
    parser.add_argument("--dispatch-pattern", default=_DEFAULT_DISPATCH_PATTERN,
                        help="regex for dispatch/combine-lane event names (DeepEP)")
    parser.add_argument("--min-overlap", type=float, default=0.8,
                        help="overlap ratio at or above which the verdict is OVERLAPPED")
    parser.add_argument("--max-windows", type=int, default=20,
                        help="maximum GIL-suspect windows to include in the report")
    args = parser.parse_args(argv[1:])

    try:
        with open(args.trace, "rb") as handle:
            payload = json.load(handle)
        events = _complete_events(payload)
    except (OSError, ValueError) as error:
        print(json.dumps({"probe": "timeline_overlap_probe", "trace": args.trace,
                          "error": str(error), "verdict": "INSUFFICIENT_DATA"}))
        return 2

    report = analyze(events, transfer_pattern=args.transfer_pattern,
                     dispatch_pattern=args.dispatch_pattern,
                     min_overlap=args.min_overlap, max_windows=args.max_windows)
    print(json.dumps({"probe": "timeline_overlap_probe", "trace": args.trace, **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
