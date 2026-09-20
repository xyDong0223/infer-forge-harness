"""Independent accounting checks for the post-run efficiency report."""

from __future__ import annotations

from collections import Counter
import math


def validate_efficiency(report: dict, snapshot: dict) -> list[str]:
    errors = []
    for key in ("run", "window", "outcome"):
        if report.get(key) != snapshot.get(key):
            errors.append(f"{key} does not match the source snapshot")
    if report.get("schema_version") != 1 or report.get("status") != "RECORDED":
        errors.append("unsupported report schema or status")
    metrics = report["metrics"]
    for key, value in metrics.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            errors.append(f"invalid nonnegative metric: {key}")
    for key in ("token_usage", "model_call_count", "cost", "device_utilization"):
        if metrics.get(key) is not None:
            errors.append(f"{key} must remain unknown without telemetry")
    events = {e["event_id"]: e for e in snapshot["events"]}
    claims = [e for e in snapshot["events"] if e["event_type"] == "task_claimed"]
    attempts = report["worker_attempts"]
    # Derive ownership from source order, including reclaimed leases. Merely
    # checking that an alleged terminal/queue event exists would allow a report
    # to omit observed completions or waits and lower its measured time.
    expected_attempts, current, pending = [], {}, {}
    for event in snapshot["events"]:
        kind, task = event["event_type"], event["task_id"]
        if kind in {"task_created", "diagnosis_task_created", "diagnosis_task_reopened"}:
            pending[task] = event
        elif kind == "diagnosis_applied" and event["payload"].get("next_action") == "RETRY":
            pending[event["payload"]["source_task_id"]] = event
        elif kind == "task_claimed":
            expected = {"claim": event, "queue": pending.pop(task, None), "terminal": None}
            expected_attempts.append(expected)
            current[task] = expected
        elif kind in {"task_succeeded", "task_failed"}:
            expected = current.pop(task, None)
            if expected is None:
                errors.append("source terminal event has no current claim")
            else:
                expected["terminal"] = event
    for attempt, expected in zip(attempts, expected_attempts):
        terminal, queue = expected["terminal"], expected["queue"]
        expected_ids = [expected["claim"]["event_id"]] + ([terminal["event_id"]] if terminal else [])
        if (attempt["event_ids"] != expected_ids
                or attempt["finished_at"] != (terminal["timestamp"] if terminal else None)
                or attempt["queue_event_id"] != (queue["event_id"] if queue else None)
                or attempt["queue_wait_seconds"] != (
                    expected["claim"]["timestamp"] - queue["timestamp"] if queue else None)):
            errors.append("attempt terminal or queue binding does not match source event order")
    if [a["event_ids"][0] for a in attempts] != [e["event_id"] for e in claims]:
        errors.append("worker attempt inventory does not match claim events")
    source_blocks = snapshot["graph_blocks"]
    if [{k: v for k, v in b.items() if k != "duration_seconds"} for b in report["graph_blocks"]] != source_blocks:
        errors.append("graph block inventory does not match Task Memory")
    intervals = []
    for attempt in attempts:
        claim = events.get(attempt["event_ids"][0], {})
        if (attempt["started_at"] != claim.get("timestamp")
                or attempt["task_id"] != claim.get("task_id")
                or attempt["stage"] != claim.get("payload", {}).get("stage")
                or attempt["attempt"] != claim.get("payload", {}).get("attempt")):
            errors.append("worker identity or start time does not match claim")
        if attempt["finished_at"] is None:
            if attempt["duration_seconds"] is not None or attempt["outcome"] != "unclosed":
                errors.append("unclosed attempt must have unknown duration")
        else:
            terminal = events.get(attempt["event_ids"][-1], {})
            if (terminal.get("task_id") != attempt["task_id"]
                    or terminal.get("event_type") != "task_" + attempt["outcome"]
                    or terminal.get("timestamp") != attempt["finished_at"]):
                errors.append("worker completion does not match terminal event")
        if attempt["queue_event_id"] is not None:
            pending = events.get(attempt["queue_event_id"], {})
            if pending.get("timestamp") is None or not math.isclose(
                attempt["queue_wait_seconds"], attempt["started_at"] - pending["timestamp"],
                abs_tol=1e-6,
            ):
                errors.append("queue wait does not match source events")
    for row in [*attempts, *report["graph_blocks"]]:
        if row["finished_at"] is None:
            continue
        left, right = row["started_at"], row["finished_at"]
        if not snapshot["window"]["started_at"] <= left <= right <= snapshot["window"]["ended_at"]:
            errors.append("interval is outside the observation window")
        if not math.isclose(row["duration_seconds"], right - left, abs_tol=1e-6):
            errors.append("interval duration does not match timestamps")
        intervals.extend([(left, 1), (right, -1)])
    # Sweep interval boundaries independently of the producer's merging algorithm.
    total, depth, previous = 0.0, 0, None
    for timestamp, change in sorted(intervals):
        if depth > 0 and previous is not None:
            total += timestamp - previous
        depth += change
        previous = timestamp
    elapsed = snapshot["window"]["ended_at"] - snapshot["window"]["started_at"]
    counted = Counter(e["task_id"] for e in claims)
    graph_counts = Counter(b["sub_target"] for b in source_blocks if b["activity"] == "execution")
    expected = {
        "elapsed_seconds": elapsed, "recorded_wall_seconds": total,
        "unobserved_wall_seconds": elapsed - total,
        "worker_attempts": len(claims), "worker_retries": sum(n - 1 for n in counted.values()),
        "failed_worker_attempts": sum(e["event_type"] == "task_failed" for e in events.values()),
        "unclosed_worker_attempts": sum(a["terminal"] is None for a in expected_attempts),
        "diagnosis_attempts": sum(e["payload"]["stage"] == "diagnosis" for e in claims),
        "graph_executions": sum(b["activity"] == "execution" for b in source_blocks),
        "graph_bookkeeping_blocks": sum(b["activity"] == "bookkeeping" for b in source_blocks),
        "graph_reuses": sum(b["reused"] for b in source_blocks),
        "repeated_graph_executions": sum(n - 1 for n in graph_counts.values()),
        "known_queue_wait_seconds": sum(a["claim"]["timestamp"] - a["queue"]["timestamp"]
                                        for a in expected_attempts if a["queue"] is not None),
        "unknown_queue_wait_count": sum(a["queue"] is None for a in expected_attempts),
    }
    for key, value in expected.items():
        actual = metrics.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(actual, value, abs_tol=1e-6):
            errors.append(f"{key} does not match source accounting")
    references = set(events) | {b["block_id"] for b in source_blocks}
    for row in [*report["stages"], *report["findings"]]:
        if not row.get("references") or not set(row["references"]) <= references:
            errors.append("stage or finding lacks source references")
    expected_stages = {}
    for row, source, stage, reference in [
        *[(a, "worker", a["stage"], a["event_ids"][0]) for a in attempts],
        *[(b, {"execution": "graph", "reuse": "graph_reuse", "bookkeeping": "graph_bookkeeping"}[b["activity"]],
           b["sub_target"], b["block_id"]) for b in report["graph_blocks"]],
    ]:
        group = expected_stages.setdefault((source, stage), {
            "count": 0, "closed_interval_seconds": 0.0, "unclosed_count": 0, "references": [],
        })
        group["count"] += 1
        group["references"].append(reference)
        if row["finished_at"] is None:
            group["unclosed_count"] += 1
        else:
            group["closed_interval_seconds"] += row["finished_at"] - row["started_at"]
    actual_keys = [(r["source"], r["stage"]) for r in report["stages"]]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != set(expected_stages):
        errors.append("stage inventory does not match recorded work")
    for row in report["stages"]:
        expected_stage = expected_stages.get((row["source"], row["stage"]), {})
        for key in ("count", "unclosed_count", "references"):
            if row.get(key) != expected_stage.get(key):
                errors.append(f"stage {key} does not match source accounting")
        duration = row.get("closed_interval_seconds")
        if not isinstance(duration, (int, float)) or not math.isclose(
            duration, expected_stage.get("closed_interval_seconds", -1), abs_tol=1e-6,
        ):
            errors.append("stage duration does not match source accounting")
    return errors
