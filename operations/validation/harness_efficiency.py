"""Post-execution accounting from scheduler events and Task Memory.

Durations describe observed orchestration intervals, not CPU/device utilization.
This operation never changes scheduler state or functional acceptance criteria.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from core.storage import ArtifactStore, RunPaths, ensure_external
from engine.scheduler import EventStore
from engine.state import task_memory


TASK_ID = "harness-efficiency"
VERSION = 1


def _option(command: list, name: str, default):
    if name in command:
        index = command.index(name) + 1
        if index < len(command):
            return command[index]
    return default


def capture(state: Path, run_id: str, *, memory_path: Path | None = None,
            workflow_id: str | None = None) -> dict:
    """Copy minimal source records from a read-only, consistent DB snapshot."""
    store = EventStore(state, readonly=True)
    try:
        store.db.execute("BEGIN")
        run = store.run(run_id)
        if run is None:
            raise ValueError(f"unknown run: {run_id}")
        root = ensure_external(run.metadata["artifact_root"])
        marker = json.loads((root / "run.json").read_text())
        if marker != {"schema_version": 1, "run_id": run_id}:
            raise ValueError("run directory identity does not match scheduler")
        events = store.events(run_id)
        progress = run.metadata.get("graph_progress", {})
        replay = progress.get("resume_command") or []
        memory_path = ensure_external(memory_path or _option(
            replay, "--loop-state", root / "task_memory.json",
        ))
        workflow_id = workflow_id or Path(_option(
            replay, "--workflow", "model_adaptation.yaml",
        )).stem
        warnings = []
        memory = task_memory.load(memory_path, workflow_id, run.model_id)
        if not memory_path.is_file():
            warnings.append("Task Memory is unavailable; graph timing is unknown.")
        end = max([run.created_at, *[event.timestamp for event in events]])
        blocks = []
        for block in memory["completed_loop_blocks"]:
            if block.get("finished_at", end) > end:
                warnings.append("A graph block is newer than the scheduler snapshot and was excluded.")
                continue
            mode = block.get("routing", {}).get("mode")
            activity = "execution" if mode in (None, "recovery_execution") else (
                "reuse" if mode == "reuse_journal_fact" else "bookkeeping"
            )
            blocks.append({key: block.get(key) for key in (
                "block_id", "sub_target", "state", "started_at", "finished_at", "artifacts",
            )} | {"reused": activity == "reuse", "activity": activity, "routing_mode": mode})
        # A stale delivery receipt must not label later work as completed.
        completed = progress.get("reason_code") == "DELIVERY_RECORDED"
        if not events or events[-1].event_type != "graph_progress":
            completed = False
        scope = "completed" if completed else "snapshot"
        return {
            "schema_version": 1, "captured_at": time.time(),
            "source": {"state": str(Path(state).resolve()), "memory": str(memory_path)},
            "run": {
                "run_id": run_id, "model_id": run.model_id,
                "model_revision": run.model_revision, "plugin_revision": run.plugin_revision,
                "backend": run.backend, "artifact_root": str(root),
                "evidence_mode": run.metadata.get("evidence_mode", "real"),
                "environment_fingerprint": run.environment.get("environment_proof", {}).get("fingerprint"),
                "scheduler_status": run.status,
            },
            "window": {"started_at": run.created_at, "ended_at": end,
                       "scope": scope, "endpoint": "last_persisted_event"},
            "outcome": {"graph_status": progress.get("status"),
                        "reason_code": progress.get("reason_code"),
                        "delivery_state": run.metadata.get("graph_delivery", {}).get("state")
                        if completed else None},
            "events": [{
                "event_id": event.event_id, "event_type": event.event_type,
                "task_id": event.task_id, "timestamp": event.timestamp,
                "payload": {key: value for key, value in event.payload.items() if key in {
                    "stage", "attempt", "source_task_id", "next_action", "status", "reason_code",
                }},
            } for event in events],
            "graph_blocks": blocks,
            "open_graph_block": memory.get("current_loop_block", {}).get("block_id")
            if memory.get("current_loop_block") else None,
            "warnings": warnings,
        }
    finally:
        store.close()


def _duration(start, end) -> float:
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (start, end)) or end < start:
        raise ValueError("timing records require finite, ordered timestamps")
    return end - start


def _union(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    right = None
    for start, end in sorted(intervals):
        total += max(0.0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total


def evaluate(snapshot: dict) -> dict:
    """Project a frozen source snapshot; never infer missing execution time."""
    start, end = (snapshot["window"][key] for key in ("started_at", "ended_at"))
    elapsed = _duration(start, end)
    attempts, active, queued = [], {}, {}
    for event in snapshot["events"]:
        task, timestamp, payload = event["task_id"], event["timestamp"], event["payload"]
        _duration(start, timestamp)
        _duration(timestamp, end)
        kind = event["event_type"]
        if kind in {"task_created", "diagnosis_task_created", "diagnosis_task_reopened"}:
            queued[task] = (timestamp, event["event_id"])
        elif kind == "diagnosis_applied" and payload.get("next_action") == "RETRY":
            queued[payload["source_task_id"]] = (timestamp, event["event_id"])
        elif kind == "task_claimed":
            pending = queued.pop(task, None)
            attempt = {
                "task_id": task, "stage": payload["stage"], "attempt": payload["attempt"],
                "started_at": timestamp, "finished_at": None, "duration_seconds": None,
                "outcome": "unclosed", "event_ids": [event["event_id"]],
                "queue_wait_seconds": _duration(pending[0], timestamp) if pending else None,
                "queue_event_id": pending[1] if pending else None,
            }
            # Reclaiming a lease does not prove when the old worker stopped.
            active[task] = attempt
            attempts.append(attempt)
        elif kind in {"task_succeeded", "task_failed"}:
            attempt = active.pop(task, None)
            if attempt is None:
                raise ValueError(f"terminal event has no preceding claim: {event['event_id']}")
            attempt.update(finished_at=timestamp,
                           duration_seconds=_duration(attempt["started_at"], timestamp),
                           outcome="succeeded" if kind == "task_succeeded" else "failed")
            attempt["event_ids"].append(event["event_id"])

    intervals, graph, stages = [], [], {}

    def add(source, stage, duration, reference, interval=None):
        key = (source, stage)
        row = stages.setdefault(key, {
            "source": source, "stage": stage, "count": 0,
            "closed_interval_seconds": 0.0, "unclosed_count": 0, "references": [],
        })
        row["count"] += 1
        row["references"].append(reference)
        if duration is None:
            row["unclosed_count"] += 1
        else:
            row["closed_interval_seconds"] += duration
            intervals.append(interval)

    for attempt in attempts:
        add("worker", attempt["stage"], attempt["duration_seconds"], attempt["event_ids"][0],
            (attempt["started_at"], attempt["finished_at"]))
    for block in snapshot["graph_blocks"]:
        duration = _duration(block["started_at"], block["finished_at"])
        _duration(start, block["started_at"])
        _duration(block["finished_at"], end)
        graph.append({**block, "duration_seconds": duration})
        source = {"reuse": "graph_reuse", "execution": "graph", "bookkeeping": "graph_bookkeeping"}[block["activity"]]
        add(source, block["sub_target"],
            duration, block["block_id"], (block["started_at"], block["finished_at"]))
    claimed = Counter(attempt["task_id"] for attempt in attempts)
    executed = Counter(block["sub_target"] for block in graph if block["activity"] == "execution")
    closed_wall = _union(intervals)
    metrics = {
        "elapsed_seconds": elapsed, "recorded_wall_seconds": closed_wall,
        "unobserved_wall_seconds": max(0.0, elapsed - closed_wall),
        "worker_attempts": len(attempts),
        "worker_retries": sum(count - 1 for count in claimed.values()),
        "failed_worker_attempts": sum(a["outcome"] == "failed" for a in attempts),
        "unclosed_worker_attempts": sum(a["outcome"] == "unclosed" for a in attempts),
        "diagnosis_attempts": sum(a["stage"] == "diagnosis" for a in attempts),
        "graph_executions": sum(block["activity"] == "execution" for block in graph),
        "graph_bookkeeping_blocks": sum(block["activity"] == "bookkeeping" for block in graph),
        "graph_reuses": sum(block["reused"] for block in graph),
        "repeated_graph_executions": sum(count - 1 for count in executed.values()),
        "known_queue_wait_seconds": sum(a["queue_wait_seconds"] or 0 for a in attempts),
        "unknown_queue_wait_count": sum(a["queue_wait_seconds"] is None for a in attempts),
        "token_usage": None, "model_call_count": None, "cost": None,
        "device_utilization": None,
    }
    ranking = sorted(stages.values(), key=lambda row: (-row["closed_interval_seconds"],
                                                       row["source"], row["stage"]))
    findings = []
    if ranking:
        top = ranking[0]
        findings.append({"code": "LARGEST_RECORDED_STAGE", "references": top["references"],
                         "message": f"Largest summed closed intervals: {top['source']}/{top['stage']}. "
                         "This is a review candidate, not proof of inefficiency."})
    for task, count in claimed.items():
        if count > 1:
            findings.append({"code": "WORKER_RETRY", "message": f"{task} was claimed {count} times.",
                             "references": [a["event_ids"][0] for a in attempts if a["task_id"] == task]})
    for node, count in executed.items():
        if count > 1:
            findings.append({"code": "REPEATED_GRAPH_NODE",
                             "message": f"{node} executed {count} times; required regressions may explain repeats.",
                             "references": [b["block_id"] for b in graph
                                            if b["sub_target"] == node and b["activity"] == "execution"]})
    warnings = list(snapshot["warnings"])
    if metrics["unclosed_worker_attempts"] or snapshot["open_graph_block"]:
        warnings.append("Unclosed attempts have unknown durations and are excluded from recorded wall time.")
    return {
        "schema_version": 1, "evaluator_version": VERSION, "status": "RECORDED",
        "run": snapshot["run"], "window": snapshot["window"], "outcome": snapshot["outcome"],
        "metrics": metrics, "stages": ranking, "worker_attempts": attempts,
        "graph_blocks": graph, "findings": findings, "warnings": warnings,
        "limitations": [
            "Elapsed time ends at the last persisted event; report generation and later idle time are excluded.",
            "Claim-to-result and graph-block intervals include waiting and overhead, not just computation.",
            "Recorded wall time is the union of closed intervals; overlapping work is counted once.",
            "Unobserved time is not evidence of idle or wasted time. Stage sums may overlap.",
            "Repeated graph nodes may be required correctness regressions; repeats are not automatically waste.",
            "Scheduler handoffs and recovery receipts are bookkeeping blocks, not additional node executions.",
            "Historical recovery reruns without recovery_execution blocks have unknown count and duration.",
            "Token usage, model calls, cost and device utilization are not collected in this version.",
            "Efficiency accounting neither establishes hardware readiness nor changes the functional verdict.",
        ],
    }


def render(report: dict) -> str:
    metrics = report["metrics"]
    lines = ["# Harness 执行效率报告", "",
             f"Run: `{report['run']['run_id']}` · 证据模式: `{report['run']['evidence_mode']}`",
             f"范围: `{report['window']['scope']}` · 工作流状态: `{report['outcome']['graph_status']}`", "",
             f"- 截至最后记录的总耗时：{metrics['elapsed_seconds']:.3f} 秒",
             f"- 已记录执行区间的并集：{metrics['recorded_wall_seconds']:.3f} 秒",
             f"- 未归因时间：{metrics['unobserved_wall_seconds']:.3f} 秒（不能认定为空闲或浪费）",
             f"- Worker 领取 / 重复领取 / 失败 / 未闭合：{metrics['worker_attempts']} / "
             f"{metrics['worker_retries']} / {metrics['failed_worker_attempts']} / {metrics['unclosed_worker_attempts']}",
             f"- 诊断执行次数：{metrics['diagnosis_attempts']}",
             f"- Graph 执行 / 再次执行 / 复用：{metrics['graph_executions']} / "
             f"{metrics['repeated_graph_executions']} / {metrics['graph_reuses']}", "",
             "阶段按已闭合区间总时长排序；并行区间可能重叠，不能直接相加为总耗时。", "",
             "| 来源 | 阶段 | 次数 | 区间总秒数 | 未闭合 |", "| --- | --- | ---: | ---: | ---: |"]
    for stage in report["stages"]:
        name = str(stage['stage']).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {stage['source']} | {name} | {stage['count']} | "
                     f"{stage['closed_interval_seconds']:.3f} | {stage['unclosed_count']} |")
    lines += ["", "审视线索（不自动判定低效）：", ""]
    lines += [f"- {finding['message']}" for finding in report["findings"]] or ["- 暂无执行记录。"]
    lines += ["", "数据边界：", ""]
    lines += [f"- {message}" for message in report["warnings"] + report["limitations"]]
    return "\n".join(lines) + "\n"


def execute(state: Path, run_id: str, *, memory_path: Path | None = None,
            workflow_id: str | None = None) -> dict:
    from validators.harness_efficiency_validator import validate_efficiency

    snapshot = capture(state, run_id, memory_path=memory_path, workflow_id=workflow_id)
    attempt = RunPaths(snapshot["run"]["artifact_root"], run_id).allocate_attempt(TASK_ID)
    inputs, outputs = ArtifactStore(attempt.input), ArtifactStore(attempt.output)
    source = inputs.write_json("snapshot.json", snapshot)
    try:
        report = evaluate(snapshot)
        errors = validate_efficiency(report, snapshot)
    except (ValueError, KeyError, TypeError) as error:
        errors = [str(error)]
        report = None
    validation = outputs.write_json("validation.json", {"valid": not errors, "errors": errors})
    if errors:
        ArtifactStore(attempt.root).register(identity=attempt.identity, outcome="REWORK",
                                             required=["input/snapshot.json", "output/validation.json"])
        raise ValueError(f"efficiency evaluation rejected: {validation}: {'; '.join(errors)}")
    report["provenance"] = {"snapshot_path": str(source),
                            "snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "validation_path": str(validation)}
    path = outputs.write_json("efficiency_report.json", report)
    markdown = outputs.write_text("efficiency_report.md", render(report))
    manifest = ArtifactStore(attempt.root).register(
        identity=attempt.identity, outcome="RECORDED",
        required=["input/snapshot.json", "output/efficiency_report.json",
                  "output/efficiency_report.md", "output/validation.json"],
    )
    return {"status": "RECORDED", "run_id": run_id, "artifact_root": str(attempt.output),
            "report_path": str(path), "markdown_path": str(markdown), "manifest_path": str(manifest)}
