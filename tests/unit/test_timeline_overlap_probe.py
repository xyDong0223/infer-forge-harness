import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe.timeline_overlap_probe import (
    _complete_events,
    _intersect,
    _merge,
    _union_us,
    analyze,
    main,
)


def _event(name, ts, dur, tid=1, cat="cpu_op"):
    return {"name": name, "ph": "X", "ts": ts, "dur": dur,
            "pid": 0, "tid": tid, "cat": cat}


def _analyze(events, **kwargs):
    kwargs.setdefault("transfer_pattern", r"(?i)(mooncake|kv[_-]?transfer)")
    kwargs.setdefault("dispatch_pattern", r"(?i)(deep[_-]?ep|dispatch|combine)")
    kwargs.setdefault("min_overlap", 0.8)
    return analyze(events, **kwargs)


class IntervalMathTest(unittest.TestCase):
    def test_merge_collapses_overlapping_intervals(self):
        self.assertEqual(_merge([(0, 10), (5, 15), (20, 25)]), [(0, 15), (20, 25)])

    def test_union_and_intersection(self):
        a = _merge([(0, 10), (20, 30)])
        b = _merge([(5, 25)])
        self.assertEqual(_union_us(a), 20)
        self.assertEqual(_union_us(_intersect(a, b)), 10)


class VerdictTest(unittest.TestCase):
    def test_overlapped_transfer_and_dispatch(self):
        events = [
            _event("MooncakeTransfer", 0, 100, tid=2),
            _event("deep_ep::dispatch", 0, 100, tid=1),
            _event("deep_ep_dispatch_kernel", 0, 100, tid=11, cat="kernel"),
        ]
        result = _analyze(events)
        self.assertEqual(result["verdict"], "OVERLAPPED")
        self.assertAlmostEqual(result["lanes"]["overlap_ratio"], 1.0)

    def test_gil_blocked_transfer_gap_is_flagged(self):
        # Transfer thread is idle 50..150 while the launcher thread sits inside
        # one long CPU dispatch op and the GPU kernel runs: the case-2 GIL
        # signature, invisible in an end-to-end latency number.
        events = [
            _event("MooncakeTransfer", 0, 50, tid=2),
            _event("MooncakeTransfer", 150, 50, tid=2),
            _event("deep_ep::combine", 40, 120, tid=1),
            _event("deep_ep_combine_kernel", 60, 80, tid=11, cat="kernel"),
        ]
        result = _analyze(events)
        self.assertEqual(result["verdict"], "GIL_SUSPECT")
        self.assertAlmostEqual(result["gil"]["blocked_transfer_gap_us"], 100)
        window = result["gil"]["windows"][0]
        self.assertEqual(window["thread"], "0/2")
        self.assertAlmostEqual(window["gpu_dispatch_during_blocked_us"], 80)

    def test_serialized_without_gil_signature(self):
        events = [
            _event("MooncakeTransfer", 0, 50, tid=2),
            _event("deep_ep::dispatch", 200, 50, tid=1),
        ]
        result = _analyze(events)
        self.assertEqual(result["verdict"], "SERIALIZED")
        self.assertEqual(result["lanes"]["serialized_transfer_us"], 50)

    def test_missing_lane_is_insufficient_data(self):
        result = _analyze([_event("deep_ep::dispatch", 0, 50)])
        self.assertEqual(result["verdict"], "INSUFFICIENT_DATA")
        self.assertIn("verdict_reason", result)


class CliTest(unittest.TestCase):
    def _run_main(self, argv):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["timeline_overlap_probe", *argv])
        return code, json.loads(stdout.getvalue().strip().splitlines()[-1])

    def test_trace_object_and_bare_list_are_both_accepted(self):
        events = [_event("kv_transfer", 0, 10, tid=2),
                  _event("deep_ep::dispatch", 0, 10, tid=1)]
        self.assertEqual(len(_complete_events({"traceEvents": events})), 2)
        self.assertEqual(len(_complete_events(events)), 2)

    def test_end_to_end_report_on_temp_trace(self):
        trace = {"traceEvents": [
            _event("MooncakeTransfer", 0, 100, tid=2),
            _event("deep_ep::dispatch", 0, 100, tid=1),
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "trace.json")
            path.write_text(json.dumps(trace), encoding="utf-8")
            code, report = self._run_main([str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(report["probe"], "timeline_overlap_probe")
        self.assertEqual(report["verdict"], "OVERLAPPED")

    def test_unreadable_trace_exits_2_with_report(self):
        code, report = self._run_main(["/nonexistent/trace.json"])
        self.assertEqual(code, 2)
        self.assertEqual(report["verdict"], "INSUFFICIENT_DATA")


if __name__ == "__main__":
    unittest.main()
