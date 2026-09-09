"""Unit tests for the MAT-029 shim handoff gate.

Fixtures are the real GLM-5.2 outcome: three torch shims written mid-loop
(kv_spans_from_batches, kunlun_convert_req_index_to_global_index,
kunlun_concat_and_cache_mla), all three missed by the operator dispatch path,
because nothing connected the place shims are born to it. The validator and the
signal matcher are what now refuse that outcome.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.probe.torch_shim_probe import scan_file  # noqa: E402
from tools.scan_torch_shims import match_signals_to_entries  # noqa: E402
from validators.shim_validator import validate_shim_handoff  # noqa: E402

GLM52_SHIM_MODULE = '''
import torch

def kv_spans_from_batches(start_seq_loc, seq_len_per_batch, device):
    """vLLM used to export this; 0.25.1 replaced it with a Triton kernel.

    The span arithmetic is kept here in torch.
    """
    return torch.zeros(2)

def kunlun_convert_req_index_to_global_index(req_id, block_table, token_indices):
    # shim for the triton _convert_req_index_to_global_index_kernel
    return token_indices // 64

def _unrelated_helper(x):
    return x + 1
'''


def _entry(name, location, status="REGISTERED", **extra):
    entry = {
        "name": name,
        "location": location,
        "replaced_kernel": "_build_prefill_chunk_metadata_kernel",
        "call_frequency": "every prefill metadata build",
        "semantics_basis": "tests/topk/test_topk_per_row.py",
        "status": status,
    }
    entry.update(extra)
    return entry


def _report(entries, signals, unmapped, state="HANDOFF_CLEAR", dispatch=None):
    report = {
        "plugin": "vllm_kunlun",
        "scanned_in": "dongxinyu03-glm52",
        "files_scanned": 149,
        "state": state,
        "signals": signals,
        "entries": entries,
        "unmapped_signals": unmapped,
    }
    if dispatch is not None:
        report["dispatch"] = dispatch
    return report


def _contract():
    return {"acceptance": {"require_dispatch_for_unwaived": True, "evidence_required": True}}


class SignalMatcherTest(unittest.TestCase):
    def setUp(self):
        self.signals = [
            {"kind": "KERNEL_REPLACEMENT_DOC", "file": "v1.attention.backends.mla.indexer",
             "symbol": "kv_spans_from_batches", "line": 18, "excerpt": "Triton kernel"},
            {"kind": "KUNLUN_PREFIX", "file": "v1.attention.backends.mla.flashmla_sparse",
             "symbol": "kunlun_convert_req_index_to_global_index", "line": 305, "excerpt": ""},
            {"kind": "KUNLUN_PREFIX", "file": "v1.attention.backends.mla.flashmla_sparse",
             "symbol": "kunlun_concat_and_cache_mla", "line": 357, "excerpt": ""},
        ]

    def test_a_declared_entry_explains_its_signal(self):
        entries = [_entry("kv_spans_from_batches", "mla/indexer.py:18")]
        unmapped, matched = match_signals_to_entries(self.signals[:1], entries)
        self.assertEqual(unmapped, [])
        self.assertEqual(len(matched), 1)

    def test_an_undeclared_signal_is_unmapped(self):
        # This is the GLM-5.2 failure: shims written, registry empty.
        unmapped, matched = match_signals_to_entries(self.signals, [])
        self.assertEqual(len(unmapped), 3)
        self.assertEqual(matched, [])

    def test_a_name_match_in_the_wrong_file_does_not_count(self):
        entries = [_entry("kv_spans_from_batches", "models/deepseek_v2.py:18")]
        unmapped, matched = match_signals_to_entries(self.signals[:1], entries)
        self.assertEqual(len(unmapped), 1)
        self.assertEqual(matched, [])

    def test_declared_entries_without_signals_are_kept(self):
        # The declaration is the contract; the probe is only a net.
        entries = [_entry("torch_paged_decode", "patches/torch_paged_decode.py:1")]
        unmapped, matched = match_signals_to_entries([], entries)
        self.assertEqual(unmapped, [])
        self.assertEqual(matched, [])


class ProbeTest(unittest.TestCase):
    def test_the_measured_glm52_module_produces_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "indexer.py"
            path.write_text(GLM52_SHIM_MODULE, encoding="utf-8")
            signals, parsed = scan_file(path, "v1.attention.backends.mla.indexer")
        self.assertTrue(parsed)
        self.assertEqual(
            sorted(signal["symbol"] for signal in signals),
            ["kunlun_convert_req_index_to_global_index", "kv_spans_from_batches"],
        )
        kinds = {signal["symbol"]: signal["kind"] for signal in signals}
        self.assertEqual(kinds["kunlun_convert_req_index_to_global_index"], "KUNLUN_PREFIX")
        self.assertEqual(kinds["kv_spans_from_batches"], "KERNEL_REPLACEMENT_DOC")

    def test_a_clean_module_produces_no_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clean.py"
            path.write_text("def helper(x):\n    return x + 1\n", encoding="utf-8")
            signals, parsed = scan_file(path, "utils.clean")
        self.assertTrue(parsed)
        self.assertEqual(signals, [])


class ValidatorTest(unittest.TestCase):
    def test_the_glm52_outcome_is_refused(self):
        # Three shims, zero entries, all signals unmapped, yet claimed clear.
        signals = [{"kind": "KUNLUN_PREFIX", "file": "mla.indexer", "symbol": "kunlun_x",
                    "line": 1, "excerpt": ""}]
        report = _report([], signals, signals, state="HANDOFF_CLEAR")
        errors = validate_shim_handoff(report, _contract())
        self.assertTrue(any("HANDOFF_CLEAR" in error for error in errors))

    def test_fully_hand_offed_report_passes(self):
        entry = _entry("kv_spans_from_batches", "mla/indexer.py:18", status="DISPATCHED",
                       request_id="glm52-op-001", dispatched_this_run=True)
        report = _report(
            [entry], [], [], state="HANDOFF_CLEAR",
            dispatch={"state": "DISPATCHED", "request_count": 1},
        )
        self.assertEqual(validate_shim_handoff(report, _contract()), [])

    def test_audited_waiver_passes_without_dispatch(self):
        entry = _entry("bind_kv_cache", "v1/worker/utils.py:1", status="WAIVED",
                       reason="plumbing, not arithmetic; no vendor kernel equivalent")
        report = _report([entry], [], [], state="HANDOFF_CLEAR")
        self.assertEqual(validate_shim_handoff(report, _contract()), [])

    def test_waiver_without_reason_is_refused(self):
        entry = _entry("bind_kv_cache", "v1/worker/utils.py:1", status="WAIVED")
        report = _report([entry], [], [], state="HANDOFF_CLEAR")
        errors = validate_shim_handoff(report, _contract())
        self.assertTrue(any("WAIVED without a reason" in error for error in errors))

    def test_registered_entry_blocks_a_clear_report(self):
        entry = _entry("kv_spans_from_batches", "mla/indexer.py:18")
        report = _report([entry], [], [], state="HANDOFF_CLEAR")
        errors = validate_shim_handoff(report, _contract())
        self.assertTrue(any("still REGISTERED" in error for error in errors))

    def test_handoff_found_is_the_honest_incomplete_state(self):
        entry = _entry("kv_spans_from_batches", "mla/indexer.py:18")
        signal = {"kind": "KERNEL_REPLACEMENT_DOC", "file": "mla.indexer",
                  "symbol": "kv_spans_from_batches", "line": 18, "excerpt": ""}
        report = _report([entry], [signal], [signal], state="HANDOFF_FOUND")
        self.assertEqual(validate_shim_handoff(report, _contract()), [])

    def test_dispatch_record_must_match_this_run(self):
        entry = _entry("kv_spans_from_batches", "mla/indexer.py:18", status="DISPATCHED",
                       request_id="glm52-op-001", dispatched_this_run=True)
        report = _report([entry], [], [], state="HANDOFF_CLEAR",
                         dispatch={"state": "DISPATCHED", "request_count": 2})
        errors = validate_shim_handoff(report, _contract())
        self.assertTrue(any("does not match" in error for error in errors))

    def test_an_empty_scan_proves_nothing(self):
        report = _report([], [], [])
        report["files_scanned"] = 0
        errors = validate_shim_handoff(report, _contract())
        self.assertTrue(any("empty scan" in error for error in errors))

    def test_an_entry_without_replacement_facts_is_refused(self):
        entry = {"name": "kv_spans_from_batches", "location": "mla/indexer.py:18",
                 "status": "REGISTERED"}
        report = _report([entry], [], [], state="HANDOFF_FOUND")
        errors = validate_shim_handoff(report, _contract())
        self.assertTrue(any("replaced_kernel" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
