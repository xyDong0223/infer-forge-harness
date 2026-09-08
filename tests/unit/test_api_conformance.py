"""MAT-009: parser profile resolution and what a conformance claim requires."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.check_api_conformance import ConformanceFailed, probe_argv, profile_for  # noqa: E402
from validators.conformance_validator import validate_conformance  # noqa: E402

CONTRACT = yaml.safe_load(
    (ROOT / "tasks" / "mat-009-api-conformance" / "task.yaml").read_text(encoding="utf-8")
)


class ProfileResolutionTest(unittest.TestCase):
    def test_a_model_resolves_through_its_family_prefix(self):
        profile = profile_for(CONTRACT, "Qwen3-30B-A3B")
        self.assertEqual(profile["reasoning_parser"], "qwen3")
        self.assertEqual(profile["tool_parser"], "hermes")

    def test_the_longest_declared_prefix_wins(self):
        contract = {"checks": {"models": {"Qwen3": {"reasoning_parser": "qwen3"},
                                          "Qwen3-Coder": {"reasoning_parser": "coder"}}}}
        self.assertEqual(profile_for(contract, "Qwen3-Coder-30B")["reasoning_parser"], "coder")
        self.assertEqual(profile_for(contract, "Qwen3-8B")["reasoning_parser"], "qwen3")

    def test_an_undeclared_model_stops_instead_of_guessing_a_marker(self):
        """Guessing the marker would make the whole check a statement about a format
        the model may never emit."""
        with self.assertRaises(ConformanceFailed):
            profile_for(CONTRACT, "SomeNewModel-7B")

    def test_every_marker_appears_in_the_samples_that_claim_it(self):
        for family, profile in CONTRACT["checks"]["models"].items():
            samples = " ".join(str(profile.get(key, "")) for key in
                               ("reasoning_sample", "reasoning_moved", "tool_sample"))
            for marker in profile.get("markers", []):
                with self.subTest(family=family, marker=marker):
                    self.assertIn(marker, samples)

    def test_the_moved_control_really_moves_the_marker(self):
        for family, profile in CONTRACT["checks"]["models"].items():
            with self.subTest(family=family):
                self.assertNotEqual(profile["reasoning_sample"], profile["reasoning_moved"])
                # Same words, different split point: otherwise the control changes two
                # things at once and says nothing about the marker. Markers become
                # spaces, since moving one necessarily moves the whitespace with it.
                def words(text: str) -> list[str]:
                    for marker in profile.get("markers", []) + ["</think>", "</tool_call>"]:
                        text = text.replace(marker, " ")
                    return sorted(text.split())

                self.assertEqual(words(profile["reasoning_sample"]),
                                 words(profile["reasoning_moved"]))

    def test_the_probe_argv_carries_every_marker_and_sample(self):
        argv = probe_argv(profile_for(CONTRACT, "Qwen3-8B"), "/mnt/cluster/Qwen3-8B")
        self.assertIn("--reasoning-moved", argv)
        self.assertEqual(argv.count("--marker"), 2)
        self.assertIn("--expected-tool-name", argv)


def conformant_report(**overrides) -> dict:
    report = {
        "state": "CONFORMANT",
        "registries": {"reasoning": ["qwen3"], "tool": ["hermes"],
                       "kunlun_oot_reasoning": [], "kunlun_oot_tool": []},
        "profile_source": "tasks/mat-009-api-conformance/task.yaml",
        "template_confirms_markers": True,
        "cases": [
            {"case": "reasoning_parser_splits_a_marked_output", "parser": "qwen3",
             "implementation": "vllm.parser.engine.adapters.Qwen3ParserReasoningAdapter",
             "separated": True,
             "control": {"case": "the_closing_marker_moved", "discriminates": True}},
            {"case": "tool_parser_extracts_a_call", "parser": "hermes",
             "implementation": "vllm.tool_parsers.hermes_tool_parser.Hermes2ProToolParser",
             "tools_called": True, "arguments_are_json": True, "name_matches": True,
             "control": {"case": "plain_prose_with_no_tool_call", "discriminates": True}},
        ],
    }
    report.update(overrides)
    return report


class ConformanceValidatorTest(unittest.TestCase):
    def test_a_well_formed_conformant_report_is_accepted(self):
        self.assertEqual(validate_conformance(conformant_report(), CONTRACT), [])

    def test_a_control_that_behaved_the_same_blocks_the_claim(self):
        """The real bug: removing the markers was the first control, and Qwen3's parser
        correctly reports unmarked text as reasoning, so the control passed."""
        report = conformant_report()
        report["cases"][0]["control"]["discriminates"] = False
        errors = validate_conformance(report, CONTRACT)
        self.assertTrue(any("cannot tell a working parser" in error for error in errors))

    def test_an_unconfirmed_marker_blocks_the_claim(self):
        report = conformant_report(template_confirms_markers=False)
        self.assertTrue(any("chat template did not confirm" in error
                            for error in validate_conformance(report, CONTRACT)))

    def test_tool_arguments_that_are_not_json_are_not_conformant(self):
        report = conformant_report()
        report["cases"][1]["arguments_are_json"] = False
        self.assertTrue(any("not JSON" in error
                            for error in validate_conformance(report, CONTRACT)))

    def test_a_missing_registry_listing_is_rejected(self):
        report = conformant_report()
        report.pop("registries")
        self.assertTrue(any("registries are required" in error
                            for error in validate_conformance(report, CONTRACT)))

    def test_an_absent_parser_is_a_valid_report_with_a_reason(self):
        report = {"state": "PARSER_ABSENT", "error": "not registered in this runtime",
                  "registries": {"reasoning": [], "tool": []},
                  "profile_source": "tasks/mat-009-api-conformance/task.yaml"}
        self.assertEqual(validate_conformance(report, CONTRACT), [])
        report.pop("error")
        self.assertTrue(any("must carry the reason" in error
                            for error in validate_conformance(report, CONTRACT)))


class ContractTest(unittest.TestCase):
    def test_the_task_is_its_own_family(self):
        self.assertEqual(CONTRACT["metadata"]["family"], "api_conformance")

    def test_the_contract_says_no_server_is_required(self):
        self.assertFalse(CONTRACT["context"]["requires_running_server"])

    def test_the_contract_names_the_control_that_discriminates(self):
        self.assertEqual(CONTRACT["checks"]["method"]["reasoning_control"],
                         "move_the_closing_marker")

    def test_a_registry_name_is_not_accepted_as_evidence(self):
        self.assertTrue(CONTRACT["checks"]["method"]["forbid_registry_name_as_evidence"])


if __name__ == "__main__":
    unittest.main()
