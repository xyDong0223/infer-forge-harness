"""In-pod parser conformance probe. Prints JSON to stdout.

A different failure mode from every numerical Task in this repo: the model computes
correctly, the text is right, and the API response is still wrong because
`reasoning_content` is empty or `tool_calls` never got parsed. Nothing in the
numbers can catch that, which is why parsers belong to their own family.

What it checks, per requested parser:

- that the name is actually registered in the *installed* runtime, not just
  documented — the vendor keeps its own OOT registries
  (`vllm_kunlun.reasoning.REASONING_PARSERS`, `.tool_parsers.TOOL_PARSERS`), so both
  they and upstream's lazy registries are reported;
- that the marker the sample uses appears in the model's own chat template, which is
  what stops a sample from being invented;
- that the parser splits a marked output, and — the control — that it reports nothing
  on the same text with the markers removed. A parser that always finds reasoning is
  as broken as one that never does, and only the control separates them.
"""

from __future__ import annotations

import argparse
import json


def registries() -> dict:
    import vllm_kunlun

    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    def names(manager) -> list[str]:
        eager = set(getattr(manager, "reasoning_parsers", None)
                    or getattr(manager, "tool_parsers", None) or {})
        return sorted(eager | set(getattr(manager, "lazy_parsers", {})))

    import vllm_kunlun.reasoning as kunlun_reasoning
    import vllm_kunlun.tool_parsers as kunlun_tools

    return {
        "reasoning": names(ReasoningParserManager),
        "tool": names(ToolParserManager),
        # Recorded even when empty: "no Kunlun-specific parser" is the fact that says
        # upstream's parsers are the ones that run here.
        "kunlun_oot_reasoning": sorted(kunlun_reasoning.REASONING_PARSERS),
        "kunlun_oot_tool": sorted(kunlun_tools.TOOL_PARSERS),
    }


def chat_template(model_path: str) -> str:
    import os

    for name in ("chat_template.jinja", "chat_template.json"):
        path = os.path.join(model_path, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            return text
    path = os.path.join(model_path, "tokenizer_config.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("chat_template") or ""


def blank_request():
    """A request object is required by both parser APIs but not read by them here.

    The class moved: it is `vllm.entrypoints.openai.chat_completion.protocol` in this
    build, not the `...openai.protocol` most examples import.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    return ChatCompletionRequest(model="probe", messages=[])


def check_reasoning(name: str, tokenizer, marked: str, moved: str, unmarked: str) -> dict:
    """Split a marked output, and prove the split follows the marker.

    The obvious control — the same text with the markers removed — turned out to be
    the wrong one. Qwen3's chat template opens `<think>` for the model, so vLLM's
    parser treats everything before a closing tag as reasoning: unmarked text comes
    back as reasoning with `content=None`, and that is correct behaviour, not a bug.
    The control that does discriminate moves the closing marker: if the parser reads
    it, the split moves with it.

    The unmarked case is still recorded, because it is an API trap worth knowing —
    a client reading only `content` sees nothing when the model never closes its tag.
    """
    from vllm.reasoning import ReasoningParserManager

    parser = ReasoningParserManager.get_reasoning_parser(name)(tokenizer)
    request = blank_request()
    reasoning, content = parser.extract_reasoning(marked, request)
    moved_reasoning, moved_content = parser.extract_reasoning(moved, request)
    bare_reasoning, bare_content = parser.extract_reasoning(unmarked, request)
    return {
        "case": "reasoning_parser_splits_a_marked_output",
        "parser": name,
        "implementation": f"{type(parser).__module__}.{type(parser).__name__}",
        "reasoning": reasoning,
        "content": content,
        "separated": bool(reasoning) and bool(content) and reasoning != content,
        "unmarked_output": {
            "reasoning": bare_reasoning,
            "content": bare_content,
            "note": "recorded, not gated: an unclosed thinking tag legitimately puts the "
                    "whole output in reasoning_content and leaves content empty",
        },
        "control": {
            "case": "the_closing_marker_moved",
            "reasoning": moved_reasoning,
            "content": moved_content,
            # If the split does not follow the marker, the parser is not reading it.
            "discriminates": moved_reasoning != reasoning,
        },
    }


def check_tools(name: str, tokenizer, marked: str, unmarked: str, expected: str) -> dict:
    from vllm.tool_parsers import ToolParserManager

    parser = ToolParserManager.get_tool_parser(name)(tokenizer)
    request = blank_request()
    extracted = parser.extract_tool_calls(marked, request)
    control = parser.extract_tool_calls(unmarked, request)
    calls = list(getattr(extracted, "tool_calls", None) or [])
    arguments_parse = None
    names = []
    for call in calls:
        function = getattr(call, "function", None)
        names.append(getattr(function, "name", None))
        raw = getattr(function, "arguments", None)
        try:
            json.loads(raw) if isinstance(raw, str) else None
            arguments_parse = True if arguments_parse is not False else False
        except Exception:
            arguments_parse = False
    return {
        "case": "tool_parser_extracts_a_call",
        "parser": name,
        "implementation": f"{type(parser).__module__}.{type(parser).__name__}",
        "tools_called": bool(getattr(extracted, "tools_called", False)),
        "call_names": names,
        "expected_name": expected,
        "arguments_are_json": arguments_parse,
        "name_matches": expected in names if expected else None,
        "control": {
            "case": "plain_prose_with_no_tool_call",
            "tools_called": bool(getattr(control, "tools_called", False)),
            "discriminates": not bool(getattr(control, "tools_called", False)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--tool-parser")
    parser.add_argument("--reasoning-sample", required=True,
                        help="a model output containing the thinking markers")
    parser.add_argument("--reasoning-moved", required=True,
                        help="the same words with the closing marker at a different offset")
    parser.add_argument("--reasoning-control", required=True, help="the same text, unmarked")
    parser.add_argument("--tool-sample", help="a model output containing a tool call")
    parser.add_argument("--tool-control", help="prose with no tool call")
    parser.add_argument("--expected-tool-name")
    parser.add_argument("--marker", action="append", default=[],
                        help="substring that must appear in the model's chat template")
    args = parser.parse_args()

    import vllm_kunlun  # noqa: F401 - activates the plugin so OOT registries are real

    result: dict = {"dimension": "parser", "model_path": args.model_path, "cases": []}
    try:
        result["registries"] = registries()
    except Exception as error:
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"registry lookup: {type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    requested = {"reasoning": args.reasoning_parser, "tool": args.tool_parser}
    missing = [f"{kind}:{name}" for kind, name in requested.items()
               if name and name not in result["registries"][kind]]
    if missing:
        result["state"] = "PARSER_ABSENT"
        result["error"] = f"not registered in this runtime: {missing}"
        print(json.dumps(result))
        return 1

    template = chat_template(args.model_path)
    result["template_markers"] = {
        marker: marker in template for marker in args.marker
    }
    result["template_confirms_markers"] = all(result["template_markers"].values()) \
        if args.marker else None
    if args.marker and not result["template_confirms_markers"]:
        # Without this the samples below would be assertions about a format the model
        # never emits.
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["error"] = ("the model's chat template does not contain the markers this "
                           "sample uses, so the sample is not evidence about this model")
        print(json.dumps(result))
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    try:
        if args.reasoning_parser:
            result["cases"].append(check_reasoning(
                args.reasoning_parser, tokenizer, args.reasoning_sample,
                args.reasoning_moved, args.reasoning_control))
        if args.tool_parser and args.tool_sample:
            result["cases"].append(check_tools(
                args.tool_parser, tokenizer, args.tool_sample, args.tool_control or "",
                args.expected_tool_name or ""))
    except Exception as error:
        result["state"] = "EVALUATION_ERROR"
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1

    conformant = []
    for case in result["cases"]:
        control_ok = case["control"]["discriminates"]
        if case["case"].startswith("reasoning"):
            conformant.append(case["separated"] and control_ok)
        else:
            conformant.append(bool(case["tools_called"]) and case["arguments_are_json"] is True
                              and (case["name_matches"] is not False) and control_ok)
    if not result["cases"]:
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["error"] = "no parser was requested, so nothing was exercised"
    elif not all(case["control"]["discriminates"] for case in result["cases"]):
        result["state"] = "EVALUATION_INCONCLUSIVE"
        result["error"] = ("a control also parsed, so the check cannot tell a working "
                           "parser from one that always fires")
    else:
        result["state"] = "CONFORMANT" if all(conformant) else "NONCONFORMANT"
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
