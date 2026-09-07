"""In-pod model support probe. Prints JSON to stdout.

Answers one question with five possible answers, because "supported / not
supported" is the wrong shape: an architecture missing from the Kunlun registry
is usually *not* a gap — it means the upstream generic implementation is used and
any failure lies elsewhere, typically in an operator. Conflating the two sends
someone off to write a model file that is not needed.

Escalates only as far as it must: installed stack first, then vLLM main, then
open pull requests. Most adaptation work happens on models already merged into
main but not yet released, so stopping at the installed version would report a
gap that is really a version lag.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

MAIN_REGISTRY_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py"
)
PR_SEARCH_URL = (
    "https://api.github.com/search/issues?q=repo:vllm-project/vllm+is:pr+is:open+in:title+{arch}"
)
REGISTER_CALL = re.compile(r'register_model\(\s*"([A-Za-z0-9_]+)"')
TIMEOUT = 30


def fetch(url: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "kunlun-inference-agent"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.reason or ""
    except Exception as error:  # network, DNS, proxy
        return 0, f"{type(error).__name__}: {error}"


def installed_state(arch: str) -> dict:
    import vllm
    import vllm_kunlun  # noqa: F401 - activates the platform plugin and its registry
    from vllm import ModelRegistry

    import inspect

    from vllm_kunlun.models import register_model

    oot = sorted(set(REGISTER_CALL.findall(inspect.getsource(register_model))))
    supported = sorted(ModelRegistry.get_supported_archs())
    return {
        "vllm_version": vllm.__version__,
        "kunlun_oot_archs": oot,
        "in_kunlun_oot": arch in oot,
        "in_installed_vllm": arch in supported,
        "installed_arch_count": len(supported),
    }


def upstream_state(arch: str) -> dict:
    status, body = fetch(MAIN_REGISTRY_URL)
    if status != 200:
        # An unreachable upstream is unknown, not absent. Reporting ABSENT here
        # would send someone to write a model that may already exist.
        return {"main_lookup": "UNKNOWN", "detail": f"{MAIN_REGISTRY_URL} -> {status} {body[:200]}"}
    return {"main_lookup": "FOUND" if arch in body else "NOT_FOUND", "detail": f"{len(body)} bytes"}


def pull_request_state(arch: str) -> dict:
    status, body = fetch(PR_SEARCH_URL.format(arch=arch))
    if status != 200:
        return {"pr_lookup": "UNKNOWN", "detail": f"search -> {status} {body[:200]}"}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        return {"pr_lookup": "UNKNOWN", "detail": str(error)}
    items = [
        {"number": item.get("number"), "title": item.get("title"), "url": item.get("html_url")}
        for item in payload.get("items", [])
    ]
    return {"pr_lookup": "FOUND" if items else "NOT_FOUND", "pull_requests": items}


def classify(arch: str) -> dict:
    result: dict = {"architecture": arch}
    result.update(installed_state(arch))

    if result["in_kunlun_oot"]:
        result["verdict"] = "KUNLUN_OOT"
        result["meaning"] = "a Kunlun-specific implementation is registered and will be used"
        return result
    if result["in_installed_vllm"]:
        result["verdict"] = "UPSTREAM_GENERIC"
        result["meaning"] = (
            "the installed vLLM implementation is used; the absence of a Kunlun OOT model is "
            "not a gap, so a runtime failure points at an operator or backend, not at networking"
        )
        return result

    result.update(upstream_state(arch))
    if result.get("main_lookup") == "FOUND":
        result["verdict"] = "MAIN_ONLY"
        result["meaning"] = "merged upstream but not in the installed version: upgrade or cherry-pick"
        return result

    result.update(pull_request_state(arch))
    if result.get("pr_lookup") == "FOUND":
        result["verdict"] = "PR_PENDING"
        result["meaning"] = "an open upstream pull request adds it: wait for merge or cherry-pick"
        return result
    if "UNKNOWN" in (result.get("main_lookup"), result.get("pr_lookup")):
        result["verdict"] = "UNKNOWN_UPSTREAM"
        result["meaning"] = "upstream could not be reached; absence cannot be concluded"
        return result

    result["verdict"] = "ABSENT"
    result["meaning"] = "no implementation anywhere: a model file has to be written"
    return result


def main(archs: list[str]) -> int:
    out = {
        "state": "SCAN_READY",
        "proxy": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "",
        "results": [],
    }
    try:
        for arch in archs:
            out["results"].append(classify(arch))
    except Exception as error:  # an unusable runtime is not a scan result
        print(json.dumps({"state": "SCAN_FAILED", "reason": f"{type(error).__name__}: {error}"}))
        return 0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "usage: probe <Arch> [Arch ...]"}))
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
