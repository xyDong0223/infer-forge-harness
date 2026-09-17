"""Explicitly simulated external boundaries for the unmodified model graph.

No operation, validator, task outcome, journal, or scheduler is replaced here.
The adapter answers remote command/probe requests, including the remote probes'
JSON wire format. Those observations are simulation, never hardware evidence.
Unknown commands fail closed. All mutable fixture state lives under the caller's
external root, including the pod identity shared by independent CLI processes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys

from adapters.kunlun_p800.adapter import ClusterConfig, KunlunP800Adapter
from core.paths import REPO_ROOT
from core.storage import ensure_external

SETTINGS_ENV = "INFER_FORGE_E2E_SETTINGS"
PLUGIN_REVISION = "e" * 40
SUBJECT = "Qwen3-Simulation"
POD = "simulation-kdp001-pod"
_installed = False


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fingerprint(model: Path) -> dict:
    from tools.probe.model_fingerprint_probe import main

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert main(str(model)) == 0
    return json.loads(output.getvalue())


def prepare_environment(root: Path) -> dict:
    """Create synthetic inputs outside source and return subprocess configuration.

    ``env`` is an override mapping, to merge with ``os.environ``. ``graph_args``
    supplies subject/target/context but not execution, scheduler, or run options.
    """
    import yaml

    root = ensure_external(root.resolve(), REPO_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    model = root / "model"
    model.mkdir()
    config = {
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "torch_dtype": "float32", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4,
        "max_position_embeddings": 128, "vocab_size": 8, "sliding_window": 4,
    }
    _json(model / "config.json", config)
    _json(model / "tokenizer_config.json", {
        "chat_template": "<think>{{ messages }}</think><tool_call></tool_call>",
    })
    # A valid, tiny safetensors container, not a mislabeled text weight file.
    inputs = [-1.0, 0.0, 0.5, 2.0]
    header = json.dumps({
        "__metadata__": {"evidence_mode": "simulation"},
        "scale": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
    }).encode()
    header += b" " * (-len(header) % 8)
    (model / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<4f", *inputs),
    )
    fingerprint = _fingerprint(model)
    observed = {
        "evidence_mode": "simulation", "call_site": "synthetic_workload.scale",
        "input": inputs, "output": [2.0 * value for value in inputs],
        "dtype": "float32", "shape": [len(inputs)], "layout": "contiguous",
        "device": "simulated-xpu", "exception": "MissingOperator: scale",
    }
    _json(root / "scale_observation.json", observed)
    report = {
        "evidence_mode": "simulation", "model_id": SUBJECT,
        "model_revision": fingerprint["revision"], "plugin_revision": PLUGIN_REVISION,
        "backend": "kunlun",
        "entries": [{
            "name": "scale", "location": "synthetic_workload.scale",
            "source": "simulated measured workload", "class": "MISSING_OPERATOR",
            "inputs": [{"name": "x", "shape": [len(inputs)], "dtype": "float32",
                        "layout": "contiguous"}],
            "outputs": [{"name": "y", "shape": [len(inputs)], "dtype": "float32",
                         "layout": "contiguous"}],
            "semantics": {"formula": "y = 2 * x", "fake_op": "double",
                          "basis": "scale_observation.json: elementwise measured pairs"},
            "evidence": {"observation": str(root / "scale_observation.json"),
                         "failure": observed["exception"], "evidence_mode": "simulation"},
        }],
    }
    report_path = root / "operators.json"
    _json(report_path, report)
    from operations.deployment.plan_deployment import plan, render_instance
    request = {"model": {"id": SUBJECT, "source": str(model), "pvc": "simulation-models",
                         "revision": fingerprint["revision"], "total_weight_bytes": 16},
               "identity": config,
               "target": {"hardware": "kunlun/p800", "vllm_kunlun_commit": PLUGIN_REVISION}}
    report = plan(request, {"classification": "OPERATOR_GAP"}, {"hbm_mib": 98304}, None)
    contract = yaml.safe_load(render_instance(report, request, root / "deployment", user_id="simulation"))
    # Legacy replay exercises all phases; the production Graph uses its own
    # environment generator and MAT-005 output instead of this boundary fixture.
    contract["metadata"].update(name="kdp-001-simulation", task_type="deployment_proof", evidence_mode="simulation")
    ctx = contract["context"]
    ctx["model"].update(name=SUBJECT, path=str(model), pvc="simulation-models")
    ctx["target"].update(device_count=1, namespace="pd-test",
                         volcano_queue="simulation", dedicated_pool="simulation")
    ctx["repository"] = {"vllm_kunlun_ref": PLUGIN_REVISION}
    ctx["software"] = {"image": "simulation.invalid/runtime:never-pulled"}
    ctx["server"].update(served_model_name=SUBJECT, max_model_len=128, dtype="float32")
    # This target-model validation entry must never override the profile's
    # MiniMax baseline when the environment phase normalizes the contract.
    ctx["validation"] = {"base_model": {
        "name": SUBJECT, "path": str(model), "served_model_name": SUBJECT,
        "dtype": "float32", "tensor_parallel_size": 1, "max_model_len": 128,
        "max_num_batched_tokens": 128, "max_num_seqs": 1, "block_size": 16,
        "gpu_memory_utilization": 0.5, "required": True,
    }}
    execution = contract["execution"]
    execution.update(namespace="pd-test", resource_name="simulation-kdp001",
                     health_interval_seconds=0, health_successes_required=1)
    execution["commands"]["serve"] = [
        "python -m vllm.entrypoints.openai.api_server "
        f"--model {shlex.quote(str(model))} --served-model-name {SUBJECT} "
        "--port 8356 --tensor-parallel-size 1 --dtype float32 --max-model-len 128",
    ]
    contract["checks"]["chat"]["payload"]["model"] = SUBJECT
    contract["artifacts"]["directory"] = str(root / "deployment")
    contract_path = root / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    target = {
        "model": SUBJECT, "hardware": "kunlun/p800",
        "runtime": {"engine": "vllm", "backend": "kunlun", "plugin": "vllm-kunlun"},
    }
    target_path = root / "target.yaml"
    target_path.write_text(yaml.safe_dump({"target": target}))
    settings_path = root / "settings.json"
    _json(settings_path, {
        "evidence_mode": "simulation", "root": str(root), "model_path": str(model),
        "subject": SUBJECT, "pod": POD, "plugin_revision": PLUGIN_REVISION,
    })
    _json(root / "cluster.json", {"pods": {}, "serving": False})
    _json(root / "kubeconfig.json", {
        "apiVersion": "v1", "kind": "Config", "clusters": [], "users": [],
        "contexts": [], "simulation": True,
    })
    bootstrap = REPO_ROOT / "tests/e2e/bootstrap"
    env = {
        SETTINGS_ENV: str(settings_path),
        "PYTHONPATH": os.pathsep.join([str(bootstrap), str(REPO_ROOT)]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "AI_INFRA_SKILLS_DIR": str(Path(__file__).resolve()),
        "USER_ID": "simulation",
        "KUBECONFIG": str(root / "kubeconfig.json"),
    }
    context = {"model_path": str(model), "weights": str(model),
               "proof_health_interval": "0",
               "server_log": f"/workspace/server_{SUBJECT.lower()}.log", "served_model_name": SUBJECT,
               "port": "8356"}
    environment = {"evidence_mode": "simulation", "stack": PLUGIN_REVISION}
    graph_args = ["--subject", SUBJECT, "--target", str(target_path)]
    for key, value in context.items():
        graph_args.extend(["--set", f"{key}={value}"])
    for key, value in environment.items():
        graph_args.extend(["--env", f"{key}={value}"])
    return {
        "root": root, "subject": SUBJECT, "model_path": str(model),
        "model_revision": fingerprint["revision"], "plugin_revision": PLUGIN_REVISION,
        "backend": "kunlun", "contract_instance": str(contract_path),
        "operator_report": report_path, "env": env, "graph_args": graph_args,
        "target": target, "environment": environment,
        "settings": settings_path, "events": root / "events.jsonl", "pod": POD,
    }


def _settings() -> dict:
    path = Path(os.environ[SETTINGS_ENV]).resolve()
    settings = json.loads(path.read_text())
    if settings.get("evidence_mode") != "simulation":
        raise ValueError("external doubles require evidence_mode=simulation")
    root = ensure_external(Path(settings["root"]), REPO_ROOT)
    if path.parent != root or not (root / "cluster.json").is_file():
        raise ValueError("invalid external fixture root")
    return settings


def _event(settings: dict, operation: str, **details) -> None:
    record = {"evidence_mode": "simulation", "pid": os.getpid(),
              "operation": operation, **details}
    # One append write keeps records whole across the child process boundaries.
    with (Path(settings["root"]) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def _result(args, stdout="", returncode=0, stderr=""):
    if not isinstance(stdout, str):
        stdout = json.dumps({"evidence_mode": "simulation", **stdout})
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def _distribution(prompt: str, top_k: int) -> list[dict]:
    # Small deterministic arithmetic, not recorded model outputs. The candidate
    # and reference boundaries expose distributions for the real comparator.
    offset = sum(prompt.encode("utf-8")) % 8
    logits = [float((index + offset) % 8) for index in range(8)]
    norm = math.log(sum(math.exp(value) for value in logits))
    return [{"token": f"sim-token-{index}", "logprob": logits[index] - norm}
            for index in sorted(range(8), key=lambda index: -logits[index])[:top_k]]


class SimulatedCluster(KunlunP800Adapter):
    """Raw cluster transport double; all inherited ownership checks stay real."""

    def __init__(self, config=None, timeout=120):
        self.settings = _settings()
        super().__init__(config or ClusterConfig(
            kubeconfig=str(Path(self.settings["root"]) / "no-credentials"),
            namespace="pd-test", container="runtime",
            resource_prefix="simulation-", deployment_kind="FedDeployment",
        ), timeout)

    def _load(self):
        return json.loads((Path(self.settings["root"]) / "cluster.json").read_text())

    def _save(self, state):
        _json(Path(self.settings["root"]) / "cluster.json", state)

    def run(self, args, timeout=None):
        import yaml

        _event(self.settings, "cluster", args=args)
        state = self._load()
        if args == ["auth", "can-i", "create", "pods"]:
            return _result(args, "yes")
        if args[:2] == ["get", "FedDeployment"] and "--ignore-not-found" in args:
            for manifest in state["pods"].values():
                if manifest["metadata"]["name"] == args[2]:
                    return _result(args, manifest)
            return _result(args, "")
        if args[:2] == ["apply", "-f"]:
            manifest = yaml.safe_load(Path(args[2]).read_text())
            # Kubernetes returns labels as strings, including zero-padded IDs.
            manifest["metadata"]["labels"] = yaml.load(
                Path(args[2]).read_text(), Loader=yaml.BaseLoader,
            )["metadata"].get("labels", {})
            name = manifest["metadata"]["name"]
            if manifest["kind"].lower() != "pod":
                name = self.settings["pod"]
            state["pods"][name] = manifest
            self._save(state)
            return _result(args, f"simulation: {name} created")
        if args[:2] == ["get", "pods"] and "-l" in args:
            return _result(args, "pod/" + self.settings["pod"])
        if args[:2] == ["get", "pod"]:
            name = args[2]
            if name not in state["pods"]:
                raise RuntimeError(f"unknown simulated pod: {name}")
            if args[-1] == "jsonpath={.metadata.labels}":
                return _result(args, state["pods"][name]["metadata"].get("labels", {}))
            if "Ready" in args[-1]:
                return _result(args, "True")
            if args[-1] == "yaml":
                return _result(args, yaml.safe_dump(state["pods"][name]))
        if args[:2] == ["delete", "pod"] and args[2] in state["pods"]:
            del state["pods"][args[2]]
            self._save(state)
            return _result(args, "simulation: deleted")
        raise RuntimeError(f"unexpected external cluster command: {args!r}")

    def copy_into(self, pod, local, remote):
        self.assert_owned(pod)
        if pod not in self._load()["pods"]:
            raise RuntimeError(f"copy into unprepared simulated pod: {pod}")
        if not Path(local).is_file() or not remote.endswith("/install_vllm_kunlun.sh"):
            raise RuntimeError(f"unexpected copy: {local} -> {remote}")
        _event(self.settings, "copy", pod=pod, local=str(local), remote=remote)
        return _result(["copy", pod, remote], "simulation: installer staged")

    def http_probe(self, pod, path, port):
        self.assert_owned(pod)
        if pod not in self._load()["pods"]:
            raise RuntimeError(f"HTTP against unprepared simulated pod: {pod}")
        _event(self.settings, "http", pod=pod, path=path, port=port)
        if path not in ("/health", "/v1/models") or port != 8356:
            raise RuntimeError(f"unexpected external HTTP probe: {path}:{port}")
        if not self._load()["serving"]:
            return 503, "simulation: server not started"
        return 200, json.dumps({"data": [{"id": self._load().get("served_model", self.settings["subject"])}],
                                "evidence_mode": "simulation"})

    def exec(self, pod, script, timeout=None):
        self.assert_owned(pod)
        if pod not in self._load()["pods"]:
            raise RuntimeError(f"exec against unprepared simulated pod: {pod}")
        # Log the actual script once, not a node name or expected task outcome.
        _event(self.settings, "exec", pod=pod, script=script)
        result = self._exec(script)
        return _result(["exec", pod, script], **result) if isinstance(result, dict) and (
            "stdout" in result or "returncode" in result
        ) else _result(["exec", pod, script], result)

    def _exec(self, script):
        model = Path(self.settings["model_path"])
        config = json.loads((model / "config.json").read_text())
        # Only known, versioned probe entrypoints are admitted. Never execute a
        # pushed shell snippet locally; its paths describe the simulated pod.
        match = re.search(r"python3 (/tmp/[A-Za-z0-9_]+\.py)(?: (.*))?$", script, re.DOTALL)
        if match:
            filename = Path(match[1]).name
            argv = shlex.split((match[2] or "").replace("2>/dev/null", "").strip())
            return self._probe(filename, argv, config)
        if script == "cat " + shlex.quote(str(model / "config.json")):
            return config
        if script == "xpu_smi -m":
            return "\n".join(" ".join([str(i), "0", "0"] + ["0"] * 14 + ["8192", "98304"]) for i in range(8))
        if script.startswith("test -f ") and " -name '*.safetensors'" in script:
            return "\n".join(str(path) for path in model.glob("*.safetensors"))
        if script.startswith("test -d ") and "git rev-parse HEAD" in script:
            return self.settings["plugin_revision"]
        if "uv pip list | grep -iE" in script:
            return ("SIMULATION: no runtime or device installed\n"
                    "torch 2.9.0\nvllm 0.25.1\nvllm-kunlun 0.25.1\n"
                    + self.settings["plugin_revision"] + "\n")
        if 'python3 -c "import json, torch, vllm, vllm_kunlun' in script:
            if self.settings.get("runtime_import_failure"):
                return {"returncode": 1, "stderr": "ImportError: synthetic runtime import failure"}
            return {"torch": "2.9.0-simulation", "vllm": "0.25.1-simulation",
                    "vllm_kunlun": "0.25.1-simulation"}
        if 'python3 -c "import torch, vllm, vllm_kunlun"' in script:
            return "SIMULATION: runtime imports"
        if script == "bash /workspace/install_vllm_kunlun.sh":
            return "SIMULATION: runtime installation; no packages installed"
        if script.endswith("2>&1 & echo started $!") and (
            "python -m vllm.entrypoints.openai.api_server" in script
        ):
            state = self._load()
            state["serving"] = True
            state["served_model"] = re.search(r"--served-model-name\s+(\S+)", script).group(1)
            self._save(state)
            return "SIMULATION: started"
        if script.startswith("pkill -9 -f '[v]llm.entrypoints.openai.api_server'"):
            state = self._load()
            state["serving"] = False
            self._save(state)
            return ""
        if script.startswith("ps -eo pid=,args= | grep -E"):
            return ""
        if script.startswith("pgrep -f '[v]llm.entrypoints.openai.api_server'"):
            return "up" if self._load()["serving"] else "dead"
        if script.startswith("cat > /workspace/chat_payload.json <<'JSON'\n"):
            payload = json.loads(script.split("\n", 1)[1].rsplit("\nJSON", 1)[0])
            state = self._load()
            state["chat_payload"] = payload
            self._save(state)
            return ""
        if script.startswith("curl -sS -X POST ") and script.endswith(
            "http://127.0.0.1:8356/v1/chat/completions"
        ):
            state = self._load()
            if not state["serving"]:
                raise RuntimeError("chat requested before simulated server startup")
            payload = state.get("chat_payload", {})
            if payload.get("model") != state.get("served_model") or not payload.get("messages"):
                raise RuntimeError("chat requires the served simulated model and messages")
            return {"choices": [{"message": {"content": "simulation: hello"},
                                 "finish_reason": "stop"}]}
        if script.startswith("curl -s http://127.0.0.1:8356/v1/completions "):
            if not self._load()["serving"]:
                raise RuntimeError("completion requested before simulated server startup")
            argv = shlex.split(script)
            payload = json.loads(argv[argv.index("-d") + 1])
            if payload["model"] != self.settings["subject"]:
                raise RuntimeError("completion requested for a different simulated model")
            values = _distribution(payload["prompt"], payload["logprobs"])
            return {"choices": [{"logprobs": {
                "top_logprobs": [{item["token"]: item["logprob"] for item in values}],
            }}]}
        server_logs = ("/workspace/server.log", f"/workspace/server_{SUBJECT.lower()}.log")
        if script in (f"cat {path}" for path in server_logs):
            return self._server_log()
        if script.startswith("stat -c %s ") and any(path in script for path in server_logs) and "tail -n 1" in script:
            return f"{len(self._server_log())}\n{self._server_log()}"
        raise RuntimeError(f"unexpected external exec command: {script[-1500:]}")

    @staticmethod
    def _server_log():
        return (
            "SIMULATION: kunlun backend, tensor_parallel_size=1\n"
            "Model loading took 1.0 GiB\nAvailable KV cache memory: 6.0 GiB\n"
            "GPU KV cache size: 131072 tokens\nmax_model_len=128\n"
            "Maximum concurrency: 1024.0x\n"
        )

    def _probe(self, filename, argv, config):
        model = self.settings["model_path"]
        if filename == "mat001_probe.py":
            if self.settings.get("intake_failure"):
                return {"state": "MODEL_UNAVAILABLE", "reason": "synthetic missing checkpoint shard"}
            if argv != [model]:
                raise RuntimeError(f"unexpected intake path: {argv!r}")
            return _fingerprint(Path(model))
        if filename == "mat027_probe.py":
            return {"state": "DRIFT_CLEAR",
                    "engine": {"name": "vllm", "path": "simulation://runtime/vllm"},
                    "plugin": {"modules_scanned": 1, "path": "simulation://runtime/plugin"},
                    "failures": [], "imported": ["simulation.runtime"]}
        if filename == "mat002_probe.py":
            if argv != config["architectures"]:
                raise RuntimeError(f"unexpected model architecture: {argv!r}")
            return {"state": "SCAN_READY", "results": [{
                "architecture": argv[0], "verdict": "UPSTREAM_GENERIC",
                "meaning": "simulation: reuse registered Qwen3 network; operator gap only",
                "in_installed_vllm": True, "in_kunlun_oot": False,
                "kunlun_oot_archs": [], "vllm_version": "0.25.1-simulation",
            }]}
        if filename == "mat003_probe.py":
            from tools.probe.capability_match_probe import match, required_capabilities

            if argv != [model]:
                raise RuntimeError(f"capability scan uses a different model: {argv!r}")
            required = required_capabilities(config)
            provided = {
                "attention_backends": {"gqa": "MODULE", "mha": "MODULE",
                                       "sliding_window": "MODULE"},
                "quantization_methods": [], "quantization_evidence": "REGISTRY",
                "moe": "ABSENT", "lora": "ABSENT", "reasoning_parsers": "MODULE",
                "tool_parsers": "MODULE", "pd_disaggregation": "ABSENT",
            }
            axes = match(required, provided)
            return {"state": "MATCH_READY", "runtime_verified": False,
                    "model_path": model, "required": required, "provided": provided,
                    "axes": axes, "verdict": "MATCHED", "blocking_axes": [],
                    "unknown_axes": []}
        if filename == "mat008_sliding_window_decode_probe.py":
            def argument(name, cast=int):
                return cast(argv[argv.index(name) + 1])

            context = argument("--context-len")
            window = argument("--window")
            batch = argument("--batch")
            # Zero Q/K -> uniform softmax. V[t] = t + batch index, broadcast
            # over the requested heads/dimensions. Compute the reduced mean
            # directly and independently via the arithmetic-series formula.
            candidate = [sum(range(context - window, context)) / window + b
                         for b in range(batch)]
            reference = [(2 * context - window - 1) / 2 + b for b in range(batch)]
            control = [sum(range(context)) / context + b for b in range(batch)]

            def metrics(values):
                dot = sum(a * b for a, b in zip(values, reference))
                norm = math.sqrt(sum(x * x for x in reference))
                cosine = dot / (math.sqrt(sum(x * x for x in values)) * norm)
                error = math.sqrt(sum((a - b) ** 2 for a, b in zip(values, reference))) / norm
                return {"cosine": cosine, "relative_l2": error}

            ceiling = argument("--max-relative-l2", float)
            floor = argument("--cosine-floor", float)
            measured = metrics(candidate)
            wrong = metrics(control)
            passed = measured["relative_l2"] <= ceiling and measured["cosine"] >= floor
            discriminates = wrong["relative_l2"] > ceiling
            return {
                "dimension": "swa",
                "state": "EXERCISED_PASS" if passed and discriminates else "EXERCISED_FAIL",
                "operators": ["simulation.uniform_window_attention"],
                "thresholds": {"min_cosine": floor, "max_relative_l2": ceiling},
                "geometry": {key.removeprefix("--"): argument(key) for key in (
                    "--heads", "--kv-heads", "--head-dim", "--block-size",
                    "--batch", "--context-len", "--window",
                )},
                "cases": [{"case": "uniform_window", "reference": "arithmetic series",
                           **measured, "candidate": candidate, "expected": reference}],
                "control": {"description": "ignore the window", **wrong,
                            "discriminates": discriminates, "output": control},
            }
        if filename == "mat028_probe.py":
            if argv[0] != model:
                raise RuntimeError("bring-up uses a different synthetic checkpoint")
            tokens = [int(2 * value) for value in (1, 2, 3)]
            stages = ["CONFIG_DERIVED", "ENGINE_CONSTRUCTED", "PREFILL_OK", "DECODE_OK"]
            report = {"stage": stages[-1], "stages_passed": stages, "complete": True,
                    "config": {"architectures": config["architectures"],
                               "model_type": config["model_type"],
                               "num_hidden_layers": {"real": 2, "toy": 2},
                               "kept_dimensions": {"hidden_size": config["hidden_size"],
                                                   "num_attention_heads": 2}},
                    "decode": {"count": len(tokens), "token_ids": tokens},
                    "simulation_workload": "scale([1,2,3])"}
            if self.settings.get("toy_failure"):
                report.update(stage="PREFILL_OK", stages_passed=stages[:-1], complete=False,
                              error={"type": "RuntimeError", "message": "simulated decode failure"})
            return report
        if filename == "mat029_probe.py":
            return {"plugin_path": "simulation://runtime/plugin", "files_scanned": 1,
                    "files_unparsable": [], "signals": []}
        if filename == "mat013_reference.py":
            if argv[0] != model:
                raise RuntimeError("reference uses the wrong synthetic checkpoint")
            top_k = int(argv[1])
            return {"state": "REFERENCE_READY", "dtype": "float32",
                    "transformers_version": "simulation-no-transformers",
                    "results": {prompt: _distribution(prompt, top_k) for prompt in argv[2:]}}
        if filename == "mat009_probe.py":
            return self._parser_observation(argv)
        raise RuntimeError(f"unexpected external runtime probe: {filename} {argv!r}")

    def _parser_observation(self, argv):
        def value(flag):
            return argv[argv.index(flag) + 1]

        def split_reasoning(text):
            reasoning, marker, content = text.partition("</think>")
            return reasoning.removeprefix("<think>"), content if marker else ""

        if value("--model-path") != self.settings["model_path"]:
            raise RuntimeError("parser probe uses a different synthetic checkpoint")
        sample = split_reasoning(value("--reasoning-sample"))
        moved = split_reasoning(value("--reasoning-moved"))
        tool_text = value("--tool-sample")
        parsed = json.loads(tool_text.split("<tool_call>")[1].split("</tool_call>")[0])
        template = json.loads((
            Path(self.settings["model_path"]) / "tokenizer_config.json"
        ).read_text())["chat_template"]
        markers = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--marker"]
        return {
            "state": "CONFORMANT", "model_path": self.settings["model_path"],
            "registries": {"reasoning": ["qwen3"], "tool": ["hermes"]},
            "template_confirms_markers": all(marker in template for marker in markers),
            "cases": [
                {"case": "reasoning", "parser": "qwen3", "implementation": "simulation.split",
                 "separated": bool(sample[0] and sample[1]), "reasoning": sample[0],
                 "content": sample[1], "control": {"discriminates": sample != moved}},
                {"case": "tool", "parser": "hermes", "implementation": "simulation.json",
                 "tools_called": [parsed], "arguments_are_json": isinstance(parsed["arguments"], dict),
                 "name_matches": parsed["name"] == value("--expected-tool-name"),
                 "control": {"discriminates": "<tool_call>" not in value("--tool-control")}},
            ],
        }


def install() -> None:
    """Activate only the hardware transport, source lookup and network guard."""
    global _installed
    if _installed:
        return
    settings = _settings()
    import adapters

    adapters._HARDWARE.update({"kunlun-p800": SimulatedCluster, "kunlun/p800": SimulatedCluster})
    original_run = subprocess.run

    def run(args, *positional, **kwargs):
        argv = list(args) if not isinstance(args, str) else []
        if len(argv) > 2 and argv[:2] == ["git", "ls-remote"]:
            if argv[2] != "https://github.com/baidu/vLLM-Kunlun":
                raise RuntimeError(f"unexpected external source lookup: {argv!r}")
            _event(settings, "source_revision", args=argv)
            return _result(argv, settings["plugin_revision"] + "\trefs/heads/simulation\n")
        return original_run(args, *positional, **kwargs)

    def audit(event, args):
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo"):
            raise RuntimeError("network access is forbidden in the local E2E simulation")
        if event in ("os.system", "os.exec"):
            raise RuntimeError("direct process execution is forbidden in the local E2E simulation")
        if event in ("subprocess.Popen", "os.posix_spawn"):
            executable, argv = args[:2]
            name = Path(os.fsdecode(executable)).name
            if name.startswith("python"):
                return
            if name == "git" and len(argv) > 1 and argv[1] in (
                "ls-files", "status", "diff", "rev-parse",
            ):
                return
            raise RuntimeError(f"unexpected external process in simulation: {argv!r}")

    subprocess.run = run
    sys.addaudithook(audit)
    _installed = True
    _event(settings, "bootstrap")


def _capacity_provider() -> int:
    """Test substitute for the optional external capacity-planner executable."""
    _settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--nvidia-smi-file", type=Path, required=True)
    parser.add_argument("--format", choices=["json"], required=True)
    args = parser.parse_args()
    log = args.log_file.read_text()
    weights = float(re.search(r"Model loading took ([\d.]+) GiB", log)[1])
    kv = float(re.search(r"Available KV cache memory: ([\d.]+) GiB", log)[1])
    used_mib = int(args.nvidia_smi_file.read_text().split(",")[1].split()[0])
    print(json.dumps({
        "evidence_mode": "simulation",
        "memory_breakdown": {"model_weights_gib": weights, "kv_pool_gib": kv,
                             "other_gib": used_mib / 1024 - weights - kv},
        "vllm": {"gpu_kv_cache_tokens": int(re.search(r"GPU KV cache size: (\d+)", log)[1]),
                 "max_model_len": 128, "maximum_concurrency": 1024.0},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(_capacity_provider())
