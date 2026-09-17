"""Fixed subprocess probe: call one frozen entry and capture raw observations.

The entry implements run_case(inputs) -> output tensor mapping. It cannot supply
the validator's verdict, measured dtype/device, dispatch record, or HTTP record.
"""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.request import Request, urlopen

from core.storage import ArtifactStore, ensure_external
from operations.validation.managed_worker import (
    ValidationContractError, check_supported_mode, digest, python_dtype, tensor_shape,
    validate_input_strides, validate_tensor_values,
)


def _load_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValidationContractError(f"non-finite JSON constant: {value}")))


def _cpu_metadata(value):
    import torch
    if not isinstance(value, torch.Tensor):
        raise ValidationContractError("real measurements require actual torch.Tensor outputs")
    if value.device.type != "cpu":
        raise ValidationContractError("builtin real worker probe supports the CPU torch stage only")
    return {"shape": list(value.shape), "strides": list(value.stride()),
            "dtype": str(value.dtype).removeprefix("torch."),
            "layout": "contiguous" if value.is_contiguous() else "noncontiguous",
            "device": str(value.device)}


def _tensor(value, mode):
    if mode == "simulation":
        # Python scalar/list precision is observed, not relabelled as float32/XPU.
        return {"values": value, "shape": tensor_shape(value), "dtype": python_dtype(value),
                "layout": "contiguous", "device": "simulation-cpu"}
    metadata = _cpu_metadata(value)
    return {"values": value.detach().cpu().tolist(), **metadata}


def _inputs(case, mode):
    if mode == "simulation":
        # Each invocation owns its data, even when an implementation mutates it.
        return json.loads(json.dumps({name: tensor["values"] for name, tensor in case["inputs"].items()}))
    import torch
    result = {}
    for name, tensor in case["inputs"].items():
        validate_input_strides(tensor, name)
        validate_tensor_values(tensor["values"], tensor["shape"], f"{name}.values")
        value = torch.tensor(tensor["values"], dtype=getattr(torch, tensor["dtype"]), device="cpu")
        # [] does not encode the trailing dimensions of [0, N]; geometry comes
        # from the frozen OperatorSpec, never from an inferred Python-list shape.
        value = value.reshape(tensor["shape"])
        if "strides" in tensor:
            strided = torch.empty_strided(tensor["shape"], tensor["strides"],
                                          dtype=value.dtype, device="cpu")
            strided.copy_(value)
            value = strided
        observed = _cpu_metadata(value)
        for field in ("shape", "dtype", "layout", "strides"):
            if field in tensor and observed[field] != tensor[field]:
                raise ValidationContractError(f"{name}: materialized input {field} differs from frozen case")
        result[name] = value
    return result


def _entry(request):
    entry = request["entry"]
    path = Path(entry["path"])
    if path.is_symlink() or str(path.resolve()) != str(path) or not path.is_file():
        raise ValidationContractError("probe entry must be a canonical regular source file")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != entry["sha256"]:
        raise ValidationContractError("probe entry changed before execution")
    namespace = {"__name__": "infer_forge_frozen_entry", "__file__": str(path)}
    # Compile the checked source itself; a stale entry .pyc is not executable evidence.
    sys.path.insert(0, str(path.parent))
    exec(compile(source, str(path), "exec"), namespace)
    function = namespace.get("run_case")
    if not callable(function):
        raise ValidationContractError("frozen entry must define callable run_case(inputs)")
    return function


def _invoke(function, inputs, case, request):
    inputs_sha256 = digest(case["inputs"])
    input_metadata = ({name: _cpu_metadata(value) for name, value in inputs.items()}
                      if request["evidence_mode"] == "real" else None)
    values = function(inputs)
    if not isinstance(values, dict) or set(values) != set(case["outputs"]):
        raise ValidationContractError("entry must return exactly the declared output tensor names")
    outputs = {name: _tensor(value, request["evidence_mode"]) for name, value in values.items()}
    measured = {"id": case["id"], "inputs_sha256": inputs_sha256, "outputs": outputs}
    if input_metadata is not None:
        measured["input_metadata"] = input_metadata
    if request["role"] == "candidate" and request["plan"]["stage"] in {"xpu", "integration"}:
        devices = sorted({tensor["device"] for tensor in outputs.values()})
        measured["dispatch"] = {
            "symbol": "run_case", "entry_sha256": request["entry"]["sha256"],
            "rank": 0, "calls": 1, "device": devices[0] if len(devices) == 1 else "mixed",
            "observer": "builtin-local-python-dispatch",
        }
        measured["fallback"] = {"observed_devices": devices,
                                "scope": "builtin-local-python-dispatch"}
    return measured


def _service_case(function, case, request):
    """Exercise a real loopback HTTP request, explicitly not a model/XPU service."""
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body != {"case_id": case["id"], "inputs": {
                    name: tensor["values"] for name, tensor in case["inputs"].items()
                }}:
                    raise ValidationContractError("HTTP request differs from frozen case")
                measured = _invoke(function, body["inputs"], case, request)
                captured["measurement"] = measured
                response = {"outputs": measured["outputs"]}
                payload = json.dumps(response, allow_nan=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:
                captured["error"] = str(error)
                self.send_error(500, "candidate execution failed")

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = {"case_id": case["id"], "inputs": {
        name: tensor["values"] for name, tensor in case["inputs"].items()}}
    try:
        req = Request(f"http://127.0.0.1:{server.server_port}/validate", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=request.get("timeout", 300)) as response:
            status, response_body = response.status, json.loads(response.read())
        measured = captured["measurement"]
        measured["service"] = {"transport": "http-loopback-simulation", "request": body,
                               "http_status": status, "response": response_body}
        return measured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _control_case(case, reference, settings, mode):
    outputs = {}

    def change(value):
        if isinstance(value, list):
            return [change(item) for item in value]
        if settings["kind"] == "zeros":
            return 0.0 if type(value) is float else 0
        return value + settings["value"]

    for name, original in reference["outputs"].items():
        values = change(original["values"])
        if mode == "simulation":
            outputs[name] = _tensor(values, mode)
        else:
            import torch
            value = torch.tensor(values, dtype=getattr(torch, original["dtype"]), device="cpu")
            outputs[name] = _tensor(value.reshape(original["shape"]), mode)
    return {"id": case["id"], "inputs_sha256": digest(case["inputs"]), "outputs": outputs}


def measure(request: dict) -> dict:
    plan, mode, role = request["plan"], request["evidence_mode"], request["role"]
    check_supported_mode(plan, mode)
    if role not in {"candidate", "reference", "control"}:
        raise ValidationContractError("unknown probe role")
    measured = []
    if role == "control":
        reference = _load_json(request["reference_observations"])
        if (reference.get("validation_id") != request["validation_id"]
                or reference.get("role") != "reference"):
            raise ValidationContractError("negative control must use this validation's measured reference")
        indexed = {case["id"]: case for case in reference["cases"]}
        for case in plan["contract"]["cases"]:
            measured.append(_control_case(case, indexed[case["id"]], plan["contract"]["negative_control"], mode))
    else:
        function = _entry(request)
        for case in plan["contract"]["cases"]:
            if role == "candidate" and plan["stage"] == "integration":
                measured.append(_service_case(function, case, request))
            else:
                measured.append(_invoke(function, _inputs(case, mode), case, request))
    return {
        "schema_version": 1, "validation_id": request["validation_id"], "role": role,
        "stage": plan["stage"], "evidence_mode": mode,
        "contract_sha256": plan["contract_sha256"], "entry": request.get("entry"),
        "cases": measured,
        "limitations": (["Python/loopback simulation; no actual XPU or model service evidence"]
                        if mode == "simulation" else ["CPU torch stage only"]),
    }


def execute(request_path: Path, output_path: Path) -> dict:
    output_path = ensure_external(output_path)
    if output_path.exists():
        raise ValidationContractError("probe observations must have a fresh output path")
    request = _load_json(request_path)
    observed = measure(request)
    # Serialization rejects nonfinite values rather than writing misleading JSON.
    ArtifactStore(output_path.parent).write_text(
        output_path.name, json.dumps(observed, ensure_ascii=False, allow_nan=False) + "\n")
    return observed
