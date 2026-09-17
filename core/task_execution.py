"""Read-only, typed execution metadata from versioned Task definitions.

Task definitions describe commands; they never execute them while loading. The
Graph owns environment-bound fact resolution, specialized adapters and fan-out.
Deployment instance contracts keep their separate schema and validation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib
from pathlib import Path, PurePosixPath
import re
from string import Formatter
from types import MappingProxyType
from typing import Mapping

import yaml

from core.paths import REPO_ROOT


CONTEXT_FIELDS = frozenset({
    "subject", "model_path", "attempt", "artifacts", "contract_instance", "pod",
    "server_log", "served_model_name", "port", "environment_text", "dimension", "weights",
})
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_STATE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_FLAG = re.compile(r"--[a-z][a-z0-9-]*\Z")
_FORMATTER = Formatter()


class TaskExecutionError(ValueError):
    """A source Task cannot safely describe an executable node."""


def _require(condition, message):
    if not condition:
        raise TaskExecutionError(message)


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()) and "\x00" not in value,
             f"{label} must be a nonempty string without NUL")
    return value


def _identifier(value, label):
    value = _text(value, label)
    _require(_IDENTIFIER.fullmatch(value), f"{label} must be a plain identifier")
    return value


def _relative_file(value, label):
    value = _text(value, label)
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts and "\\" not in value
             and value == path.as_posix() and value not in {".", ""}
             and not any(char in value for char in "{}*?[]:"),
             f"{label} must be a safe relative file path without traversal or templates")
    return value


def _argument(value):
    value = _text(value, "argv token")
    try:
        fields = list(_FORMATTER.parse(value))
    except ValueError as error:
        raise TaskExecutionError(f"invalid argv placeholder: {value}") from error
    for _, field, format_spec, conversion in fields:
        if field is not None:
            _require(field in CONTEXT_FIELDS and not format_spec and conversion is None,
                     f"unsupported argv placeholder: {field!r}")
    return value


@dataclass(frozen=True)
class FactInput:
    flag: str
    kind: str
    artifact: str
    optional: bool = False


@dataclass(frozen=True)
class TaskExecution:
    task_id: str
    task_type: str
    task_path: Path
    task_sha256: str
    produces: str | None
    consumes: tuple[str, ...]
    validator: str | None
    exit_states: Mapping[str, str]
    argv: tuple[str, ...] | None = None
    inputs: tuple[FactInput, ...] = ()
    status_file: str | None = None
    success_exit_keys: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.argv is not None

    @property
    def success_states(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.exit_states[key] for key in self.success_exit_keys))

    def to_node_spec(self) -> dict:
        """Return an isolated compatibility view, never mutable source state."""
        if not self.executable:
            raise TaskExecutionError(f"Task {self.task_id} has no execution descriptor")
        result = {
            "command": list(self.argv), "state_file": self.status_file,
            "produces": self.produces, "consumes": list(self.consumes),
            "validator": self.validator, "success_states": list(self.success_states),
            "task_path": str(self.task_path), "task_sha256": self.task_sha256,
        }
        for label, optional in (("needs", False), ("optional", True)):
            bindings = {item.flag: f"fact:{item.kind}:{item.artifact}"
                        for item in self.inputs if item.optional is optional}
            if bindings:
                result[label] = bindings
        return result


def load_task(path: str | Path) -> TaskExecution:
    """Read a source Task, including standalone definitions with no descriptor."""
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "Task definition must be a regular file")
    _require(path.absolute() == path.resolve(), "Task definition path must not traverse symbolic links or parent directories")
    path = path.resolve()
    content = path.read_bytes()
    try:
        definition = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise TaskExecutionError(f"invalid Task YAML: {path}: {error}") from error
    _require(isinstance(definition, dict) and definition.get("kind") == "Task",
             "Task definition must be a mapping with kind: Task")
    api_version = definition.get("api_version")
    _require(isinstance(api_version, str) and re.fullmatch(r"infer\.kunlun/v[0-9]+(?:alpha|beta)[0-9]+", api_version),
             "Task api_version must identify a versioned infer.kunlun definition")
    _require(isinstance(definition.get("acceptance"), dict) and bool(definition["acceptance"]),
             "Task acceptance must be a nonempty mapping")
    _require(isinstance(definition.get("artifacts"), (list, dict)), "Task artifacts must be declared")
    metadata = definition.get("metadata")
    _require(isinstance(metadata, dict), "Task metadata must be a mapping")
    task_id = _text(metadata.get("name"), "metadata.name")
    task_type = _identifier(metadata.get("task_type"), "metadata.task_type")
    spec = definition.get("spec", {})
    _require(isinstance(spec, dict), "Task spec must be a mapping")
    produces = spec.get("produces")
    if produces is not None:
        _identifier(produces, "spec.produces")
    consumes = spec.get("consumes", [])
    _require(isinstance(consumes, list), "spec.consumes must be a list")
    consumes = tuple(_identifier(item, "spec.consumes item") for item in consumes)
    _require(len(consumes) == len(set(consumes)), "spec.consumes must not contain duplicates")
    validator = definition.get("validator")
    if validator is not None:
        validator = _text(validator, "validator")
        validator_path, separator, function = validator.partition(":")
        _relative_file(validator_path, "validator path")
        _require(validator_path.startswith("validators/") and validator_path.endswith(".py"),
                 "validator must reference a repository validator module")
        if separator:
            _identifier(function, "validator function")
    exit_states = definition.get("exit_states", {})
    _require(isinstance(exit_states, dict), "exit_states must be a mapping")
    for key, value in exit_states.items():
        _identifier(key, "exit_states key")
        _require(isinstance(value, str) and _STATE.fullmatch(value), "exit_states values must be state identifiers")
    common = dict(
        task_id=task_id, task_type=task_type, task_path=path,
        task_sha256=hashlib.sha256(content).hexdigest(), produces=produces,
        consumes=consumes, validator=validator, exit_states=MappingProxyType(dict(exit_states)),
    )
    descriptor = spec.get("execution_descriptor")
    if descriptor is None:
        return TaskExecution(**common)
    _require(isinstance(descriptor, dict), "execution_descriptor must be a mapping")
    required = {"schema_version", "argv", "inputs", "status_file", "success_exit_keys"}
    _require(set(descriptor) == required, "execution_descriptor must contain exactly the supported fields")
    _require(type(descriptor["schema_version"]) is int and descriptor["schema_version"] == 1,
             "execution_descriptor schema_version must be integer 1")
    _require(produces is not None and validator is not None,
             "executable Task requires spec.produces and validator")
    _require("consumes" in spec, "executable Task requires spec.consumes")
    argv = descriptor["argv"]
    _require(isinstance(argv, list) and len(argv) >= 2, "argv must be a list of command tokens")
    argv = tuple(_argument(item) for item in argv)
    _require(argv[0] == "python3", "descriptor executable must be python3, never a shell")
    entry = _relative_file(argv[1], "argv entrypoint")
    _require(entry.startswith("cli/") and entry.endswith(".py"),
             "descriptor entrypoint must be a repository cli Python command")
    inputs = descriptor["inputs"]
    _require(isinstance(inputs, dict), "descriptor inputs must be a flag mapping")
    bindings = []
    for flag, binding in inputs.items():
        _require(isinstance(flag, str) and _FLAG.fullmatch(flag), "input flag must be a long CLI option")
        _require(isinstance(binding, dict) and {"kind", "artifact"} <= binding.keys()
                 and set(binding) <= {"kind", "artifact", "optional"},
                 "input binding requires kind/artifact and optional boolean only")
        kind = _identifier(binding["kind"], "input kind")
        artifact = _relative_file(binding["artifact"], "input artifact")
        optional = binding.get("optional", False)
        _require(type(optional) is bool, "input optional must be boolean")
        _require(optional or kind in consumes, "required input kind must be declared in spec.consumes")
        _require(flag not in argv, "input flag must not duplicate an argv option")
        bindings.append(FactInput(flag, kind, artifact, optional))
    status_file = _relative_file(descriptor["status_file"], "descriptor status_file")
    success_keys = descriptor["success_exit_keys"]
    _require(isinstance(success_keys, list) and bool(success_keys), "success_exit_keys must be a nonempty list")
    success_keys = tuple(_identifier(key, "success exit key") for key in success_keys)
    _require(len(set(success_keys)) == len(success_keys) and set(success_keys) <= set(exit_states),
             "success_exit_keys must uniquely reference Task exit_states keys")
    return TaskExecution(**common, argv=argv, inputs=tuple(bindings), status_file=status_file,
                         success_exit_keys=success_keys)


def load_execution_catalog(root: str | Path = REPO_ROOT) -> dict[str, TaskExecution]:
    """Load executable definitions only; reject ambiguous Task/fact identities."""
    paths = sorted((Path(root) / "tasks").glob("*/task.yaml"))
    result = {}
    task_types, task_ids, facts = set(), set(), set()
    for path in paths:
        descriptor = load_task(path)
        _require(descriptor.task_type not in task_types, f"duplicate task_type: {descriptor.task_type}")
        _require(descriptor.task_id not in task_ids, f"duplicate Task name: {descriptor.task_id}")
        task_types.add(descriptor.task_type)
        task_ids.add(descriptor.task_id)
        if not descriptor.executable:
            continue
        _require(descriptor.produces not in facts, f"ambiguous produced fact: {descriptor.produces}")
        facts.add(descriptor.produces)
        result[descriptor.task_type] = descriptor
    return result


@cache
def default_execution_catalog() -> Mapping[str, TaskExecution]:
    """Share one immutable source snapshot across consumers in this process.

    Graph execution checks source digests and requires a fresh process after a
    definition changes. Explicit load_execution_catalog remains an uncached read.
    """
    return MappingProxyType(load_execution_catalog())
