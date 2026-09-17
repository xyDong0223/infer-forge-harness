"""Fail closed at legacy runtime entry points during the managed-v2 rollout."""


def require_supported_runtime(run):
    if (run.metadata.get("worker_protocol") == "managed-v2"
            and run.metadata.get("evidence_mode", "real") != "simulation"):
        raise ValueError(
            "managed-v2 real Pod execution is BLOCKED: the trusted remote process, "
            "device-dispatch and service observation drivers are not installed; "
            "do not downgrade this run or use legacy shell execution to bypass the gate")
