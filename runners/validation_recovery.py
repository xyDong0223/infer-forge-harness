"""Observe an interrupted local validation coordinator without rerunning it."""
from engine.managed_validation import reconcile_managed_validation
from runners.managed_execution import host_identity, process_identity


def reconcile_local_validation(scheduler, run_id, validation_id):
    def observe(receipt):
        controller = receipt.get("recipe", {}).get("controller_process", {})
        result = {"controller_absent": False, "controller_process": controller}
        if (controller.get("host_id") != host_identity()
                or type(controller.get("pid")) is not int
                or not controller.get("process_identity")):
            return result
        # PID reuse is conservative too: a different live PID is not permission
        # to stop a process or to override an unknown controller's receipt.
        if process_identity(controller["pid"]) is None:
            result["controller_absent"] = True
        return result
    return reconcile_managed_validation(scheduler, run_id, validation_id, observer=observe)
