"""Opt-in bootstrap inherited by the real workflow's Python subprocesses."""

import os
import sys

if os.environ.get("INFER_FORGE_E2E_SETTINGS"):
    try:
        from tests.e2e.external import install

        install()
    except BaseException:
        # Python normally swallows sitecustomize exceptions. Never let a broken
        # simulation bootstrap fall through to the actual cluster adapter.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(78)
