"""Runtime axis: one package entry per inference framework stack.

`registry.get_runtime(name)` is how every other layer reaches a runtime —
importing `runtimes.vllm_kunlun` directly from runners or tools would
recreate the concrete-import coupling the adapter factory removed.
"""

from runtimes.registry import RegistryError, default_runtime, get_runtime
from runtimes.vllm_kunlun import RuntimeProfile, VllmKunlunRuntime

__all__ = [
    "RuntimeProfile",
    "VllmKunlunRuntime",
    "RegistryError",
    "default_runtime",
    "get_runtime",
]
