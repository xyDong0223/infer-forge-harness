"""Install the torch decode fallback into a running pod, and remove it again.

Placement decision. vLLM-Kunlun offers four override mechanisms; this uses
**post-import patching of the vendor symbol**, appended to the plugin's own
`register()` which already runs in every process:

- editing `kunlun_attn.py` (what the first experiment did) is rejected: an
  install-time file edit is invisible, survives nothing, and silently disappears
  on reinstall;
- OOT model registration is the wrong layer — the model is fine, the decode
  kernel is not;
- module redirection would replace more of the backend than the one call that
  fails.

Wrapping `kunlun_ops.speculative_attention` itself means both call sites are
covered and the routing decision is made from `qlen`: only regular decode
(`qlen == 1`) goes to torch, so the speculative path keeps the vendor kernel.

Reversible by design: every edited file gets a `.kdp_backup`, and `--remove`
restores it. `KDP_DECODE_KERNEL=speculative` disables the routing at runtime
without uninstalling, which is how the original failure is reproduced.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.kunlun_p800.adapter import KunlunP800Adapter  # noqa: E402

FALLBACK = REPO_ROOT / "patches" / "torch_paged_decode.py"
SITE = "/opt/vllm_kunlun/lib/python3.10/site-packages"

HOOK = '''

def _kdp_install_torch_decode():
    """Route regular decode to a torch implementation of paged attention.

    Both vendor decode kernels return non-zero inside the server at Qwen3-8B's
    decode geometry while accepting identical arguments in isolation. Set
    KDP_DECODE_KERNEL=speculative to reproduce that failure.
    """
    import os
    if os.environ.get("KDP_DECODE_KERNEL", "torch") != "torch":
        return
    import kunlun_ops
    if getattr(kunlun_ops.speculative_attention, "_kdp_wrapped", False):
        return
    from kdp_torch_paged_decode import torch_paged_decode

    _vendor = kunlun_ops.speculative_attention

    def _dispatch(**kwargs):
        # Only regular decode is rerouted; the speculative path (qlen > 1) keeps
        # the vendor kernel, which this fallback does not implement.
        if kwargs.get("qlen") == 1:
            return torch_paged_decode(**kwargs)
        return _vendor(**kwargs)

    _dispatch._kdp_wrapped = True
    kunlun_ops.speculative_attention = _dispatch


_kdp_install_torch_decode()
'''

INSTALL = '''
import ast, shutil, sys
site, hook = sys.argv[1], sys.stdin.read()
init = site + "/vllm_kunlun/__init__.py"
backup = init + ".kdp_backup"
try:
    open(backup)
except FileNotFoundError:
    shutil.copy(init, backup)
source = open(init).read()
if "_kdp_install_torch_decode" in source:
    print("ALREADY_INSTALLED")
    raise SystemExit(0)
source = source + hook
ast.parse(source)
open(init, "w").write(source)
print("INSTALLED")
'''

REMOVE = '''
import os, shutil, sys
site = sys.argv[1]
init = site + "/vllm_kunlun/__init__.py"
backup = init + ".kdp_backup"
if not os.path.exists(backup):
    print("NO_BACKUP")
    raise SystemExit(1)
shutil.move(backup, init)
for path in (site + "/kdp_torch_paged_decode.py", site + "/vllm_kunlun/__pycache__"):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)
print("REMOVED")
'''


def in_pod(adapter: KunlunP800Adapter, pod: str, script: str, args: str, stdin: str = "") -> str:
    payload = base64.b64encode(script.encode()).decode()
    feed = (
        f"echo {base64.b64encode(stdin.encode()).decode()} | base64 -d | " if stdin else ""
    )
    command = (
        f"echo {payload} | base64 -d > /tmp/kdp_placement_step.py && "
        f"{feed}python3 /tmp/kdp_placement_step.py {args}"
    )
    result = adapter.exec(pod, command, timeout=300)
    return (result.stdout + result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--site", default=SITE)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    adapter = KunlunP800Adapter()
    adapter.assert_owned(args.pod)
    if args.remove:
        print(in_pod(adapter, args.pod, REMOVE, args.site))
        return 0

    copied = adapter.exec(
        args.pod,
        "echo {} | base64 -d > {}/kdp_torch_paged_decode.py && echo COPIED".format(
            base64.b64encode(FALLBACK.read_bytes()).decode(), args.site
        ),
        timeout=120,
    )
    print((copied.stdout + copied.stderr).strip())
    print(in_pod(adapter, args.pod, INSTALL, args.site, stdin=HOOK))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
