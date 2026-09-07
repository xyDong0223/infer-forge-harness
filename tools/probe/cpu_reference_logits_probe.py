"""In-pod CPU reference: next-token distributions from `transformers`.

Runs the same checkpoint on CPU so the comparison has an authority that does not
share the accelerator's kernels. Slow by nature — one forward pass per prompt, no
generation — which is why it compares distributions instead of generated text.

Accelerator visibility is cleared by the caller; this refuses to run if torch can
still see a device, because a reference that quietly used the accelerator would
agree with the candidate for the wrong reason.
"""

from __future__ import annotations

import json
import sys


def main(model_path: str, top_k: int, prompts: list[str]) -> int:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:
        print(json.dumps({"state": "REFERENCE_UNAVAILABLE", "reason": str(error)}))
        return 0

    if torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "state": "REFERENCE_UNAVAILABLE",
                    "reason": "an accelerator is still visible; the reference must run on CPU",
                }
            )
        )
        return 0

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # No device_map: it would pull in `accelerate`, and the point is to stay on
        # plain CPU torch with as little machinery as possible between the weights
        # and the logits.
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32, trust_remote_code=True
        )
        model.eval()

        results = {}
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            values, indices = torch.topk(torch.log_softmax(logits, dim=-1), k=top_k)
            results[prompt] = [
                {"token": tokenizer.decode([int(index)]), "logprob": float(value)}
                for value, index in zip(values, indices)
            ]
        print(
            json.dumps(
                {
                    "state": "REFERENCE_READY",
                    # float32 on purpose: the reference should not inherit the
                    # candidate's reduced precision.
                    "dtype": "float32",
                    "transformers_version": transformers.__version__,
                    "results": results,
                }
            )
        )
    except Exception as error:
        print(json.dumps({"state": "REFERENCE_FAILED", "reason": f"{type(error).__name__}: {error}"}))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(json.dumps({"state": "CONTRACT_INVALID", "reason": "usage: probe <path> <top_k> <prompt>..."}))
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]), sys.argv[3:]))
