#!/usr/bin/env python3
"""Offline vLLM 1-best ug->en KD decode with an instruction MT model (Hy-MT2).

1-best (greedy) on purpose: extract-best is near-dead for uig (3.2% gate), so
n-best buys nothing here and 1-best halves the generated tokens. vLLM's
continuous batching handles throughput; input order is preserved in the output.

Model id from HY_MODEL (baked into the image env). Prints a throughput line so
the same script doubles as the throughput bench.

The `if __name__ == "__main__"` guard is REQUIRED: vLLM's v1 engine uses `spawn`
multiprocessing, which re-imports this module in each worker — without the guard
the worker re-runs LLM() and dies with "process before bootstrapping".

    vllm_decode.py SRC OUT [LIMIT=0] [--max-tokens N] [--drop-truncated]
"""

import argparse
import os
import sys
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    from vllm_truncation import print_truncation_summary, write_decode_outputs

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("limit", nargs="?", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--drop-truncated", action="store_true")
    args = ap.parse_args()

    model = os.environ["HY_MODEL"]

    # Hy-MT2's own English instruction template (matches hy_mt2_gate.py exactly).
    template = ("Translate the following text into English. Note that you should only "
                "output the translated result without any additional explanation: {text}")

    lines = [l for l in open(args.src, encoding="utf-8").read().splitlines() if l.strip()]
    if args.limit:
        lines = lines[: args.limit]

    # enforce_eager: skip torch.compile + the 51-size cudagraph capture. That
    # warmup is ~10-20min per box with no compile cache (and can hang), which for
    # a one-pass bulk KD decode costs more than the ~10-20% eager generation
    # penalty it would save across the corpus.
    llm = LLM(model=model, trust_remote_code=True, max_model_len=2048,
              gpu_memory_utilization=0.92, enforce_eager=True)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": template.format(text=s)}],
            add_generation_prompt=True, tokenize=False,
        )
        for s in lines
    ]

    t0 = time.time()
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    )
    dt = time.time() - t0

    truncated, n = write_decode_outputs(
        args.out, outs, drop_truncated=args.drop_truncated
    )
    print_truncation_summary(truncated, n)

    print(f"THROUGHPUT: {len(lines)} lines / {dt:.1f}s = {len(lines)/dt:.1f} l/s", file=sys.stderr)


if __name__ == "__main__":
    main()
