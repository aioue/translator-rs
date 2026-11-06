"""Sidecar finish reasons and stderr summary for vLLM KD/decode scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def finish_reason_sidecar_path(out_path: str) -> str:
    """Path for one finish reason per output line (sibling of the hyp file)."""
    return f"{out_path}.finish_reason"


def _flatten_hyp(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\t", " ")


def write_decode_outputs(
    out_path: str,
    outs,
    *,
    drop_truncated: bool = False,
) -> tuple[int, int]:
    """Write hyp lines and `.finish_reason` sidecar; return (truncated_count, n)."""
    hyp_lines: list[str] = []
    reasons: list[str] = []
    truncated = 0
    for o in outs:
        reason = o.outputs[0].finish_reason
        reasons.append(reason)
        if reason == "length":
            truncated += 1
        flat = _flatten_hyp(o.outputs[0].text)
        if drop_truncated and reason != "stop":
            hyp_lines.append("")
        else:
            hyp_lines.append(flat)
    n = len(outs)
    Path(out_path).write_text("".join(line + "\n" for line in hyp_lines), encoding="utf-8")
    Path(finish_reason_sidecar_path(out_path)).write_text(
        "".join(r + "\n" for r in reasons), encoding="utf-8"
    )
    return truncated, n


def print_truncation_summary(truncated: int, n: int) -> None:
    print(f"TRUNCATED {truncated} of {n}", file=sys.stderr)
