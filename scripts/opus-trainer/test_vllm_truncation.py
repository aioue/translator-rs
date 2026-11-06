"""Tests for vLLM decode truncation sidecar (stub outputs, no GPU)."""

import io
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[1] / "wt" / "scripts" / "opus-trainer"
sys.path.insert(0, str(WT))

from vllm_truncation import (  # noqa: E402
    finish_reason_sidecar_path,
    print_truncation_summary,
    write_decode_outputs,
)


class _Out:
    def __init__(self, text: str, finish_reason: str):
        self.text = text
        self.finish_reason = finish_reason


class _Req:
    def __init__(self, text: str, finish_reason: str = "stop"):
        self.outputs = [_Out(text, finish_reason)]


def test_sidecar_alignment_and_summary(tmp_path):
    outs = [
        _Req("line one", "stop"),
        _Req("cut\noff\there", "length"),
        _Req("line three", "stop"),
    ]
    out = tmp_path / "hyp.txt"
    truncated, n = write_decode_outputs(str(out), outs)
    assert (truncated, n) == (1, 3)
    assert out.read_text(encoding="utf-8") == (
        "line one\n" "cut off here\n" "line three\n"
    )
    side = Path(finish_reason_sidecar_path(str(out))).read_text(encoding="utf-8")
    assert side == "stop\nlength\nstop\n"
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        print_truncation_summary(truncated, n)
    finally:
        sys.stderr = old
    assert buf.getvalue().strip() == "TRUNCATED 1 of 3"


def test_drop_truncated_preserves_rows(tmp_path):
    outs = [_Req("keep", "stop"), _Req("drop me", "length")]
    out = tmp_path / "hyp.txt"
    truncated, n = write_decode_outputs(str(out), outs, drop_truncated=True)
    assert (truncated, n) == (1, 2)
    assert out.read_text(encoding="utf-8") == "keep\n\n"
    assert Path(finish_reason_sidecar_path(str(out))).read_text() == "stop\nlength\n"


def test_default_matches_legacy_line_format(tmp_path):
    """Without drop_truncated, truncated lines still emit flattened text (legacy)."""
    outs = [_Req("a\tb\nc", "length")]
    out = tmp_path / "hyp.txt"
    write_decode_outputs(str(out), outs, drop_truncated=False)
    assert out.read_text(encoding="utf-8") == "a b c\n"
    assert Path(finish_reason_sidecar_path(str(out))).read_text() == "length\n"
