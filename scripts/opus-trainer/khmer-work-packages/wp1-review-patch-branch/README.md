# WP1: independent review of the Khmer patch branch

Goal: find bugs and risks in the code on this branch before it goes to the upstream author. You are a second pair of eyes.
An earlier review already found that one patch was missing a file and that two others clashed, so more is likely there.

Branch: `km-prep-2026-11-05`, on top of `khmer-smoke` @ `9eed645` (already in this branch's history). Review the eight product commits after `9eed645` and before `bd7ec84`. The commits from `bd7ec84` on are these instructions.

```bash
git log --oneline 9eed645..HEAD
git diff 9eed645..HEAD --stat
```

## The commits and what to look at

| Commit subject | Files | Please check |
|---|---|---|
| vLLM decode: write finish reasons, report truncation ... | `vllm_kd.py`, `vllm_decode.py`, `hy_mt2_gate.py`, `vllm_truncation.py` | Is there exactly one finish reason per output line, in order, also with `--drop-truncated`? Is the default cap (512) applied everywhere? What happens on an empty or failed generation? |
| translate: strip U+200B from Khmer source text ... | `crates/translator-translate/src/translate.rs` | Does any path send Khmer text to the engine without `prepare_source_text`? Pivot paths (km to en to X) should strip once. Are non-Khmer languages unchanged, including `source_text`? Offsets and `steer`: still consistent? |
| opustrainer.student.km.yml: comment ... | `configs/opustrainer.student.km.yml` | Are the comments true? The header claims a measurement from 3 Nov 2026 (about 0.3% of augmented lines get a malformed Khmer cluster). Check that claim against the code. |
| prep_drop_report: parallel worker pool, LF-only line splitting | `prep_drop_report.py` | Does it give the same output as the serial version at `9eed645` on a small corpus? Any ordering or counting difference? |
| compare_filters: split lines on LF only | `compare_filters.py` | Same LF-only reasoning. |
| gen_pairs: Khmer short-register block v1 | `gen_pairs.py`, `pair_spec.py`, `check_short_block.py`, `configs/gen_pairs.km.v1.json`, `configs/km_short_block.switch.json` | Can bulk generation run while the switch file says `bulk_generation_allowed: false`? Do the number, script, duplicate and probe near-duplicate gates do what the docs say? |
| gen_pairs: delegate backend ... | `gen_pairs.py`, `check_handoff.py`, `D-templates/` | Does `import-responses` reject bad rows per row (not silently drop or accept)? Does `check_handoff.py` catch a missing manifest key (exit 2) and an extra response file (exit 1)? |
| test: vllm_truncation | `test_vllm_truncation.py` | Does the test exercise the real helper? |

Not in the Python test list below: the parity of the parallel drop report with the serial one, and the OpusTrainer config. The Rust change has a unit test, `khmer_source_strips_zwsp_only` (see the Docker command below).

## Known issues (already found, do not report again)

- The truncation patch originally imported `vllm_truncation.py` without creating it. The file is now in the first commit.
- Two upstream-facing fixes for `prep_drop_report.py` overlapped. The branch uses the parallel version and takes only the `compare_filters.py` part of the other.
- `test_khmer.py` is not in the loop below: without the `sentencepiece` package its vocab test fails (the other 15 pass and 6 skip for lack of `fasttext`).
- vLLM cannot be run in this review; the truncation code is only tested with a stub.

## How to run what can be run

```bash
cd scripts/opus-trainer
python -m venv /tmp/wp1 && /tmp/wp1/bin/pip install pytest   # Python 3.11 to 3.13; pip needs network, otherwise use any environment that has pytest
for t in test_short_block.py test_handoff.py test_gen_pairs.py test_vllm_truncation.py test_registers.py test_exclude_eval.py test_build_tranche.py test_number_fidelity.py test_perturb_numbers.py; do /tmp/wp1/bin/python -m pytest -q $t; done
```

Optional, if you have Docker: `cargo test -p translator-translate --lib khmer_source_strips_zwsp_only` (the Rust crate needs cmake and clang).

## Which model to use

Prefer the strongest code-capable model you have (for example Claude Opus or Sonnet in Claude Code, or a GPT-5-class model in Codex),
and let it run the tests rather than only read the diff. Anything you can kindly provide is welcome, including a human read.
Please say which model and version you used in `FINDINGS.md`.

## What to hand back

One file, `FINDINGS.md` (send it in your reply; do not commit it), using the template in [FINDINGS-TEMPLATE.md](FINDINGS-TEMPLATE.md). Rank findings by severity. Give the file and line, what goes wrong, and how you know (a command, a test you wrote, or reasoning from the code). Say clearly which findings you verified by running something and which you inferred.

You may write throwaway tests, but keep them out of the diff you hand back. Do not change this branch.
