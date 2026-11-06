#!/usr/bin/env python3
"""Run exclude_eval.py over one training TSV with the full Model A exclusion set (works on the km box and on the rented box).

Why a wrapper: exclude_eval.py takes ~25 CLI specs (digest lists, probe lists, eval pairs) and the SAME set must be applied to
EVERY training TSV (backward-model data, KD set, finetune set, rehearsal sample); finetune_student.sh refuses to start without
<tsv>.evalclean.json. Set = pool-v3's (run_exclude.py: 7 eval sets by digest + probe lists + pair rule on the ZWSP-stripped eval
texts) PLUS the frozen Marian validation file as an eval pair, so a validation row never trains the model it validates.

usage: model_a_exclude.py TRAIN_TSV OUT_TSV [--text-columns 2]
Env (defaults = km box): MA_EVAL (dir with <name>.en/.km.zwsp-stripped, probes-en.txt), MA_HASH (dir with <name>.sha256),
MA_PROBES_KM (probes_en_to_km.tsv), MA_VALID (dir with valid.en / valid.km).
Both column orders work (pair rule is order-free).  The sibling exclude_eval.py/pair_spec.py must be importable from this directory."""
import enum, os, pathlib, runpy, sys
# The marian-cuda image ships Python 3.10; pair_spec.py imports enum.StrEnum (3.11+). Same semantics, injected only when missing
# (found by the km-box smoke run 2026-11-06, which would otherwise have failed on the rented box after hours of training).
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self): return str(self.value)
        @staticmethod
        def _generate_next_value_(name, start, count, last_values): return name.lower()
    enum.StrEnum = StrEnum
HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(os.environ.get("MA_REPO", HERE))
EVAL = pathlib.Path(os.environ.get("MA_EVAL", "/home/km/work/pool-v3/eval"))
HASH = pathlib.Path(os.environ.get("MA_HASH", "/home/km/work/pool-v3/hashes"))
PKM = pathlib.Path(os.environ.get("MA_PROBES_KM", "/home/km/work/pool-v3/probes_en_to_km.tsv"))
VALID = pathlib.Path(os.environ.get("MA_VALID", "/home/km/work/model-a/valid"))
names = ["flores-plus-dev", "flores-plus-devtest", "alt-dev", "alt-test", "ntrex-128", "holdout-dialogue-300", "holdout-ui-300"]
specs = [str(HASH / f"{n}.sha256") for n in names] + [str(EVAL / "probes-en.txt"), str(PKM)]
pairs = [(str(EVAL / f"{n}.en.zwsp-stripped"), str(EVAL / f"{n}.km.zwsp-stripped")) for n in names]
pairs.append((str(VALID / "valid.en"), str(VALID / "valid.km")))
for p in [*specs, *[x for pr in pairs for x in pr]]:
    if not os.path.exists(p): sys.exit(f"missing exclusion input {p}")
train, out = sys.argv[1], sys.argv[2]
cols = sys.argv[sys.argv.index("--text-columns") + 1] if "--text-columns" in sys.argv else "2"
argv = ["exclude_eval.py", "--train", train, "--out", out, "--text-columns", cols, "--eval", *specs]
for l, r in pairs: argv += ["--eval-pair", l, r]
sys.path.insert(0, str(REPO)); sys.argv = argv
runpy.run_path(str(REPO / "exclude_eval.py"), run_name="__main__")
