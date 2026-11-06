# Model A (km->en, no native review) run scripts

Wrappers around David's scripts for the 2026-11-06 Khmer->English baseline. Nothing here replaces `train_backward.sh`,
`cefilter_score.sh`, `align.sh`, `exclude_eval.py`, `train_student.sh` or `finetune_student.sh`; those run unchanged.
The bundle builder copies this directory over `scripts/opus-trainer/` so everything sits in one folder on the box.

| file | what / why |
|---|---|
| `p1_backward_data.py` | km box. Swaps `valid.tsv` (en/km order) to km/en for the km->en student; builds the backward-model (en->km) data from LaBSE-trusted refs |
| `p2_screen.py`, `p3_rule_pass.py` | km box. ZWSP-stripped Khmer + teacher English + rule screen (EMPTY/COPY/KHMER_IN_EN/LEN/REPEAT/NUM_MISMATCH) over all teacher rows; flagged rows go to a rejects file |
| `p4_finetune_nonkd.py` | km box. Finetune block that is not KD rehearsal (trusted human-ish rows + generated short rows, columns swapped) |
| `model_a_exclude.py` | `exclude_eval.py` with the full eval + validation exclusion set, the same for every training TSV |
| `ce_cut.py` | CE-filter cut per (register x English-length bucket) instead of one global tail; dropped rows to a rejects file |
| `diff_rejects.py` | rows removed by `exclude_eval.py` -> rejects file |
| `run_model_a.sh` | the one-box stage runner (stage markers, heartbeat in `/w/logs/job.log`, `ALL_DONE`) |
| `make_bundle.sh` | builds the upload bundle with `MANIFEST.sha256` |
| `smoke_wrapper.sh` | km-box smoke only: runs the real marian binaries on CPU for a few batches |
| `inspect_aug.py` | inspects the OpusTrainer-augmented Khmer stream (Typos can break clusters) |
| `score_model_a.py` | Gate 4 scoring: chrF++/spBLEU vs the teacher, paired bootstrap, number fidelity |
| `launch_job_a.sh`, `guard_loop.sh` | Mac side: launch one Vast box with the marian image; guard loop for my boxes only |
| `marian_short.sh`, `ce_sanity.py` | first-stage GPU smoke wrapper (150 updates per stage); CE-filter sanity (real vs shuffled English win rate) |
| `start_job_a.sh` | copy of helpers/start_job.sh without the sacrebleu pip install |

Found by the km-box smoke run before any rental: the image has Python 3.10 but `pair_spec.py` needs `enum.StrEnum`; `model_a_exclude.py` injects a shim.
The CPU build of marian segfaults in `CorpusBatch::fakeBatch` when training the 3-field guided-alignment student, so the student path is smoke-tested on the GPU box (stage `gpu_smoke`) instead.
