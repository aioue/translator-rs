#!/bin/bash
# GPU smoke stage only: run the real marian trainer for a handful of updates with frequent validation/saving, so every code path
# (fp16, guided alignment from OpusTrainer, validation, keep-best, quantize-bits 8 in the finetune config, resume markers) is exercised
# in about a minute per stage before the multi-hour runs. Extra flags come last so they override the config files.
exec /usr/local/bin/marian "$@" --after "${SHORT_AFTER:-150u}" --valid-freq 50 --save-freq 50 --disp-freq 10
