#!/bin/bash
# SMOKE ONLY (km box, CPU): make the real marian binaries run a few CPU batches. Installed as smokebin/{marian,marian-scorer,marian-decoder}.
# Strips GPU-only flags (--devices, --fp16, --workspace) and bounds the run. Not used on rented boxes.
real=/usr/local/bin/$(basename "$0"); out=()
while [ $# -gt 0 ]; do case "$1" in
  --devices) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do shift; done;;
  --fp16) shift;;
  --workspace|--mini-batch|--mini-batch-words|--maxi-batch|--valid-mini-batch) shift 2;;
  --mini-batch-fit|--mini-batch-fit=*) shift;;
  *) out+=("$1"); shift;;
esac; done
extra=(--cpu-threads 4 --workspace 1500)
case "$(basename "$0")" in marian) extra+=(--mini-batch 16 --maxi-batch 20 --mini-batch-fit=false --valid-mini-batch 16 --guided-alignment none --after 30u --valid-freq 15 --save-freq 15 --disp-freq 5);;
  marian-scorer) extra+=(--mini-batch 16 --maxi-batch 20);; marian-decoder) extra+=(--mini-batch 8 --maxi-batch 20);; esac
exec "$real" "${out[@]}" "${extra[@]}"
