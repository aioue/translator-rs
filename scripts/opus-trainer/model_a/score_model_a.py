#!/usr/bin/env python3
"""Gate 4 scoring (light CPU, no COMET): chrF++ (word_order=2) and spBLEU (flores200 tokenizer) of the teacher and each student
checkpoint on the SAME sources/refs, plus number-fidelity (repo number_fidelity.py, km->en) per slice and per-slice corpus
bootstrap 95% CI for the student-minus-teacher spBLEU gap (paired, 100 resamples, seed 1).
usage: uv run --with sacrebleu python -I score_model_a.py --decode DIR_WITH_kd_AND_ft --out report.md --json report.json
Teacher hyps: gpu-session-2026-11-03/w_out/hy-mt2-7b (Hy-MT2-7B-FP8 greedy, raw Khmer sources incl. ZWSP). Student sources:
eval/<slice>.km.zwsp-stripped (what the runtime would send). References identical for both (checked by cmp)."""
import argparse, json, pathlib, random, subprocess, sys, tempfile
import sacrebleu
R = pathlib.Path("/Users/tom/src/github/translator-rs/scripts/opus-trainer/data")
E = R / "eval-2026-11-03/eval"; T = R / "gpu-session-2026-11-03/w_out/hy-mt2-7b"
NF = pathlib.Path("/tmp/km-model-a/worktree/scripts/opus-trainer/number_fidelity.py")
SL = {"flores-plus-dev": "slice_flores-plus-dev", "flores-plus-devtest": "devtest", "holdout-ui-300": "slice_holdout-ui-300",
      "holdout-dialogue-300": "slice_holdout-dialogue-300", "alt-dev": "slice_alt-dev", "alt-test": "slice_alt-test", "ntrex-128": "slice_ntrex-128"}
ap = argparse.ArgumentParser(); ap.add_argument("--decode", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--json", type=pathlib.Path, required=True)
a = ap.parse_args()
rd = lambda p: pathlib.Path(p).read_text(encoding="utf-8").split("\n")[:-1] if str(p).endswith("") else None
def lines(p): 
    t = pathlib.Path(p).read_text(encoding="utf-8"); L = t.split("\n"); return L[:-1] if L and L[-1] == "" else L
def m(h, r): return (sacrebleu.corpus_chrf(h, [r], word_order=2).score, sacrebleu.corpus_bleu(h, [r], tokenize="flores200").score)
def nf(src, hyp):
    with tempfile.TemporaryDirectory() as d:
        j = pathlib.Path(d) / "x.json"
        subprocess.run([sys.executable, str(NF), "--slice", "s", str(src), str(hyp), "--src-lang", "km", "--hyp-lang", "en", "--json", str(j)], check=True, capture_output=True)
        o = json.loads(j.read_text())
        return o
res = {}; systems = [d.name for d in sorted(a.decode.iterdir()) if d.is_dir()]
for s, td in SL.items():
    td = td + "_khm_Khmr-eng_Latn"
    ref = lines(E / f"{s}.en"); tref = lines(T / td / "khm_Khmr-eng_Latn.ref"); assert ref == tref, s
    th = lines(T / td / "khm_Khmr-eng_Latn.hyp"); assert len(th) == len(ref)
    row = {"n": len(ref), "teacher": dict(zip(("chrf", "spbleu"), m(th, ref)))}
    row["teacher"]["numfid"] = nf(T / td / "khm_Khmr-eng_Latn.src", T / td / "khm_Khmr-eng_Latn.hyp")
    for sysn in systems:
        p = a.decode / sysn / f"{s}.hyp"
        if not p.exists(): continue
        h = lines(p); assert len(h) == len(ref), (s, sysn, len(h), len(ref))
        d = dict(zip(("chrf", "spbleu"), m(h, ref)))
        rng = random.Random(1); gaps = []
        for _ in range(100):
            ix = [rng.randrange(len(ref)) for _ in ref]
            hh = [h[i] for i in ix]; tt = [th[i] for i in ix]; rr = [ref[i] for i in ix]
            gaps.append(sacrebleu.corpus_bleu(hh, [rr], tokenize="flores200").score - sacrebleu.corpus_bleu(tt, [rr], tokenize="flores200").score)
        gaps.sort(); d["gap_spbleu_ci95"] = [round(gaps[2], 2), round(gaps[97], 2)]
        d["numfid"] = nf(E / f"{s}.km.zwsp-stripped", p)
        row[sysn] = d
    res[s] = row
a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
def nfs(x): 
    t = x["total"] if "total" in x else None
    return t
L = ["| slice | n | teacher chrF++ / spBLEU | " + " | ".join(f"{s} chrF++ / spBLEU (gap, 95% CI)" for s in systems) + " |", "|---|---|---|" + "---|" * len(systems)]
for s, r in res.items():
    cells = [f"{r['teacher']['chrf']:.1f} / {r['teacher']['spbleu']:.1f}"]
    for sysn in systems:
        d = r.get(sysn)
        cells.append("-" if not d else f"{d['chrf']:.1f} / {d['spbleu']:.1f} ({d['spbleu']-r['teacher']['spbleu']:+.1f} [{d['gap_spbleu_ci95'][0]:+.1f},{d['gap_spbleu_ci95'][1]:+.1f}])")
    L.append(f"| {s} | {r['n']} | " + " | ".join(cells) + " |")
a.out.write_text("\n".join(L) + "\n"); print("\n".join(L))
