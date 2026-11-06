# Khmer pack: work packages for outside helpers

Context: we are adding Khmer (km) to David Ventura's Offline Translator. A large "teacher" model translates
millions of sentences and a small "student" model learns from them. The pipeline, numbers and the current
state are described in the `scripts/opus-trainer/NEW_LANGUAGE.md` guide and, for Khmer, in the commits on this branch.

This directory holds three self-contained jobs that need no credentials, no GPU and no access to our machines.
Each can be done in a single Claude or ChatGPT session (or a few), by a person or an agent.

| Package | What you do | Output | Rough effort |
|---|---|---|---|
| [WP1: review the patch branch](wp1-review-patch-branch/README.md) | Read and test the Khmer patches in this branch and report problems | `FINDINGS.md` | 2 to 3 hours |
| [WP2: judge generated rows](wp2-judge-rows/README.md) | Rate 21,984 short English-Khmer pairs, 400 per sheet | `ratings-NNN.tsv` files | about 15 minutes per sheet, any number of sheets |
| [WP3: write short text](wp3-generate-short-text/README.md) | Fill in prompts for signs, labels, menus and notices (25 prompts per batch, 29 batches) | `responses/*.json` per batch | about an hour per batch, any number of batches |

Data for WP2 and WP3 (two zip files with checksums) is in this Google Drive folder: https://drive.google.com/drive/folders/1TepAqJr712lQUq-N0VC_8YEIrNnUxdBL?usp=sharing

## Ground rules

- Nothing here needs a password, API key or account of ours. If something asks for one, stop and say so.
- Do not push to this repository, open pull requests on it, or contact the upstream author. Send results to the person who gave you the link.
- WP2 asks you to rate Khmer and WP3 asks you to write it. Say "unsure" where you cannot tell, and do not pad. A short honest batch beats a long invented one.
- None of the data contains personal information. It is machine-generated text (signs, labels, menus, notices, app strings).
- WP3 output is used as training data for a small translation model. Only take WP3 on if your model's terms allow you to hand its output to someone else for that use; if you are not sure, do WP1 or WP2 instead.
- Models: each package says which to prefer. In short, use the strongest model you have, and anything you can kindly provide is welcome.
- The ratings and findings are inputs to a human decision; nothing is merged or trained on automatically.

## What is known to be weak (so you do not need to rediscover it)

- No native Khmer speaker has read any of the data yet. The existing judges are AI models too.
- The generated rows come from two sources (a hosted model and an open-weights pipeline) that are mixed together in the sheets, in random order. Please do not try to guess the source.
