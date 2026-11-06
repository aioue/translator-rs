# Khmer pack: work packages for outside helpers

Context: we are adding Khmer (km) to David Ventura's Offline Translator. A large "teacher" model translates
millions of sentences and a small "student" model learns from them. The pipeline, numbers and the current
state are described in the `scripts/opus-trainer/NEW_LANGUAGE.md` guide and, for Khmer, in the commits on this branch.

This directory holds two self-contained jobs that need no credentials, no GPU and no access to our machines.
Each can be done in a single Claude or ChatGPT session (or a few), by a person or an agent.

| Package | What you do | Output | Rough effort |
|---|---|---|---|
| [WP1: review the patch branch](wp1-review-patch-branch/README.md) | Read and test the Khmer patches in this branch and report problems | `FINDINGS.md` | 2 to 3 hours |
| [WP2: judge generated rows](wp2-judge-rows/README.md) | Rate 21,984 short English-Khmer pairs, 400 per sheet | `ratings-NNN.tsv` files | about 15 minutes per sheet, any number of sheets |

## Ground rules

- Nothing here needs a password, API key or account of ours. If something asks for one, stop and say so.
- Do not push to this repository, open pull requests on it, or contact the upstream author. Send results to the person who gave you the link.
- Neither package asks you to write Khmer. WP2 asks you to rate it; say "unsure" where you cannot tell.
- None of the data contains personal information. It is machine-generated text (signs, labels, menus, notices, app strings).
- The ratings and findings are inputs to a human decision; nothing is merged or trained on automatically.

## What is known to be weak (so you do not need to rediscover it)

- No native Khmer speaker has read any of the data yet. The existing judges are AI models too.
- The generated rows come from two sources (a hosted model and an open-weights pipeline) that are mixed together in the sheets, in random order. Please do not try to guess the source.
