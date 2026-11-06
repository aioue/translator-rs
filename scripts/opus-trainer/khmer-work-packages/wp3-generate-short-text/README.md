# WP3: write short Khmer text for signs, labels, menus and notices

Goal: more realistic short English-Khmer pairs. Real data for these is almost non-existent, so we have asked AI models to write
it. We have about 38,500 rows and want more variety, in the registers with the fewest repeats (menus, notices, labels).

There are 707 prompts ("cells", each a situation such as allergen labels or vaccination notices, a surface form and a length band),
split into 29 batches: 25 prompts in batches 01 to 28 and 7 in batch 29. Every key is unique across batches, so fill the keys listed in each batch's `manifest.json` and nothing else. The batches are not in this repository: download `khmer-wp3-prompt-batches.zip` from this Google Drive folder: https://drive.google.com/drive/folders/1TepAqJr712lQUq-N0VC_8YEIrNnUxdBL?usp=sharing
(SHA-256 `45a5c1efcadb7d64aec67b4521105d3922e84122b7d0762edd33685795fd7bba`) and unzip it into this directory so that `batches/batch-01/manifest.json` exists. (Each prompt contains a list of text we already have, so you do not repeat it. Please do not republish the prompts.) Start with `batch-01`: it doubles as a pilot, and we will check
its output before asking you for more.

## Your job, per batch

1. In `batches/batch-NN/`, read `manifest.json` (the list of keys) and one `prompts/<key>.txt` per key.
2. For each key, produce a JSON array of objects with `"en"` and `"km"` string fields, exactly as the user prompt describes.
   The text after `---SYSTEM---` is the system message and the text after `---USER---` is the user message.
3. Save only that array to `batches/batch-NN/responses/<key>.json`: no markdown fences, no commentary, no wrapper object.
4. Write the model name and version you used, on one line, to `batches/batch-NN/responses/MODEL.txt`.
5. Check it from `scripts/opus-trainer/`:
   `python3 check_handoff.py --batch-dir khmer-work-packages/wp3-generate-short-text/batches/batch-NN --code km`
   Run it from `scripts/opus-trainer/`. Exit 0 means every response file is a JSON array of objects whose `km` contains Khmer and whose `en` is non-empty, with no markdown fences. It does not check the row count, the word band, distinctness or `MODEL.txt`, so follow the prompt yourself. Exit 1: a bad or unexpected file (the output lists `missing_keys` too). Exit 2: only keys are missing. Fix what it reports.

The full instructions an agent should follow are in [`../../D-templates/DELEGATE-INSTRUCTIONS.md`](../../D-templates/DELEGATE-INSTRUCTIONS.md).
Do not run `gen_pairs.py gen`, call a model API from a script, or use the web: just answer the prompts.

## Rules that matter

- Every `km` value must contain Khmer script, written the way it is printed in Cambodia: no spaces inside a phrase, no zero-width spaces.
- Follow the row count, word band, register and numeric rules in each prompt. Rows must be distinct.
- If you cannot write a row well, leave it out. Fewer good rows are better than padded ones.
- Do not copy rows from other batches or from your own earlier batches.

## Which models to use

Use the strongest model you have with good Khmer. Our baseline was a GPT-5.6-class model; we have not tested a Claude Opus or Sonnet,
a Gemini Pro or other models for Khmer here, so the first batch from a new model will be checked before we use more. Anything you can
kindly provide is welcome, including a smaller model: just record which one in `MODEL.txt`.

## Before you start

Check that your model's terms let you give its output to someone else to use as training data for a small translation model.
If you are not sure, please take WP1 or WP2 instead.

## Returning results

Send the `responses/` folders you filled in (zip them, one folder per batch) to the person who gave you this link. Do not open a pull request here.
What comes back is validated again on our side, judged by a second model, and only then used.
