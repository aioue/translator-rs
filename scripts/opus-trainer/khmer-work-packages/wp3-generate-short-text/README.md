# WP3: write short Khmer text for signs, labels, menus and notices

Goal: more realistic short English-Khmer pairs. Real data for these is almost non-existent, so we have asked AI models to write
it. We have about 38,500 rows and want more variety, in the registers with the fewest repeats (menus, notices, labels).

There are 707 prompts ("cells", each a situation such as allergen labels or vaccination notices, a surface form and a length band),
split into 29 batches of 25. The batches are not in this repository: download `khmer-wp3-prompt-batches.zip` from the link the person who sent you here gave you
(SHA-256 `8313020e480e4b66d4a903217ac4d8d5d78fc0c5dcafc26c571f776f98f7d94a`) and unzip it into this directory so that `batches/batch-01/manifest.json` exists. (The prompts contain lists of text already written, which we share separately.) Batches 1 to 13 are cells nobody has filled yet, batch 14 is a mix, and later batches repeat cells that an
open-weights model already filled, so we get a second view of them. Start with `batch-01`: it doubles as a pilot, and we will check
its output before asking you for more.

## Your job, per batch

1. In `batches/batch-NN/`, read `manifest.json` (the list of keys) and one `prompts/<key>.txt` per key.
2. For each key, produce a JSON array of objects with `"en"` and `"km"` string fields, exactly as the user prompt describes.
   The text after `---SYSTEM---` is the system message and the text after `---USER---` is the user message.
3. Save only that array to `batches/batch-NN/responses/<key>.json`: no markdown fences, no commentary, no wrapper object.
4. Write the model name and version you used, on one line, to `batches/batch-NN/responses/MODEL.txt`.
5. Check it from `scripts/opus-trainer/`:
   `python3 check_handoff.py --batch-dir khmer-work-packages/wp3-generate-short-text/batches/batch-NN --code km`
   (exit 0 means every response is fine; 1 means a bad file; 2 means a key is missing). Fix what it reports.

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
