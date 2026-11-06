# WP2: rate 21,984 generated English-Khmer pairs

Goal: give every short generated pair a quality score, so wrong ones can be set aside before training.
The 55 sheets hold 21,984 pairs (400 per sheet, the last has 384): signage, notices and app strings,
mixed in random order. Each pair is an English phrase and its Khmer translation, machine-generated.

The sheets are not in this repository. Download `khmer-wp2-judging-sheets.zip` from the link the person who sent you here gave you
(SHA-256 `958a69e6d774a7891a11daf56d48555ff60ed3a4aa9176626655fb6fb7c372e1`), unzip it into this directory so that `sheets/sheet-001.tsv` exists, and carry on.

## Your job, per sheet

1. Open `sheets/sheet-NNN.tsv` (tab separated: `id`, `en`, `km`).
2. Rate every row from 1 to 3:
   - **3**: the Khmer is a correct, natural translation of the English for a sign, label or app string.
   - **2**: understandable, but with a small error, odd wording or a missing detail.
   - **1**: wrong, garbled, unrelated, untranslated, or has stray Latin letters or markup stuck into the Khmer that the English does not call for.
3. Add a flag when one applies, otherwise leave it empty: `STRAY_LATIN`, `MISALIGNED`, `WRONG_TERM`, `UNTRANSLATED`, `OTHER`.
4. Write `ratings/ratings-NNN.tsv` with the header `id<TAB>score<TAB>flag<TAB>note` and one line per input id, in the same order. Keep each note under 15 words.
5. Check it: `python3 check_ratings.py sheets/sheet-NNN.tsv ratings/ratings-NNN.tsv`. Fix anything it reports.

The prompt to give an AI rater is in [PROMPT.md](PROMPT.md). You can do any number of sheets; start with `sheet-001`.

## Judging guidance

- Do not rewrite the Khmer. Only rate it.
- If you cannot tell, give 2 and put `unsure` in the note. Do not guess a 1.
- Both Western digits (0 to 9) and Khmer digits are fine. Riel amounts such as `៛5,000` or `5,000៛` are fine.
- Names, brand names and technical terms left in Latin letters are fine when the English contains them.
- Safety wording matters more than style: allergens, warnings, medicine and electrical notices that are wrong deserve a 1.
- Rate each row on its own. Do not try to work out where it came from.

## Which models to use

Use the strongest model you have, for example a current Claude Opus or Sonnet, a GPT-5-class model, or a Gemini Pro-class model.
In our own checks a small, fast Claude Haiku-class model and a Gemini Flash-class model were both usable judges, so a smaller model
is fine too. None of these judges is a native Khmer speaker. Anything you can kindly provide is welcome: please write the model and
version you used at the top of the message you send back.

## Returning results

Send the `ratings-NNN.tsv` files to the person who gave you this link. Do not open a pull request here.
The sheets are for rating only: please do not republish them or use them as training data.
