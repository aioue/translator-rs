# Prompt for an AI rater (copy everything below the line)

---

You are a blind rater of English-Khmer translation pairs. Read the file `sheets/sheet-NNN.tsv`
(tab separated: id, en, km; 400 rows; short phrases for signs, labels, notices and app screens).

For EVERY row give a score:
- 3 = the Khmer is a correct, natural translation of the English.
- 2 = understandable but with a minor error, odd wording or a missing detail.
- 1 = wrong, garbled, unrelated, untranslated, or the Khmer has Latin letters or markup glued in that the English does not call for.

Also give a flag from this list when it applies, else leave it empty: STRAY_LATIN, MISALIGNED (the Khmer is about something else),
WRONG_TERM (a key word is mistranslated), UNTRANSLATED, OTHER.

Rules: do not rewrite the Khmer. If you cannot tell, give 2 and write `unsure` in the note. Western and Khmer digits are both fine.
Names and brands left in Latin letters are fine when the English has them. Wrong safety wording (allergens, warnings, medicine, electrical) is a 1.

Write ONLY one file, `ratings/ratings-NNN.tsv`, with the header line `id<TAB>score<TAB>flag<TAB>note` and exactly one line per input id,
in input order, each note under 15 words. Then run
`python3 check_ratings.py sheets/sheet-NNN.tsv ratings/ratings-NNN.tsv` and fix anything it reports.
Report how many rows scored 1, 2 and 3. Do not edit any other file and do not use the network.
