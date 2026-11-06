# Handoff generation (external model agent)

You are filling in model responses for Khmer pair generation. Do not run
`gen_pairs.py gen`, do not call Codex, Claude, or any other model API, and do
not use the web.

## Layout

The operator gives you a batch directory `D` with:

- `manifest.json` — list of cell keys to fill
- `prompts/<key>.txt` — one file per cell

Write your answers under `D/responses/<key>.json`.

## For each key in `manifest.json`

1. Open `prompts/<key>.txt`. It contains the same system and user messages
   `gen_pairs.py gen` would send:
   - text after `---SYSTEM---` is the system message
   - text after `---USER---` is the user message
2. Produce a JSON **array** of objects, each with `"en"` and `"km"` string
   fields, matching the format described in the user prompt.
3. Save **only** that JSON array to `responses/<key>.json` (no markdown fences,
   no commentary, no wrapper object).

## Quality

- Every `"km"` value must contain Khmer script characters.
- Follow the word-band, register, and numeric rules in the prompt.
- One response file per manifest key; use the exact key as the filename stem.

When finished, the operator runs `check_handoff.py` and
`gen_pairs.py import-responses` on your `responses/` directory.
