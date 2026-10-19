---
name: literature-check
description: Check the published literature before committing to a sensorium design choice, hypothesis or experiment - search, read primary sources, verify every number, and record it in docs/literature-review.md and docs/references.md. Use before building anything with a design choice, before claiming novelty, and whenever a doc marks a claim (verify).
---

# Literature check

The supervisor's rule: **check the literature before committing, and don't
redo published experiments.** Every number we rely on comes from a primary
source we read, or from our own measurement.

## Procedure

1. **Frame the question.** One line: what are we deciding, and what would
   change our choice? (E.g. "What resolution and crop do robot world models
   use, and why?")
2. **Search broadly, then narrow.** Run several searches at once: the
   method's name, the problem phrased plainly, and the closest benchmark.
   Include the last 12 months. Look for released code, data and policies we
   could reuse instead of building.
3. **Read primary sources.** The paper (arXiv HTML is easiest to fetch), the
   official repo or model card. Blog posts and surveys only point to sources.
   Read past the abstract for numbers: tables, appendices, code.
4. **Verify every number** you will use: quote it with where it appears
   (section, table, file and line). If the source doesn't state it, say so
   ("the paper gives no image size") rather than repeating folklore.
5. **Answer the question.** What the field does, where it disagrees, what is
   still open, and the recommended default for us, with the reason.
6. **Record it** in the same PR as the change it justifies:
   - `docs/literature-review.md`: a numbered section "(checked YYYY-MM-DD)"
     with one bullet per source and the verified facts;
   - `docs/references.md`: an "Added for ... (YYYY-MM-DD)" list with titles
     and URLs;
   - the design doc (e.g. `docs/touch.md`, `docs/vision.md`): a "what the
     literature does" table with a "used here" column.

## Red flags

- Only abstracts read: mark the claim *(verify)*.
- A number from memory or a secondary source: verify it or drop it.
- "Nobody has done X" from one search: search again with other wording
  before telling the supervisor something is novel.
- A paper that already answers our hypothesis: stop and escalate; the plan
  may need to change (as with "acting beats watching" on 2026-10-11).
