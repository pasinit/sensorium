---
name: supervisor-report
description: Report to the sensorium supervisor (the user) - progress updates, decision requests, and end-of-step summaries in a fixed, short format with evidence and recommendations. Use whenever finishing a step, asking the supervisor to decide something, or answering "where are we".
---

# Reporting to the supervisor

The supervisor reads on a phone, often between other things. Lead with the
answer, keep it short, and make every decision request answerable in one word.

## Progress or end-of-step report

1. **One line:** what is done, and whether it worked.
2. **What changed:** at most five bullets, in plain words, each with its
   evidence (a number measured, a PR link, a file).
3. **Who did what:** which parts the PhDs did, which you did or fixed.
4. **What isn't good yet:** problems found, with numbers. Never bury these.
5. **Decisions I took:** each with a one-line reason; the supervisor can
   overrule.
6. **Ideas from the team:** PhD ideas worth the supervisor's attention,
   credited, with your assessment. Omit if none.
7. **Decisions I need:** see below; anything about direction or compute
   money always goes here. Omit the section if there are none.
8. **Next:** the next step, its scope and rough time, and whether you are
   starting it or waiting for a go.
9. **Link:** a page they can open on a phone when there is data or a long
   document to look at (publish as a private artifact).

## Results: always tables and loss curves

The supervisor's standing rule (2026-10-14): **every time results are
reported, show tables and loss curves.**

- **Tables:** every measured result as a table (conditions as rows, metrics
  as columns, mean ± spread and the number of seeds or episodes), not prose.
- **Loss curves:** for any training run, send a figure (`SendUserFile`, so
  it reaches the phone) with the training loss terms against steps, the
  held-out loss, and the held-out evaluation score against steps with the
  random-init floor as a line. Every seed is drawn. Data results (no
  training) get a table plus the relevant plot (frame strip, histogram).
- Figures come from the run logs (`runs/<study>/<run>/log.jsonl` and the
  probe files); say which runs they show and the training set size in the
  title.
- Briefs to the PhDs require the same figures in their reports, so they
  exist before you report.

## Decision request

For each decision:

- the question, in one sentence;
- the options (two or three), each with its cost, its risk and what the
  literature or our measurements say;
- **my recommendation**, first, with the reason;
- what happens if they don't answer (the reversible default you will take).

## Style

- Plain words; explain jargon on first use. Numbers with units.
- Say "measured", "verified in <source>" or "estimated" for every number.
- No hedging on things that are done and verified; no false confidence on
  things that are not.
- After reporting, update `docs/roadmap.md`: status column, decision log
  (date, decision, why) and open questions.
