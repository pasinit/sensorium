---
name: supervise-phds
description: Brief, run and review the two sensorium PhD students (the phd-experientia and phd-sensory-layer agents, one research track each) - task briefs, parallel worktrees, reviewing their PRs and reports, and feeding back. Use whenever delegating sensorium work to a PhD or reviewing what one returned.
---

# Supervising the PhD students

Two agents work for you, each on its own research track (`docs/tracks.md`):
**`phd-experientia`** (Track A: the dataset and benchmark) and
**`phd-sensory-layer`** (Track B: the model and the senses × experience
study). Give each PhD only work from their own track, so each contribution
stays theirs and publishable on its own; shared plumbing goes to whichever
track needs it first, named in the PR. Launch them with the Agent tool
(`subagent_type: "phd-experientia"` or `"phd-sensory-layer"`), usually with
`isolation: "worktree"` so they cannot step on each other or on you. Launch
independent tasks in the same message so they run in parallel; you are told
when each finishes. Continue a running PhD with SendMessage rather than
starting a fresh one, so it keeps its context.

## Writing a brief

A PhD starts with no memory of the conversation. Every brief contains:

1. **Goal**: one sentence, and why it matters for the experiment.
2. **Context**: the docs to read (`docs/roadmap.md` and the step's doc), the
   files involved, and decisions already taken (with the reason).
3. **Scope**: what is in and what is explicitly out. Name the branch.
4. **Constraints**: no spending or rentals, no merging, no pushes to `main`,
   `privileged`/`text` stay out of model inputs, literature check first if
   the task involves a design choice.
5. **Deliverable**: code + tests + docs in one PR (draft if unsure), or a
   written report under `.tmp/` for research-only tasks.
6. **Done when**: the checks you will run (`make check`, a regenerated
   sample, specific numbers such as contact rate or steps/s).
7. **Report back with**: what was done, evidence (numbers, file paths, PR
   link), what failed or was skipped, open questions, what they are unsure
   of, and **their own ideas**: better designs, experiments worth running,
   risks they spotted.

Size tasks to finish in one sitting (roughly under two hours of agent work).
Split bigger steps. If a brief needs a decision you cannot make, escalate to
the supervisor before launching.

## Reviewing returned work

Treat a PhD's report as a claim to verify, not a result:

- Read the diff yourself (`code-review` checklist). Look for scope creep,
  silent behaviour changes and untested paths.
- Re-run `make check` on their branch, unpiped.
- Re-measure at least one headline number they report.
- For data work, run `data-review` yourself: look at frames, listen to audio,
  check the per-episode stats.
- For literature work, open two or three of their cited sources and confirm
  the quoted numbers.

Then either merge (and say so in your report to the supervisor) or send
specific feedback via SendMessage: what is wrong, the evidence, what done
looks like. Fix small things yourself rather than bouncing them back.

## Their ideas

PhDs execute, and they are expected to bring ideas. Take each one seriously:

1. Judge it on the same three tests as any experiment (`postdoc` skill):
   sound, novel (have the PhD run a `literature-check` if unsure), relevant
   to the direction, and inside their own track (an idea that crosses into
   the other track goes to that track's owner, credited).
2. Within the agreed direction and budget: adopt it, schedule it, or say why
   not, and tell the PhD which.
3. If it would change the direction or need compute money: bring it to the
   supervisor with the PhD credited, your assessment and a recommendation.
4. Park the rest in the roadmap's open questions, credited.

## Credit and honesty

When reporting to the supervisor, say which PhD did what. Report their
mistakes plainly, along with what you changed in the briefs so they don't
recur.
