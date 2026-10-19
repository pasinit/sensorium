---
name: code-review
description: Pre-push self-review checklist for a code change — correctness, scope, tests, leftovers, and security. Use before opening or updating a pull request, before pushing, or whenever asked to review your own diff before sharing it.
---

# Pre-push self-review

Run this pass on your own diff before pushing or opening a PR. Read the actual
diff first (`git diff` / `git diff --staged`), then check each item.

## 1. Correctness

- Does the change do what was asked — the whole thing, edge cases included?
- Off-by-one, null/`None`, empty-collection, and boundary cases handled?
- Concurrency / ordering assumptions valid?
- Error paths: failures raise or are handled, not silently swallowed.

## 2. Scope

- The diff contains **only** what this change needs — no unrelated edits,
  reformatting churn, or drive-by renames mixed in.
- No commented-out code, leftover debug `print`/`logger.debug`, or `TODO`s you
  meant to resolve.
- No secrets, tokens, or local paths committed.

## 3. Tests

- New behavior has tests; fixed bugs have a regression test.
- Tests actually exercise the change (would fail without it).
- The suite passes locally. Lint and type-check are clean.
- Gates were run **bare, not piped** — `cmd | tail` masks the exit code.

## 4. Clarity & fit

- Names, structure, and comment density match the surrounding code.
- Imports are absolute — no relative imports (`from .` / `from ..`).
- Modules and public functions/methods have contract docstrings; types are
  precise (no stray `object`/`Any`, no `dict` where a dataclass fits).
- Layering respected: no lower-level module imports a higher one, and any new
  cross-module dependency edge is intentional, not casual.
- No coupling to a single concrete backend where the contract supports several
  (route through an adapter/registry instead).
- No needless new dependency for something the stdlib/existing deps cover.
- The simplest version that fully solves it — no speculative generality.

## 5. Report honestly

- If tests fail or a step was skipped, say so with the evidence.
- Note anything you changed that the reviewer wouldn't expect from the title.

For deeper, automated review, the host project may also offer `/code-review`.
This skill is the fast manual gate you run yourself first.
