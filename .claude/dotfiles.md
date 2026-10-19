# Engineering guidelines (dotfiles)

Reusable, project-agnostic guidelines vendored from
[dotfiles](https://github.com/pasinit/dotfiles). Project-specific context
belongs in the host project's own `CLAUDE.md`, which imports this file.

## Working style

- Prefer the smallest change that fully solves the problem. Match the
  surrounding code's naming, structure, and comment density.
- Read before you edit. Understand the existing pattern, then follow it.
- Keep functions focused. Split a function that grows past a screen or mixes
  layers (parsing + transforming + persisting) into named helpers.
- Don't add a dependency for something the stdlib or existing deps already do.
- When work is done and verified, say so plainly. If tests fail or a step was
  skipped, report it with the evidence — don't paper over it.
- Parallelise wherever possible. Run independent work concurrently — batch
  independent tool calls, fan out parallel subagents, and use concurrency in
  code — rather than one step at a time. Only serialize steps that genuinely
  depend on each other.

## Environment & workflows

- Use the repository-local environment (`.venv`), not a global one.
- Run tests, linting, formatting, and type-checking through **`make`** targets —
  `make test`, `make lint`, `make format`, `make typecheck` — never by calling
  `pytest`/`ruff`/`mypy` directly. If a needed target is missing, add it as part
  of the same change. (Fall back to the project's runner only if it's
  `just`/npm/`tox` instead of `make`.)
- Use the GitHub CLI (`gh`) for GitHub operations (remotes, auth, PRs, pushing).
- Keep generated artifacts out of git — data, caches, and run outputs belong
  under ignored paths (`data/`, `runs/`, `.tmp/`).

## Background & long-running tasks

If you start a script in the background or one that runs for a while, make its
progress observable — never leave a silent process the user can't follow:

- Redirect its output to a predictable log file (e.g. `.tmp/<task>.log`) and tell
  the user the command to watch it (`tail -f .tmp/<task>.log`).
- Emit progress as it runs — a step count, percentage, or at minimum a periodic
  heartbeat line — not just a single line at the end.
- Prefer the tool's managed background mechanism (one that streams output back and
  reports completion) over a detached `nohup … &` the user can't see.
- Surface the PID / job handle and how to stop it, so a runaway task can be killed.

## Python (≥ 3.11)

Details in the `python-conventions` skill. Essentials:

- **Never** add `from __future__ import annotations` (or any `__future__`
  import). Modern syntax is native in 3.11+.
- **Absolute imports only** — no relative imports (`from .x import y`).
- Type-hint public APIs. Use `X | None`, not `Optional[X]`. Avoid `object`/`Any`
  except where the value is genuinely dynamic.
- Every module and every public function/method gets a docstring describing the
  *contract* (inputs, side effects, errors raised), not a summary of the code.
- Format/lint with **ruff**, type-check with **mypy (strict)**, test with
  **pytest**, manage deps with **uv** — all configured in `pyproject.toml`.

## Quality gates (before handoff)

- Run `make lint`, `make typecheck`, and `make test` before handing off a change.
- The rules above are enforced by the kit-installed `[tool.ruff]` section in
  `pyproject.toml` and the vendored `.claude/mypy.ini` that those targets run
  against — not just by this prompt.
- Run gates **bare, never piped** — `cmd | tail` masks the exit code and lets a
  red suite slip through. Read the output after the bare run passes. CI is
  authoritative.

## Git & PRs

See `commit-pr` (conventional commits, branching, PR template) and `code-review`
(pre-push self-review). Commit/push only when asked; branch first if on the
default branch; never commit secrets.

## Skills in this kit

- `python-conventions` — tooling, layout, typing, docs, and style for Python.
- `commit-pr` — branching, commit message format, PR descriptions.
- `code-review` — pre-push self-review checklist.
