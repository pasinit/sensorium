---
name: commit-pr
description: Commit and pull-request hygiene — branching, conventional-commit messages, PR descriptions, and co-author trailers. Use when committing changes, writing a commit message, creating a branch, or opening/updating a pull request.
---

# Commit & PR hygiene

## Before committing

- Commit or push **only when asked**. If the user said to commit, but you're on
  the default branch (`main`/`master`), create a feature branch first.
- Branch names: `type/short-description` (e.g. `feat/oauth-login`,
  `fix/null-deref`).
- Never commit secrets, `.env` files, credentials, or large generated artifacts.
- Stage deliberately. Don't `git add -A` over unrelated changes; commit one
  logical change at a time.

## Commit messages

Use **Conventional Commits** for the subject line:

```
<type>(<optional scope>): <imperative summary>

<optional body: what changed and WHY, wrapped at ~72 cols>

<optional footers / trailers>
```

- **Types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `build`,
  `ci`.
- Subject in **imperative mood**, lowercase, no trailing period, ≤ ~72 chars.
  ("add retry to client", not "added" / "adds".)
- The body explains *why* and any non-obvious tradeoffs — not a restatement of
  the diff. Skip the body for trivial changes.
- When Claude authored the change, end the message with a co-author trailer:

  ```
  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

## Pull requests

Keep PRs focused — one concern per PR. Description template:

```markdown
## What
<one-paragraph summary of the change>

## Why
<the motivation / issue being solved; link the issue>

## How
<key implementation decisions reviewers should know>

## Testing
<what you ran / added, and the result>
```

- Link the issue it closes (`Closes #123`).
- End the PR body with:
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
- Use the `gh` CLI for PR operations. Before opening, run the `code-review`
  self-check.

## Dependencies & generated files

- Keep generated artifacts out of git — data, caches, model outputs, and run
  results belong under ignored paths (`data/`, `runs/`, `.tmp/`). Add them to
  `.gitignore`; don't commit them.
- Treat a lockfile change as a deliberate, separately reviewable change. After
  editing dependencies, regenerate the lock (`uv lock`) in its own commit — the
  lockfile diff is the review surface.
