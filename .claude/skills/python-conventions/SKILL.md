---
name: python-conventions
description: Python project conventions for Python ≥ 3.11 — tooling (ruff, mypy strict, pytest, uv, make), project layout, typing, docstrings, implementation discipline, and style. Use when writing, refactoring, or reviewing Python code, setting up a Python project, or deciding on Python tooling/structure.
---

# Python conventions

Conventions for Python ≥ 3.11 projects. Apply unless the host project's own
`CLAUDE.md` overrides them.

## Hard rules

- **Never** add `from __future__ import annotations` (or any `__future__`
  import). The syntax it enables (`list[int]`, `X | None`, deferred eval) is
  native in 3.11+; adding it is noise and can change annotation runtime behavior.
- **Never** write a bare `except:` or `except Exception: pass`. Catch the
  narrowest exception that can actually be raised, then handle it or re-raise
  with context.
- **Never** commit secrets, `.env` files, or credentials.
- **Always use absolute imports.** No relative imports (`from .module import x`,
  `from ..pkg import y`) — import from the package root. ruff's `TID` rules
  (`ban-relative-imports = "all"`) enforce this.

## Tooling

Run tests, linting, formatting, and type-checking through **`make`** — never
call the underlying tools directly:

| Task       | Command          | Runs          |
| ---------- | ---------------- | ------------- |
| Test       | `make test`      | pytest        |
| Lint       | `make lint`      | ruff check    |
| Format     | `make format`    | ruff format   |
| Type-check | `make typecheck` | mypy (strict) |

If a target is missing, add it as part of your change. Fall back to `just`/npm
scripts/`tox` only when that's what the project already uses.

- **Enforcement config is kit-installed**: ruff rules live in the project's
  `pyproject.toml` under `[tool.ruff]` (appended by `install.sh`), mypy strict
  config is vendored as `.claude/mypy.ini`; the `make` targets run against them,
  so the rules below are enforced, not just advisory.
- **mypy runs in `strict` mode.**
- Tests live in `tests/`, named `test_*.py`.
- **Deps**: `uv` (`uv venv`, `uv sync`, `uv run`). Regenerate the lock
  (`uv lock`) as a deliberate, separately reviewed change — the lockfile diff is
  the review surface. Don't switch a project's package manager unasked.

### What ruff + mypy enforce

`make lint` runs ruff against `pyproject.toml`'s `[tool.ruff]`; `make typecheck`
runs mypy strict against `.claude/mypy.ini`. These turn the rules above into
real gates:

| Rule | Enforced by |
| ---- | ----------- |
| No `from __future__ import annotations` | ruff `TID251` (banned-api) |
| Absolute imports only | ruff `TID252` (`ban-relative-imports`) |
| No bare / blind except | ruff `E722`, `BLE001` |
| Modern syntax (`X \| None`, `list[int]`, f-strings) | ruff `UP` |
| `pathlib` over `os.path` | ruff `PTH` |
| No `print` (use logging) | ruff `T20` |
| Comprehensions | ruff `C4` |
| Module + public docstrings | ruff `D100`–`D103` |
| No `Any` in annotations | ruff `ANN401` |
| No lazy/optional runtime imports | ruff `PLC0415` |
| Keep functions focused | ruff `C901` (max-complexity 10) |
| Full, precise type coverage | mypy `strict` |

Prompt-only (no lint rule covers these): `Protocol` over `Callable` aliases,
dataclass over `dict`, adapter over hardcoded backend. **Layering** is enforced
per-project with **import-linter** when the project has package layers.

## Project layout

Prefer the `src/` layout; configure everything in `pyproject.toml` (no
`setup.py`/`setup.cfg` for new projects):

```
project/
├── pyproject.toml
├── src/mypackage/__init__.py
└── tests/test_*.py
```

## Typing

- **Don't type values as `object`.** Reach for a precise type, a generic type
  parameter, or a `Protocol`. Use `typing.Any` only where the value is genuinely
  dynamic (untyped JSON, pandas record payloads) and the looser type is the
  intended contract.
- **Express callable shapes with a `typing.Protocol`** that has an explicit
  `__call__`. Don't introduce `type Foo = Callable[..., X]` aliases — they hide
  the contract, make refactors harder, and bypass docstring enforcement.
- **Prefer generic type parameters** over `object`/`Any`/weakly typed
  collections. On Python ≥ 3.12 use PEP 695 syntax (`class Foo[T]: ...`,
  `def bar[T](...) -> T`, `class P[T](Protocol): ...`) so each entry point
  preserves the caller's concrete type.
- **Prefer a typed `@dataclass`** over a `dict[str, X]` with a fixed key set. If
  a function returns the same named fields every call, model it as a dataclass.
  Use `dataclasses.asdict` only at the serialization boundary, not as the
  in-process contract.
- Modern syntax throughout: `list[int]`, `X | None` (not `Optional`), `A | B`
  (not `Union`), `collections.abc` types (not deprecated `typing` aliases),
  `typing.Self`.

## Documentation

- Every module under `src/` starts with a module-level docstring: its purpose
  and the contract it owns. One paragraph is enough.
- Every public function and method has a docstring (ruff `D102` enforces
  methods; hold public module-level functions to the same bar).
- Docstrings describe the **contract** — what the caller can rely on, the side
  effects, the errors raised — not a restatement of the code.

## Implementation

- **No lazy/optional imports for runtime dependencies.** Declare every runtime
  dep and assume it is importable at module load. The only exception is a
  genuine optional extra: defer that import to the call site that needs it and
  raise `ModuleNotFoundError` with the exact `pip install` command when missing.
- **Don't couple code to one concrete backend.** When a contract supports
  several implementations (model frameworks, exchanges, storage), route through
  an adapter behind an enum or registry. Hardcoded backend names (e.g.
  `"sklearn"`) inside business logic are a smell.
- **Respect layering.** A lower-level module must never import a higher-level
  one. Adding a cross-module dependency edge is a deliberate, reviewable change.
- **Keep functions focused.** Split when one grows past a screen or starts
  mixing layers (parsing + transforming + persisting) into named helpers.

## Style

- **Strings**: f-strings only — never `%` or `str.format()`.
- **Paths**: `pathlib.Path`, not `os.path`.
- **Logging**: the `logging` module, not `print`, for diagnostics; use
  `logger.exception()` inside `except` blocks to capture tracebacks.
- **Resources**: context managers (`with`) for files, locks, connections.
- **Comprehensions** over `map`/`filter` + lambda when readable.
- **Constants**: module-level `UPPER_SNAKE_CASE`.

## Quality gates

- ruff clean (import/order/bug rules + public docstrings), mypy strict clean,
  pytest green.
- Run gates **bare, never piped**: `cmd | tail` masks the exit code, and red
  suites get shipped exactly that way. Read output after the bare run passes;
  CI is authoritative.
- Run lint, typecheck, and test before handing off a change.
