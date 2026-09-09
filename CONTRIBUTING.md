# Contributing to Eugene Plexus `agent`

Thanks for your interest. The agent is the outermost process of an Eugene Plexus install — it spawns and supervises the body components and serves the UI's static assets. It implements the `agent` OpenAPI contract from [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs); please read that contract before opening a PR.

## Developer Certificate of Origin (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/) instead of a CLA. **Every commit must be signed off** with `git commit -s`:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must match your `git config user.name` and `git config user.email`. CI blocks PRs whose commits are missing matching sign-offs.

If you forgot to sign off, fix the most recent commit:

```bash
git commit --amend -s --no-edit
```

…or for a whole branch:

```bash
git rebase --signoff main
```

The full DCO text is in [the specs CONTRIBUTING.md](https://github.com/eugene-plexus/specs/blob/main/CONTRIBUTING.md).

## Wire contract changes go in `specs`, not here

If your change touches the HTTP API — endpoints, request/response shapes, schemas — it belongs in [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs), not here. Land that PR first; bump `SPECS_REF` and re-run codegen here in a follow-up.

PRs to this repo should generally cover one or more of:

- **Implementation** of an existing spec endpoint (e.g. wiring up a stub).
- **Supervisor work** — the subprocess spawn / restart / signal-handling loop.
- **Reliability / safety** — clean shutdown, log rotation, error reporting.
- **Tooling** — CI, type-checking, lint config, codegen script.

## Local setup

```bash
git clone https://github.com/eugene-plexus/agent
cd agent
python -m venv .venv
. .venv/bin/activate           # or: .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Git hooks

We use [pre-commit](https://pre-commit.com/) to auto-format staged files with Ruff before they reach CI. Enable it once per clone:

```bash
pip install pre-commit
pre-commit install
```

After that, `git commit` runs `ruff check --fix` and `ruff format` on staged Python files; if a hook reformats anything, re-stage and commit again.

## Style

- **Python 3.12+** features are fine. We use `from __future__ import annotations` only where it materially helps.
- **Ruff** for lint and format. `ruff check .` and `ruff format .` should both be clean before you push. CI enforces.
- **Mypy strict** for type-checking. New code must type-check; the `_generated/` directory is excluded.
- **No comments explaining what code does** — let names do the work. Reserve comments for *why* a non-obvious choice was made.
- **Async-first.** Every route handler is `async`. The supervisor's process-management logic uses asyncio's subprocess primitives, not blocking `subprocess.Popen`.

## Running checks

```bash
ruff check .                       # lint
ruff format --check .              # formatting
mypy src/                          # type-check
pytest                             # tests
python scripts/codegen.py          # regenerate models from pinned specs
git diff --exit-code src/.../_generated/   # codegen freshness
```

## Reporting issues

File issues at <https://github.com/eugene-plexus/agent/issues>. Useful issues include:

- Concrete supervisor failures (a child crashed and wasn't respawned, etc.) with reproduction steps.
- Spec-vs-impl divergence (the impl drifted from `agent.yaml`).
- Resource-leak / shutdown-cleanliness regressions.

For broader architectural questions about Eugene Plexus, file the issue on the [orchestrator repo](https://github.com/eugene-plexus/orchestrator) instead.
