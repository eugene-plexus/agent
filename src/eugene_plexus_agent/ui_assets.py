"""Where the browser half of an install comes from.

`agent.yaml` has said since v0.2 that the agent "serves the UI's static
assets at the root path". It never did — the agent repo has no `ui/`
and nothing was mounted — and the sentence survived four milestones
because no path item contradicted it. This module is the sentence
becoming true.

**The assets live in a separate distribution, `eugene-plexus-ui`,** and
are found through `importlib.resources` rather than a path in this
repo. Two reasons, and neither is packaging taste:

  * The agent is a Python package installed into a venv, per
    `watchdog-venv-is-runtime`. A wheel is the one artifact that
    arrives in that venv by the same mechanism as everything else, so
    `pip install eugene-plexus-ui` is the whole of "add a browser" and
    `pip uninstall` is the whole of removing one.
  * It keeps the build that produces them — Node, npm, `next build` —
    out of the runtime entirely. That is the payoff the whole install
    story hangs on: Node disappears from every platform.

    Precedent: Open WebUI ships its built frontend inside its Python
    wheel. Known-good, not invented here.

**Absence is a degradation, not a failure** (`degraded-mode-required`).
An install without the distribution serves its entire API normally,
answers `/healthz` 200, and says at `/` what to install. An operator
who deployed a headless node gets a working node; one who wanted a
browser gets a sentence naming the fix rather than a connection reset.

The `EUGENE_PLEXUS_AGENT_UI_DIR` override exists for development, where
the assets are `ui/out` in a checkout that is being rebuilt every few
seconds. Startup-only bootstrap, which is the sanctioned use of an env
var — there is nothing here a running install would want to change.
**Acceptance runs must not set it**, or they test the directory and not
the wheel, which is this project's recurring failure shape.
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from ._generated.common_models import Problem

log = logging.getLogger(__name__)

UI_DISTRIBUTION = "eugene-plexus-ui"
UI_PACKAGE = "eugene_plexus_ui"


class UIAssets:
    """The resolved location of the UI bundle, or the reason there isn't one."""

    def __init__(self, directory: Path | None, reason: str, source: str) -> None:
        self.directory = directory
        self.reason = reason
        """Human-readable: why there is no UI, or where the UI came from."""
        self.source = source
        """`package`, `override`, or `absent` — for logs and tests."""

    def __bool__(self) -> bool:
        return self.directory is not None


def _validate(directory: Path, source: str) -> UIAssets:
    """A directory is only a UI if it has an entry point in it.

    The distinction matters: "the package is not installed" and "the
    package is installed but its build produced nothing" are different
    problems with different fixes, and a mount over an empty directory
    reports neither — it 404s every route and looks like a routing bug.
    """
    if not directory.is_dir():
        return UIAssets(None, f"{source} {directory} is not a directory", "absent")
    if not (directory / "index.html").is_file():
        return UIAssets(
            None,
            f"{source} {directory} has no index.html — the UI build produced nothing",
            "absent",
        )
    return UIAssets(directory, str(directory), source)


def locate(override: Path | None = None) -> UIAssets:
    """Find the UI bundle: the override first, then the distribution."""
    if override is not None:
        return _validate(Path(override).expanduser().resolve(), "EUGENE_PLEXUS_AGENT_UI_DIR")

    try:
        module = import_module(UI_PACKAGE)
    except ImportError:
        return UIAssets(
            None,
            f"{UI_DISTRIBUTION} is not installed in this interpreter "
            f"(pip install {UI_DISTRIBUTION})",
            "absent",
        )

    static_dir = getattr(module, "static_dir", None)
    if static_dir is None:
        # A *function* is the contract, not a directory layout, so the
        # UI distribution can move its own files without this file
        # knowing. A build old enough to lack it is a real mismatch and
        # says so rather than guessing at a path that may not be there.
        return UIAssets(
            None,
            f"{UI_DISTRIBUTION} is installed but exposes no static_dir(); "
            "it is too old for this agent",
            "absent",
        )

    try:
        directory = Path(static_dir())
    except Exception as exc:  # pragma: no cover - defensive
        return UIAssets(None, f"{UI_DISTRIBUTION}.static_dir() failed: {exc}", "absent")
    return _validate(directory, UI_DISTRIBUTION)


UNAVAILABLE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eugene Plexus — no web UI installed</title>
<style>
  body {{ font: 15px/1.6 system-ui, sans-serif; margin: 0; padding: 3rem 1.5rem;
          background: #0f1115; color: #e6e8ee; }}
  main {{ max-width: 34rem; margin: 0 auto; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 1rem; }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  pre {{ background: #191d26; padding: .75rem 1rem; border-radius: 6px; overflow-x: auto; }}
  p {{ color: #aeb4c2; }}
  .why {{ color: #7f8797; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <h1>This Eugene Plexus agent has no web UI installed.</h1>
  <p>The API is running normally — only the browser half is missing.</p>
  <pre>pip install {distribution}</pre>
  <p>…into the same interpreter this agent runs from, then restart it.</p>
  <p class="why">{reason}</p>
</main>
</body>
</html>
"""


def unavailable_page(assets: UIAssets) -> str:
    return UNAVAILABLE_PAGE.format(distribution=UI_DISTRIBUTION, reason=assets.reason)


# Paths that belong to the API even when no route matched them. A
# catch-all mount at "/" reports Match.FULL for every path, and
# Starlette takes the first full match — so without this guard a wrong
# *method* on a real endpoint, or a typo under /v1/, would be answered
# with the UI's HTML 404 page instead of a Problem document. The status
# is still 404 rather than the 405 it was before the mount existed;
# that is the one behaviour a catch-all cannot preserve, and it is
# recorded here rather than discovered later.
_API_PREFIXES = ("/v1/", "/api/", "/healthz", "/openapi.json", "/docs", "/redoc")


class _UIStaticFiles(StaticFiles):
    """The UI bundle, refusing to answer for the API's paths."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        request_path = str(scope.get("path", ""))
        if request_path.startswith(_API_PREFIXES):
            raise HTTPException(
                status_code=404,
                detail=Problem(
                    type="https://github.com/eugene-plexus/agent#no-such-endpoint",
                    title="No such endpoint",
                    status=404,
                    detail=(f"The agent serves no endpoint at {request_path!r} for this method."),
                    component="agent",
                ).model_dump(exclude_none=True),
            )
        return await super().get_response(path, scope)


def mount(app: FastAPI, ui_dir: Path | None) -> None:
    """Serve the web UI, or explain at `/` why there isn't one."""
    assets = locate(ui_dir)
    app.state.ui_assets = assets

    if assets.directory is None:
        log.warning(
            "no web UI will be served: %s. The API is unaffected; install "
            "%s into this interpreter to get one.",
            assets.reason,
            UI_DISTRIBUTION,
        )

        @app.get("/", include_in_schema=False)
        async def ui_unavailable() -> HTMLResponse:
            return HTMLResponse(unavailable_page(assets), status_code=503)

        return

    log.info("serving the web UI from %s (%s)", assets.directory, assets.source)
    # `html=True` is what makes deep links work: the static export emits
    # one index.html per route, so `/nodes/` resolves to
    # `nodes/index.html` and `/nodes` redirects to `/nodes/` rather than
    # 404ing. A page that works when clicked and fails when pasted is
    # the symptom of getting this wrong.
    app.mount("/", _UIStaticFiles(directory=assets.directory, html=True), name="ui")
