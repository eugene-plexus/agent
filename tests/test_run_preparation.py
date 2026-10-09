"""LS5: the run worker's `preparing` step (library-sources-and-engines.md §6.6)."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import httpx
import pytest

from eugene_plexus_agent.model_paths import PathRule
from eugene_plexus_agent.preparation import PreparationJobs, PreparationResult, Progress
from eugene_plexus_agent.routes import runtimes as actions
from eugene_plexus_agent.run_worker import NodeActions, RunWorker

GGUF = {
    "id": "g",
    "name": "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002",
    "path": "/models/ISTA-DASLab/repo/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
    "root": "/models",
    "format": "gguf",
}
PREPARE = {"modelId": "g", "preparation": {"engine": "strata"}}


def _verdict(engine, verdict, *, available=True, reason="runs it after preparing it"):
    return {"engine": engine, "verdict": verdict, "available": available, "reason": reason}


class _Library:
    """Answers the judge with given verdicts and records what else is asked."""

    def __init__(self, verdicts=(), *, refuse: tuple[int, str] | None = None):
        self.verdicts, self.refuse, self.asked = list(verdicts), refuse, []

    async def operation_request(self, method, path, **kwargs):
        self.asked.append((method, path, kwargs.get("json")))
        if path == "/v1/eligibility":
            return {"models": [{"modelId": "g", "engines": self.verdicts}]}
        if self.refuse:
            request = httpx.Request(method, "http://library" + path)
            status, detail = self.refuse
            response = httpx.Response(status, json={"detail": detail}, request=request)
            raise httpx.HTTPStatusError("refused", request=request, response=response)
        return {}


class _Engines(NodeActions):
    def __init__(self, app, *, strata_available=True, installable=True, progress=None):
        super().__init__(app)
        self.strata_available, self.installable = strata_available, installable
        self.progress, self.planned = progress, 0

    async def engines(self):
        return [
            {"engine": "llama_cpp", "available": True, "modelFormats": ["gguf"]},
            {
                "engine": "strata",
                "available": self.strata_available,
                "modelFormats": ["prepared"],
                "acquisition": {"installable": self.installable, "reason": "needs Windows"},
            },
        ]

    async def prepare(self, job):
        self.planned += 1
        return self.progress


def _job(step="checking", **extra):
    return {
        "id": "op",
        "step": step,
        "engine": "strata" if step != "checking" else None,
        "model": GGUF,
        "intent": PREPARE,
        "lease": "L",
        **extra,
    }


@pytest.mark.asyncio
async def test_a_preparation_goes_from_checking_to_preparing(app):
    library = _Library([_verdict("llama_cpp", "runs"), _verdict("strata", "after_preparation")])
    worker = RunWorker(app, node_actions=_Engines(app))
    # llama.cpp runs it as it is, and still the person asked for Strata's.
    assert await worker.advance(library, "/b", _job(), {}) == {
        "step": "preparing",
        "engine": "strata",
    }


@pytest.mark.asyncio
async def test_a_preparation_installs_its_engine_first(app):
    library = _Library([_verdict("strata", "after_preparation", available=False)])
    worker = RunWorker(app, node_actions=_Engines(app, strata_available=False))
    assert (await worker.advance(library, "/b", _job(), {}))["step"] == "awaiting-install"
    actions_ = _Engines(app, strata_available=True)
    worker = RunWorker(app, node_actions=actions_)
    done = await worker.advance(library, "/b", _job("installing"), {})
    assert done["step"] == "preparing" and done["install"] is None


@pytest.mark.asyncio
async def test_no_preparation_for_an_engine_that_cannot_prepare_it(app):
    library = _Library([_verdict("strata", "no", reason="prepares only the files on its list")])
    worker = RunWorker(app, node_actions=_Engines(app))
    with pytest.raises(ValueError, match=r"strata does not prepare .*its list"):
        await worker.advance(library, "/b", _job(), {})
    library = _Library([_verdict("strata", "after_preparation", available=False)])
    worker = RunWorker(app, node_actions=_Engines(app, strata_available=False, installable=False))
    with pytest.raises(ValueError, match="needs Windows"):
        await worker.advance(library, "/b", _job(), {})


@pytest.mark.asyncio
async def test_run_never_prepares_by_itself_and_names_the_action(app):
    library = _Library([_verdict("strata", "after_preparation")])
    worker = RunWorker(app, node_actions=_Engines(app))
    plain = _job(intent={"modelId": "g"})
    with pytest.raises(ValueError, match="strata can run it after preparing it: choose Prepare"):
        await worker.advance(library, "/b", plain, {})


@pytest.mark.asyncio
async def test_preparing_reports_its_progress(app):
    progress = Progress(state="running", step="Step 6 of 7: preparing", bytes_needed=8)
    library = _Library()
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    result = await worker.advance(library, "/b", _job("preparing"), {})
    assert result["step"] == "preparing"
    assert result["preparation"]["step"] == "Step 6 of 7: preparing"
    assert result["preparation"]["bytesNeeded"] == 8
    assert library.asked == []  # nothing listed until it is done


def _done(entry: Path) -> Progress:
    progress = Progress(state="done")
    progress.result = PreparationResult(
        entry=entry,
        name="qwen3.8-flash-next-iq2_xs",
        recipe="strata-prepare",
        recipe_version="v0.1.39",
        source={"path": GGUF["path"], "repoId": "ISTA-DASLab/repo"},
    )
    return progress


@pytest.mark.asyncio
async def test_done_lists_the_prepared_model_as_the_library_spells_it(app, monkeypatch, tmp_path):
    # The Library folder is /models in the library's container and a share here.
    local = tmp_path / "share"
    monkeypatch.setattr(
        actions, "effective_rules_for", lambda _c: [PathRule(source="/models", target=str(local))]
    )
    entry = local / "Strata-data" / "strata-iq2_xs.json"
    library = _Library()
    worker = RunWorker(app, node_actions=_Engines(app, progress=_done(entry)))
    params = {"node": "amish"}
    result = await worker.advance(library, "/v1/run-operations/op", _job("preparing"), params)
    assert result["step"] == "settings"
    method, path, body = library.asked[0]
    assert (method, path) == ("POST", "/v1/run-operations/op/prepared")
    assert body["lease"] == "L"
    assert body["name"] == "qwen3.8-flash-next-iq2_xs"
    assert body["provenance"]["entry"] == "/models/Strata-data/strata-iq2_xs.json"
    assert body["provenance"]["engine"] == "strata"
    assert body["provenance"]["recipe"] == "strata-prepare"
    assert body["provenance"]["source"]["path"] == GGUF["path"]


@pytest.mark.asyncio
async def test_a_failed_preparation_fails_the_run_with_its_cause(app):
    progress = Progress(state="failed", error="Strata's setup stopped: not enough RAM")
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    with pytest.raises(ValueError, match="not enough RAM"):
        await worker.advance(_Library(), "/b", _job("preparing"), {})


@pytest.mark.asyncio
async def test_the_librarys_refusal_to_list_it_is_the_cause(app, monkeypatch, tmp_path):
    monkeypatch.setattr(
        actions,
        "effective_rules_for",
        lambda _c: [PathRule(source="/models", target=str(tmp_path))],
    )
    progress = _done(tmp_path / "Strata-data" / "strata-iq2_xs.json")
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    library = _Library(refuse=(409, "x.eugene-prepared.json already exists and was not made"))
    with pytest.raises(ValueError, match="already exists and was not made"):
        await worker.advance(library, "/b", _job("preparing"), {})


@pytest.mark.asyncio
async def test_an_entry_outside_the_library_folder_is_not_sent(app, monkeypatch, tmp_path):
    monkeypatch.setattr(
        actions,
        "effective_rules_for",
        lambda _c: [PathRule(source="/models", target=str(tmp_path / "share"))],
    )
    progress = _done(tmp_path / "elsewhere" / "strata-iq2_xs.json")
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    library = _Library()
    with pytest.raises(ValueError, match="not under the Library folder"):
        await worker.advance(library, "/b", _job("preparing"), {})
    assert library.asked == []


@pytest.mark.asyncio
async def test_a_lost_checkpoint_after_listing_goes_on_without_preparing_again(app):
    node = _Engines(app, progress=Progress(state="running"))
    worker = RunWorker(app, node_actions=node)
    job = _job("preparing", preparedFrom=GGUF)
    assert await worker.advance(_Library(), "/b", job, {}) == {"step": "settings"}
    assert node.planned == 0


class _Slow:
    def __init__(self):
        self.output, self.bytes_needed = Path("."), None

    def run(self, progress):
        while True:
            progress.check_cancelled()
            time.sleep(0.01)


class _Listing:
    def __init__(self, operations):
        self.operations = operations

    async def operation_request(self, method, path, **kwargs):
        if path.endswith("/assigned"):
            return {"operations": self.operations}
        request = httpx.Request(method, "http://library" + path)
        raise httpx.HTTPStatusError(
            "gone", request=request, response=httpx.Response(409, request=request)
        )


@pytest.mark.asyncio
async def test_a_cancelled_operation_stops_its_preparation(app):
    node = NodeActions(app)
    progress = node.preparations.start("op", _Slow(), engine="strata")
    worker = RunWorker(app, node_actions=node)
    # Still preparing: kept.
    await worker.tick(_Listing([{"id": "op", "step": "preparing"}]), node=None)
    assert progress.state == "running"
    # Cancelled in the console: no longer assigned, so it stops.
    await worker.tick(_Listing([]), node=None)
    for _ in range(200):
        if progress.finished:
            break
        await asyncio.sleep(0.01)
    assert progress.state == "cancelled"


def test_uninstall_refuses_while_strata_prepares(authed_client):
    jobs = authed_client.app.state.preparations = PreparationJobs()
    gate = threading.Event()

    class _Held:
        output, bytes_needed = Path("."), None

        def run(self, progress):
            while not gate.wait(0.01):
                progress.check_cancelled()
            return None

    progress = jobs.start("op", _Held(), engine="strata")
    try:
        response = authed_client.post("/v1/engines/strata/uninstall")
        assert response.status_code == 409
        assert "preparing a model" in response.json()["detail"]["detail"]
    finally:
        gate.set()
        for _ in range(200):
            if progress.finished:
                break
            time.sleep(0.01)


@pytest.mark.asyncio
async def test_a_library_folder_this_node_cannot_reach_is_named(app, monkeypatch, tmp_path):
    async def refreshed(_context):
        return False

    monkeypatch.setattr(actions, "refresh_library_folders", refreshed)
    absent = tmp_path / "not-mounted"
    monkeypatch.setattr(
        actions, "effective_rules_for", lambda _c: [PathRule(source="/models", target=str(absent))]
    )
    with pytest.raises(ValueError, match=r"cannot reach the Library folder /models \(here "):
        await NodeActions(app).prepare(_job("preparing"))
