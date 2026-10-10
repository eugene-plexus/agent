"""LS5: the run worker's `preparing` step (library-sources-and-engines.md §6.6).

Since LS10 (§6.13) a finished preparation's files go to the library, which
writes them into the Library folder: a node never does.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from pathlib import Path

import httpx
import pytest

from eugene_plexus_agent import preparation_send
from eugene_plexus_agent.model_paths import PathRule
from eugene_plexus_agent.preparation import PreparationJobs, PreparationResult, Progress
from eugene_plexus_agent.routes import runtimes as actions
from eugene_plexus_agent.run_worker import NodeActions, RunWorker, preparation_folder

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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Library:
    """Answers the judge with given verdicts, keeps the files a preparation
    sends as the library does (LS10), and records what is asked."""

    def __init__(self, verdicts=(), *, refuse: tuple[int, str] | None = None):
        self.verdicts, self.refuse, self.asked = list(verdicts), refuse, []
        self.held: dict[str, bytes] = {}
        self.partial: dict[str, bytes] = {}
        self.damage_once: set[str] = set()
        self.refuse_files: tuple[int, str] | None = None

    def _refused(self, method, path, status, detail):
        request = httpx.Request(method, "http://library" + path)
        response = httpx.Response(status, json={"detail": detail}, request=request)
        return httpx.HTTPStatusError("refused", request=request, response=response)

    async def operation_request(self, method, path, **kwargs):
        body = kwargs.get("json")
        params = kwargs.get("params") or {}
        self.asked.append((method, path, body if body is not None else dict(params)))
        if path == "/v1/eligibility":
            return {"models": [{"modelId": "g", "engines": self.verdicts}]}
        if "/files" in path:
            return self._file(method, path, body, params, kwargs.get("content"))
        if self.refuse:
            raise self._refused(method, path, *self.refuse)
        return {}

    def _file(self, method, path, body, params, content):
        if self.refuse_files:
            raise self._refused(method, path, *self.refuse_files)
        name = (body or params)["path"]
        if path.endswith("/files/state"):
            held = self.held.get(name)
            return {
                "path": name,
                "sizeBytes": None if held is None else len(held),
                "sha256": None if held is None else _sha(held),
                "receivedBytes": len(self.partial.get(name, b"")),
            }
        if method == "PUT":
            offset = int(params["offset"])
            have = self.partial.get(name, b"")
            if offset not in (0, len(have)):
                raise self._refused(method, path, 409, f"{len(have)} bytes have arrived")
            self.partial[name] = (b"" if offset == 0 else have) + content
            return {"path": name, "receivedBytes": len(self.partial[name])}
        data = self.partial.pop(name, None)
        if data is None:
            raise self._refused(method, path, 409, "Nothing has arrived")
        if name in self.damage_once:
            self.damage_once.discard(name)
            raise self._refused(method, path, 409, "arrived damaged")
        assert len(data) == body["sizeBytes"] and _sha(data) == body["sha256"]
        self.held[name] = data
        return {"path": name, "sizeBytes": len(data), "sha256": _sha(data), "receivedBytes": 0}

    def puts(self, name: str) -> list[int]:
        return [int(b["offset"]) for m, _p, b in self.asked if m == "PUT" and b["path"] == name]


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


PACK = b"P" * 2500
CONFIG = b'{"args": ["--pack", "packs/iq2_xs"]}'
MTP = b"M" * 700


def _done(work: Path) -> Progress:
    """A finished preparation in this node's folder `work` (LS10)."""
    data = work / "Strata-data"
    (data / "packs" / "iq2_xs").mkdir(parents=True, exist_ok=True)
    (data / "mtp" / "rt").mkdir(parents=True, exist_ok=True)
    entry = data / "strata-iq2_xs.json"
    entry.write_bytes(CONFIG)
    (data / "packs" / "iq2_xs" / "dense.bin").write_bytes(PACK)
    (data / "mtp" / "rt" / "experts.bin").write_bytes(MTP)
    files = (entry, data / "packs" / "iq2_xs" / "dense.bin", data / "mtp" / "rt" / "experts.bin")
    progress = Progress(state="done")
    progress.result = PreparationResult(
        root=work,
        files=files,
        discard=files[:2],
        entry=entry,
        name="qwen3.8-flash-next-iq2_xs",
        recipe="strata-prepare",
        recipe_version="v0.1.39",
        source={"path": GGUF["path"], "repoId": "ISTA-DASLab/repo"},
        facts={
            "title": "Qwen3.8-Flash-Next IQ2_XS",
            "contextLength": 131072,
            "files": [{"path": "mtp/rt/experts.bin", "sizeBytes": 20, "shared": True}],
        },
    )
    return progress


@pytest.mark.asyncio
async def test_done_sends_the_files_and_lists_the_model_as_the_library_spells_it(app, tmp_path):
    work = tmp_path / "node" / "preparing" / "strata" / "k"
    library = _Library()
    worker = RunWorker(app, node_actions=_Engines(app, progress=_done(work)))
    params = {"node": "amish"}
    result = await worker.advance(library, "/v1/run-operations/op", _job("preparing"), params)
    assert result["step"] == "settings"
    # B103/B104: every file, at its place in the Library folder, whole.
    assert library.held == {
        "Strata-data/strata-iq2_xs.json": CONFIG,
        "Strata-data/packs/iq2_xs/dense.bin": PACK,
        "Strata-data/mtp/rt/experts.bin": MTP,
    }
    assert all(p.startswith("/v1/run-operations/op/") for _m, p, _b in library.asked)
    assert all(b["lease"] == "L" for _m, _p, b in library.asked)
    assert all(b.get("node") == "amish" for m, _p, b in library.asked if m == "PUT")
    # B107: what was sent is gone from this node, the MTP helper kept.
    assert not (work / "Strata-data" / "strata-iq2_xs.json").exists()
    assert not (work / "Strata-data" / "packs" / "iq2_xs" / "dense.bin").exists()
    assert (work / "Strata-data" / "mtp" / "rt" / "experts.bin").is_file()
    method, path, body = library.asked[-1]
    assert (method, path) == ("POST", "/v1/run-operations/op/prepared")
    assert body["lease"] == "L"
    assert body["name"] == "qwen3.8-flash-next-iq2_xs"
    assert body["provenance"]["entry"] == "/models/Strata-data/strata-iq2_xs.json"
    assert body["provenance"]["engine"] == "strata"
    assert body["provenance"]["recipe"] == "strata-prepare"
    assert body["provenance"]["source"]["path"] == GGUF["path"]
    # What the engine's files say of it goes into the provenance (LS7b).
    assert body["provenance"]["title"] == "Qwen3.8-Flash-Next IQ2_XS"
    assert body["provenance"]["contextLength"] == 131072
    assert body["provenance"]["files"][0]["shared"] is True


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
    progress = _done(tmp_path / "work")
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    library = _Library(refuse=(409, "x.eugene-prepared.json already exists and was not made"))
    with pytest.raises(ValueError, match="already exists and was not made"):
        await worker.advance(library, "/b", _job("preparing"), {})


@pytest.mark.asyncio
async def test_an_entry_outside_the_nodes_folder_is_not_listed(app, tmp_path):
    progress = _done(tmp_path / "work")
    assert progress.result is not None
    elsewhere = tmp_path / "elsewhere" / "strata-iq2_xs.json"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(CONFIG)
    progress.result = PreparationResult(
        **{**progress.result.__dict__, "entry": elsewhere, "files": progress.result.files[1:]}
    )
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    library = _Library()
    with pytest.raises(ValueError, match="is not under"):
        await worker.advance(library, "/b", _job("preparing"), {})
    assert not [a for a in library.asked if a[1].endswith("/prepared")]


@pytest.mark.asyncio
async def test_a_send_is_a_slice_per_tick_and_carries_on_where_it_stopped(
    app, monkeypatch, tmp_path
):
    """B104: a claim lasts two minutes and a checkpoint lets it go, so each
    tick sends a slice and says how far it got; the library's count is the
    state, so a send cut off (an agent restart) carries on from it."""
    monkeypatch.setattr(preparation_send, "CHUNK_BYTES", 1000)
    monkeypatch.setattr(preparation_send, "SLICE_SECONDS", 0.0)
    library = _Library()
    library.held["Strata-data/mtp/rt/experts.bin"] = MTP  # another model's: not sent
    library.partial["Strata-data/packs/iq2_xs/dense.bin"] = PACK[:1000]  # an earlier tick's
    progress = _done(tmp_path / "work")
    node = _Engines(app, progress=progress)
    worker = RunWorker(app, node_actions=node)
    seen = []
    for _ in range(20):
        result = await worker.advance(library, "/b", _job("preparing"), {})
        if result["step"] != "preparing":
            break
        status = result["preparation"]
        assert status["step"] == "Sending the prepared model to the Library"
        seen.append(status["bytesWritten"])
    assert result["step"] == "settings"
    assert library.held["Strata-data/packs/iq2_xs/dense.bin"] == PACK
    assert library.puts("Strata-data/packs/iq2_xs/dense.bin") == [1000, 2000]
    assert library.puts("Strata-data/mtp/rt/experts.bin") == []
    # One file, then a chunk of the next (it carried on at 1000), then the rest.
    assert seen == [len(CONFIG), len(CONFIG) + 2000, len(CONFIG) + len(PACK)]
    assert node.senders == {}


@pytest.mark.asyncio
async def test_a_file_that_arrives_damaged_is_sent_again(app, tmp_path):
    library = _Library()
    library.damage_once.add("Strata-data/strata-iq2_xs.json")
    worker = RunWorker(app, node_actions=_Engines(app, progress=_done(tmp_path / "work")))
    result = await worker.advance(library, "/b", _job("preparing"), {})
    assert result["step"] == "settings"
    assert library.held["Strata-data/strata-iq2_xs.json"] == CONFIG
    assert library.puts("Strata-data/strata-iq2_xs.json") == [0, 0]


@pytest.mark.asyncio
async def test_the_librarys_refusal_of_a_file_fails_the_run_naming_it(app, tmp_path):
    library = _Library()
    library.refuse_files = (409, "Strata-data is not an engine's own folder, so it is yours")
    node = _Engines(app, progress=_done(tmp_path / "work"))
    worker = RunWorker(app, node_actions=node)
    with pytest.raises(ValueError) as refused:
        await worker.advance(library, "/b", _job("preparing"), {})
    assert "would not take the prepared file Strata-data/strata-iq2_xs.json" in str(refused.value)
    assert "it is yours" in str(refused.value)
    assert node.senders == {}


@pytest.mark.asyncio
async def test_a_failed_preparation_sends_its_log_and_names_the_librarys_copy(
    app, monkeypatch, tmp_path
):
    share = tmp_path / "share"
    monkeypatch.setattr(
        actions, "effective_rules_for", lambda _c: [PathRule(source="/models", target=str(share))]
    )
    log = tmp_path / "work" / "Strata-data" / "strata-iq2_xs.setup.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"=== Step 1: checking your PC ===\nTraceback: PermissionError\n")
    progress = Progress(state="failed", error="Strata's setup exited 1: PermissionError")
    progress.log_file, progress.log_name = log, "Strata-data/strata-iq2_xs.setup.log"
    library = _Library()
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    with pytest.raises(ValueError) as stopped:
        await worker.advance(library, "/b", _job("preparing"), {})
    assert library.held["Strata-data/strata-iq2_xs.setup.log"] == log.read_bytes()
    where = str(share) + ("\\" if "\\" in str(share) else "/") + "Strata-data"
    assert str(stopped.value).startswith(
        "Strata's setup exited 1: PermissionError (its whole output: " + where
    )
    assert str(stopped.value).endswith("strata-iq2_xs.setup.log)")


@pytest.mark.asyncio
async def test_a_log_the_library_would_not_take_is_named_where_it_is(app, tmp_path):
    log = tmp_path / "work" / "Strata-data" / "strata-iq2_xs.setup.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"x")
    progress = Progress(state="failed", error="Strata's setup exited 1: x")
    progress.log_file, progress.log_name = log, "Strata-data/strata-iq2_xs.setup.log"
    library = _Library()
    library.refuse_files = (409, "Run was cancelled, completed, or its lease expired")
    worker = RunWorker(app, node_actions=_Engines(app, progress=progress))
    with pytest.raises(ValueError) as stopped:
        await worker.advance(library, "/b", _job("preparing"), {})
    assert str(stopped.value).endswith(f"(its whole output: {log})")


def test_each_library_folder_has_its_own_folder_under_the_engines():
    one = preparation_folder("strata", "/models")
    assert one.parent.name == "strata" and one.parent.parent.name == "preparing"
    assert one == preparation_folder("strata", "/models")
    assert one != preparation_folder("strata", "/models2")


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


def test_uninstall_refuses_while_a_preparation_sends_its_files(authed_client):
    """LS10: a done job is kept until the library lists its model, and its
    folder must stay until then."""
    jobs = authed_client.app.state.preparations = PreparationJobs()

    class _Made:
        output, bytes_needed = Path("."), None

        def run(self, progress):
            return None

    progress = jobs.start("op", _Made(), engine="strata")
    for _ in range(200):
        if progress.finished:
            break
        time.sleep(0.01)
    assert progress.state == "done"
    response = authed_client.post("/v1/engines/strata/uninstall")
    assert response.status_code == 409


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
