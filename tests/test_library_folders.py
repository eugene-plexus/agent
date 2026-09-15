"""The Library's folders, as this node reaches them (2026-09-14).

The claims: a node inherits, for each Library folder, the mount of its
own OS shape as a rule and nothing for a folder with no mount of that
shape; its own `pathMappings` override the inherited rule for the same
folder and nothing else; a declared model under no Library folder is a
400 with the remedy, before any companion is declared, and `force` does
not bypass it; a node that has never read the folder list skips the
check with a warning rather than refusing; the copy survives a restart
with its age; the check endpoint says which rule applied per folder;
and an override whose `from` is no Library folder is rejected at PATCH.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent.library_folders import (
    FolderRecord,
    LibraryFolderCache,
    check_reach,
    effective_rules,
    not_in_library_detail,
    parse_folders,
    refresh,
)
from eugene_plexus_agent.model_paths import PathRule, resolve_model_path

MODELS = FolderRecord(path="/models", mounts=("/mnt/models", "\\\\NAS\\models"))
ARCHIVE = FolderRecord(path="/archive", mounts=("/mnt/archive",))
LOCAL = FolderRecord(path="D:\\local-models")


class _FakeLibrary:
    """Anything with `folders()` (and `list_models()` for the check)."""

    base_url = "http://library.test"

    def __init__(
        self, folders: list[dict[str, Any]] | None, models: list[dict[str, Any]] | None = None
    ):
        self._folders = folders
        self._models = models or []
        self.asked = 0

    async def folders(self) -> list[dict[str, Any]] | None:
        self.asked += 1
        return self._folders

    async def list_models(self) -> list[dict[str, Any]] | None:
        return self._models

    async def fit(self, model_path: str, **kwargs: Any) -> None:
        return None


def _cache(tmp_path: Path, *folders: FolderRecord, windows: bool) -> LibraryFolderCache:
    cache = LibraryFolderCache(tmp_path / "library_folders.json", windows=windows)
    cache.update(list(folders), library_url="http://library.test")
    return cache


# --- inheritance --------------------------------------------------------------


def test_a_windows_node_inherits_the_windows_shaped_mount_and_a_posix_node_the_other(
    tmp_path: Path,
) -> None:
    windows = _cache(tmp_path / "w", MODELS, windows=True)
    posix = _cache(tmp_path / "p", MODELS, windows=False)

    assert windows.inherited_rules() == [PathRule(source="/models", target="\\\\NAS\\models")]
    assert posix.inherited_rules() == [PathRule(source="/models", target="/mnt/models")]
    assert (
        resolve_model_path("/models/q/x.gguf", windows.inherited_rules()).local_path
        == "\\\\NAS\\models\\q\\x.gguf"
    )


def test_a_posix_decoy_is_not_picked_on_windows_whatever_its_position(tmp_path: Path) -> None:
    """The shape decides, not the order: the first entry is POSIX here."""
    cache = _cache(tmp_path, MODELS, ARCHIVE, windows=True)

    rules = cache.inherited_rules()

    assert rules == [PathRule(source="/models", target="\\\\NAS\\models")]
    # `/archive` has no Windows-shaped mount, so a Windows node opens it as written.
    assert resolve_model_path("/archive/a.gguf", rules).local_path == "/archive/a.gguf"


def test_a_folder_with_no_mount_of_this_shape_yields_no_rule(tmp_path: Path) -> None:
    cache = _cache(tmp_path, LOCAL, windows=False)
    assert cache.inherited_rules() == []
    assert cache.folder_for("D:\\local-models\\q.gguf") is LOCAL


def test_a_mount_equal_to_the_folders_own_path_is_the_identical_mount_convention(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path, FolderRecord("/models", ("/models",)), windows=False)
    assert cache.inherited_rules() == []


def test_an_override_wins_over_the_inherited_rule_for_the_same_folder_only(tmp_path: Path) -> None:
    cache = _cache(tmp_path, MODELS, ARCHIVE, windows=False)
    override = PathRule(source="/models", target="/srv/elsewhere")

    rules = effective_rules([override], cache.inherited_rules())

    assert rules == [override, PathRule(source="/archive", target="/mnt/archive")]
    assert resolve_model_path("/models/q.gguf", rules).local_path == "/srv/elsewhere/q.gguf"
    assert resolve_model_path("/archive/q.gguf", rules).local_path == "/mnt/archive/q.gguf"


def test_folder_for_matches_by_component_and_prefers_the_deepest(tmp_path: Path) -> None:
    deep = FolderRecord("/models/gguf")
    cache = _cache(tmp_path, MODELS, deep, windows=False)

    assert cache.folder_for("/models/gguf/q.gguf") is deep
    assert cache.folder_for("/models/other/q.gguf") is MODELS
    assert cache.folder_for("/models2/q.gguf") is None
    assert cache.folder_for("D:\\elsewhere\\q.gguf") is None


# --- the copy -----------------------------------------------------------------


def test_the_copy_survives_a_restart_with_its_age(tmp_path: Path) -> None:
    first = LibraryFolderCache(tmp_path / "library_folders.json", windows=True)
    assert not first.known and first.age_seconds() is None
    first.update([MODELS], library_url="http://library.test", now=1000.0)

    second = LibraryFolderCache(tmp_path / "library_folders.json", windows=True)
    second.load()

    assert second.known
    assert second.folders == [MODELS]
    assert second.library_url == "http://library.test"
    assert second.age_seconds(now=1090.0) == 90
    written = json.loads((tmp_path / "library_folders.json").read_text(encoding="utf-8"))
    assert written["folders"] == [{"path": "/models", "mounts": ["/mnt/models", "\\\\NAS\\models"]}]


def test_a_malformed_copy_is_unknown_not_an_error(tmp_path: Path) -> None:
    (tmp_path / "library_folders.json").write_text("{not json", encoding="utf-8")
    cache = LibraryFolderCache(tmp_path / "library_folders.json")
    cache.load()
    assert not cache.known


def test_parsing_takes_strings_objects_and_skips_junk() -> None:
    assert parse_folders(["/a", {"path": "/b", "mounts": ["/m", 3]}, 7, {"mounts": []}]) == [
        FolderRecord("/a"),
        FolderRecord("/b", ("/m",)),
    ]


@pytest.mark.anyio
async def test_refresh_holds_an_answer_and_keeps_the_copy_when_there_is_none(
    tmp_path: Path,
) -> None:
    cache = LibraryFolderCache(tmp_path / "f.json", windows=False)

    assert await refresh(cache, None) is False
    assert await refresh(cache, object()) is False  # nothing with `folders()`
    assert await refresh(cache, _FakeLibrary([{"path": "/models", "mounts": ["/mnt/models"]}]))
    assert cache.folders == [FolderRecord("/models", ("/mnt/models",))]
    assert await refresh(cache, _FakeLibrary(None)) is False
    assert cache.folders == [FolderRecord("/models", ("/mnt/models",))]


# --- the check ----------------------------------------------------------------


def test_the_check_names_the_rule_that_applied_per_folder(tmp_path: Path) -> None:
    cache = _cache(tmp_path, MODELS, ARCHIVE, LOCAL, windows=False)
    override = PathRule(source="/archive", target="/srv/archive")
    models = [{"path": "/models/a.gguf"}, {"path": "/models/b.gguf"}, {"path": "/archive/c.gguf"}]
    present = {"/mnt/models", "/mnt/models/a.gguf", "/srv/archive", "/srv/archive/c.gguf"}

    reach = check_reach(
        cache,
        [override],
        models,
        library_consulted=True,
        exists=lambda p: p in present,
        isdir=lambda p: not p.endswith(".gguf"),
    )

    by_path = {row.path: row for row in reach.folders}
    assert reach.libraryConsulted is True and reach.folderListAgeSeconds == 0
    inherited = by_path["/models"]
    assert inherited.source.value == "inherited"
    assert inherited.localPath == "/mnt/models" and inherited.mount == "/mnt/models"
    assert inherited.exists is True and inherited.isDirectory is True
    assert (inherited.modelsUnder, inherited.modelsReachable) == (2, 1)
    assert (
        inherited.problem
        == "1 of 2 library models under /models are not at their resolved path here"
    )
    overridden = by_path["/archive"]
    assert overridden.source.value == "override"
    assert overridden.override is not None and overridden.override.to == "/srv/archive"
    assert (overridden.modelsUnder, overridden.modelsReachable) == (1, 1)
    assert overridden.problem is None
    same = by_path["D:\\local-models"]
    assert same.source.value == "same_path"
    assert same.exists is False and same.isDirectory is None
    assert same.problem == "D:\\local-models does not exist on this host"


def test_the_check_without_the_library_reports_the_copys_age_and_no_model_counts(
    tmp_path: Path,
) -> None:
    cache = LibraryFolderCache(tmp_path / "f.json", windows=False)
    cache.update([MODELS], library_url=None, now=500.0)

    reach = check_reach(
        cache,
        [],
        None,
        library_consulted=False,
        exists=lambda p: True,
        isdir=lambda p: True,
        now=560.0,
    )

    assert reach.libraryConsulted is False and reach.folderListAgeSeconds == 60
    assert reach.folders[0].modelsUnder is None and reach.folders[0].problem is None


# --- the routes ----------------------------------------------------------------


def _install_library(
    client: TestClient, folders: list[dict[str, Any]] | None, **kw: Any
) -> _FakeLibrary:
    library = _FakeLibrary(folders, **kw)
    client.app.state.library_fit_client = library  # type: ignore[attr-defined]
    return library


def _spec(path: str, name: str = "m") -> dict[str, Any]:
    return {"name": name, "engine": "llama_cpp", "modelPath": path}


def test_a_model_under_no_library_folder_is_a_400_with_the_remedy_and_no_companion(
    authed_client: TestClient,
) -> None:
    _install_library(authed_client, [{"path": "/models", "mounts": []}])

    response = authed_client.post("/v1/runtimes", json=_spec("/elsewhere/q.gguf", "stray"))

    assert response.status_code == 400
    body = response.json()["detail"]
    assert body["title"] == "Not a Library model"
    assert body["detail"] == not_in_library_detail("/elsewhere/q.gguf")
    assert "Library -> Folders" in body["detail"]
    assert authed_client.get("/v1/runtimes").json()["runtimes"] == []
    names = [c["name"] for c in authed_client.get("/v1/components").json()["components"]]
    assert "stray-driver" not in names


def test_force_does_not_bypass_the_folder_rule(authed_client: TestClient) -> None:
    _install_library(authed_client, [{"path": "/models", "mounts": []}])
    response = authed_client.post(
        "/v1/runtimes?force=true", json=_spec("/elsewhere/q.gguf", "stray")
    )
    assert response.status_code == 400


def test_a_model_under_a_library_folder_is_declared_and_opens_at_the_inherited_mount(
    authed_client: TestClient, tmp_path: Path
) -> None:
    cache = authed_client.app.state.library_folders  # type: ignore[attr-defined]
    cache._windows = False  # this test is about a POSIX node whatever the desk runs
    _install_library(authed_client, [{"path": "/models", "mounts": ["/mnt/models", "Z:\\models"]}])

    response = authed_client.post("/v1/runtimes", json=_spec("/models/q.gguf", "kept"))

    assert response.status_code == 201, response.text
    assert response.json()["modelPath"] == "/models/q.gguf"
    # The copy was refreshed by the request and written beside agent.yaml.
    assert cache.known and cache.folders[0].path == "/models"
    assert cache.path.exists()


def test_an_update_to_a_path_outside_the_library_is_refused_too(authed_client: TestClient) -> None:
    _install_library(authed_client, [{"path": "/models", "mounts": []}])
    assert (
        authed_client.post("/v1/runtimes", json=_spec("/models/q.gguf", "kept")).status_code == 201
    )

    response = authed_client.patch("/v1/runtimes/kept", json=_spec("/elsewhere/q.gguf", "kept"))

    assert response.status_code == 400
    assert authed_client.get("/v1/runtimes/kept").json()["modelPath"] == "/models/q.gguf"


def test_a_node_that_has_never_read_the_folders_launches_with_a_warning(
    authed_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """No library in this topology and none through the install: M11's
    behaviour, said out loud."""
    caplog.set_level(logging.WARNING, logger="eugene_plexus_agent.routes.runtimes")

    response = authed_client.post("/v1/runtimes", json=_spec("/anywhere/q.gguf", "faith"))

    assert response.status_code == 201
    assert "was not checked against the Library's folders" in caplog.text


def test_the_check_endpoint_reports_per_folder_and_takes_unsaved_overrides(
    authed_client: TestClient, tmp_path: Path
) -> None:
    mount = tmp_path / "mnt"
    mount.mkdir()
    cache = authed_client.app.state.library_folders  # type: ignore[attr-defined]
    cache._windows = str(mount).startswith("\\\\") or str(mount)[1:3] == ":\\"
    _install_library(
        authed_client,
        [{"path": "/models", "mounts": [str(mount)]}, {"path": "/archive", "mounts": []}],
        models=[{"path": "/models/q.gguf"}],
    )

    saved = authed_client.post("/v1/library/folders/check", json={}).json()

    rows = {row["path"]: row for row in saved["folders"]}
    assert saved["libraryConsulted"] is True
    assert rows["/models"]["source"] == "inherited"
    assert rows["/models"]["localPath"] == str(mount)
    assert rows["/models"]["exists"] is True and rows["/models"]["isDirectory"] is True
    assert rows["/models"]["modelsUnder"] == 1 and rows["/models"]["modelsReachable"] == 0
    assert rows["/archive"]["source"] == "same_path" and rows["/archive"]["exists"] is False

    unsaved = authed_client.post(
        "/v1/library/folders/check",
        json={"pathMappings": [{"from": "/archive", "to": str(tmp_path)}]},
    ).json()
    archive = next(row for row in unsaved["folders"] if row["path"] == "/archive")
    assert archive["source"] == "override" and archive["exists"] is True
    assert archive["override"] == {"from": "/archive", "to": str(tmp_path)}
    # Nothing was saved by asking.
    assert authed_client.get("/v1/config").json()["pathMappings"] == []


def test_an_override_for_a_directory_that_is_no_library_folder_is_rejected_at_patch(
    authed_client: TestClient,
) -> None:
    _install_library(authed_client, [{"path": "/models", "mounts": []}])

    result = authed_client.patch(
        "/v1/config",
        json={
            "pathMappings": [{"from": "/srv/models", "to": "Z:\\models"}],
            "firstRunComplete": True,
        },
    ).json()

    assert result["applied"] == ["firstRunComplete"]
    assert result["rejected"][0]["key"] == "pathMappings"
    assert "'/srv/models' is not a Library folder" in result["rejected"][0]["message"]
    assert authed_client.get("/v1/config").json()["pathMappings"] == []

    good = authed_client.patch(
        "/v1/config", json={"pathMappings": [{"from": "/models", "to": "Z:\\models"}]}
    ).json()
    assert good["applied"] == ["pathMappings"] and good["rejected"] == []


def test_the_labels_say_library(authed_client: TestClient) -> None:
    schema = authed_client.get("/v1/config/schema").json()
    field = next(f for f in schema["fields"] if f["key"] == "pathMappings")
    assert field["label"] == "Library folder overrides"
    assert schema["categories"][field["category"]] == "Library"
