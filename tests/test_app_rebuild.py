"""After a move to the Mac's daemon, apps are rebuilt, never carried (row 8).

The move carries an app's data and keys, never its environment, and lists
the app in `apps/rebuild.json`: a catalogue app is installed again from the
agent's own catalogue; one the person added is not started until it is
installed again by hand. And what runs an app, or a component, is checked
and starts with nothing imported from the folder it starts in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import apps, exec_trust
from eugene_plexus_agent._generated.models import AppOrigin, State1
from eugene_plexus_agent.supervisor import _ComponentPlanner

from .conftest import standalone_auth
from .test_apps import _fake_python, _manager, _manifest, _record
from .test_launch_boundary import _component


def _listed(manager: apps.AppManager, *ids: str) -> Path:
    marker = manager.store.root / apps.REBUILD_FILE
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"apps": list(ids)}), encoding="utf-8")
    return marker


def _catalogue_record(manifest: Any) -> apps.InstalledApp:
    record = _record(manifest)
    return apps.InstalledApp(
        manifest=record.manifest,
        origin=AppOrigin.catalogue,
        port=record.port,
        installed_at=record.installed_at,
    )


def test_a_moved_catalogue_app_is_installed_again_from_the_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shipped = _manifest()
    manager = _manager(tmp_path, catalogue=[shipped])
    manager.store.put(_catalogue_record(shipped))
    marker = _listed(manager, "tiny")
    monkeypatch.setattr(manager, "uv", lambda: Path("/usr/local/eugene-plexus/bin/uv"))
    started: list[Any] = []

    def start(manifest: Any, **kw: Any) -> None:
        started.append(manifest)

    async def wait(app_id: str) -> Any:
        return SimpleNamespace(state=State1.done, error=None)

    monkeypatch.setattr(manager.installer, "start", start)
    monkeypatch.setattr(manager.installer, "wait", wait)
    assert asyncio.run(manager.rebuild_moved()) == ["tiny"]
    assert [m.id for m in started] == ["tiny"] and started[0] is manager.catalogue["tiny"]
    assert not marker.exists()


def test_a_moved_app_the_person_added_is_not_installed_or_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The catalogue has an app by that id; the record says the person added
    # theirs, so it is theirs to install again, not the catalogue's.
    manager = _manager(tmp_path, catalogue=[_manifest()])
    manager.store.put(_record())  # origin: custom
    marker = _listed(manager, "tiny")
    monkeypatch.setattr(manager, "uv", lambda: Path("/usr/local/eugene-plexus/bin/uv"))
    started: list[Any] = []
    monkeypatch.setattr(manager.installer, "start", lambda manifest, **kw: started.append(manifest))
    assert asyncio.run(manager.rebuild_moved()) == []
    assert started == []
    assert json.loads(marker.read_text(encoding="utf-8")) == {"apps": ["tiny"]}
    asyncio.run(manager.start_enabled())
    assert manager.supervisor.started == []  # type: ignore[attr-defined]


def test_an_app_s_interpreter_is_checked_and_the_checked_path_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    record = _record()
    _fake_python(store, record)
    planner = apps._AppPlanner(
        record, store=store, gateway_url=lambda: None, bind_host=lambda: None
    )
    checked = Path("/usr/local/eugene-plexus/apps/tiny/versions/v1/venv/bin/python")
    monkeypatch.setattr(exec_trust, "trusted_program", lambda path, **kw: checked)
    plan = planner.base_plan()
    assert plan.argv[0] == str(checked)
    assert plan.env["PYTHONSAFEPATH"] == "1"

    def refuse(path: Any, **kw: Any) -> Path:
        raise exec_trust.Untrusted(
            f"{path} belongs to alice; Eugene's own account runs programs only"
        )

    monkeypatch.setattr(exec_trust, "trusted_program", refuse)
    with pytest.raises(apps.SpawnPlanError, match="belongs to alice"):
        planner.base_plan()


def test_a_component_imports_nothing_from_the_folder_it_starts_in() -> None:
    auth = standalone_auth(Path(tempfile.mkdtemp()), master_key=b"m" * 32)
    plan = _ComponentPlanner(_component("gateway"), logging.getLogger("test"), auth).plan()
    assert plan is not None
    assert plan.env["PYTHONSAFEPATH"] == "1"
    assert plan.argv[1:3] == ["-m", "eugene_plexus_gateway"]
