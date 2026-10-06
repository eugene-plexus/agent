"""The site host on a node (job-sites-own-enrollment.md, J19, J21, J23): the
agent installs and keeps it when an administrator turned it on here, tells
it only what the agent may say, says which site it hosts (J32), and holds
none of the site's identity. `site join` runs the site host's own join."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_accounts, site_cli, site_host
from eugene_plexus_agent._generated.models import AppOrigin
from eugene_plexus_agent.apps import InstalledApp, is_site_host
from eugene_plexus_agent.site_host import HELPER_ID, RETIRED_ID, SiteHostSupervisor


def test_the_host_is_told_what_only_the_agent_may_say(app: FastAPI) -> None:
    """The install's private directory, the links and the local servers, where
    they live; never who the site is or who owns it, which is the site's own
    enrollment. The agent no longer copies the server list for the host (2b.2):
    the host reads the protected file itself, and is told when it changes."""
    supervisor = SiteHostSupervisor(app)
    folder = supervisor.config_dir / "site"
    plain = supervisor.environment("user")
    assert set(plain) == {
        "SITE_HOST_PROTECTED_ROOTS",
        "SITE_HOST_LINKS_FILE",
        "SITE_HOST_LOCAL_SERVERS_FILE",
        "SITE_HOST_CHANNEL",
        "SITE_HOST_SERVERS_STAMP",
    }
    assert plain["SITE_HOST_LINKS_FILE"] == str(folder / "links.json")
    assert plain["SITE_HOST_LOCAL_SERVERS_FILE"] == str(folder / "servers.yaml")
    assert plain["SITE_HOST_SERVERS_STAMP"] == ""  # no list yet
    assert plain["SITE_HOST_CHANNEL"] == site_host.channel_name(supervisor.config_dir)
    # A long list, and braces the launcher would otherwise read as a placeholder.
    servers = [
        {
            "id": f"tool-{n}",
            "name": f"Tool {n}",
            "command": sys.executable,
            "args": ["{config}", "x" * 1500],
            "sha256": "a" * 64,
        }
        for n in range(4)
    ]
    folder.mkdir()
    listed = folder / "servers.yaml"
    listed.write_text(yaml.safe_dump({"servers": servers}), encoding="utf-8")
    site = supervisor.environment("user")
    # Not a copy: the very file, and a stamp that moves when it does.
    assert Path(site["SITE_HOST_LOCAL_SERVERS_FILE"]) == listed
    assert site["SITE_HOST_SERVERS_STAMP"] == hashlib.sha256(listed.read_bytes()).hexdigest()
    listed.write_text(yaml.safe_dump({"servers": servers[:1]}), encoding="utf-8")
    assert (
        supervisor.environment("user")["SITE_HOST_SERVERS_STAMP"] != site["SITE_HOST_SERVERS_STAMP"]
    )
    assert json.loads(site["SITE_HOST_PROTECTED_ROOTS"]) == [str(supervisor.config_dir)]
    assert not any("OWNER" in key or "MODE" in key for key in site)
    assert not any("SHA256" in key for key in site), "the old copy's hash is gone"
    assert "SITE_HOST_LINK_PAGE" not in site, "no link page unless a service install offers it"


def test_the_site_host_runs_only_when_an_administrator_turned_it_on(tmp_path: Path) -> None:
    assert site_host.wanted(tmp_path) is False
    site_host.set_wanted(tmp_path, True)
    assert site_host.wanted(tmp_path) is True
    site_host.set_wanted(tmp_path, False)
    assert site_host.wanted(tmp_path) is False
    (tmp_path / site_host.WANTED_FILE).write_text("{not json", encoding="utf-8")
    assert site_host.wanted(tmp_path) is False


async def test_unsupported_never_installs_and_the_host_is_not_an_app(
    app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.apps
    manager.accounts = app_accounts.AccountSupport(
        None, "Isolated accounts are not available here."
    )
    install = AsyncMock()
    monkeypatch.setattr(manager.installer, "start", install)
    supervisor = SiteHostSupervisor(app)
    try:
        await supervisor.reconcile(True)
        install.assert_not_called()
        for app_id in (HELPER_ID, RETIRED_ID):
            for suffix in ["install", "start", "stop", "restart"]:
                response = authed_client.post(f"/v1/apps/{app_id}/{suffix}")
                assert response.status_code == 409 and "site join" in response.text
    finally:
        await supervisor._host.aclose()
    assert is_site_host(HELPER_ID, site_host.ENTRY)
    assert is_site_host(RETIRED_ID, "eugene_plexus_node_helper")
    assert not is_site_host("workbench", site_host.ENTRY)
    record = InstalledApp(
        manifest=site_host.manifest({}),
        origin=AppOrigin.catalogue,
        port=8300,
        installed_at=datetime.now(UTC),
    )
    manager.store.put(record)
    assert all(v.id != HELPER_ID for v in manager.views())
    assert authed_client.delete(f"/v1/apps/{HELPER_ID}").status_code == 409


async def test_turned_on_it_installs_and_slice_2s_host_is_removed(
    app: FastAPI, authed_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = app.state.apps
    manager.accounts = app_accounts.AccountSupport("windows_service", None)
    started: list[Any] = []
    monkeypatch.setattr(manager.installer, "start", lambda want, **_: started.append(want))
    monkeypatch.setattr(manager, "uv", lambda: Path(sys.executable))
    removed: list[str] = []

    async def uninstall(app_id: str, *, purge: bool = False) -> None:
        removed.append(app_id)

    monkeypatch.setattr(manager, "uninstall", uninstall)
    old = InstalledApp(
        manifest=site_host.manifest({}).model_copy(update={"id": RETIRED_ID}),
        origin=AppOrigin.catalogue,
        port=8301,
        installed_at=datetime.now(UTC),
    )
    manager.store.put(old)
    supervisor = SiteHostSupervisor(app)
    try:
        await supervisor.reconcile(True)
    finally:
        await supervisor._host.aclose()
    assert removed == [RETIRED_ID]
    assert [want.id for want in started] == [HELPER_ID]
    assert started[0].name == "Job site" and started[0].entry == site_host.ENTRY


async def test_the_node_says_which_site_it_hosts(app: FastAPI) -> None:
    supervisor = SiteHostSupervisor(app)
    sent: list[tuple[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.content)))
        return httpx.Response(204)

    app.state.node_identity = SimpleNamespace(
        record=SimpleNamespace(enrolled=True, control_url="http://root", name="amish")
    )
    app.state.auth_state = SimpleNamespace(trust=SimpleNamespace(agent_token=lambda audience: "t"))
    supervisor._root = "http://root"
    supervisor._root_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        supervisor.site = "s-" + "a" * 26
        await supervisor._report()
        await supervisor._report()  # unchanged: not sent again
        supervisor.site = None
        await supervisor._report()
    finally:
        await supervisor._root_client.aclose()
        await supervisor._host.aclose()
    assert sent == [
        ("/v1/nodes/amish/hosted-sites", {"sites": ["s-" + "a" * 26]}),
        ("/v1/nodes/amish/hosted-sites", {"sites": []}),
    ]


def test_join_runs_the_site_hosts_own_join_with_the_password_piped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    data = tmp_path / "data"
    data.mkdir()
    ran: list[tuple[list[str], str | None]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        ran.append((command, kwargs.get("input")))
        return subprocess.CompletedProcess(command, 0, "Joined as ada's job site desk.", "")

    monkeypatch.setattr(site_cli.subprocess, "run", run)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(readline=lambda: "secret\n"))
    args = SimpleNamespace(
        url="https://nodes.example.test",
        token="invitation",
        owner="ada",
        label="desk",
        root_key="KEY",
        password_stdin=True,
        python=sys.executable,
        data_dir=str(data),
    )
    said = site_cli.join(tmp_path, args)  # type: ignore[arg-type]
    assert "ada's job site" in said and site_host.wanted(tmp_path) is True
    ((command, given),) = ran
    assert command[:4] == [sys.executable, "-m", "eugene_plexus_site_host", "join"]
    assert command[command.index("--data-dir") + 1] == str(data)
    assert command[command.index("--root-key") + 1] == "KEY"
    assert given == "secret\n" and "secret" not in command


def test_leave_on_a_system_install_needs_the_machines_administrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2b.2: only a system install (ProgramData, /var/lib) needs elevation."""
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    monkeypatch.setattr(site_cli, "_system_install", lambda config_dir: True)
    args = SimpleNamespace(python=None, data_dir=None, password_stdin=True)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.leave(tmp_path, args)  # type: ignore[arg-type]
    assert site_host.wanted(tmp_path) is False


@pytest.mark.skipif(sys.platform == "linux", reason="a Linux system install refuses join outright")
def test_join_on_a_system_install_needs_the_machines_administrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    monkeypatch.setattr(site_cli, "_system_install", lambda config_dir: True)
    args = SimpleNamespace(python=None, data_dir=None, password_stdin=True)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.join(tmp_path, args)  # type: ignore[arg-type]
    assert site_host.wanted(tmp_path) is False


def test_a_refused_join_says_why_and_leave_turns_the_host_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(readline=lambda: "secret\n"))
    monkeypatch.setattr(
        site_cli.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(
            command, 1, "", "eugene-plexus-site-host: The join was refused: wrong password"
        ),
    )
    args = SimpleNamespace(
        url="http://192.168.1.5:8083",
        token="t",
        owner="ada",
        label="desk",
        root_key=None,
        password_stdin=True,
        python=sys.executable,
        data_dir=str(data),
    )
    with pytest.raises(site_cli.SiteError, match="wrong password"):
        site_cli.join(tmp_path, args)  # type: ignore[arg-type]
    monkeypatch.setattr(
        site_cli.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
    )
    assert "no longer a job site" in site_cli.leave(tmp_path, args)  # type: ignore[arg-type]
    assert site_host.wanted(tmp_path) is False


# --- the `site` command ------------------------------------------------------------


def program(tmp_path: Path) -> Path:
    path = tmp_path / "server.exe"
    path.write_bytes(b"a program")
    return path


def test_adding_a_server_needs_the_machines_administrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: False)
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.add_server(
            tmp_path,
            server_id="notes",
            name="Notes",
            command=str(program(tmp_path)),
            args=[],
            env=[],
            system=False,
        )
    assert not site_host.servers_path(tmp_path).exists()
    with pytest.raises(site_cli.SiteError, match="administrator"):
        site_cli.remove_server(tmp_path, "notes")


def test_a_server_is_written_with_its_programs_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    exe = program(tmp_path)
    said = site_cli.add_server(
        tmp_path,
        server_id="notes",
        name="Notes",
        command=str(exe),
        args=["--root", "{here}"],
        env=["MODE=fast"],
        system=False,
    )
    assert "off until" in said
    saved = yaml.safe_load(site_host.servers_path(tmp_path).read_text(encoding="utf-8"))
    entry = saved["servers"][0]
    assert entry["sha256"] == hashlib.sha256(b"a program").hexdigest()
    assert entry["args"] == ["--root", "{here}"] and entry["env"] == {"MODE": "fast"}
    assert entry["system"] is False and "consentedAt" not in entry
    assert site_host.local_servers(tmp_path)[0]["id"] == "notes"
    for bad, match in (("files-x", "Eugene's own"), ("Notes", "lower-case"), ("notes", "already")):
        with pytest.raises(site_cli.SiteError, match=match):
            site_cli.add_server(
                tmp_path, server_id=bad, name="X", command=str(exe), args=[], env=[], system=False
            )
    with pytest.raises(site_cli.SiteError, match="absolute path"):
        site_cli.add_server(
            tmp_path, server_id="rel", name="X", command="server.exe", args=[], env=[], system=False
        )
    assert "Removed" in site_cli.remove_server(tmp_path, "notes")
    assert site_host.local_servers(tmp_path) == []


def test_a_system_server_records_the_administrators_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_cli, "elevated", lambda: True)
    said = site_cli.add_server(
        tmp_path,
        server_id="settings",
        name="Settings",
        command=str(program(tmp_path)),
        args=[],
        env=[],
        system=True,
    )
    assert "consent" in said
    entry = site_host.local_servers(tmp_path)[0]
    assert entry["system"] is True and entry["consentedAt"]


def test_status_and_audit_read_what_the_host_keeps(tmp_path: Path) -> None:
    data = tmp_path / "apps" / HELPER_ID / "data"
    data.mkdir(parents=True)
    (data / "policy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "folders": [
                    {
                        "id": "a" * 32,
                        "name": "Notes",
                        "path": "/srv/notes",
                        "identity": "1:2",
                        "writable": True,
                        "people": [{"subject": "p-bo", "writable": True}],
                    }
                ],
                "access": [],
                "enabled": {},
                "ownerInDevMode": False,
            }
        ),
        encoding="utf-8",
    )
    lines = [
        {
            "at": "2026-10-05T12:00:00Z",
            "subject": "p-bo",
            "kind": "mcp",
            "server": "files",
            "tool": "read_text",
            "decision": "allowed",
        },
        {
            "at": "2026-10-05T12:01:00Z",
            "subject": "operator",
            "kind": "mcp",
            "server": "files",
            "tool": "read_text",
            "decision": "refused",
            "reason": "Dev mode alone opens nothing",
        },
    ]
    (data / "audit.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), "utf-8")
    shown = site_cli.status(tmp_path)
    assert "Notes: /srv/notes" in shown and "p-bo (may change files)" in shown
    log = site_cli.audit(tmp_path, 10).splitlines()
    assert log[0].startswith("2026-10-05T12:01") and "refused" in log[0]
    assert "allowed" in log[1]


def test_the_site_command_never_starts_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--command` once shared its dest with the subcommand, so `site server
    add --command X` fell through and started a whole agent."""
    from eugene_plexus_agent import __main__ as entry

    ran: list[Any] = []
    monkeypatch.setenv("EUGENE_PLEXUS_AGENT_CONFIG_FILE", str(tmp_path / "agent.yaml"))
    monkeypatch.setattr(entry, "_serve", lambda *a, **k: ran.append("serve"))
    monkeypatch.setattr(site_cli, "run", lambda args, settings: ran.append(args) or 0)
    with pytest.raises(SystemExit) as done:
        entry.main(
            [
                "site",
                "server",
                "add",
                "notes",
                "--name",
                "Notes",
                "--command",
                str(program(tmp_path)),
                "--arg=-I",
                "--arg",
                "x.py",
            ]
        )
    assert done.value.code == 0 and ran and ran[0] != "serve"
    assert ran[0].program == str(program(tmp_path)) and ran[0].arg == ["-I", "x.py"]


def test_only_the_site_host_itself_is_kept_out_of_the_owners_apps() -> None:
    """The id and the entry must both be the host's: an app that merely takes
    its id is the owner's to see and manage."""
    assert is_site_host("site-host", "eugene_plexus_site_host")
    assert is_site_host(RETIRED_ID, "eugene_plexus_site_host")
    assert not is_site_host("site-host", "something_else")
    assert not is_site_host("workbench", "eugene_plexus_site_host")


# --- the starter's part, 2b.2 ---------------------------------------------------------------


def test_the_site_host_is_exempt_from_the_own_account_rule_and_nothing_else_is(
    app: FastAPI, client: TestClient
) -> None:
    """On a per-user install there is no account of its own to give an app, and
    the site host is allowed because it runs nothing a model chooses (§3.2)."""
    manager = app.state.apps
    manager.accounts = app_accounts.AccountSupport(None, "No accounts here.")
    host = site_host.manifest({})
    assert manager.local_actions_refusal(host) is None
    other = host.model_copy(update={"id": "notes", "entry": "notes_app"})
    assert "needs an account of its own" in (manager.local_actions_refusal(other) or "")


def prepared_host(app: FastAPI, version: str = "local-abc") -> Path:
    """The host as the agent installed it: a record, enabled, with an interpreter."""
    manager = app.state.apps
    manifest = site_host.manifest({}).model_copy(update={"version": version})
    manager.store.put(
        InstalledApp(
            manifest=manifest,
            origin=AppOrigin.catalogue,
            port=8302,
            installed_at=datetime.now(UTC),
            enabled=True,
        )
    )
    python = manager.store.version_dir(HELPER_ID, version) / "venv"
    interpreter = (
        (python / "Scripts" / "python.exe")
        if sys.platform == "win32"
        else python / "bin" / "python"
    )
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"")
    return interpreter


def test_a_per_users_worker_program_is_the_hosts_own_python_for_the_installer(
    app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(site_host, "_own_account", lambda: "S-1-5-21-1-2-3-1001")
    supervisor = SiteHostSupervisor(app)
    assert supervisor._program("user") is None, "nothing installed yet"
    python = prepared_host(app)
    wanted = supervisor._program("user")
    assert wanted is not None and wanted.python == python
    assert wanted.host == "S-1-5-21-1-2-3-1001"
    # The per-user starter says the worker shares the host's account (J38);
    # without it the worker refuses, as it must on a service install.
    assert wanted.shared is True and wanted.argv("S-1-5-21-1-2-3-1001")[-1] == "--shared-account"
    assert wanted.servers == supervisor.config_dir / "site" / "servers.yaml"
    assert wanted.channel == site_host.channel_name(supervisor.config_dir)
    assert supervisor.config_dir in wanted.protect
    record = app.state.apps.store.get(HELPER_ID)
    assert record is not None
    record.enabled = False
    app.state.apps.store.put(record)
    assert supervisor._program("user") is None, "a host that is off has no workers"


async def test_a_per_users_install_runs_one_child_worker_for_its_person(
    app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    class Child:
        def __init__(self, account: str) -> None:
            self.account, self.program = account, None
            seen.append(self)
            self.steps = 0

        async def step(self) -> None:
            self.steps += 1

        async def stop(self) -> None:
            self.program = "stopped"

    monkeypatch.setattr(site_host, "ChildStarter", Child)
    monkeypatch.setattr(site_host, "_own_account", lambda: "1001")
    supervisor = SiteHostSupervisor(app)
    prepared_host(app)
    await supervisor._workers("user")
    await supervisor._workers("user")
    (child,) = seen
    assert child.account == "1001" and child.steps == 2
    assert child.program is not None and child.program.python.name.startswith("python")
    await supervisor._stop_workers()
    assert child.program == "stopped"
    await supervisor._host.aclose()


def test_a_container_hosts_no_site_and_a_linux_system_install_is_roots(
    app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_agent import install_info
    from eugene_plexus_agent._generated.models import InstallMechanism

    manager = app.state.apps
    supervisor = SiteHostSupervisor(app)
    manager.accounts = app_accounts.AccountSupport(None, "none")
    monkeypatch.setattr(install_info, "mechanism", lambda: InstallMechanism.container)
    assert supervisor.mode() is None and supervisor.link_store() is None
    monkeypatch.setattr(install_info, "mechanism", lambda: InstallMechanism.none)
    assert supervisor.mode() == "user"
    manager.accounts = app_accounts.AccountSupport("systemd", None)
    assert supervisor.mode() == "root"


async def test_on_a_linux_system_install_the_agent_only_reads_which_site_it_is(
    app: FastAPI, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    where = tmp_path / "host.json"
    where.write_text(json.dumps({"port": 8411}), encoding="utf-8")
    monkeypatch.setattr(site_host, "ROOT_SITE_FILE", where)
    supervisor = SiteHostSupervisor(app)
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return httpx.Response(200, json={"site": "s-" + "b" * 26})

    await supervisor._host.aclose()
    supervisor._host = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    assert await supervisor._hosted("root") == "s-" + "b" * 26
    assert asked == ["http://127.0.0.1:8411/healthz"]
    where.write_text("not json", encoding="utf-8")
    assert await supervisor._hosted("root") is None
    where.unlink()
    assert await supervisor._hosted("root") is None
    await supervisor._host.aclose()
