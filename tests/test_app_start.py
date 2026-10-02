"""An app that is not ours (C4, specs docs/design/c4-open-webui.md §1).

The checks are about the properties the design rests on, each a thing a
plausible wrong implementation would get wrong:

* every placeholder is filled in, by **one** function, on both ways an
  app runs -- in its own account (the launcher) and in the agent's;
* **secrets are read only inside the app's account**: the spec the agent
  writes, the agent's log and what the service manager is handed hold
  none, and an argument may not name one;
* a manifest naming a placeholder nobody fills in, or a variable of ours,
  is refused when it is validated -- shipped or custom -- not at a start;
* a `pypi` entry installs one exact release;
* `module:attribute` is checked at install and called as its console
  script calls it;
* readiness is asked at `healthPath`;
* a changed connection resets the app's settings for one start, and only
  then.

Nothing here installs Open WebUI: a fake package in a temporary directory
stands in for it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import app_accounts, app_launcher, apps
from eugene_plexus_agent._generated.models import AppManifest, AppOrigin, ComponentStatus
from eugene_plexus_agent.supervisor import ProcessState

from .test_app_accounts import FakeRunner, _free_port, _Ingress

KEY = "the-client-key-value"
OIDC = "the-oidc-client-secret"
EVERY = {
    "BIND": "{bindHost}",
    "PORT": "{port}",
    "DATA": "{dataDir}",
    "GATEWAY": "{gatewayUrl}/v1",
    "APP_URL": "{appUrl}",
    "ISSUER": "{oidcIssuer}",
    "CLIENT_ID": "{oidcClientId}",
    "KEY": "{clientKey}",
    "OIDC_SECRET": "{oidcClientSecret}",
    "SECRET": "{appSecret}",
}


def _manifest(**overrides: Any) -> AppManifest:
    base: dict[str, Any] = {
        "id": "fake",
        "name": "Fake",
        "source": "pypi",
        "version": "1.2.3",
        "package": "fake-app",
        "entry": "fakeapp:main",
        "args": ["serve", "--host", "{bindHost}", "--port", "{port}"],
        "environment": dict(EVERY),
        "signIn": True,
        "healthPath": "/ready",
        "resetOnConnectionChange": "RESET_CONFIG_ON_START",
        "localActions": True,
    }
    base.update(overrides)
    return apps.normalized(AppManifest.model_validate(base))


def _store(tmp_path: Path) -> apps.AppStore:
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    store.data_dir("fake").mkdir(parents=True)
    store.key_file("fake").write_text(KEY + "\n", encoding="utf-8")
    store.oidc_secret_file("fake").write_text(OIDC, encoding="utf-8")
    store.put_oidc_client("fake", "c-made")
    return store


def _record(manifest: AppManifest | None = None, *, port: int = 8190) -> apps.InstalledApp:
    return apps.InstalledApp(
        manifest=manifest or _manifest(),
        origin=AppOrigin.catalogue,
        port=port,
        installed_at=datetime.now(UTC),
    )


def _planner(
    store: apps.AppStore, record: apps.InstalledApp, *, gateway: str = "http://10.0.0.5:8079/gw"
) -> apps._AppPlanner:
    python = apps.venv_python(store.version_dir(record.id, record.version) / "venv")
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    return apps._AppPlanner(
        record,
        store=store,
        gateway_url=lambda: gateway,
        bind_host=lambda: None,
        oidc_issuer=lambda: "http://10.0.0.5:8079/oidc",
        app_url=lambda: "http://10.0.0.5:8190",
    )


def _spec(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """A spec as the launcher reads it, with the files it is given."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    key = tmp_path / "client_key"
    if not key.exists():
        key.write_text(KEY, encoding="utf-8")
    secret = tmp_path / "oidc_secret"
    secret.write_text(OIDC, encoding="utf-8")
    spec: dict[str, Any] = {
        "argv": ["python", "-m", "fakeapp"],
        "args": ["--port", "{port}"],
        "environment": {},
        "values": {"bindHost": "127.0.0.1", "port": "8190", "gatewayUrl": "http://gw"},
        "resetOnConnectionChange": None,
        "dataDir": str(data),
        "keyFile": str(key),
        "oidcSecretFile": str(secret),
    }
    spec.update(overrides)
    return spec


# --------------------------------------------------------------------------- #
# filling in
# --------------------------------------------------------------------------- #


def test_every_placeholder_is_filled_in(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        environment=dict(EVERY),
        values={
            "bindHost": "0.0.0.0",
            "port": "8191",
            "gatewayUrl": "http://10.0.0.5:8079/api/proxy/gateway",
            "appUrl": "http://10.0.0.5:8191",
            "oidcIssuer": "http://10.0.0.5:8079/oidc",
            "oidcClientId": "c-made",
        },
        args=["serve", "--host", "{bindHost}", "--port", "{port}"],
    )
    argv, env, notes = app_launcher.render_start(spec, {"PATH": "kept"})
    secret = (tmp_path / "data" / app_launcher.APP_SECRET_FILE).read_text(encoding="utf-8")
    assert argv == ["python", "-m", "fakeapp", "serve", "--host", "0.0.0.0", "--port", "8191"]
    assert {k: env[k] for k in EVERY} == {
        "BIND": "0.0.0.0",
        "PORT": "8191",
        "DATA": str(tmp_path / "data"),
        "GATEWAY": "http://10.0.0.5:8079/api/proxy/gateway/v1",
        "APP_URL": "http://10.0.0.5:8191",
        "ISSUER": "http://10.0.0.5:8079/oidc",
        "CLIENT_ID": "c-made",
        "KEY": KEY,
        "OIDC_SECRET": OIDC,
        "SECRET": secret,
    }
    assert len(secret) >= 32 and env["PATH"] == "kept" and notes == []


def test_a_placeholder_with_no_value_leaves_its_variable_unset_and_says_so(
    tmp_path: Path,
) -> None:
    spec = _spec(
        tmp_path,
        args=["--gateway", "{gatewayUrl}", "--port", "{port}"],
        environment={"OPENAI_API_BASE_URL": "{gatewayUrl}/v1", "PLAIN": "x"},
        values={"port": "8190"},
    )
    argv, env, notes = app_launcher.render_start(spec, {"OPENAI_API_BASE_URL": "inherited"})
    # Not half a value ("/v1"), and not an inherited one either.
    assert "OPENAI_API_BASE_URL" not in env and env["PLAIN"] == "x"
    assert notes == [
        "OPENAI_API_BASE_URL is not set at this start: there is no value for {gatewayUrl}."
    ]
    # An argument keeps its place, so the ones after it still mean what they say.
    assert argv[-4:] == ["--gateway", "", "--port", "8190"]


def test_a_secret_carried_in_the_spec_is_never_used(tmp_path: Path) -> None:
    """Secrets come from the files the account is given; a value under a
    secret's name in the spec is something no agent should have written.
    It matters most when the file is missing: the variable stays unset
    rather than taking the spec's word for it."""
    spec = _spec(
        tmp_path,
        environment={"KEY": "{clientKey}", "OIDC_SECRET": "{oidcClientSecret}"},
        values={"clientKey": "written-into-the-spec", "oidcClientSecret": "this-too"},
    )
    _argv, env, _notes = app_launcher.render_start(spec, {})
    assert env["KEY"] == KEY and env["OIDC_SECRET"] == OIDC

    spec.update(keyFile=str(tmp_path / "gone"), oidcSecretFile=str(tmp_path / "gone-too"))
    _argv, env, notes = app_launcher.render_start(spec, {})
    assert "KEY" not in env and "OIDC_SECRET" not in env
    assert "written-into-the-spec" not in json.dumps([env, notes])
    assert len(notes) == 2 and "{clientKey}" in notes[0]


def test_the_app_secret_is_made_once_and_kept(tmp_path: Path) -> None:
    spec = _spec(tmp_path, environment={"SECRET": "{appSecret}"})
    first = app_launcher.render_start(spec, {})[1]["SECRET"]
    second = app_launcher.render_start(spec, {})[1]["SECRET"]
    assert first == second and len(first) >= 32
    if sys.platform != "win32":
        mode = (tmp_path / "data" / app_launcher.APP_SECRET_FILE).stat().st_mode & 0o777
        assert mode == 0o600
    # An app that does not ask for one is not given a file it never reads.
    other = tmp_path / "other"
    other.mkdir()
    app_launcher.render_start(_spec(other, environment={"KEY": "{clientKey}"}), {})
    assert not (other / "data" / app_launcher.APP_SECRET_FILE).exists()


# --------------------------------------------------------------------------- #
# a changed connection resets the app's settings, once
# --------------------------------------------------------------------------- #


def test_a_changed_connection_resets_for_one_start_only(tmp_path: Path) -> None:
    key = tmp_path / "client_key"
    spec = _spec(tmp_path, resetOnConnectionChange="RESET_CONFIG_ON_START")

    def start(inherited: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
        _argv, env, notes = app_launcher.render_start(spec, dict(inherited or {}))
        return env, notes

    # The first start has nothing to differ from.
    env, notes = start()
    assert "RESET_CONFIG_ON_START" not in env and notes == []
    stored = (tmp_path / "data" / app_launcher.CONNECTION_FILE).read_text(encoding="utf-8")
    assert KEY not in stored

    # The same connection again: no reset, even from an inherited value.
    env, notes = start({"RESET_CONFIG_ON_START": "true"})
    assert "RESET_CONFIG_ON_START" not in env and notes == []

    # A rotated key: one start resets, and says why in a single line.
    key.write_text("a-new-key", encoding="utf-8")
    env, notes = start()
    assert env["RESET_CONFIG_ON_START"] == "true"
    assert len(notes) == 1 and "RESET_CONFIG_ON_START=true" in notes[0]
    assert "a-new-key" not in notes[0]
    env, notes = start()
    assert "RESET_CONFIG_ON_START" not in env and notes == []

    # A gateway that moved is a change too.
    spec["values"] = dict(spec["values"], gatewayUrl="http://elsewhere")
    assert start()[0]["RESET_CONFIG_ON_START"] == "true"
    assert "RESET_CONFIG_ON_START" not in start()[0]


# --------------------------------------------------------------------------- #
# validating a manifest
# --------------------------------------------------------------------------- #


def test_an_unknown_placeholder_is_refused_and_named() -> None:
    with pytest.raises(ValueError, match=r"\{nope\}"):
        apps.validate_manifest(_manifest(environment={"X": "{nope}"}))
    with pytest.raises(ValueError, match=r"\{gateway\}"):
        apps.validate_manifest(_manifest(args=["--to", "{gateway}"]))
    # Braces that are not a name are text.
    apps.validate_manifest(_manifest(environment={"JSON": '{"a": 1}'}))


@pytest.mark.parametrize("name", ["EUGENE_PLEXUS_APP_KEY_FILE", "eugene_plexus_agent_x"])
def test_a_variable_of_ours_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="EUGENE_PLEXUS_"):
        apps.validate_manifest(_manifest(environment={name: "x"}))
    with pytest.raises(ValueError, match="EUGENE_PLEXUS_"):
        apps.validate_manifest(_manifest(resetOnConnectionChange=name.upper()))


def test_a_secret_may_not_ride_in_an_argument() -> None:
    for secret in sorted(app_launcher.SECRET_PLACEHOLDERS):
        with pytest.raises(ValueError, match="environment instead"):
            apps.validate_manifest(_manifest(args=["--key", f"{{{secret}}}"]))


def test_the_reset_variable_is_ours_to_set() -> None:
    with pytest.raises(ValueError, match="cannot be given a value"):
        apps.validate_manifest(_manifest(environment={"RESET_CONFIG_ON_START": "true"}, args=[]))


@pytest.mark.parametrize("version", ["1.2.3", "0.11.4", "2", "1.0rc1", "2.0.post1"])
def test_a_pypi_entry_installs_that_exact_release(version: str) -> None:
    manifest = _manifest(version=version)
    apps.validate_manifest(manifest)
    assert apps.pip_requirement(manifest) == f"fake-app=={version}"


@pytest.mark.parametrize("version", ["latest", "v1.2.3", "1.2.3+local", "5ed00016be12", "main"])
def test_a_pypi_entry_that_is_not_one_release_is_refused(version: str) -> None:
    manifest = _manifest(version=version)
    with pytest.raises(ValueError, match="not an exact release"):
        apps.validate_manifest(manifest)
    with pytest.raises(ValueError, match="not an exact release"):
        apps.pip_requirement(manifest)


def test_one_unusable_catalogue_entry_costs_that_entry(caplog: pytest.LogCaptureFixture) -> None:
    good = _manifest(id="good").model_dump(mode="json", exclude_none=True)
    bad = _manifest(id="bad", environment={"X": "{nope}"}).model_dump(
        mode="json", exclude_none=True
    )
    ours = _manifest(id="ours", environment={"EUGENE_PLEXUS_APP_KEY_FILE": "/x"}).model_dump(
        mode="json", exclude_none=True
    )
    with caplog.at_level(logging.ERROR):
        loaded = apps.load_catalogue(yaml.safe_dump([bad, good, ours]))
    assert [m.id for m in loaded] == ["good"]
    assert "'bad'" in caplog.text and "{nope}" in caplog.text
    assert "'ours'" in caplog.text and "EUGENE_PLEXUS_" in caplog.text


def test_a_custom_entry_is_validated_the_same_way(authed_client: TestClient) -> None:
    assert authed_client.patch("/v1/config", json={"allowCustomApps": True}).status_code == 200
    body = _manifest().model_dump(mode="json", exclude_none=True)

    unknown = authed_client.post(
        "/v1/app-catalogue/custom", json=dict(body, environment={"X": "{nope}"})
    )
    assert unknown.status_code == 422 and "{nope}" in unknown.json()["detail"]["detail"]
    ours = authed_client.post(
        "/v1/app-catalogue/custom", json=dict(body, environment={"EUGENE_PLEXUS_APP_ID": "x"})
    )
    assert ours.status_code == 422 and "EUGENE_PLEXUS_" in ours.json()["detail"]["detail"]
    loose = authed_client.post("/v1/app-catalogue/custom", json=dict(body, version="latest"))
    assert loose.status_code == 422 and "exact release" in loose.json()["detail"]["detail"]

    # A pypi entry has no folder to look for.
    added = authed_client.post("/v1/app-catalogue/custom", json=body)
    assert added.status_code == 201, added.text


# --------------------------------------------------------------------------- #
# module:attribute, for real, with a fake package
# --------------------------------------------------------------------------- #


def _fake_package(tmp_path: Path) -> Path:
    """`fakeapp.main` records how it was called, as `open_webui:app` would
    be: its argv and the variables the manifest set, in its data directory."""
    root = tmp_path / "pkgs"
    (root / "fakeapp").mkdir(parents=True)
    (root / "fakeapp" / "__init__.py").write_text(
        "import json, os, sys\n"
        "NOT_CALLABLE = 3\n"
        "def main():\n"
        "    names = ('DATA', 'KEY', 'SECRET', 'RESET_CONFIG_ON_START', 'GATEWAY')\n"
        "    record = {'argv': sys.argv, 'env': {n: os.environ.get(n) for n in names},\n"
        "              'cwd_on_path': '' in sys.path}\n"
        "    with open(os.path.join(os.environ['DATA'], 'record.json'), 'w') as f:\n"
        "        json.dump(record, f)\n"
        "    return 0\n",
        encoding="utf-8",
    )
    return root


def _verify(entry: str, packages: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", apps._VERIFY_SNIPPET, entry],
        env={**os.environ, "PYTHONPATH": str(packages)},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_install_checks_the_attribute_exists_and_can_be_called(tmp_path: Path) -> None:
    packages = _fake_package(tmp_path)
    assert _verify("fakeapp:main", packages).returncode == 0
    missing = _verify("fakeapp:serve", packages)
    assert missing.returncode != 0 and "has no serve" in missing.stderr
    constant = _verify("fakeapp:NOT_CALLABLE", packages)
    assert constant.returncode != 0 and "cannot be called" in constant.stderr
    absent = _verify("nosuchmodule:main", packages)
    assert absent.returncode != 0 and "nosuchmodule" in absent.stderr


def _recorded(store_or_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads((store_or_dir / "record.json").read_text())
    return result


def test_the_launcher_calls_the_attribute_with_its_arguments_filled_in(tmp_path: Path) -> None:
    """The own-account path end to end: the spec the agent writes, then the
    launcher filling it in and starting the app."""
    packages = _fake_package(tmp_path)
    ingress = _Ingress()
    data = tmp_path / "data"
    data.mkdir()
    key = tmp_path / "client_key"
    key.write_text(KEY, encoding="utf-8")
    spec: dict[str, Any] = {
        "app": "fake",
        "argv": apps.entry_argv(Path(sys.executable), _manifest()),
        "args": ["serve", "--host", "{bindHost}", "--port", "{port}"],
        "environment": {"DATA": "{dataDir}", "KEY": "{clientKey}", "SECRET": "{appSecret}"},
        "values": {"bindHost": "127.0.0.1", "port": "8190"},
        "resetOnConnectionChange": "RESET_CONFIG_ON_START",
        "env": {"PYTHONPATH": str(packages)},
        "dataDir": str(data),
        "keyFile": str(key),
        "ingress": ingress.url,
    }
    app_launcher._STOP.clear()
    try:
        assert app_launcher.run(spec) == 0
        record = _recorded(data)
        # The way its console script calls it: named after its package.
        assert record["argv"] == ["fake-app", "serve", "--host", "127.0.0.1", "--port", "8190"]
        assert record["env"]["KEY"] == KEY and record["env"]["DATA"] == str(data)
        assert record["env"]["SECRET"] == (data / app_launcher.APP_SECRET_FILE).read_text()
        assert record["env"]["RESET_CONFIG_ON_START"] is None
        assert record["cwd_on_path"] is False

        # A rotated key: the reset reaches the app, and the reason reaches
        # the Logs page through the launcher's own forwarding.
        key.write_text("a-new-key", encoding="utf-8")
        assert app_launcher.run(spec) == 0
        assert _recorded(data)["env"]["RESET_CONFIG_ON_START"] == "true"
        deadline = time.perf_counter() + 10
        while not any("RESET_CONFIG_ON_START=true" in line for line in ingress.lines):
            assert time.perf_counter() < deadline, ingress.lines
            time.sleep(0.1)
        assert not any(KEY in line or "a-new-key" in line for line in ingress.lines)
    finally:
        ingress.server.shutdown()


def test_the_agents_own_supervisor_fills_in_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The agent-account path: `plan()` fills in with the launcher's function,
    and what it builds starts the app the same way."""
    packages = _fake_package(tmp_path)
    store = _store(tmp_path)
    manifest = _manifest(
        environment={"DATA": "{dataDir}", "KEY": "{clientKey}", "SECRET": "{appSecret}"}
    )
    planner = _planner(store, _record(manifest))
    monkeypatch.setattr(apps, "venv_python", lambda venv: Path(sys.executable))
    with caplog.at_level(logging.DEBUG):
        plan = planner.plan()
    assert plan.argv[-5:] == ["serve", "--host", "127.0.0.1", "--port", "8190"]
    assert plan.env["KEY"] == KEY
    # The spawn's argv is what the agent's log prints: no secret is in it.
    assert not any(KEY in a or OIDC in a for a in plan.argv)
    assert KEY not in caplog.text and OIDC not in caplog.text

    done = subprocess.run(
        plan.argv,
        env={**plan.env, "PYTHONPATH": str(packages)},
        cwd=plan.cwd,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0
    record = _recorded(store.data_dir("fake"))
    assert record["argv"] == ["fake-app", "serve", "--host", "127.0.0.1", "--port", "8190"]
    assert record["env"]["SECRET"] == (
        store.data_dir("fake") / app_launcher.APP_SECRET_FILE
    ).read_text(encoding="utf-8")

    # A rotated key here is reset too, and the agent's log says why.
    store.key_file("fake").write_text("a-new-key", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        again = planner.plan()
    assert again.env["RESET_CONFIG_ON_START"] == "true"
    assert "RESET_CONFIG_ON_START=true" in caplog.text and "a-new-key" not in caplog.text


# --------------------------------------------------------------------------- #
# secrets stay inside the app's account
# --------------------------------------------------------------------------- #


class _RecordingRunner(FakeRunner):
    def prepare(self, plan: app_accounts.LaunchPlan, spec: dict) -> None:
        self.plan = plan
        super().prepare(plan, spec)


@pytest.mark.parametrize("kind", ["windows_service", "systemd"])
def test_the_spec_the_agent_writes_holds_no_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kind: str,
) -> None:
    store = _store(tmp_path)
    # A port nothing answers on: the supervisor polls it.
    port = _free_port()
    planner = _planner(store, _record(port=port))
    runner = _RecordingRunner(store.root)
    runner.kind = kind
    supervisor = app_accounts.OwnAccountSupervisor(
        runner,  # type: ignore[arg-type]
        ingress=lambda: "http://127.0.0.1:8079/v1/logs",
        data_dir=store.data_dir,
        key_file=store.key_file,
    )

    async def scenario() -> None:
        supervisor.start(planner)
        await supervisor._starting["fake"]
        await supervisor.stop_all()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(scenario())
    assert runner.calls == ["prepare", "start"]
    spec = runner.spec
    written = json.dumps(spec) + json.dumps(runner.plan.env) + " ".join(runner.plan.argv)
    for secret in (KEY, OIDC):
        assert secret not in written and secret not in caplog.text
    # The agent made no app secret: only the launcher does, in the account.
    assert not (store.data_dir("fake") / app_launcher.APP_SECRET_FILE).exists()
    assert spec["environment"]["KEY"] == "{clientKey}"
    assert spec["args"] == ["serve", "--host", "{bindHost}", "--port", "{port}"]
    assert spec["values"]["gatewayUrl"] == "http://10.0.0.5:8079/gw"

    # What the launcher makes of that spec, in the account, with the files
    # systemd hands over: the secrets, filled in there.
    spec_file = tmp_path / "launch.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    (credentials / "client_key").write_text(KEY, encoding="utf-8")
    (credentials / "oidc_secret").write_text(OIDC, encoding="utf-8")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(credentials))
    monkeypatch.setenv("STATE_DIRECTORY", str(tmp_path / "state"))
    loaded = app_launcher.load_spec(str(spec_file))
    argv, env, _notes = app_launcher.render_start(loaded, app_launcher.child_environment(loaded))
    assert env["KEY"] == KEY and env["OIDC_SECRET"] == OIDC and env["ISSUER"].endswith("/oidc")
    assert argv[-5:] == ["serve", "--host", "127.0.0.1", "--port", str(port)]


# --------------------------------------------------------------------------- #
# readiness at healthPath, on both supervisors
# --------------------------------------------------------------------------- #


def _ready_only_at(path: str, seen: list[str]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200 if request.url.path == path else 404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_the_own_account_supervisor_asks_at_health_path(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runner = FakeRunner(store.root)
    supervisor = app_accounts.OwnAccountSupervisor(
        runner,  # type: ignore[arg-type]
        ingress=lambda: "",
        data_dir=store.data_dir,
        key_file=store.key_file,
    )
    seen: list[str] = []

    async def scenario() -> None:
        supervisor.start(_planner(store, _record(port=_free_port())))
        await supervisor._starting["fake"]
        assert supervisor._client is not None
        await supervisor._client.aclose()
        supervisor._client = _ready_only_at("/ready", seen)
        await supervisor._poll("fake")
        assert supervisor.status("fake")[0] == ComponentStatus.running
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert seen and set(seen) == {"/ready"}


class _Process:
    """`SupervisedProcess` without a process."""

    def __init__(self, planner: Any, log: Any) -> None:
        self.state = ProcessState.starting
        self.last_error = self.last_restart = self.pid = self.last_bind_host = None

    def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def observe_readiness(self, ready: bool) -> None:
        return None


def test_the_agents_own_supervisor_asks_at_health_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(apps, "SupervisedProcess", _Process)
    store = _store(tmp_path)
    supervisor = apps.AppSupervisor()
    seen: list[str] = []

    async def scenario() -> None:
        supervisor.start(_planner(store, _record(port=_free_port())))
        assert supervisor._client is not None
        await supervisor._client.aclose()
        supervisor._client = _ready_only_at("/ready", seen)
        await supervisor._poll("fake", supervisor._health["fake"])
        assert supervisor.status("fake")[0] == ComponentStatus.running
        await supervisor.stop_all()

    asyncio.run(scenario())
    assert seen and set(seen) == {"/ready"}
    # An app that says nothing is asked where the contract has always said.
    assert _planner(store, _record(_manifest(healthPath=None))).health_url.endswith(":8190/healthz")


# --------------------------------------------------------------------------- #
# the catalogue
# --------------------------------------------------------------------------- #


def test_the_catalogue_offers_open_webui_as_designed() -> None:
    catalogue = {m.id: m for m in apps.load_catalogue()}
    assert list(catalogue) == ["workbench", "open-webui"]
    webui = catalogue["open-webui"]
    assert apps.pip_requirement(webui) == "open-webui==0.11.4"
    assert webui.entry == "open_webui:app" and webui.healthPath == "/ready"
    assert [a.root for a in webui.args or []] == [
        "serve",
        "--host",
        "{bindHost}",
        "--port",
        "{port}",
    ]
    assert webui.localActions is True and webui.signIn is True and webui.configTrio is False
    assert webui.signInCallbackPath == "/oauth/oidc/callback"
    assert webui.resetOnConnectionChange == "RESET_CONFIG_ON_START"
    assert str(webui.licenseUrl) == "https://github.com/open-webui/open-webui/blob/v0.11.4/LICENSE"
    env = webui.environment or {}
    assert env["OPENAI_API_KEY"] == "{clientKey}"
    assert env["OAUTH_CLIENT_SECRET"] == "{oidcClientSecret}"
    assert env["WEBUI_SECRET_KEY"] == "{appSecret}"
    assert env["OPENAI_API_BASE_URL"] == "{gatewayUrl}/v1"
    assert env["OPENID_PROVIDER_URL"] == "{oidcIssuer}/.well-known/openid-configuration"
    assert len(env) == 33

    # Workbench needs none of it and did not change.
    workbench = catalogue["workbench"]
    assert workbench.args is None and workbench.environment is None
    assert workbench.healthPath == "/healthz" and workbench.resetOnConnectionChange is None


def test_open_webui_is_told_where_everything_is(tmp_path: Path) -> None:
    (webui,) = [m for m in apps.load_catalogue() if m.id == "open-webui"]
    store = apps.AppStore(tmp_path / apps.APPS_FILE)
    store.data_dir("open-webui").mkdir(parents=True)
    store.key_file("open-webui").write_text(KEY, encoding="utf-8")
    store.oidc_secret_file("open-webui").write_text(OIDC, encoding="utf-8")
    store.put_oidc_client("open-webui", "c-webui")
    record = apps.InstalledApp(
        manifest=webui, origin=AppOrigin.catalogue, port=8191, installed_at=datetime.now(UTC)
    )
    planner = _planner(store, record, gateway="http://127.0.0.1:8080")
    plan = planner.plan()
    data = str(store.data_dir("open-webui"))
    assert plan.argv[1:5] == ["-c", apps._CALL_SNIPPET, "open_webui:app", "open-webui"]
    assert plan.argv[5:] == ["serve", "--host", "127.0.0.1", "--port", "8191"]
    env = plan.env
    assert env["DATA_DIR"] == data and env["STATIC_DIR"] == data + "/static"
    assert env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert env["OPENAI_API_KEY"] == KEY and env["OAUTH_CLIENT_SECRET"] == OIDC
    assert env["OAUTH_CLIENT_ID"] == "c-webui"
    assert env["OPENID_PROVIDER_URL"] == (
        "http://10.0.0.5:8079/oidc/.well-known/openid-configuration"
    )
    assert env["WEBUI_URL"] == env["CORS_ALLOW_ORIGIN"] == "http://10.0.0.5:8190"
    assert env["ENABLE_OLLAMA_API"] == "false" and env["OFFLINE_MODE"] == "true"


def test_the_listing_carries_the_licence(app: FastAPI, authed_client: TestClient) -> None:
    manager: apps.AppManager = app.state.apps
    manager.catalogue = {m.id: m for m in apps.load_catalogue()}
    listed = authed_client.get("/v1/app-catalogue").json()["apps"]
    (webui,) = [e["manifest"] for e in listed if e["manifest"]["id"] == "open-webui"]
    assert webui["licenseUrl"] == "https://github.com/open-webui/open-webui/blob/v0.11.4/LICENSE"


def test_the_app_url_is_where_the_console_opens_it(tmp_path: Path) -> None:
    async def gateway() -> tuple[str | None, str | None]:
        return None, None

    manager = apps.AppManager(
        store=apps.AppStore(tmp_path / apps.APPS_FILE),
        catalogue=[],
        get_config=lambda key: None,
        bind_host=lambda: None,
        advertise_host=lambda: "192.168.16.75",
        node_name=lambda: "node-a",
        resolve_gateway=gateway,
        accounts=app_accounts.AccountSupport(None, "tests make no accounts"),
    )
    port = manager.reserve_port("fake")
    assert manager.app_url("fake") == f"http://192.168.16.75:{port}"
    assert manager.sign_in_redirects("fake", "/cb")[0] == f"http://192.168.16.75:{port}/cb"
    manager._advertise_host = lambda: "fd7a::5"
    assert manager.app_url("fake") == f"http://[fd7a::5]:{port}"
