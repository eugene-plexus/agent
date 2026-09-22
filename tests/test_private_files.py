"""Files that hold a secret are created owner-only, and atomically.

`agent.yaml` carries the passphrase hash and the master-key salt,
`node.yaml` the install's signing key, a companion driver's config any
`apiKey` the operator saved on it, and `agent.yaml.unreadable` a
verbatim copy of the first. Before 2026-09-22 they were written with
the process umask (0644 on a stock Linux) or chmodded after the fact.

Two kinds of test, because this box is Windows and CI is Linux. The
mode assertions only mean something on POSIX, so they skip here; the
**platform-independent** ones record every `os.open` and assert each
site creates its file through one with an owner-only mode and
`O_EXCL` -- which the old code never called at all (`Path.open` and
`write_text` go through `io.open`), so they fail against it on any OS.
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from eugene_plexus_agent import _private_files, client_keys
from eugene_plexus_agent.client_keys import ClientKeyRecord, ClientKeyStore
from eugene_plexus_agent.node_identity import NODE_FILE, NodeIdentityStore
from eugene_plexus_agent.state import UNREADABLE_SUFFIX, AgentState

from .conftest import TEST_PASSPHRASE

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="file mode bits are not the access control on Windows"
)

Opened = list[tuple[str, int, int]]


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    """Every `os.open` for the rest of the test: path, flags, mode."""
    calls: Opened = []
    real = os.open

    def recording(path: Any, flags: int, mode: int = 0o777, *args: Any, **kwargs: Any) -> int:
        calls.append((os.fspath(path), flags, mode))
        return real(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording)
    return calls


def _created_privately(calls: Opened, target: Path) -> None:
    """Assert `target` (or its temp sibling) was created owner-only, exclusively."""
    mine = [
        (path, flags, mode)
        for path, flags, mode in calls
        if Path(path).parent == target.parent and Path(path).name.startswith(target.name)
    ]
    assert mine, (
        f"{target.name} was never created through os.open with an explicit mode, so it "
        f"took the process umask -- 0644 on a stock Linux"
    )
    for path, flags, mode in mine:
        assert mode & 0o077 == 0, f"{path} created with mode {oct(mode)}: others can read it"
        assert flags & os.O_EXCL, f"{path} opened without O_EXCL: it can adopt a stranger's file"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# -- the sites, each through the surface that really writes it -------------


def _agent_yaml_via_initialize(client: TestClient, tmp_path: Path) -> Path:
    assert (
        client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE}).status_code == 200
    )
    return tmp_path / "agent.yaml"


def _companion_config_via_launch(client: TestClient, tmp_path: Path) -> Path:
    init = client.post("/v1/auth/initialize", json={"passphrase": TEST_PASSPHRASE})
    client.headers["Authorization"] = f"Bearer {init.json()['sessionToken']}"
    created = client.post(
        "/v1/runtimes",
        json={
            "name": "qwen3-a",
            "engine": "llama_cpp",
            "modelPath": "/models/Qwen3-1.7B-Q8_0.gguf",
        },
    )
    assert created.status_code == 201, created.text
    comps = {c["name"]: c for c in client.get("/v1/components").json()["components"]}
    return Path(comps["qwen3-a-driver"]["spawn"]["configFile"])


def _node_yaml(_client: TestClient, tmp_path: Path) -> Path:
    store = NodeIdentityStore(tmp_path / NODE_FILE)
    store.ensure_keypair()
    return store.path


def _unreadable_copy(_client: TestClient, tmp_path: Path) -> Path:
    path = tmp_path / "damaged" / "agent.yaml"
    path.parent.mkdir()
    path.write_text(
        "auth:\n  masterSalt: c2FsdA==\n  passphraseHash: $argon2id$x\ncomponents:\n  - name: g\n",
        encoding="utf-8",
    )
    assert AgentState(path).load_or_degrade() is not None
    return path.with_suffix(path.suffix + UNREADABLE_SUFFIX)


def _client_keys(_client: TestClient, tmp_path: Path) -> Path:
    store = ClientKeyStore(tmp_path / client_keys.KEYS_FILE)
    store.add(
        ClientKeyRecord(id="one", name="phone", tail="Xk9Q0z", created_at=1.0, expires_at=9e9)
    )
    return store.path


SITES: dict[str, Callable[[TestClient, Path], Path]] = {
    "agent.yaml": _agent_yaml_via_initialize,
    "companion config": _companion_config_via_launch,
    "node.yaml": _node_yaml,
    "agent.yaml.unreadable": _unreadable_copy,
    "client_keys.json": _client_keys,
}


@pytest.mark.parametrize("site", list(SITES))
def test_every_secret_bearing_file_is_created_owner_only(
    site: str, client: TestClient, tmp_path: Path, opened: Opened
) -> None:
    target = SITES[site](client, tmp_path)
    assert target.exists()
    _created_privately(opened, target)


@posix_only
@pytest.mark.parametrize("site", list(SITES))
def test_every_secret_bearing_file_is_mode_0600_on_disk(
    site: str, client: TestClient, tmp_path: Path
) -> None:
    target = SITES[site](client, tmp_path)
    assert _mode(target) == 0o600, f"{target.name} is {oct(_mode(target))}"


@posix_only
def test_a_config_an_older_build_left_world_readable_is_tightened_on_its_next_write(
    authed_client: TestClient, tmp_path: Path
) -> None:
    """Nothing to migrate: the replace is a rename, so the target takes the
    temp file's mode on the next config write."""
    path = tmp_path / "agent.yaml"
    os.chmod(path, 0o644)
    assert authed_client.patch("/v1/config", json={"uiFontSize": "large"}).status_code == 200
    assert _mode(path) == 0o600


@posix_only
def test_a_leftover_client_keys_temp_does_not_lend_the_registry_its_mode(tmp_path: Path) -> None:
    """The one site that was already private had a fixed temp name opened
    with `O_TRUNC`, which keeps an existing file's mode. A crash-leftover
    at 0644 made every later registry 0644."""
    leftover = (tmp_path / client_keys.KEYS_FILE).with_suffix(".tmp")
    leftover.write_text("{}", encoding="utf-8")
    os.chmod(leftover, 0o644)
    path = _client_keys(None, tmp_path)  # type: ignore[arg-type]
    assert _mode(path) == 0o600


# -- the helper's own promises -----------------------------------------------


def test_write_private_replaces_the_target_and_leaves_nothing_else(tmp_path: Path) -> None:
    path = tmp_path / "secret.yaml"
    path.write_text("old\n", encoding="utf-8")
    _private_files.write_private(path, "new: 1\n")
    assert path.read_bytes() == b"new: 1\n", "the bytes given are the bytes that land"
    assert [p.name for p in tmp_path.iterdir()] == ["secret.yaml"]


def test_a_failed_write_keeps_the_old_file_and_removes_its_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interruption is inside the write this time, not in rendering
    before it -- the case `state.py`'s own test can no longer reach, since
    the YAML is rendered before any file is opened."""
    path = tmp_path / "secret.yaml"
    path.write_text("old\n", encoding="utf-8")

    def die(_fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", die)
    with pytest.raises(OSError):
        _private_files.write_private(path, "new\n")
    assert path.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["secret.yaml"]


def test_the_agent_config_still_round_trips(tmp_path: Path) -> None:
    """The rewrite changed how the file is opened, not what is in it."""
    path = tmp_path / "agent.yaml"
    state = AgentState(path)
    state.load()
    state.set_passphrase(passphrase_hash="$argon2id$h", master_salt_b64="c2FsdA==")
    reloaded = AgentState(path)
    reloaded.load()
    assert reloaded.get_passphrase_hash() == "$argon2id$h"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["auth"]["masterSalt"] == "c2FsdA=="
