"""Eugene's own account runs no program anyone else can change (row 8).

The disk is a stand-in (owners, modes, links, file heads, macOS ACL
entries and mount options), so these run on every platform: the agent is
`_eugeneplexus` (uid 400) or a person (troy, 501); alice (502) is someone
else on the machine. The last tests walk the real disk where there is one.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from eugene_plexus_agent import exec_trust
from eugene_plexus_agent._generated.models import EngineKind, RuntimeSpec
from eugene_plexus_agent.engines import EngineUnavailableError, LlamaCppAdapter
from eugene_plexus_agent.engines.vllm import VllmAdapter

NAMES = {0: "root", 400: "_eugeneplexus", 501: "troy", 502: "alice"}


def _key(path: Any) -> str:
    return str(PurePosixPath(str(path).replace("\\", "/")))


class Disk:
    """Folders, files and links, each with an owner, a mode and ACL entries."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.noowners: set[str] = set()
        self.dir("/")

    def _parents(self, path: str, uid: int = 0) -> None:
        for parent in reversed(PurePosixPath(path).parents):
            self.nodes.setdefault(
                str(parent), {"kind": stat.S_IFDIR, "uid": uid, "mode": 0o755, "acl": []}
            )

    def dir(self, path: str, uid: int = 0, mode: int = 0o755, acl: list[str] | None = None) -> Disk:
        self._parents(path)
        self.nodes[_key(path)] = {"kind": stat.S_IFDIR, "uid": uid, "mode": mode, "acl": acl or []}
        return self

    def file(
        self,
        path: str,
        uid: int = 0,
        mode: int = 0o755,
        content: bytes = b"\x7fELF",
        acl: list[str] | None = None,
    ) -> Disk:
        self._parents(path)
        self.nodes[_key(path)] = {
            "kind": stat.S_IFREG,
            "uid": uid,
            "mode": mode,
            "content": content,
            "acl": acl or [],
        }
        return self

    def link(self, path: str, target: str) -> Disk:
        self._parents(path)
        self.nodes[_key(path)] = {
            "kind": stat.S_IFLNK,
            "uid": 0,
            "mode": 0o777,
            "target": target,
            "acl": [],
        }
        return self

    # --- what exec_trust reads ---

    def lstat(self, path: Any) -> Any:
        node = self.nodes.get(_key(path))
        if node is None:
            raise FileNotFoundError(2, "No such file or directory")
        return SimpleNamespace(
            st_uid=node["uid"],
            st_mode=node["kind"] | node["mode"],
            st_dev=1,
            st_ino=hash(_key(path)),
            st_ctime_ns=0,
        )

    def readlink(self, path: Any) -> str:
        return str(self.nodes[_key(path)]["target"])

    def resolve(self, path: Any) -> str:
        key = _key(path)
        for _ in range(40):
            node = self.nodes.get(key)
            if node is None or node["kind"] != stat.S_IFLNK:
                return key
            target = PurePosixPath(node["target"])
            key = str(target if target.is_absolute() else PurePosixPath(key).parent / target)
        raise OSError("too many links")

    def head(self, path: Any) -> bytes:
        node = self.nodes.get(self.resolve(path))
        if node is None or node["kind"] != stat.S_IFREG:
            raise FileNotFoundError(2, "No such file")
        return bytes(node["content"])

    def listdir(self, path: Any) -> list[str]:
        key = self.resolve(path)
        return [
            PurePosixPath(k).name
            for k in self.nodes
            if str(PurePosixPath(k).parent) == key and k != key
        ]

    def acl(self, path: Any) -> list[str]:
        return [f"{n}: {entry}" for n, entry in enumerate(self.nodes[_key(path)]["acl"])]

    def mount(self, path: Any) -> frozenset[str]:
        key = _key(path)
        return (
            frozenset({"noowners"})
            if any(key == m or key.startswith(m + "/") for m in self.noowners)
            else frozenset()
        )


@pytest.fixture
def disk(monkeypatch: pytest.MonkeyPatch) -> Disk:
    d = Disk()
    monkeypatch.setattr(exec_trust, "_lstat", d.lstat)
    monkeypatch.setattr(exec_trust, "_readlink", d.readlink)
    monkeypatch.setattr(exec_trust, "_head", d.head)
    monkeypatch.setattr(exec_trust, "_listdir", d.listdir)
    monkeypatch.setattr(exec_trust, "_acl_entries", d.acl)
    monkeypatch.setattr(exec_trust, "_mount_options", d.mount)
    monkeypatch.setattr(exec_trust, "_acl_seen", {})
    monkeypatch.setattr(exec_trust, "_account_name", lambda uid: NAMES.get(uid))
    monkeypatch.setattr(exec_trust, "_platform", lambda: "linux")
    monkeypatch.setattr(exec_trust, "_cwd", lambda: "/usr/local/eugene-plexus")
    as_eugene(monkeypatch)
    return d


def as_eugene(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exec_trust, "_euid", lambda: 400)


def refused(path: str, *words: str) -> str:
    with pytest.raises(exec_trust.Untrusted) as caught:
        exec_trust.trusted_program(path)
    reason = str(caught.value)
    for word in words:
        assert word in reason, reason
    return reason


def ran(path: str) -> str:
    return str(exec_trust.trusted_program(path)).replace("\\", "/")


# --- owners, modes ----------------------------------------------------------


def test_a_person_s_own_install_is_not_limited(disk: Disk, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as troy, not Eugene's account: even alice's file in a folder its
    group can write is not this check's to refuse."""
    disk.dir("/opt/shared", uid=502, mode=0o775).file("/opt/shared/llama-server", uid=502)
    monkeypatch.setattr(exec_trust, "_euid", lambda: 501)
    assert exec_trust.untrusted("/opt/shared/llama-server") is None


def test_a_file_of_a_person_s_is_refused_naming_its_owner(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501).dir("/Users/troy/llama", uid=501)
    disk.file("/Users/troy/llama/llama-server", uid=501)
    refused(
        "/Users/troy/llama/llama-server",
        "belongs to troy",
        "Move it, or let Eugene fetch its own build",
    )


def test_a_folder_of_a_person_s_above_it_is_refused(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501).dir("/Users/troy/llama", uid=400)
    disk.file("/Users/troy/llama/llama-server", uid=400)
    refused("/Users/troy/llama/llama-server", "which belongs to troy")


def test_a_folder_others_can_write_is_refused(disk: Disk) -> None:
    disk.dir("/usr/local/shared", mode=0o775).file("/usr/local/shared/llama-server")
    refused("/usr/local/shared/llama-server", "can be changed by others (mode 0775)")


def test_root_s_and_its_own_folders_run_and_the_resolved_path_is_returned(disk: Disk) -> None:
    disk.dir("/usr/local/eugene-plexus", uid=400, mode=0o750)
    disk.dir("/usr/local/eugene-plexus/engines", uid=400).dir(
        "/usr/local/eugene-plexus/engines/llama_cpp", uid=400
    )
    disk.dir("/usr/local/eugene-plexus/engines/llama_cpp/b1", uid=400)
    disk.file("/usr/local/eugene-plexus/engines/llama_cpp/b1/llama-server", uid=400)
    assert ran("/usr/local/eugene-plexus/engines/llama_cpp/b1/llama-server") == (
        "/usr/local/eugene-plexus/engines/llama_cpp/b1/llama-server"
    )


# --- links --------------------------------------------------------------------


def test_a_link_to_a_person_s_file_is_refused(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501).file("/Users/troy/llama-server", uid=501)
    disk.link("/usr/local/bin/llama-server", "/Users/troy/llama-server")
    refused("/usr/local/bin/llama-server", "belongs to troy")


def test_a_link_to_its_own_file_in_a_person_s_folder_is_refused(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501).file("/Users/troy/llama-server", uid=400)
    disk.link("/usr/local/bin/llama-server", "/Users/troy/llama-server")
    refused("/usr/local/bin/llama-server", "which belongs to troy")


def test_a_link_in_a_person_s_folder_is_refused_though_it_points_at_root_s_file(disk: Disk) -> None:
    disk.file("/usr/bin/llama-server")
    disk.dir("/Users/troy", uid=501).dir("/Users/troy/bin", uid=501)
    disk.link("/Users/troy/bin/llama-server", "/usr/bin/llama-server")
    refused("/Users/troy/bin/llama-server", "which belongs to troy")


def test_a_chain_of_links_through_a_person_s_folder_is_refused(disk: Disk) -> None:
    """`/opt/safe/engine -> /Users/alice/hop -> /usr/bin/true`: the middle
    link sits in alice's folder, so she can point it anywhere before it
    runs. Every hop's folder is walked."""
    disk.file("/usr/bin/true")
    disk.dir("/Users/alice", uid=502)
    disk.link("/Users/alice/hop", "/usr/bin/true")
    disk.dir("/opt/safe").link("/opt/safe/engine", "/Users/alice/hop")
    refused("/opt/safe/engine", "belongs to alice")


def test_folders_reached_through_a_link_are_resolved_in_what_runs(disk: Disk) -> None:
    disk.dir("/opt/real").file("/opt/real/llama-server")
    disk.link("/opt/current", "real")
    assert ran("/opt/current/llama-server") == "/opt/real/llama-server"


def test_a_relative_path_runs_as_the_absolute_path_that_was_checked(disk: Disk) -> None:
    """Checked from the agent's own folder, and what runs is that absolute
    path, whatever folder the engine is started in."""
    disk.dir("/usr/local/eugene-plexus", uid=400).dir("/usr/local/eugene-plexus/engines", uid=400)
    disk.file("/usr/local/eugene-plexus/engines/llama-server", uid=400)
    assert ran("engines/llama-server") == "/usr/local/eugene-plexus/engines/llama-server"


def test_a_relative_path_into_a_person_s_folder_is_refused(disk: Disk) -> None:
    disk.dir("/usr/local/eugene-plexus", uid=400).dir("/usr/local/eugene-plexus/theirs", uid=502)
    disk.file("/usr/local/eugene-plexus/theirs/llama-server", uid=502)
    refused("theirs/llama-server", "belongs to alice")


# --- interpreters and environments ----------------------------------------------


def test_a_script_whose_interpreter_a_person_owns_is_refused(disk: Disk) -> None:
    disk.dir("/Users/alice", uid=502).dir("/Users/alice/bin", uid=502).file(
        "/Users/alice/bin/python", uid=502
    )
    disk.file("/usr/local/bin/vllm", content=b"#!/Users/alice/bin/python\nimport vllm\n")
    refused("/usr/local/bin/vllm", "belongs to alice")


def test_a_script_that_finds_its_interpreter_on_path_is_refused(disk: Disk) -> None:
    disk.file("/usr/local/bin/vllm", content=b"#!/usr/bin/env python3\nimport vllm\n")
    refused("/usr/local/bin/vllm", "names its interpreter through PATH")


def test_uv_s_long_path_script_names_its_python_on_the_second_line(disk: Disk) -> None:
    disk.file("/bin/sh")
    disk.dir("/Users/alice", uid=502).file("/Users/alice/python", uid=502)
    disk.file(
        "/usr/local/bin/vllm",
        content=b"#!/bin/sh\n'''exec' \"/Users/alice/python\" \"$0\" \"$@\"\n' '''\n",
    )
    refused("/usr/local/bin/vllm", "belongs to alice")


def _venv(disk: Disk, root: str, *, home: str, site_uid: int = 400) -> None:
    disk.dir(root, uid=400).dir(f"{root}/bin", uid=400).dir(f"{root}/lib", uid=400)
    disk.dir(f"{root}/lib/python3.12", uid=400).dir(
        f"{root}/lib/python3.12/site-packages", uid=site_uid
    )
    disk.file(f"{root}/pyvenv.cfg", uid=400, mode=0o644, content=f"home = {home}\n".encode())
    disk.link(f"{root}/bin/python", f"{home}/python3.12")
    disk.file(f"{root}/bin/vllm", uid=400, content=f"#!{root}/bin/python\nimport vllm\n".encode())


def test_an_environment_made_from_a_person_s_python_is_refused(disk: Disk) -> None:
    disk.dir("/Users/alice", uid=502).dir("/Users/alice/py", uid=502).dir(
        "/Users/alice/py/bin", uid=502
    )
    disk.file("/Users/alice/py/bin/python3.12", uid=502)
    _venv(disk, "/usr/local/eugene-plexus/vllm-env", home="/Users/alice/py/bin")
    refused("/usr/local/eugene-plexus/vllm-env/bin/vllm", "belongs to alice")


def test_an_environment_whose_packages_a_person_owns_is_refused(disk: Disk) -> None:
    disk.dir("/opt/py").dir("/opt/py/bin").file("/opt/py/bin/python3.12").dir("/opt/py/lib")
    _venv(disk, "/usr/local/eugene-plexus/vllm-env", home="/opt/py/bin", site_uid=502)
    refused("/usr/local/eugene-plexus/vllm-env/bin/vllm", "site-packages", "belongs to alice")


def test_a_trusted_environment_runs_by_its_own_python(disk: Disk) -> None:
    """A venv's `bin/python` is a link to the interpreter it was made
    from; it runs by that name, so Python finds its environment."""
    disk.dir("/opt/py").dir("/opt/py/bin").file("/opt/py/bin/python3.12").dir("/opt/py/lib")
    _venv(disk, "/usr/local/eugene-plexus/vllm-env", home="/opt/py/bin")
    assert (
        ran("/usr/local/eugene-plexus/vllm-env/bin/vllm")
        == "/usr/local/eugene-plexus/vllm-env/bin/vllm"
    )
    assert (
        ran("/usr/local/eugene-plexus/vllm-env/bin/python")
        == "/usr/local/eugene-plexus/vllm-env/bin/python"
    )


# --- macOS ---------------------------------------------------------------------


def test_macos_an_acl_entry_that_lets_someone_write_is_refused(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root's 0755 file, but `user:alice allow write`: the mode bits are
    not the whole permission on a Mac."""
    monkeypatch.setattr(exec_trust, "_platform", lambda: "darwin")
    disk.dir("/usr/local/row8").file("/usr/local/row8/llama-server", acl=["user:alice allow write"])
    refused("/usr/local/row8/llama-server", "ACL entry that lets user alice write")


def test_macos_deny_entries_and_its_own_entries_are_not_refused(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exec_trust, "_platform", lambda: "darwin")
    disk.dir(
        "/usr/local/row8", acl=["group:everyone deny delete", "user:_eugeneplexus allow add_file"]
    )
    disk.file("/usr/local/row8/llama-server", acl=["user:alice allow read,execute"])
    assert ran("/usr/local/row8/llama-server") == "/usr/local/row8/llama-server"


def test_macos_a_volume_that_ignores_ownership_is_refused(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exec_trust, "_platform", lambda: "darwin")
    disk.dir("/Volumes/USB").file("/Volumes/USB/llama-server")
    disk.noowners.add("/Volumes/USB")
    refused("/Volumes/USB/llama-server", "ignores ownership")


# --- the real disk ---------------------------------------------------------------


@pytest.mark.skipif(
    os.name != "posix" or sys.platform == "darwin", reason="Linux's own system folders"
)
def test_linux_system_programs_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real `/bin/true`: root's, through the `/bin -> usr/bin` link
    where there is one."""
    me = os.geteuid()
    monkeypatch.setattr(
        exec_trust, "_account_name", lambda uid: "_eugeneplexus" if uid == me else NAMES.get(uid)
    )
    path = "/bin/true" if os.path.exists("/bin/true") else "/usr/bin/true"
    assert exec_trust.untrusted(path) is None
    assert str(exec_trust.trusted_program(path)).endswith("/true")


# --- where it is used ------------------------------------------------------------


def _spec(**overrides: Any) -> RuntimeSpec:
    base: dict[str, Any] = {
        "name": "m",
        "engine": EngineKind.llama_cpp,
        "modelPath": "/models/m.gguf",
    }
    base.update(overrides)
    return RuntimeSpec.model_validate(base)


def test_a_runtime_s_binary_is_refused_before_it_is_run_even_for_its_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")

    def refuse(path: Any) -> Path:
        raise exec_trust.Untrusted(
            f"{path} belongs to troy; Eugene's own account runs programs only from folders"
        )

    monkeypatch.setattr(exec_trust, "trusted_program", refuse)
    ran_: list[Path] = []
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(adapter, "probe_version", lambda binary: ran_.append(binary) or "1")
    with pytest.raises(EngineUnavailableError, match="belongs to troy"):
        adapter.resolve_binary(_spec(binary=str(exe)))
    assert ran_ == []


def test_the_checked_path_is_what_runs_and_is_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not the spelling a setting gave: what the check resolved."""
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    checked = tmp_path / "checked" / "llama-server"
    monkeypatch.setattr(exec_trust, "trusted_program", lambda path: checked)
    probed: list[Path] = []
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(adapter, "probe_version", lambda binary: probed.append(binary) or "1")
    found = adapter.resolve_binary(_spec(binary=str(exe)))
    assert found.path == checked
    assert probed == [checked]


def test_a_configured_engine_binary_is_refused_before_its_environment_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "vllm"
    exe.write_text("#!/usr/bin/python3\n", encoding="utf-8")

    def refuse(path: Any) -> Path:
        raise exec_trust.Untrusted(f"{path} belongs to troy; ...")

    monkeypatch.setattr(exec_trust, "trusted_program", refuse)
    read: list[Path] = []
    from eugene_plexus_agent.engines import vllm

    monkeypatch.setattr(vllm, "inspect_python_environment", lambda path: read.append(path))
    with pytest.raises(EngineUnavailableError, match="belongs to troy"):
        VllmAdapter().discover(configured=str(exe))
    assert read == []
