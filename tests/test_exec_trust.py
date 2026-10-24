"""Eugene's own account runs no program anyone else can change (row 8).

The disk is a stand-in (owners, groups, modes, links, file heads, macOS
ACL entries and mount options), so these run on every platform: the agent
is `_eugeneplexus` (uid 400) or a person (troy, 501); alice (502) is
someone else on the machine. The last tests walk the real disk where there
is one.

Three kinds (part 8c, call R8-EXEC-TRUST): `ok` runs; `needs_approval`
(one other account can change it, and nothing else can) runs once an
owner approves it; `refused` never runs.
"""

from __future__ import annotations

import hashlib
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
GROUPS = {0: "wheel", 20: "staff"}
HOME = "/usr/local/eugene-plexus"


def _key(path: Any) -> str:
    return str(PurePosixPath(str(path).replace("\\", "/")))


class Disk:
    """Folders, files and links, each with an owner, a mode and ACL entries."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.noowners: set[str] = set()
        self.unreadable: set[str] = set()
        self.unlistable: set[str] = set()
        self.hashed: list[str] = []
        self.dir("/")

    def _parents(self, path: str, uid: int = 0) -> None:
        for parent in reversed(PurePosixPath(path).parents):
            self.nodes.setdefault(
                str(parent), {"kind": stat.S_IFDIR, "uid": uid, "mode": 0o755, "acl": []}
            )

    def dir(
        self,
        path: str,
        uid: int = 0,
        mode: int = 0o755,
        acl: list[str] | None = None,
        gid: int = 0,
    ) -> Disk:
        self._parents(path)
        self.nodes[_key(path)] = {
            "kind": stat.S_IFDIR,
            "uid": uid,
            "gid": gid,
            "mode": mode,
            "acl": acl or [],
        }
        return self

    def file(
        self,
        path: str,
        uid: int = 0,
        mode: int = 0o755,
        content: bytes = b"\x7fELF",
        acl: list[str] | None = None,
        gid: int = 0,
    ) -> Disk:
        self._parents(path)
        self.nodes[_key(path)] = {
            "kind": stat.S_IFREG,
            "uid": uid,
            "gid": gid,
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
            st_gid=node.get("gid", 0),
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
        key = self.resolve(path)
        node = self.nodes.get(key)
        if node is None or node["kind"] != stat.S_IFREG:
            raise FileNotFoundError(2, "No such file")
        if key in self.unreadable:
            raise PermissionError(13, "Permission denied")
        return bytes(node["content"])

    def sha256(self, path: Any) -> str:
        self.hashed.append(_key(path))
        return hashlib.sha256(self.head(path)).hexdigest()

    def listdir(self, path: Any) -> list[str]:
        key = self.resolve(path)
        if key not in self.nodes:
            raise FileNotFoundError(2, "No such file or directory")
        if key in self.unlistable:
            raise PermissionError(13, "Permission denied")
        return [
            PurePosixPath(k).name
            for k in self.nodes
            if str(PurePosixPath(k).parent) == key and k != key
        ]

    def acl(self, path: Any) -> list[str]:
        return [f"{n}: {entry}" for n, entry in enumerate(self.nodes[_key(path)]["acl"])]

    def mount(self, path: Any, dev: int = 0) -> frozenset[str]:
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
    monkeypatch.setattr(exec_trust, "_group_name", lambda gid: GROUPS.get(gid))
    monkeypatch.setattr(exec_trust, "_account_home", lambda uid: HOME)
    monkeypatch.setattr(exec_trust, "_sha256", d.sha256)
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


def test_a_file_of_a_person_s_is_not_run_unasked_naming_its_owner_and_what_to_do(
    disk: Disk,
) -> None:
    disk.dir("/Users/troy", uid=501).dir("/Users/troy/llama", uid=501)
    disk.file("/Users/troy/llama/llama-server", uid=501)
    refused(
        "/Users/troy/llama/llama-server",
        "Eugene's own account runs /Users/troy/llama/llama-server only once an owner approves it",
        "/Users/troy, which belongs to troy",
        f"copy its folder into {HOME} as an administrator, or let Eugene install the engine",
    )


def test_a_folder_of_a_person_s_above_it_is_not_run_unasked(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501).dir("/Users/troy/llama", uid=400)
    disk.file("/Users/troy/llama/llama-server", uid=400)
    refused("/Users/troy/llama/llama-server", "which belongs to troy")


def test_a_folder_others_can_write_is_refused(disk: Disk) -> None:
    disk.dir("/usr/local/shared", mode=0o777).file("/usr/local/shared/llama-server")
    refused(
        "/usr/local/shared/llama-server",
        "will not run /usr/local/shared/llama-server",
        "/usr/local/shared, which can be changed by others (mode 0777: anyone)",
        f"Copy its folder into {HOME} as an administrator, or let Eugene install the engine",
    )


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


# --- the three kinds (part 8c) -------------------------------------------------


def test_root_s_and_eugene_s_own_program_is_ok_and_not_hashed(disk: Disk) -> None:
    disk.dir("/usr/local/eugene-plexus", uid=400).file(
        "/usr/local/eugene-plexus/llama-server", uid=400
    )
    found = exec_trust.assess("/usr/local/eugene-plexus/llama-server")
    assert found.kind is exec_trust.Kind.ok
    assert found.program == "/usr/local/eugene-plexus/llama-server"
    assert (found.account, found.sha256, found.message()) == (None, None, None)
    assert disk.hashed == []


def test_a_program_only_one_other_account_can_change_needs_approval(disk: Disk) -> None:
    """Every item is root's, Eugene's or troy's, and no group or other can
    write any of them: troy's, to approve. The verdict carries what an
    approval binds, so no caller works it out again."""
    disk.dir("/Users/troy", uid=501).dir("/Users/troy/llama", uid=501, mode=0o700)
    disk.file("/Users/troy/llama/llama-server", uid=501, content=b"\x7fELF troy's build")
    found = exec_trust.assess("/Users/troy/llama/llama-server")
    assert found.kind is exec_trust.Kind.needs_approval
    assert (found.account, found.account_uid) == ("troy", 501)
    assert found.program == "/Users/troy/llama/llama-server"
    assert found.sha256 == hashlib.sha256(b"\x7fELF troy's build").hexdigest()
    assert [(c.path, c.uid, stat.S_IMODE(c.mode)) for c in found.chain] == [
        ("/", 0, 0o755),
        ("/Users", 0, 0o755),
        ("/Users/troy", 501, 0o755),
        ("/Users/troy/llama", 501, 0o700),
        ("/Users/troy/llama/llama-server", 501, 0o755),
    ]


def test_a_folder_a_group_can_change_is_refused_naming_who_can_write(disk: Disk) -> None:
    disk.dir("/usr/local/shared", mode=0o775, gid=20).file("/usr/local/shared/llama-server")
    found = exec_trust.assess("/usr/local/shared/llama-server")
    assert found.kind is exec_trust.Kind.refused
    assert found.sha256 is None and disk.hashed == []
    refused(
        "/usr/local/shared/llama-server",
        "can be changed by others (mode 0775: its owner root and group staff)",
    )


def test_a_person_s_program_their_group_can_change_is_refused_not_offered(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501, mode=0o775, gid=20).file("/Users/troy/llama-server", uid=501)
    assert exec_trust.assess("/Users/troy/llama-server").kind is exec_trust.Kind.refused


def test_a_program_two_accounts_can_change_is_refused(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501).dir("/Users/alice", uid=502)
    disk.file("/Users/alice/llama-server", uid=502)
    disk.link("/Users/troy/llama-server", "/Users/alice/llama-server")
    found = exec_trust.assess("/Users/troy/llama-server")
    assert found.kind is exec_trust.Kind.refused and found.account is None
    refused("/Users/troy/llama-server", "more than one account can change it (alice, troy)")


def test_a_refusal_carries_its_verdict(disk: Disk) -> None:
    disk.dir("/Users/troy", uid=501).file("/Users/troy/llama-server", uid=501)
    with pytest.raises(exec_trust.Untrusted) as caught:
        exec_trust.trusted_program("/Users/troy/llama-server")
    assert caught.value.verdict is not None
    assert caught.value.verdict.kind is exec_trust.Kind.needs_approval


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
    assert exec_trust.assess("/usr/local/bin/vllm").kind is exec_trust.Kind.refused
    refused(
        "/usr/local/bin/vllm",
        "names its interpreter through PATH",
        "Name the interpreter by its full path on the script's first line",
    )


def test_a_script_naming_a_relative_interpreter_is_refused(disk: Disk) -> None:
    """`#!python` is found from the folder the script starts in, or by PATH
    for a probe, not from the agent's folder the check would read: refused
    even with a trusted `python` there (8c review, 3)."""
    disk.dir("/usr/local/eugene-plexus", uid=400).file("/usr/local/eugene-plexus/python", uid=400)
    disk.file("/usr/local/bin/vllm", content=b"#!python\nimport vllm\n")
    found = exec_trust.assess("/usr/local/bin/vllm")
    assert found.kind is exec_trust.Kind.refused
    refused("/usr/local/bin/vllm", "names its interpreter by a relative path (#!python)")


def test_uv_s_exec_line_naming_a_relative_python_is_refused(disk: Disk) -> None:
    disk.file("/bin/sh")
    disk.file(
        "/usr/local/bin/vllm",
        content=b"#!/bin/sh\n'''exec' \"python3\" \"$0\" \"$@\"\n' '''\n",
    )
    assert exec_trust.assess("/usr/local/bin/vllm").kind is exec_trust.Kind.refused
    refused("/usr/local/bin/vllm", "names its interpreter by a relative path (python3)")


def test_an_interpreter_that_is_env_under_another_name_is_refused(disk: Disk) -> None:
    """`#!/usr/local/bin/run python3`, with `run` a link to `env`: PATH
    chooses the program all the same (8c review, 3)."""
    disk.file("/usr/bin/env")
    disk.link("/usr/local/bin/run", "/usr/bin/env")
    disk.file("/usr/local/bin/vllm", content=b"#!/usr/local/bin/run python3\nimport vllm\n")
    assert exec_trust.assess("/usr/local/bin/vllm").kind is exec_trust.Kind.refused
    refused("/usr/local/bin/vllm", "is env, which lets PATH choose the program")


def test_a_program_that_is_env_under_another_name_is_refused(disk: Disk) -> None:
    disk.file("/usr/bin/env")
    disk.link("/usr/local/bin/llama-server", "/usr/bin/env")
    assert exec_trust.assess("/usr/local/bin/llama-server").kind is exec_trust.Kind.refused
    refused("/usr/local/bin/llama-server", "is env, which lets PATH choose the program")


def test_a_script_it_cannot_read_is_refused_naming_the_cause(disk: Disk) -> None:
    """Root's execute-only script: what runs it cannot be known, so it is
    refused, not taken for a binary (8c review, 4)."""
    disk.file("/usr/local/bin/vllm", mode=0o111, content=b"#!/Users/alice/python\n")
    disk.unreadable.add("/usr/local/bin/vllm")
    assert exec_trust.assess("/usr/local/bin/vllm").kind is exec_trust.Kind.refused
    refused("/usr/local/bin/vllm", "cannot be read to find what runs it (Permission denied)")


def test_an_environment_whose_settings_or_packages_cannot_be_read_is_refused(
    disk: Disk,
) -> None:
    disk.dir("/opt/py").dir("/opt/py/bin").file("/opt/py/bin/python3.12").dir("/opt/py/lib")
    _venv(disk, "/usr/local/eugene-plexus/vllm-env", home="/opt/py/bin")
    disk.unreadable.add("/usr/local/eugene-plexus/vllm-env/pyvenv.cfg")
    assert (
        exec_trust.assess("/usr/local/eugene-plexus/vllm-env/bin/vllm").kind
        is exec_trust.Kind.refused
    )
    refused("/usr/local/eugene-plexus/vllm-env/bin/vllm", "pyvenv.cfg cannot be read")
    disk.unreadable.clear()
    disk.unlistable.add("/usr/local/eugene-plexus/vllm-env/lib")
    refused("/usr/local/eugene-plexus/vllm-env/bin/vllm", "cannot be listed")


def test_a_person_s_script_is_not_read_when_it_is_not_a_plain_file(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk goes on past a person's folder to say who else can change
    a program, so it reads files a person controls: only the first bytes
    of a plain file, so a pipe cannot hold the agent."""
    disk.dir("/Users/troy", uid=501).file("/Users/troy/llama-server", uid=501)
    disk.nodes["/Users/troy/llama-server"]["kind"] = stat.S_IFIFO
    monkeypatch.setattr(exec_trust, "_head", lambda path: pytest.fail("read a pipe"))
    assert exec_trust.assess("/Users/troy/llama-server").kind is exec_trust.Kind.refused


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


def test_macos_an_acl_entry_naming_a_group_with_a_space_is_refused(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`group:Domain Users allow add_file,delete_child` on troy's own 0755
    folder: the group's members can replace the program, so it is refused,
    not offered as troy's alone (8c review, 2)."""
    monkeypatch.setattr(exec_trust, "_platform", lambda: "darwin")
    disk.dir("/Users/troy", uid=501).dir(
        "/Users/troy/llama", uid=501, acl=["group:Domain Users allow add_file,delete_child"]
    )
    disk.file("/Users/troy/llama/llama-server", uid=501)
    assert exec_trust.assess("/Users/troy/llama/llama-server").kind is exec_trust.Kind.refused
    refused(
        "/Users/troy/llama/llama-server",
        "carries an ACL entry that lets group Domain Users add_file, delete_child",
    )


def test_macos_an_acl_entry_it_cannot_understand_is_refused(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exec_trust, "_platform", lambda: "darwin")
    disk.dir("/usr/local/row8").file(
        "/usr/local/row8/llama-server", acl=["ABCDEFAB-0000-1111-2222-333344445555 allow write"]
    )
    assert exec_trust.assess("/usr/local/row8/llama-server").kind is exec_trust.Kind.refused
    refused("/usr/local/row8/llama-server", "carries an ACL entry Eugene cannot read")


def test_macos_deny_entries_and_its_own_entries_are_not_refused(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exec_trust, "_platform", lambda: "darwin")
    disk.dir(
        "/usr/local/row8",
        acl=[
            "group:everyone deny delete",
            "user:_eugeneplexus allow add_file",
            "group:Domain Users inherited deny write,delete",
        ],
    )
    disk.file(
        "/usr/local/row8/llama-server",
        acl=["user:alice allow read,execute", "group:Domain Users allow read"],
    )
    assert ran("/usr/local/row8/llama-server") == "/usr/local/row8/llama-server"


def test_macos_a_volume_that_ignores_ownership_is_refused(
    disk: Disk, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exec_trust, "_platform", lambda: "darwin")
    disk.dir("/Volumes/USB").file("/Volumes/USB/llama-server")
    disk.noowners.add("/Volumes/USB")
    refused("/Volumes/USB/llama-server", "ignores ownership")


# --- the real disk ---------------------------------------------------------------


def test_the_real_hash_sees_one_byte_changed_with_its_size_and_time_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An approval names the file's SHA-256: one byte changed, the size the
    same and the modification time put back, it differs, every time it is
    asked (8c review). No answer is kept between checks: a file's change
    time moves on a coarse clock (a second on HFS+, a jiffy on Linux), so
    two writes can share it. Here the clock does not move at all."""
    program = tmp_path / "llama-server"
    program.write_bytes(b"\x7fELF build one")
    before = os.stat(program)
    real_fstat = os.fstat

    def same_clock(fd: int) -> Any:
        found = real_fstat(fd)
        return SimpleNamespace(
            **{name: getattr(found, name) for name in dir(found) if name.startswith("st_")}
            | {"st_mtime_ns": before.st_mtime_ns, "st_ctime_ns": before.st_ctime_ns}
        )

    monkeypatch.setattr(exec_trust.os, "fstat", same_clock)
    first = exec_trust._sha256(program)
    assert first == hashlib.sha256(b"\x7fELF build one").hexdigest()
    program.write_bytes(b"\x7fELF build two")
    os.utime(program, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(program)
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    assert exec_trust._sha256(program) == hashlib.sha256(b"\x7fELF build two").hexdigest()


def test_vllm_s_environment_probe_runs_only_the_interpreter_that_was_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In Eugene's own account the probe runs the absolute interpreter the
    script names, which the check walked; never a bare name PATH would
    choose, never `env` (8c review, 3)."""
    from eugene_plexus_agent.engines import vllm

    script = tmp_path / "vllm"
    monkeypatch.setattr(exec_trust, "own_account", lambda: True)
    monkeypatch.setattr(vllm.shutil, "which", lambda name: "/home/alice/bin/python3")
    for first_line in (b"#!python\n", b"#!/usr/bin/env python3\n"):
        script.write_bytes(first_line)
        assert vllm.interpreter_from_shebang(script) is None, first_line
    script.write_bytes(b"#!/opt/venv/bin/python\n")
    assert vllm.interpreter_from_shebang(script) == "/opt/venv/bin/python"
    # A person's own install keeps what it did: `env` through PATH.
    monkeypatch.setattr(exec_trust, "own_account", lambda: False)
    script.write_bytes(b"#!/usr/bin/env python3\n")
    assert vllm.interpreter_from_shebang(script) == "/home/alice/bin/python3"


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

    def refuse(path: Any, **kw: Any) -> Path:
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
    monkeypatch.setattr(exec_trust, "trusted_program", lambda path, **kw: checked)
    probed: list[Path] = []
    adapter = LlamaCppAdapter()
    monkeypatch.setattr(adapter, "probe_version", lambda binary: probed.append(binary) or "1")
    found = adapter.resolve_binary(_spec(binary=str(exe)))
    assert found.path == checked
    assert probed == [checked]


def test_a_refused_runtime_binary_is_named_for_the_console_and_cleared_when_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Runtime.refusedBinary`: the runtime's own `binary`, which the
    console offers to clear so Eugene's own build runs instead."""
    import logging

    from eugene_plexus_agent.runtimes import RuntimeSupervisor, _RuntimePlanner
    from eugene_plexus_agent.supervisor import SpawnPlanError

    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    spec = _spec(binary=str(exe), port=8090)
    verdict = exec_trust.Verdict(
        kind=exec_trust.Kind.needs_approval,
        given=str(exe),
        program="/Users/troy/llama-server",
        account="troy",
        account_uid=501,
        sha256="ab" * 32,
    )

    def refuse(path: Any, **kw: Any) -> Path:
        raise exec_trust.Untrusted(
            f"{path} belongs to troy; Eugene's own account runs programs only", verdict
        )

    monkeypatch.setattr(exec_trust, "trusted_program", refuse)
    planner = _RuntimePlanner(
        spec,
        LlamaCppAdapter(),
        logging.getLogger("test"),
        get_config=lambda key: [str(tmp_path)] if key == "engineBinaryRoots" else None,
    )
    with pytest.raises(SpawnPlanError, match="belongs to troy"):
        planner.plan()
    assert planner.refused_binary == str(exe)
    supervisor = RuntimeSupervisor(log=logging.getLogger("test"))
    supervisor._planners[spec.name] = planner
    shown = supervisor.compose(spec)
    assert shown.refusedBinary == str(exe)
    # Part 8c: what the console asks about, and what an approval names.
    assert shown.execTrust is not None
    assert shown.execTrust.model_dump(mode="json", exclude_none=True) == {
        "kind": "needs_approval",
        "path": "/Users/troy/llama-server",
        "account": "troy",
        "sha256": "ab" * 32,
    }

    monkeypatch.setattr(exec_trust, "trusted_program", lambda path, **kw: Path(str(path)))
    monkeypatch.setattr(LlamaCppAdapter, "probe_version", lambda self, binary: "1")
    planner.plan()
    assert planner.refused_binary is None
    assert supervisor.compose(spec).execTrust is None


def test_an_engine_found_but_not_run_says_why_in_the_engine_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GET /v1/engines`: vLLM's configured program is one only troy can
    change, so the listing carries what to approve (part 8c)."""
    from eugene_plexus_agent import runtimes
    from eugene_plexus_agent.engines.base import UntrustedBinaryError

    verdict = exec_trust.Verdict(
        kind=exec_trust.Kind.needs_approval,
        given="/Users/troy/venv/bin/vllm",
        program="/Users/troy/venv/bin/vllm",
        account="troy",
        account_uid=501,
        sha256="cd" * 32,
    )

    def discover(self: Any, configured: str | None = None) -> Any:
        raise UntrustedBinaryError("only once an owner approves it", str(configured), verdict)

    monkeypatch.setattr(VllmAdapter, "discover", discover)
    listed = runtimes.describe_engines(
        get_config=lambda key: "/Users/troy/venv/bin/vllm" if key == "vllmBinary" else None
    )
    vllm = next(e for e in listed if e.engine is EngineKind.vllm)
    assert vllm.available is False and vllm.execTrust is not None
    assert (vllm.execTrust.kind.value, vllm.execTrust.path, vllm.execTrust.sha256) == (
        "needs_approval",
        "/Users/troy/venv/bin/vllm",
        "cd" * 32,
    )


def test_a_configured_engine_binary_is_refused_before_its_environment_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "vllm"
    exe.write_text("#!/usr/bin/python3\n", encoding="utf-8")

    def refuse(path: Any, **kw: Any) -> Path:
        raise exec_trust.Untrusted(f"{path} belongs to troy; ...")

    monkeypatch.setattr(exec_trust, "trusted_program", refuse)
    read: list[Path] = []
    from eugene_plexus_agent.engines import vllm

    monkeypatch.setattr(vllm, "inspect_python_environment", lambda path: read.append(path))
    with pytest.raises(EngineUnavailableError, match="belongs to troy"):
        VllmAdapter().discover(configured=str(exe))
    assert read == []
