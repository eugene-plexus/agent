"""Another account reading node.yaml, or adding files to the install (2026-09-24).

The ACE lists below are the ones measured on a live Windows install that
day, SIDs and flags as `icacls` printed them: the service install under
`%ProgramData%` (every local account could read the signing key and add
files anywhere), and a per-user install whose profile hands
`CodexSandboxUsers` Modify.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import install_permissions as ip
from eugene_plexus_agent.install_permissions import Ace, Grant, foreign_grants

SYSTEM = "S-1-5-18"
ADMINS = "S-1-5-32-544"
USERS = "S-1-5-32-545"
CREATOR_OWNER = "S-1-3-0"
ME = "S-1-5-21-1-2-3-1001"
CODEX = "S-1-5-21-1-2-3-2603643044"

OI, CI, IO, INHERITED = 0x1, 0x2, 0x8, 0x10
FULL = 0x1F01FF
MODIFY = 0x1301BF
READ_EXECUTE = 0x1200A9
ADD_FILES = 0x0002 | 0x0004 | 0x0010 | 0x0100  # WD, AD, WEA, WA as icacls prints them
LIST = 0x0001

DIR = Path("C:/ProgramData/EugenePlexus")
NODE = DIR / "node.yaml"


def ace(sid: str, mask: int, flags: int = 0, *, deny: bool = False, account: str = "") -> Ace:
    return Ace(1 if deny else 0, flags, mask, sid, account or sid)


SERVICE_INSTALL_DIR = [
    ace(SYSTEM, FULL, INHERITED | OI | CI),
    ace(ADMINS, FULL, INHERITED | OI | CI),
    ace(CREATOR_OWNER, FULL, INHERITED | OI | CI | IO),
    ace(USERS, READ_EXECUTE, INHERITED | OI | CI, account="BUILTIN\\Users"),
    ace(USERS, ADD_FILES, INHERITED | CI, account="BUILTIN\\Users"),
]
SERVICE_INSTALL_NODE_YAML = [
    ace(SYSTEM, FULL, INHERITED),
    ace(ADMINS, FULL, INHERITED),
    ace(USERS, READ_EXECUTE, INHERITED, account="BUILTIN\\Users"),
]


def test_the_service_install_as_found_lets_every_account_read_and_plant() -> None:
    on_dir = foreign_grants(SERVICE_INSTALL_DIR, own_sid=SYSTEM, is_directory=True, path=DIR)
    on_file = foreign_grants(
        SERVICE_INSTALL_NODE_YAML, own_sid=SYSTEM, is_directory=False, path=NODE
    )
    assert on_dir == [
        Grant(DIR, "BUILTIN\\Users", "read new files in"),
        Grant(DIR, "BUILTIN\\Users", "add files to"),
    ]
    assert on_file == [Grant(NODE, "BUILTIN\\Users", "read")]


def test_a_protected_service_install_is_clean() -> None:
    protected = [ace(SYSTEM, FULL, OI | CI), ace(ADMINS, FULL, OI | CI)]
    assert foreign_grants(protected, own_sid=SYSTEM, is_directory=True, path=DIR) == []
    inherited = [ace(SYSTEM, FULL, INHERITED), ace(ADMINS, FULL, INHERITED)]
    assert foreign_grants(inherited, own_sid=SYSTEM, is_directory=False, path=NODE) == []


def test_the_agents_own_account_is_trusted_and_nobody_elses_is() -> None:
    """A person's account is the agent on a per-user install and a
    stranger to a service install. The same ACE is fine in one and a
    finding in the other."""
    user_full = [ace(SYSTEM, FULL, OI | CI), ace(ME, FULL, OI | CI, account="PC\\troy")]
    assert foreign_grants(user_full, own_sid=ME, is_directory=True, path=DIR) == []
    assert foreign_grants(user_full, own_sid=SYSTEM, is_directory=True, path=DIR) == [
        Grant(DIR, "PC\\troy", "add files to"),
        Grant(DIR, "PC\\troy", "read new files in"),
    ]


def test_a_sandbox_group_with_modify_on_a_per_user_install_is_a_finding() -> None:
    profile = [
        ace(CODEX, MODIFY, INHERITED | OI | CI, account="PC\\CodexSandboxUsers"),
        ace(SYSTEM, FULL, INHERITED | OI | CI),
        ace(ME, FULL, INHERITED | OI | CI),
    ]
    node = [
        ace(CODEX, MODIFY, INHERITED, account="PC\\CodexSandboxUsers"),
        ace(ME, FULL, INHERITED),
    ]
    assert foreign_grants(profile, own_sid=ME, is_directory=True, path=DIR) == [
        Grant(DIR, "PC\\CodexSandboxUsers", "add files to"),
        Grant(DIR, "PC\\CodexSandboxUsers", "read new files in"),
    ]
    assert foreign_grants(node, own_sid=ME, is_directory=False, path=NODE) == [
        Grant(NODE, "PC\\CodexSandboxUsers", "read"),
        Grant(NODE, "PC\\CodexSandboxUsers", "change"),
    ]


def test_creator_owner_is_not_a_finding() -> None:
    """It is replaced on each new file by whoever made it: this agent."""
    only = [ace(CREATOR_OWNER, FULL, OI | CI | IO)]
    assert foreign_grants(only, own_sid=SYSTEM, is_directory=True, path=DIR) == []


def test_listing_a_directory_is_not_a_finding() -> None:
    """Knowing a file called node.yaml exists is not reading it."""
    listing = [ace(USERS, LIST, CI)]
    assert foreign_grants(listing, own_sid=SYSTEM, is_directory=True, path=DIR) == []


def test_a_write_that_only_inherits_does_not_apply_to_the_directory_itself() -> None:
    inherit_only = [ace(USERS, ADD_FILES, CI | IO)]
    assert foreign_grants(inherit_only, own_sid=SYSTEM, is_directory=True, path=DIR) == []


def test_deny_aces_are_not_grants() -> None:
    denied = [ace(USERS, FULL, OI | CI, deny=True)]
    assert foreign_grants(denied, own_sid=SYSTEM, is_directory=True, path=DIR) == []


def test_one_account_in_two_aces_is_one_finding() -> None:
    twice = [ace(USERS, READ_EXECUTE, INHERITED), ace(USERS, READ_EXECUTE)]
    assert foreign_grants(twice, own_sid=SYSTEM, is_directory=False, path=NODE) == [
        Grant(NODE, USERS, "read")
    ]


def test_off_windows_the_check_claims_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ip.sys, "platform", "linux")
    assert ip.check(tmp_path) == []


@pytest.mark.skipif(sys.platform != "win32", reason="reads a real Windows ACL")
def test_a_real_directory_is_read_the_way_icacls_prints_it(tmp_path: Path) -> None:
    """Build the service install's ACL on a real directory and read it
    back through `check`, then protect it and read again. Only findings
    under `tmp_path` count: `check` also reads this test's own venv,
    which is the developer's business."""
    win32security = pytest.importorskip("win32security")
    import win32api

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32security.TOKEN_QUERY)
    me = win32security.GetTokenInformation(token, win32security.TokenUser)[0]

    def set_dacl(entries: list[tuple[str | object, int, int]]) -> None:
        dacl = win32security.ACL()
        for sid, flags, mask in entries:
            if isinstance(sid, str):
                sid = win32security.ConvertStringSidToSid(sid)
            dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, flags, mask, sid)
        win32security.SetNamedSecurityInfo(
            str(tmp_path),
            win32security.SE_FILE_OBJECT,
            win32security.DACL_SECURITY_INFORMATION
            | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            dacl,
            None,
        )

    (tmp_path / "node.yaml").write_text("signingKey: not-a-real-key\n")
    set_dacl(
        [
            (me, OI | CI, FULL),
            (SYSTEM, OI | CI, FULL),
            (USERS, OI | CI, READ_EXECUTE),
            (USERS, CI, ADD_FILES),
        ]
    )
    as_found = {
        (g.path, g.right) for g in ip.check(tmp_path) if tmp_path in (g.path, *g.path.parents)
    }
    assert as_found == {
        (tmp_path, "read new files in"),
        (tmp_path, "add files to"),
        (tmp_path / "node.yaml", "read"),
    }

    set_dacl([(me, OI | CI, FULL), (SYSTEM, OI | CI, FULL), (ADMINS, OI | CI, FULL)])
    assert [g for g in ip.check(tmp_path) if tmp_path in (g.path, *g.path.parents)] == []


def _wait_for_the_check(app: FastAPI) -> None:
    task = app.state.install_permissions_task
    deadline = time.perf_counter() + 5
    while not task.done() and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert task.done()


def test_healthz_names_the_account_and_stays_ok(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    grant = Grant(NODE, "BUILTIN\\Users", "read")
    monkeypatch.setattr(ip, "check", lambda config_dir: [grant])
    caplog.set_level(logging.WARNING, logger="eugene_plexus_agent.app")
    with TestClient(app) as client:
        _wait_for_the_check(app)
        body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["details"]["installPermissions"] == [grant.sentence()]
    assert any(
        "Re-run the installer" in r.getMessage() and grant.sentence() in r.getMessage()
        for r in caplog.records
    )


def test_healthz_says_nothing_when_nothing_is_wrong(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ip, "check", lambda config_dir: [])
    with TestClient(app) as client:
        _wait_for_the_check(app)
        body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert "installPermissions" not in (body.get("details") or {})


def test_reading_the_interpreter_is_not_a_finding_and_adding_to_it_is() -> None:
    """A venv's files are code, not secrets; a `.pth` added to one runs
    as this account at the next start."""
    venv = Path("C:/ProgramData/EugenePlexus/venv")
    found = foreign_grants(
        SERVICE_INSTALL_DIR, own_sid=SYSTEM, is_directory=True, path=venv, holds_secrets=False
    )
    assert found == [Grant(venv, "BUILTIN\\Users", "add files to")]
