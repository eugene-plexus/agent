"""Logins for the file servers a Library folder lives on (R2.6).

The measurement this exists for, taken on the live install 2026-09-18:
the same share opens from a logon session holding a Credential Manager
entry and answers `WinError 1272` from one that does not — because the
share is guest-open and **Windows 11 refuses the guest fallback**. A
LocalSystem service is a session that does not hold the entry, so
without this the install comes back from a reboot unable to read a
single model with every health check green.

Most of what can go wrong here is not a failed login. It is the field
losing a password nobody meant to change.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent import security, share_credentials

MASTER_KEY = b"0" * 32
OTHER_KEY = b"1" * 32


# --------------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------------- #


def test_writing_back_a_redacted_row_keeps_the_password() -> None:
    """**The reproduction for the defect that costs an install its models.**

    `GET /v1/config` redacts, so every UI holds a row whose password is
    null. Saving a change to the user name sends that row back. If the
    merge did not exist, the secret would be gone — and nothing would
    say so until the next reboot, when the share stops opening.
    """
    stored, error = share_credentials.merge_and_seal(
        [{"host": "nas", "username": "tcorbin", "password": "hunter2"}], [], MASTER_KEY
    )
    assert error is None

    redacted = share_credentials.redact_entries(stored)
    assert redacted == [{"host": "nas", "username": "tcorbin", "password": None}]

    # The UI edits the user name on the row it was shown and PATCHes it.
    written_back = [{"host": "nas", "username": "someone-else", "password": None}]
    merged, error = share_credentials.merge_and_seal(written_back, stored, MASTER_KEY)
    assert error is None
    assert merged[0]["username"] == "someone-else"

    opened = share_credentials.unseal_entries(merged, MASTER_KEY)
    assert opened[0]["password"] == "hunter2", "the password did not survive the round trip"


def test_an_empty_password_clears_it_and_a_missing_one_does_not() -> None:
    """The only way to remove a stored password is to say so.

    Absent means *keep*; empty string means *clear*. Collapsing the two
    is what makes the round trip above unsafe, so they are pinned apart.
    """
    stored, _ = share_credentials.merge_and_seal(
        [{"host": "nas", "username": "u", "password": "p"}], [], MASTER_KEY
    )
    cleared, _ = share_credentials.merge_and_seal(
        [{"host": "nas", "username": "u", "password": ""}], stored, MASTER_KEY
    )
    assert cleared[0]["password"] is None
    assert share_credentials.has_password(cleared[0]) is False


def test_the_password_is_sealed_at_rest() -> None:
    """Not readable from the config file, and not openable with another key."""
    stored, _ = share_credentials.merge_and_seal(
        [{"host": "nas", "username": "u", "password": "hunter2"}], [], MASTER_KEY
    )
    assert "hunter2" not in repr(stored)
    assert security.is_envelope(stored[0]["password"])

    wrong = share_credentials.unseal_entries(stored, OTHER_KEY)
    assert wrong[0]["password"] is None, "a different master key opened the envelope"

    missing = share_credentials.unseal_entries(stored, None)
    assert missing[0]["password"] is None


def test_a_locked_agent_refuses_to_store_a_new_password() -> None:
    """Rather than writing it in the clear, which is the tempting bug."""
    _, error = share_credentials.merge_and_seal(
        [{"host": "nas", "username": "u", "password": "hunter2"}], [], None
    )
    assert error is not None
    assert "locked" in error


def test_a_password_written_by_hand_still_works() -> None:
    """An operator who edited `agent.yaml` is not punished for it.

    Plaintext is honoured on read and re-sealed the next time the field
    is saved, so the migration from "I put it in the file" happens on
    its own.
    """
    opened = share_credentials.unseal_entries(
        [{"host": "nas", "username": "u", "password": "typed-by-hand"}], MASTER_KEY
    )
    assert opened[0]["password"] == "typed-by-hand"


# --------------------------------------------------------------------------- #
# what the field will accept
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        ({"username": "u"}, "needs a host"),
        ({"host": "  ", "username": "u"}, "needs a host"),
        ({"host": "nas"}, "needs a user name"),
        ({"host": r"\\nas\downloads", "username": "u"}, "not a UNC path"),
        ({"host": "nas/downloads", "username": "u"}, "with no share name"),
        ({"host": "nas", "username": "u", "realm": "x"}, "unexpected field"),
        ({"host": "nas", "username": "u", "password": 7}, "must be text"),
    ],
)
def test_the_shapes_that_are_refused(entry: dict, fragment: str) -> None:
    error = share_credentials.validate_entries([entry])
    assert error is not None
    assert fragment in error


def test_two_rows_for_one_server_are_refused_rather_than_silently_ignored() -> None:
    """Windows allows one session per server (1219).

    A second row could never take effect, so accepting it would be a
    field that lies: an operator would see their credential saved and
    their share still refused.
    """
    error = share_credentials.validate_entries(
        [
            {"host": "nas", "username": "a", "password": "1"},
            {"host": "NAS", "username": "b", "password": "2"},
        ]
    )
    assert error is not None
    assert "already listed" in error


def test_a_sealed_password_survives_validation() -> None:
    """The validator sees the value on the way out of the file too.

    Refusing an envelope here would make the agent unable to load its
    own config the first time it restarted after a password was saved —
    a config that writes successfully and will not load.
    """
    stored, _ = share_credentials.merge_and_seal(
        [{"host": "nas", "username": "u", "password": "p"}], [], MASTER_KEY
    )
    assert share_credentials.validate_entries(stored) is None


# --------------------------------------------------------------------------- #
# what a person is told
# --------------------------------------------------------------------------- #


def test_1272_is_explained_as_the_thing_it_actually_is() -> None:
    """The code a guest-open share produces, which is the confusing one.

    *"The share has no password"* and *"anything can open it"* are
    different sentences, and an operator who has just been refused needs
    the second one named or they will go looking at the NAS.
    """
    said = share_credentials.explain(1272)
    assert "user name" in said
    assert "does not need a password" in said


def test_an_unknown_code_is_reported_by_number() -> None:
    """A number that can be searched for beats a sentence that guesses."""
    assert "4242" in share_credentials.explain(4242)


def test_already_connected_is_not_a_failure(monkeypatch) -> None:
    """1219 means this session already has a connection to that server.

    Tearing it down to replace it would break whatever opened it — on a
    logon-task install, the person's own Explorer window.
    """
    monkeypatch.setattr(
        share_credentials,
        "connect",
        lambda host, username, password: share_credentials.ConnectResult(
            host=host, username=username, code=1219, ok=True, detail="already"
        ),
    )
    results = share_credentials.connect_all([{"host": "nas", "username": "u"}])
    assert results[0].ok is True


def test_a_second_row_for_one_server_is_not_dialled_twice(monkeypatch) -> None:
    """Belt to `validate_entries`' braces: stored config can predate it."""
    dialled: list[str] = []

    def record(host, username, password):
        dialled.append(host)
        return share_credentials.ConnectResult(
            host=host, username=username, code=0, ok=True, detail="ok"
        )

    monkeypatch.setattr(share_credentials, "connect", record)
    share_credentials.connect_all(
        [{"host": "nas", "username": "a"}, {"host": "NAS", "username": "b"}]
    )
    assert dialled == ["nas"]


# --------------------------------------------------------------------------- #
# the Test button
# --------------------------------------------------------------------------- #


def test_test_tries_the_row_being_typed_not_the_row_last_saved() -> None:
    """`POST /v1/config/test` exists to check a value before saving it."""
    saved = [{"host": "nas", "username": "old", "password": "stored"}]
    overlaid = share_credentials.overlay_typed(
        [{"host": "nas", "username": "new", "password": "typed"}], saved
    )
    assert overlaid == [{"host": "nas", "username": "new", "password": "typed"}]


def test_test_falls_back_to_the_stored_password_for_an_untouched_row() -> None:
    """The box was shown a redaction, so it has nothing else to send."""
    saved = [{"host": "nas", "username": "u", "password": "stored"}]
    overlaid = share_credentials.overlay_typed(
        [{"host": "nas", "username": "u", "password": None}], saved
    )
    assert overlaid[0]["password"] == "stored"


def test_test_does_not_try_a_row_the_operator_deleted() -> None:
    saved = [
        {"host": "nas", "username": "u", "password": "p"},
        {"host": "other", "username": "u", "password": "p"},
    ]
    overlaid = share_credentials.overlay_typed([{"host": "nas", "username": "u"}], saved)
    assert [e["host"] for e in overlaid] == ["nas"]


# --------------------------------------------------------------------------- #
# through the route
# --------------------------------------------------------------------------- #


def test_the_config_endpoint_never_hands_back_a_password(authed_client: TestClient) -> None:
    """Driven through the route, because a module test is not a wiring test.

    The redaction lives in `GET /v1/config` and the sealing in `PATCH`.
    A test of `redact_entries` alone passes just as happily against a
    route that forgot to call it.
    """
    saved = authed_client.patch(
        "/v1/config",
        json={"shareCredentials": [{"host": "nas", "username": "u", "password": "hunter2"}]},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["rejected"] == []

    read = authed_client.get("/v1/config")
    assert read.status_code == 200
    rows = read.json()["shareCredentials"]
    assert rows == [{"host": "nas", "username": "u", "password": None}]
    assert "hunter2" not in read.text
