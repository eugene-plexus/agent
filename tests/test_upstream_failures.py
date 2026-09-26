"""A failed upstream request says what actually happened.

Found 2026-09-26 on a friend's first install: *"llama.cpp cannot be
installed: could not reach the upstream release list. Check network
access"*, on a machine whose network, and whose copy of our own Python,
both reached GitHub fine. Every failure — a rate limit, a replaced
certificate, a DNS failure, a timeout — became that one sentence, and the
exception went to a DEBUG log nobody runs at. Each cause has a case here,
and each asserts the words a person would act on, not merely that some
message exists.
"""

from __future__ import annotations

import email.message
import io
import json
import logging
import socket
import ssl
import time
import urllib.error
from datetime import datetime

import pytest

from eugene_plexus_agent._generated.models import (
    Accelerator,
    Arch,
    EngineKind,
    HostAccelerator,
    Os,
)
from eugene_plexus_agent.engines import acquisition as acq
from eugene_plexus_agent.engines.acquisition import (
    AcquisitionError,
    GitHubReleases,
    ReleaseAsset,
    Unavailable,
    describe_fetch_failure,
)
from eugene_plexus_agent.engines.llama_cpp import LlamaCppAdapter

URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
GENERIC = "Check network access"


def _http_error(code: int, headers: dict[str, str] | None = None, body: object = None):
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    payload = json.dumps(body).encode() if body is not None else b""
    return urllib.error.HTTPError(URL, code, "Forbidden", message, io.BytesIO(payload))


def _rate_limited(reset: int) -> urllib.error.HTTPError:
    return _http_error(
        403,
        {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(reset),
        },
        {"message": "API rate limit exceeded for 203.0.113.9."},
    )


# What Windows says through `truststore`, verbatim from a live run against
# badssl.com on 2026-09-26, and what OpenSSL says for the same failures.
UNTRUSTED_WINDOWS = (
    "A certificate chain processed, but terminated in a root certificate "
    "which is not trusted by the trust provider."
)
EXPIRED_WINDOWS = (
    "A required certificate is not within its validity period when verifying "
    "against the current system clock or the timestamp in the signed file."
)
WRONG_NAME_WINDOWS = "The certificate's CN name does not match the passed value."


def _cert_error(message: str = UNTRUSTED_WINDOWS) -> urllib.error.URLError:
    error = ssl.SSLCertVerificationError(1, "certificate verify failed")
    error.verify_message = message
    return urllib.error.URLError(error)


def _describe(error: BaseException) -> str:
    return describe_fetch_failure(error, url=URL, timeout=10.0).sentence()


# --------------------------------------------------------------------------- #
# Each cause, in its own words
# --------------------------------------------------------------------------- #


def test_a_rate_limit_names_the_limit_and_when_it_resets() -> None:
    """The case a person cannot guess: they made no requests. The limit is
    per ADDRESS, and a shared one (Starlink, mobile data) is spent by others."""
    reset = int(time.time()) + 600
    said = _describe(_rate_limited(reset))
    assert "limit of 60 requests an hour" in said
    assert f"{datetime.fromtimestamp(reset):%H:%M}" in said
    assert "in 10 minutes" in said
    assert "shared by many customers" in said
    assert GENERIC not in said


def test_a_secondary_rate_limit_says_how_long_to_wait() -> None:
    said = _describe(_http_error(429, {"Retry-After": "120"}))
    assert "limiting requests" in said
    assert "2 minute(s)" in said


@pytest.mark.parametrize(
    "message", [UNTRUSTED_WINDOWS, "unable to get local issuer certificate"], ids=["win", "openssl"]
)
def test_an_untrusted_certificate_points_at_what_replaced_it(message: str) -> None:
    """The OS is the verifier now, so a root it fetches on demand or an
    antivirus root it holds both pass; what is left is something whose
    certificate the OS does not trust, or an OS forbidden to fetch roots."""
    said = _describe(_cert_error(message))
    assert "could not verify the security certificate api.github.com presented" in said
    assert message.rstrip(".") in said
    assert "replacing its certificate" in said
    assert "date and time" not in said


def test_the_group_policy_that_blocks_root_downloads_is_named_on_windows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acq.sys, "platform", "win32")
    assert "Automatic Root Certificates Update" in _describe(_cert_error())
    monkeypatch.setattr(acq.sys, "platform", "linux")
    assert "Automatic Root Certificates Update" not in _describe(_cert_error())


@pytest.mark.parametrize(
    "message", [EXPIRED_WINDOWS, "certificate has expired", "certificate is not yet valid"]
)
def test_a_certificate_out_of_date_sends_a_person_to_the_clock(message: str) -> None:
    """GitHub's certificate is not expired; a clock that is wrong makes it
    look so, and that is fixed on this machine, not on the network."""
    said = _describe(_cert_error(message))
    assert "date and time" in said
    assert "replacing its certificate" not in said


@pytest.mark.parametrize("message", [WRONG_NAME_WINDOWS, "Hostname mismatch"])
def test_a_certificate_for_another_name_says_something_else_answered(message: str) -> None:
    said = _describe(_cert_error(message))
    assert "different name" in said
    assert "date and time" not in said


def test_a_dns_failure_says_the_name_could_not_be_looked_up() -> None:
    said = _describe(urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")))
    assert "could not look up the address of api.github.com" in said
    assert "DNS" in said


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))],
    ids=["read", "connect"],
)
def test_a_timeout_says_how_long_it_waited(error: BaseException) -> None:
    said = _describe(error)
    assert "did not answer within 10 seconds" in said


def test_a_refused_connection_says_refused() -> None:
    said = _describe(urllib.error.URLError(ConnectionRefusedError(10061, "refused")))
    assert "was refused" in said
    assert acq._python_that_connects() in said


def test_a_reset_connection_points_at_a_firewall_or_antivirus() -> None:
    said = _describe(ConnectionResetError(10054, "forcibly closed"))
    assert "was cut off" in said
    assert "antivirus" in said


def test_an_answer_that_is_not_json_suspects_a_sign_in_page() -> None:
    said = _describe(json.JSONDecodeError("Expecting value", "<html>", 0))
    assert "was not GitHub's release list" in said
    assert "sign-in page" in said


def test_a_server_error_carries_the_status_and_githubs_own_words() -> None:
    said = _describe(_http_error(503, body={"message": "Service Unavailable"}))
    assert "HTTP 503" in said
    assert "Service Unavailable" in said
    assert "few minutes" in said


def test_a_refusal_that_is_not_a_rate_limit_is_not_called_one() -> None:
    """A 403 without the exhausted-limit header is something else, and
    reporting it as the rate limit would send a person to wait an hour
    for nothing."""
    said = _describe(_http_error(403, body={"message": "Repository access blocked"}))
    assert "HTTP 403" in said
    assert "Repository access blocked" in said
    assert "limit" not in said


# --------------------------------------------------------------------------- #
# Through the release list, the adapter and the install
# --------------------------------------------------------------------------- #


def _failing_releases(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> GitHubReleases:
    def raising(_url: str) -> object:
        raise error

    monkeypatch.setattr(GitHubReleases, "_fetch", staticmethod(raising))
    return GitHubReleases("ggml-org/llama.cpp")


def _adapter_with(releases: GitHubReleases) -> LlamaCppAdapter:
    adapter = LlamaCppAdapter()
    # `releases` is a class attribute shared by every adapter; shadow it.
    adapter.releases = releases
    return adapter


_HOST = HostAccelerator(
    os=Os.windows, arch=Arch.x64, accelerator=Accelerator.cuda, acceleratorVersion="13.3"
)


def test_the_install_card_says_what_happened_not_check_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The friend's card, end to end: the reason reaching the UI is the
    classified cause, not the generic guess."""
    adapter = _adapter_with(_failing_releases(monkeypatch, _rate_limited(int(time.time()) + 900)))
    plan = adapter.plan_latest(_HOST)
    assert isinstance(plan, Unavailable)
    assert plan.reason.startswith("could not get the list of llama.cpp builds from GitHub:")
    assert "limit of 60 requests an hour" in plan.reason
    assert "set `binary`" in plan.reason
    assert GENERIC not in plan.reason


def test_the_reason_survives_the_back_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside the five-minute back-off nothing is fetched, and the card is
    polled every 15 s: it must keep saying why, not fall back to a guess."""
    adapter = _adapter_with(_failing_releases(monkeypatch, _cert_error()))
    adapter.plan_latest(_HOST)
    again = adapter.plan_latest(_HOST)
    assert isinstance(again, Unavailable)
    assert "could not verify the security certificate" in again.reason


def test_an_answer_with_no_builds_is_not_called_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub answered; nothing it listed has a download. That is a
    different sentence from "could not get the list"."""
    monkeypatch.setattr(GitHubReleases, "_fetch", staticmethod(lambda _url: []))
    adapter = _adapter_with(GitHubReleases("ggml-org/llama.cpp"))
    plan = adapter.plan_latest(_HOST)
    assert isinstance(plan, Unavailable)
    assert "has no build with downloads" in plan.reason
    assert "could not get the list" not in plan.reason


def test_a_named_build_with_no_list_gives_the_lists_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`plan_for(version=...)` said *"build 'b10867' is not among the recent
    releases"* when there were no releases at all, which sends a person to
    pick another build rather than at the network."""
    from eugene_plexus_agent import runtimes

    adapter = _adapter_with(
        _failing_releases(
            monkeypatch, urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))
        )
    )
    monkeypatch.setattr(runtimes, "adapter_for", lambda _kind: adapter)
    plan = runtimes.plan_for(EngineKind.llama_cpp, version="b10867", host=_HOST)
    assert isinstance(plan, Unavailable)
    assert "could not look up the address" in plan.reason
    assert "not among the recent releases" not in plan.reason


def test_a_failure_is_logged_as_a_warning_once_per_cause(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The only record of why something a person asked for did not happen
    cannot live at DEBUG. Once per cause, though: the check repeats every
    five minutes for as long as it fails."""
    releases = _failing_releases(monkeypatch, _cert_error())
    with caplog.at_level(logging.DEBUG, logger=acq.__name__):
        releases.list_releases(force=True)
        releases.list_releases(force=True)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "could not verify the security certificate" in warnings[0].getMessage()
    assert "SSLCertVerificationError" in warnings[0].getMessage()


def test_a_new_cause_is_warned_about_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    errors = iter([_cert_error(), TimeoutError("timed out")])

    def raising(_url: str) -> object:
        raise next(errors)

    monkeypatch.setattr(GitHubReleases, "_fetch", staticmethod(raising))
    releases = GitHubReleases("ggml-org/llama.cpp")
    with caplog.at_level(logging.WARNING, logger=acq.__name__):
        releases.list_releases(force=True)
        releases.list_releases(force=True)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_success_clears_the_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes: list[object] = [TimeoutError("timed out"), []]

    def fetch(_url: str) -> object:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(GitHubReleases, "_fetch", staticmethod(fetch))
    releases = GitHubReleases("ggml-org/llama.cpp")
    releases.list_releases(force=True)
    assert releases.last_failure is not None
    releases.list_releases(force=True)
    assert releases.last_failure is None


def test_a_failed_download_is_described_too(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same classes of failure meet the download, and the install's
    error is what the tray shows."""

    def raising(*_a: object, **_k: object) -> object:
        raise _cert_error()

    monkeypatch.setattr(acq.urllib.request, "urlopen", raising)
    asset = ReleaseAsset(
        name="llama-b10867-bin-win-cpu-x64.zip",
        url="https://github.com/ggml-org/llama.cpp/releases/download/b10867/x.zip",
        size=1,
        digest=None,
    )
    progress = acq._Progress(engine=EngineKind.llama_cpp)
    with pytest.raises(AcquisitionError) as caught:
        acq._download(asset, tmp_path / "x.zip", progress)
    assert "could not verify the security certificate" in str(caught.value)
    assert "github.com" in str(caught.value)


# --------------------------------------------------------------------------- #
# The OS verifies: the fix itself, not only its message
# --------------------------------------------------------------------------- #


class _Captured(Exception):
    pass


def _capture_context(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    seen: list[object] = []

    def capturing(_request: object, *, timeout: float, context: object = None) -> object:
        seen.append(context)
        raise _Captured

    monkeypatch.setattr(acq.urllib.request, "urlopen", capturing)
    return seen


def test_the_release_list_is_verified_by_the_os(monkeypatch: pytest.MonkeyPatch) -> None:
    """The friend's failure: OpenSSL's read of the Windows store has no root
    Windows has not downloaded yet. The OS verifier fetches it."""
    import truststore

    seen = _capture_context(monkeypatch)
    with pytest.raises(_Captured):
        GitHubReleases._fetch(URL)
    assert isinstance(seen[0], truststore.SSLContext)


def test_the_engine_download_is_verified_by_the_os(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`github.com/.../releases/download` chains to the same root as the API."""
    import truststore

    seen = _capture_context(monkeypatch)
    asset = ReleaseAsset(name="x.zip", url="https://github.com/x.zip", size=1, digest=None)
    with pytest.raises(_Captured):
        acq._download(asset, tmp_path / "x.zip", acq._Progress(engine=EngineKind.llama_cpp))
    assert isinstance(seen[0], truststore.SSLContext)


def test_egress_clients_use_the_os_verifier_and_internal_ones_do_not() -> None:
    """Internal hops are plain HTTP to this install and keep the cheap
    shared context; anything bound for the internet is verified by the OS."""
    import truststore

    from eugene_plexus_agent import _http

    egress = _http.egress_ssl_context()
    assert isinstance(egress, truststore.SSLContext)
    assert _http.egress_ssl_context() is egress, "built once, like ssl_context()"
    assert _http.sync_client_for("https://api.github.com/")._transport._pool._ssl_context is egress
    assert (
        _http.sync_client_for("http://127.0.0.1:8081/")._transport._pool._ssl_context
        is _http.ssl_context()
    )
