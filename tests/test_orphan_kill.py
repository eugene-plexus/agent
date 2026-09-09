"""Tests for the cross-platform orphan-prevention helpers.

The platform-specific paths (prctl, Job Objects) require real syscalls
to exercise meaningfully, so we don't try here. These tests just verify
the dispatch logic — that the right preexec / Job Object setup is
selected based on `sys.platform`.
"""

from __future__ import annotations

import sys

import pytest

from eugene_plexus_agent import orphan_kill


def test_kwargs_for_platform_includes_preexec_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    kwargs = orphan_kill.kwargs_for_platform()
    assert "preexec_fn" in kwargs
    assert kwargs["preexec_fn"] is orphan_kill.linux_pdeathsig_preexec


def test_kwargs_for_platform_omits_preexec_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    kwargs = orphan_kill.kwargs_for_platform()
    # Windows uses Job Object, not preexec.
    assert "preexec_fn" not in kwargs


def test_kwargs_for_platform_omits_preexec_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS has no pdeathsig equivalent; we don't pass preexec_fn there."""
    monkeypatch.setattr(sys, "platform", "darwin")
    kwargs = orphan_kill.kwargs_for_platform()
    assert "preexec_fn" not in kwargs


def test_windows_job_no_op_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    orphan_kill._reset_for_tests()
    job = orphan_kill.windows_job()
    assert job is None


def test_is_orphan_kill_supported_reports_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert orphan_kill.is_orphan_kill_supported() is True

    monkeypatch.setattr(sys, "platform", "win32")
    assert orphan_kill.is_orphan_kill_supported() is True

    monkeypatch.setattr(sys, "platform", "darwin")
    assert orphan_kill.is_orphan_kill_supported() is False
