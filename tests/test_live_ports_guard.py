"""Row 39 review: no test reaches the live install's ports."""

from __future__ import annotations

import httpx
import pytest


def test_a_call_to_the_live_agent_port_is_refused_in_tests() -> None:
    with (
        httpx.Client(trust_env=False) as client,
        pytest.raises(httpx.ConnectError, match=r"never reach the live install.s port 8079"),
    ):
        client.get("http://127.0.0.1:8079/healthz")
