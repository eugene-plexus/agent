"""Restoring the identity must not start enrollment, even in safe mode."""

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent.__main__ import build_server
from eugene_plexus_agent.app import create_app
from eugene_plexus_agent.settings import Settings


@pytest.mark.parametrize("safe_mode", [False, True])
def test_quarantined_copy_refuses_before_writing_state(tmp_path, safe_mode):
    (tmp_path / ".recovery-quarantine").write_text("pending")
    directory = tmp_path / "state"
    directory.mkdir()
    settings = Settings(config_file=directory / "agent.yaml", safe_mode=safe_mode)
    with pytest.raises(RuntimeError, match="quarantined"):
        build_server(settings, unattended=True)
    with pytest.raises(RuntimeError, match="quarantined"), TestClient(create_app(settings)):
        pass
    assert list(directory.iterdir()) == []
