"""Row 15 (DI2, call DI-C4): the prompt cache on disk, as the console reads and clears it.

`GET /v1/prompt-cache` says what is kept (counts, sizes, dates; never the text) and the
limits; `DELETE` deletes every file. Both are the owner's. The settings are on the agent's
config with DI-C4's defaults: on, 3 days, 20 GiB.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import disk_prompt_cache as dpc

GIB = 1024**3


def _file(folder: Path, name: str, size: int, age: float = 0.0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"x" * size)
    when = time.time() - age
    os.utime(path, (when, when))
    return path


def _root(app: FastAPI, tmp_path: Path) -> Path:
    root = tmp_path / "prompt-cache"
    app.state.runtime_supervisor.prompt_cache_root = root
    return root


def test_the_owner_reads_what_is_kept_and_never_the_text(
    app: FastAPI, authed_client: TestClient, tmp_path: Path
) -> None:
    root = _root(app, tmp_path)
    _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 100, age=60)
    _file(root / "qwen" / "k1", "1-aaaaaa-s1.bin", 50, age=30)
    _file(root / "gemma" / "k2", "1-bbbbbb-s0.bin", 10)
    response = authed_client.get("/v1/prompt-cache")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["enabled"] is True
    assert body["days"] == 3
    assert body["capBytes"] == 20 * GIB
    assert (body["conversations"], body["bytes"]) == (3, 160)
    assert body["folder"] == str(root)
    assert sorted((r["runtime"], r["conversations"], r["bytes"]) for r in body["runtimes"]) == [
        ("gemma", 1, 10),
        ("qwen", 2, 150),
    ]
    assert body["oldest"] < body["newest"]


def test_clear_deletes_every_file(app: FastAPI, authed_client: TestClient, tmp_path: Path) -> None:
    root = _root(app, tmp_path)
    one = _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 100)
    response = authed_client.delete("/v1/prompt-cache")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "conversations": 1,
        "bytesFreed": 100,
        "notDeleted": 0,
        "reason": None,
    }
    assert not one.exists()
    assert authed_client.get("/v1/prompt-cache").json()["conversations"] == 0


def test_nobody_else_reads_or_clears_it(app: FastAPI, client: TestClient, tmp_path: Path) -> None:
    root = _root(app, tmp_path)
    kept = _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 100)
    assert client.get("/v1/prompt-cache").status_code == 401
    assert client.delete("/v1/prompt-cache").status_code == 401
    assert kept.exists()


def test_the_settings_carry_the_answered_defaults(authed_client: TestClient) -> None:
    fields = {f["key"]: f for f in authed_client.get("/v1/config/schema").json()["fields"]}
    assert fields[dpc.ENABLED_KEY]["default"] is True
    assert fields[dpc.DAYS_KEY]["default"] == 3
    assert fields[dpc.CAP_KEY]["default"] == 20
    assert {fields[k]["category"] for k in dpc.KEYS} == {dpc.CATEGORY}
    refused = authed_client.patch("/v1/config", json={dpc.DAYS_KEY: 0}).json()
    assert [r["key"] for r in refused["rejected"]] == [dpc.DAYS_KEY]


def test_the_setting_says_what_is_kept_now(
    app: FastAPI, authed_client: TestClient, tmp_path: Path
) -> None:
    root = _root(app, tmp_path)
    _file(root / "qwen" / "k1", "1-aaaaaa-s0.bin", 3 * 1024**2)
    fields = {f["key"]: f for f in authed_client.get("/v1/config/schema").json()["fields"]}
    status = fields[dpc.ENABLED_KEY].get("status") or {}
    assert "1 saved conversation" in status.get("text", ""), status
