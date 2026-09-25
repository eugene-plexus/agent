"""`GET /v1/directories`: the picker behind every path field (M11)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_agent.directory_listing import ListingError, list_directory, starting_points

from .conftest import local_service_token


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    (root / "zeta").mkdir(parents=True)
    (root / "Alpha").mkdir()
    (root / ".hidden").mkdir()
    (root / "a.gguf").write_bytes(b"x")
    (root / ".dotfile").write_bytes(b"x")
    return root


# --- the pure function ------------------------------------------------------


def test_directories_come_first_sorted_without_regard_to_case(tmp_path: Path) -> None:
    listing = list_directory(str(_tree(tmp_path)), host="box")
    assert [e.name for e in listing.entries] == ["Alpha", "zeta"]
    assert all(e.kind.value == "directory" for e in listing.entries)
    assert listing.host == "box"
    assert listing.path == str(tmp_path / "models")
    assert listing.parent == str(tmp_path)
    assert all(e.hidden is None for e in listing.entries)


def test_files_and_hidden_entries_only_on_request(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with_files = list_directory(str(root), include_files=True)
    assert [e.name for e in with_files.entries] == ["Alpha", "zeta", "a.gguf"]
    assert with_files.entries[-1].kind.value == "file"

    everything = list_directory(str(root), include_files=True, show_hidden=True)
    names = [e.name for e in everything.entries]
    assert names == [".hidden", "Alpha", "zeta", ".dotfile", "a.gguf"]
    assert {e.name: e.hidden for e in everything.entries}[".hidden"] is True
    assert {e.name: e.hidden for e in everything.entries}["zeta"] is False


def test_each_entry_carries_a_path_ready_to_be_a_config_value(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    listing = list_directory(str(root))
    assert listing.entries[0].path == str(root / "Alpha")


def test_a_path_that_is_not_there_is_404(tmp_path: Path) -> None:
    with pytest.raises(ListingError) as caught:
        list_directory(str(tmp_path / "nope"), host="box")
    assert caught.value.status == 404
    assert "does not exist on box" in caught.value.detail


def test_a_file_is_400(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with pytest.raises(ListingError) as caught:
        list_directory(str(root / "a.gguf"))
    assert caught.value.status == 400


def test_the_starting_points_are_the_roots_and_home() -> None:
    listing = starting_points(host="box")
    names = [e.name for e in listing.entries]
    assert names[-1] == "Home"
    assert listing.entries[-1].path == str(Path.home())
    if sys.platform == "win32":
        assert any(n.endswith(":\\") for n in names)
    else:
        assert names[0] == "/"
    assert listing.path is None and listing.parent is None


def test_no_path_means_the_starting_points(tmp_path: Path) -> None:
    assert list_directory(None, host="box").entries[-1].name == "Home"
    assert list_directory("  ", host="box").entries[-1].name == "Home"


def test_a_filesystem_root_has_no_parent() -> None:
    root = Path.home().anchor
    listing = list_directory(root)
    assert listing.parent is None


# --- the route --------------------------------------------------------------


def test_the_route_lists_for_the_operator(authed_client: TestClient, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    response = authed_client.get("/v1/directories", params={"path": str(root)})
    assert response.status_code == 200
    body = response.json()
    assert [e["name"] for e in body["entries"]] == ["Alpha", "zeta"]
    # `hidden` is present only in a listing that asked for it.
    assert "hidden" not in body["entries"][0]
    assert body["parent"] == str(tmp_path)


def test_the_route_honours_the_flags(authed_client: TestClient, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    response = authed_client.get(
        "/v1/directories", params={"path": str(root), "includeFiles": "true", "showHidden": "true"}
    )
    names = [e["name"] for e in response.json()["entries"]]
    assert ".dotfile" in names and "a.gguf" in names
    assert response.json()["entries"][0]["hidden"] is True


def test_the_route_maps_the_errors(authed_client: TestClient, tmp_path: Path) -> None:
    missing = authed_client.get("/v1/directories", params={"path": str(tmp_path / "nope")})
    assert missing.status_code == 404
    assert missing.json()["detail"]["title"] == "No such directory"
    root = _tree(tmp_path)
    not_dir = authed_client.get("/v1/directories", params={"path": str(root / "a.gguf")})
    assert not_dir.status_code == 400


def test_the_route_is_operator_only(client: TestClient, tmp_path: Path) -> None:
    """A service token reads runtimes; it does not walk the disk."""
    init = client.post("/v1/auth/initialize", json={"passphrase": "correct horse battery staple"})
    assert init.status_code == 200
    anonymous = client.get("/v1/directories")
    assert anonymous.status_code == 401
    service = local_service_token(client.app, "gateway")  # type: ignore[arg-type]
    refused = client.get("/v1/directories", headers={"Authorization": f"Bearer {service}"})
    assert refused.status_code in (401, 403)
