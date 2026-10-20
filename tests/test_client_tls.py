"""This node's client certificate (LT2, `client_tls`): asked for, kept,
renewed and reported; plain connections counted for 7 days."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from eugene_plexus_agent import client_doors, client_tls


def _key_pem() -> tuple[ec.EllipticCurvePrivateKey, str]:
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return key, pem


def _sign(csr_pem: str, *, days: int = 397) -> tuple[str, str]:
    """What the root answers: a leaf for the CSR's key, and a CA."""
    ca_key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test CA")])
    now = datetime.datetime.now(datetime.UTC)
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    csr = x509.load_pem_x509_csr(csr_pem.encode())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([]))
        .issuer_name(name)
        .public_key(csr.public_key())
        .serial_number(2)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("gpu-box.local")]), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    return leaf.public_bytes(pem).decode(), ca.public_bytes(pem).decode()


def _app(key_pem: str, *, advertise: str = "http://192.168.1.20:8079") -> Any:
    record = SimpleNamespace(
        enrolled=True,
        name="gpu-box",
        tls_private_key=key_pem,
        control_url="http://192.168.1.10:8083",
        advertise_url=advertise,
    )
    config: dict[str, Any] = {}
    return SimpleNamespace(
        state=SimpleNamespace(
            node_identity=SimpleNamespace(record=record),
            agent_state=SimpleNamespace(get_config=config.get),
            auth_state=SimpleNamespace(
                trust=SimpleNamespace(agent_token=lambda audience: "node-token")
            ),
            config=config,
        )
    )


def _root(seen: list[httpx.Request], *, status: int = 200, detail: str = "") -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"title": "no", "detail": detail})
        leaf, ca = _sign(json.loads(request.content)["csr"])
        return httpx.Response(
            200,
            json={
                "certificate": leaf,
                "ca": ca,
                "names": ["gpu-box.local"],
                "notAfter": (
                    datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=397)
                ).isoformat(),
            },
        )

    return httpx.MockTransport(handler)


async def test_a_joined_node_asks_for_a_leaf_and_its_doors_present_it(tmp_path: Path) -> None:
    _, key = _key_pem()
    doors = client_doors.Doors()
    tls = client_tls.ClientTls(tmp_path, _app(key), doors)
    assert tls.wanted() == "this node holds none"
    seen: list[httpx.Request] = []
    await tls.request("this node holds none", transport=_root(seen))
    assert str(seen[0].url) == "http://192.168.1.10:8083/v1/nodes/gpu-box/certificate"
    assert seen[0].headers["authorization"] == "Bearer node-token"
    csr = x509.load_pem_x509_csr(json.loads(seen[0].content)["csr"].encode())
    assert csr.is_signature_valid
    assert tls.state == "issued" and doors.context() is not None
    assert tls.wanted() is None
    # Kept on disk: a restart presents it with the root down.
    again = client_tls.ClientTls(tmp_path, _app(key), client_doors.Doors())
    again.load()
    assert again.state == "issued" and again.doors.context() is not None
    assert again.ca_pem() == tls.ca_pem()


async def test_a_new_address_or_two_thirds_of_its_life_asks_again(tmp_path: Path) -> None:
    _, key = _key_pem()
    tls = client_tls.ClientTls(tmp_path, _app(key), client_doors.Doors())
    await tls.request("first", transport=_root([]))
    moved = client_tls.ClientTls(
        tmp_path, _app(key, advertise="http://192.168.1.77:8079"), client_doors.Doors()
    )
    moved.load()
    assert moved.wanted() == "this node's address changed"
    assert tls.held is not None
    now = datetime.datetime.now(datetime.UTC)
    tls.held["issuedAt"] = (now - datetime.timedelta(days=300)).isoformat()
    tls.held["notAfter"] = (now + datetime.timedelta(days=97)).isoformat()
    assert tls.wanted() == "it is two thirds through its life"


async def test_a_leaf_for_another_key_is_not_presented(tmp_path: Path) -> None:
    _, key = _key_pem()
    tls = client_tls.ClientTls(tmp_path, _app(key), client_doors.Doors())
    await tls.request("first", transport=_root([]))
    _, other = _key_pem()
    rekeyed = client_tls.ClientTls(tmp_path, _app(other), client_doors.Doors())
    rekeyed.load()
    assert rekeyed.doors.context() is None and rekeyed.wanted() == "this node holds none"


async def test_a_refusal_is_reported_and_not_asked_again_at_once(tmp_path: Path) -> None:
    _, key = _key_pem()
    tls = client_tls.ClientTls(tmp_path, _app(key), client_doors.Doors())
    await tls.request("first", transport=_root([], status=403, detail="not its names"))
    assert tls.state == "error" and tls.error == "not its names"
    assert tls.wanted() is None


async def test_a_sealed_root_leaves_it_pending(tmp_path: Path) -> None:
    _, key = _key_pem()
    tls = client_tls.ClientTls(tmp_path, _app(key), client_doors.Doors())
    await tls.request("first", transport=_root([], status=503, detail="locked"))
    assert tls.state == "pending" and tls.wanted() == "this node holds none"


def test_a_node_alone_wants_nothing(tmp_path: Path) -> None:
    _, key = _key_pem()
    app = _app(key)
    app.state.node_identity.record.enrolled = False
    tls = client_tls.ClientTls(tmp_path, app, client_doors.Doors())
    assert tls.wanted() is None and tls.status()["state"] == "none"


def test_plain_counts_are_kept_by_door_and_client_for_seven_days(tmp_path: Path) -> None:
    path = tmp_path / client_tls.PLAIN_FILE
    counts = client_tls.PlainCounts(path)
    counts.add("gateway", "192.168.1.40")
    counts.add("gateway", "192.168.1.40")
    counts.add("console", "192.168.1.41")
    counts.save()
    kept = client_tls.PlainCounts(path)
    assert [(c["address"], c["count"]) for c in kept.clients("gateway")] == [("192.168.1.40", 2)]
    old = (datetime.date.today() - datetime.timedelta(days=8)).isoformat()
    kept._data["console"]["192.168.1.41"] = {old: 5}
    assert kept.clients("console") == []


def _self_signed(directory: Path) -> tuple[Path, Path]:
    key, key_pem = _key_pem()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "eugene.example.com")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(3)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=90))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("eugene.example.com")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = directory / "mine.crt", directory / "mine.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_text(key_pem, encoding="utf-8")
    return cert_path, key_path


async def test_the_owners_own_certificate_replaces_the_install_leaf_and_gives_way(
    tmp_path: Path,
) -> None:
    _, key = _key_pem()
    app = _app(key)
    doors = client_doors.Doors()
    tls = client_tls.ClientTls(tmp_path, app, doors)
    await tls.request("first", transport=_root([]))
    leaf = doors.context()
    cert_path, key_path = _self_signed(tmp_path)
    app.state.config.update(clientCertificateFile=str(cert_path), clientKeyFile=str(key_path))
    tls.check_supplied()
    status = tls.status()
    assert status["source"] == "supplied" and status["names"] == ["eugene.example.com"]
    assert doors.context() is not None and doors.context() is not leaf
    # A key that does not open: said, and the install's leaf comes back.
    key_path.write_text("not a key", encoding="utf-8")
    tls.check_supplied()
    assert tls.status()["source"] == "installCa"
    assert "could not be loaded" in str(tls.status()["error"])
    app.state.config.clear()
    tls.check_supplied()
    assert tls.status()["error"] is None and tls.status()["source"] == "installCa"


async def test_apps_get_a_bundle_with_the_install_ca(tmp_path: Path) -> None:
    _, key = _key_pem()
    tls = client_tls.ClientTls(tmp_path, _app(key), client_doors.Doors())
    told: list[bool] = []
    tls.on_new_ca = lambda: told.append(True)
    await tls.request("first", transport=_root([]))
    bundle = tls.bundle_path()
    assert bundle is not None and told == [True]
    text = bundle.read_text(encoding="utf-8")
    assert text.rstrip().endswith(str(tls.ca_pem()).rstrip())
    assert text.count("BEGIN CERTIFICATE") > 50  # certifi's roots come first
