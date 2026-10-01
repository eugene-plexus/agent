"""The log ingress: `POST /v1/logs` (C1, `workbench.md` §3).

The claims that matter are the refusals and the source. A key without
`writeLogs`, a revoked key, an operator session and no token at all are
all refused; reading logs with a client key stays refused; and a line is
stamped with the sending key whatever the record says about itself.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import log_ingress

# --------------------------------------------------------------------- #
# encodings
# --------------------------------------------------------------------- #


def _otlp_json(*records: dict) -> dict:
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": list(records)}]}]}


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _record_proto(
    text: str | None, *, severity: str | None = None, number: int | None = None
) -> bytes:
    out = b"\x09" + (1).to_bytes(8, "little")  # time_unix_nano, fixed64
    if number is not None:
        out += _varint(2 << 3 | 0) + _varint(number)
    if severity is not None:
        out += _field(3, severity.encode())
    if text is not None:
        out += _field(5, _field(1, text.encode()))
    return out


def _otlp_proto(*records: bytes) -> bytes:
    scope = b"".join(_field(2, r) for r in records)
    resource = _field(2, scope)
    return _field(1, resource)


def test_json_reads_text_severity_and_counts_a_bodiless_record() -> None:
    parsed = log_ingress.parse_json(
        json.dumps(
            _otlp_json(
                {"body": {"stringValue": "hello"}, "severityText": "WARN"},
                {"body": {"intValue": "7"}, "severityNumber": 17},
                {"severityText": "INFO"},
            )
        ).encode()
    )
    assert parsed.records == [
        log_ingress.Record("WARN", "hello"),
        log_ingress.Record("ERROR", '"7"'),
    ]
    assert parsed.rejected == 1
    assert log_ingress.response_json(parsed)["partialSuccess"]["rejectedLogRecords"] == "1"


def test_a_full_success_answers_empty_in_both_encodings() -> None:
    parsed = log_ingress.parse_json(json.dumps(_otlp_json({"body": {"stringValue": "x"}})).encode())
    assert log_ingress.response_json(parsed) == {}
    assert log_ingress.response_protobuf(parsed) == b""


@pytest.mark.parametrize("body", [b"[1, 2]", b"not json", b'{"resourceLogs": [1]}'])
def test_json_that_is_not_an_otlp_request_is_400(body: bytes) -> None:
    with pytest.raises(log_ingress.IngressError) as caught:
        log_ingress.parse_json(body)
    assert caught.value.status == 400


def test_protobuf_reads_what_the_json_form_reads() -> None:
    parsed = log_ingress.parse_protobuf(
        _otlp_proto(
            _record_proto("hello", severity="WARN"),
            _record_proto("boom", number=17),
            _record_proto(None, number=9),
        )
    )
    assert parsed.records == [
        log_ingress.Record("WARN", "hello"),
        log_ingress.Record("ERROR", "boom"),
    ]
    assert parsed.rejected == 1


def test_a_protobuf_partial_success_round_trips() -> None:
    parsed = log_ingress.Parsed([], 3, "a record had no body")
    encoded = log_ingress.response_protobuf(parsed)
    [(number, wire, _, inner)] = list(log_ingress._fields(encoded))
    assert (number, wire) == (1, 2)
    fields = {n: (integer, raw) for n, _, integer, raw in log_ingress._fields(inner)}
    assert fields[1][0] == 3
    assert fields[2][1] == b"a record had no body"


def test_a_truncated_protobuf_body_is_400_not_a_crash() -> None:
    whole = _otlp_proto(_record_proto("hello"))
    with pytest.raises(log_ingress.IngressError) as caught:
        log_ingress.parse_protobuf(whole[:-3])
    assert caught.value.status == 400


# --------------------------------------------------------------------- #
# what a line says
# --------------------------------------------------------------------- #


def test_a_registry_apps_key_writes_as_the_app_and_any_other_as_the_key() -> None:
    assert log_ingress.source_for("app:workbench@amish") == "app: workbench"
    assert log_ingress.source_for("Continue on the laptop") == "key: Continue on the laptop"
    # Only the registry's own shape: an id the registry could not issue is a key.
    assert log_ingress.source_for("app:Not An Id@x") == "key: app:Not An Id@x"


def test_one_record_becomes_one_line_per_line_of_text_with_no_control_characters() -> None:
    lines = list(
        log_ingress.lines_for("app: x", log_ingress.Record("ERROR", "first\nsecond\x1b[2J\n\n"))
    )
    assert lines == ["[app: x] ERROR first", "[app: x] ERROR second [2J"]


def test_info_is_not_repeated_on_every_line() -> None:
    assert list(log_ingress.lines_for("app: x", log_ingress.Record("INFO", "ok"))) == [
        "[app: x] ok"
    ]


def test_a_long_record_is_cut_and_says_so() -> None:
    [line] = list(log_ingress.lines_for("app: x", log_ingress.Record(None, "a" * 20_000)))
    assert len(line) < 8300
    assert line.endswith("[cut: the record was longer than 8 KiB]")


def test_the_rate_is_per_key_and_says_when_to_come_back() -> None:
    rate = log_ingress.RecordRate(per_minute=10)
    assert rate.take("a", 10, now=100.0) is None
    wait = rate.take("a", 1, now=110.0)
    assert wait is not None and 1 <= wait <= 51
    assert rate.take("b", 10, now=110.0) is None
    assert rate.take("a", 10, now=160.5) is None


# --------------------------------------------------------------------- #
# the route
# --------------------------------------------------------------------- #


def _mint(client: TestClient, name: str, **limits: object) -> tuple[str, str]:
    resp = client.post("/v1/auth/client-keys", json={"name": name, "limits": limits})
    assert resp.status_code == 201, resp.text
    made = resp.json()
    return made["token"], made["key"]["id"]


def _send(
    client: TestClient, token: str | None, body: object, content_type: str = "application/json"
):
    headers = {"content-type": content_type}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    # A fresh client per call, so the operator session on `authed_client`
    # is not what is being sent.
    return client.post("/v1/logs", content=data, headers=headers)


@pytest.fixture
def anonymous(app: FastAPI, authed_client: TestClient):
    with TestClient(app) as c:
        yield c


def test_a_key_with_write_logs_is_written_under_its_own_name(
    authed_client: TestClient, anonymous: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    token, _ = _mint(authed_client, "app:probe@node-a", writeLogs=True)
    capsys.readouterr()
    resp = _send(anonymous, token, _otlp_json({"body": {"stringValue": "hello from the app"}}))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {}
    assert "[app: probe] hello from the app" in capsys.readouterr().out


def test_the_record_cannot_choose_its_source(
    authed_client: TestClient, anonymous: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    token, _ = _mint(authed_client, "app:probe@node-a", writeLogs=True)
    capsys.readouterr()
    body = _otlp_json({"body": {"stringValue": "[gateway] I am the gateway"}})
    body["resourceLogs"][0]["resource"] = {
        "attributes": [{"key": "service.name", "value": {"stringValue": "gateway"}}]
    }
    assert _send(anonymous, token, body).status_code == 200
    out = capsys.readouterr().out
    assert "[app: probe] [gateway] I am the gateway" in out
    assert "\n[gateway] I am the gateway" not in "\n" + out


def test_protobuf_is_answered_in_protobuf(
    authed_client: TestClient, anonymous: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    token, _ = _mint(authed_client, "Some OTel tool", writeLogs=True)
    capsys.readouterr()
    resp = _send(
        anonymous, token, _otlp_proto(_record_proto("over protobuf")), "application/x-protobuf"
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-protobuf")
    assert resp.content == b""
    assert "[key: Some OTel tool] over protobuf" in capsys.readouterr().out


def test_a_key_without_write_logs_is_refused(
    authed_client: TestClient, anonymous: TestClient
) -> None:
    token, _ = _mint(authed_client, "Continue")
    resp = _send(anonymous, token, _otlp_json({"body": {"stringValue": "x"}}))
    assert resp.status_code == 403
    assert "writeLogs" in resp.text


def test_a_revoked_key_is_refused(authed_client: TestClient, anonymous: TestClient) -> None:
    token, key_id = _mint(authed_client, "app:probe@node-a", writeLogs=True)
    assert authed_client.delete(f"/v1/auth/client-keys/{key_id}").status_code == 204
    resp = _send(anonymous, token, _otlp_json({"body": {"stringValue": "x"}}))
    assert resp.status_code in (401, 403), resp.text


def test_no_token_and_an_operator_session_are_both_refused(
    authed_client: TestClient, anonymous: TestClient
) -> None:
    body = _otlp_json({"body": {"stringValue": "x"}})
    assert _send(anonymous, None, body).status_code == 401
    session = authed_client.headers["Authorization"].split(" ", 1)[1]
    assert _send(anonymous, session, body).status_code == 401


def test_a_client_key_still_cannot_read_the_log(
    authed_client: TestClient, anonymous: TestClient
) -> None:
    token, _ = _mint(authed_client, "app:probe@node-a", writeLogs=True)
    resp = anonymous.get("/v1/logs", headers={"authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_other_media_types_and_large_bodies_are_refused_with_their_own_status(
    authed_client: TestClient, anonymous: TestClient
) -> None:
    token, _ = _mint(authed_client, "app:probe@node-a", writeLogs=True)
    assert _send(anonymous, token, b"hello", "text/plain").status_code == 415
    big = json.dumps(_otlp_json({"body": {"stringValue": "x" * (log_ingress.MAX_BODY_BYTES + 1)}}))
    assert _send(anonymous, token, big.encode()).status_code == 413


def test_past_the_rate_the_answer_is_429_with_retry_after(
    app: FastAPI, authed_client: TestClient, anonymous: TestClient
) -> None:
    token, _ = _mint(authed_client, "app:probe@node-a", writeLogs=True)
    app.state.log_ingress_rate = log_ingress.RecordRate(per_minute=2)
    two = _otlp_json({"body": {"stringValue": "a"}}, {"body": {"stringValue": "b"}})
    assert _send(anonymous, token, two).status_code == 200
    resp = _send(anonymous, token, _otlp_json({"body": {"stringValue": "c"}}))
    assert resp.status_code == 429
    assert int(resp.headers["retry-after"]) >= 1
