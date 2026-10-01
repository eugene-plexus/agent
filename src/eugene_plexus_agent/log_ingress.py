"""The log ingress: `POST /v1/logs`, OpenTelemetry's logs endpoint (C1).

Design: `specs/docs/design/workbench.md` §3. An app the agent runs in an
account of its own is started by the OS service manager, not by this
process, so its output no longer reaches the agent's stdout. A launcher
inside the app's account forwards it here, and so may any other tool the
operator gives a key that may send: the standard is OTLP over HTTP, so a
tool that already exports OpenTelemetry needs only an address and a key.

What this module is: the two encodings of OTLP's `ExportLogsServiceRequest`
read down to (severity, text) pairs, the answer in the encoding the request
came in, and a per-key rate. **No protobuf dependency**: the four messages
the ingress reads are decoded from the wire format directly, because the
agent's own venv is every component's runtime (`watchdog-venv-is-runtime`)
and a library for twenty lines of varint arithmetic is not worth that.

**The source is the key, never the record.** A line is written as
`[app: <id>]` for a registry app's key (the name the supervisor prefixes an
app it runs itself with, so the Logs page reads the same either way) and
`[key: <name>]` for any other. A record that claims to be `gateway` is
still from the key that sent it.
"""

from __future__ import annotations

import json
import re
import struct
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

#: OTLP's own limit is the collector's choice; this is ours.
MAX_BODY_BYTES = 1024 * 1024
#: A record's text is cut here, with a marker.
MAX_RECORD_CHARS = 8192
#: Per key, a sliding minute.
RECORDS_PER_MINUTE = 600

_APP_KEY = re.compile(r"^app:([a-z][a-z0-9-]{1,39})@")
# Control characters a sender could use to start a line of its own in the
# file, or to repaint a terminal following it. A record's text may still
# begin with `[gateway] `: the line is `[app: x] [gateway] ...`, and the
# log's parser takes the first bracket as the source.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class IngressError(Exception):
    """A request the ingress refuses, with the status that says why."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class Record:
    severity: str | None
    text: str


@dataclass(frozen=True)
class Parsed:
    records: list[Record]
    rejected: int
    reason: str | None = None


def source_for(key_name: str) -> str:
    """`app: <id>` for a registry app's key, `key: <name>` for any other."""
    found = _APP_KEY.match(key_name)
    if found is not None:
        return f"app: {found.group(1)}"
    return f"key: {key_name}"


def lines_for(source: str, record: Record) -> Iterator[str]:
    """The console lines one record becomes, each in the supervisor's
    `[source] text` form so the tee stamps it like any child's line."""
    text = record.text
    if len(text) > MAX_RECORD_CHARS:
        text = text[:MAX_RECORD_CHARS] + " [cut: the record was longer than 8 KiB]"
    severity = (record.severity or "").strip().upper()
    for line in text.splitlines() or [""]:
        line = _CONTROL.sub(" ", line).rstrip()
        if not line:
            continue
        if severity and severity not in ("INFO", "INFO2", "INFO3", "INFO4"):
            line = f"{severity} {line}"
        yield f"[{source}] {line}"


# --------------------------------------------------------------------------- #
# OTLP JSON
# --------------------------------------------------------------------------- #

_SEVERITY_NAMES = {
    **{n: "TRACE" for n in range(1, 5)},
    **{n: "DEBUG" for n in range(5, 9)},
    **{n: "INFO" for n in range(9, 13)},
    **{n: "WARN" for n in range(13, 17)},
    **{n: "ERROR" for n in range(17, 21)},
    **{n: "FATAL" for n in range(21, 25)},
}


def _severity(text: object, number: object) -> str | None:
    if isinstance(text, str) and text.strip():
        return text.strip()
    if isinstance(number, bool) or not isinstance(number, (int, str)):
        return None
    try:
        return _SEVERITY_NAMES.get(int(number))
    except ValueError:
        return None


def _any_value_json(value: object) -> str | None:
    """An OTLP `AnyValue` in its JSON encoding, as text."""
    if not isinstance(value, dict):
        return None
    text = value.get("stringValue")
    if isinstance(text, str):
        return text
    for key in ("boolValue", "intValue", "doubleValue", "arrayValue", "kvlistValue", "bytesValue"):
        if key in value:
            return json.dumps(value[key], separators=(",", ":"))
    return None


def parse_json(body: bytes) -> Parsed:
    try:
        document = json.loads(body or b"{}")
    except ValueError as exc:
        raise IngressError(400, f"the body is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise IngressError(400, "the body must be an OTLP ExportLogsServiceRequest object")
    records: list[Record] = []
    rejected = 0
    for resource in document.get("resourceLogs") or []:
        if not isinstance(resource, dict):
            raise IngressError(400, "resourceLogs must be a list of objects")
        for scope in resource.get("scopeLogs") or []:
            if not isinstance(scope, dict):
                raise IngressError(400, "scopeLogs must be a list of objects")
            for item in scope.get("logRecords") or []:
                if not isinstance(item, dict):
                    raise IngressError(400, "logRecords must be a list of objects")
                text = _any_value_json(item.get("body"))
                if text is None:
                    rejected += 1
                    continue
                records.append(
                    Record(_severity(item.get("severityText"), item.get("severityNumber")), text)
                )
    return Parsed(records, rejected, "a record had no body" if rejected else None)


def response_json(parsed: Parsed) -> dict:
    if not parsed.rejected:
        return {}
    return {
        "partialSuccess": {
            "rejectedLogRecords": str(parsed.rejected),
            "errorMessage": parsed.reason or "",
        }
    }


# --------------------------------------------------------------------------- #
# OTLP protobuf: the wire format, for the four messages read here
# --------------------------------------------------------------------------- #
#
# ExportLogsServiceRequest { repeated ResourceLogs resource_logs = 1; }
# ResourceLogs             { Resource resource = 1; repeated ScopeLogs scope_logs = 2; }
# ScopeLogs                { InstrumentationScope scope = 1; repeated LogRecord log_records = 2; }
# LogRecord                { fixed64 time_unix_nano = 1; SeverityNumber severity_number = 2;
#                            string severity_text = 3; AnyValue body = 5; ... }
# AnyValue                 { oneof value { string string_value = 1; bool bool_value = 2;
#                            int64 int_value = 3; double double_value = 4; ...;
#                            bytes bytes_value = 7; } }


def _varint(data: bytes, at: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        if at >= len(data):
            raise IngressError(400, "the protobuf body ends inside a varint")
        byte = data[at]
        at += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, at
        shift += 7
        if shift > 63:
            raise IngressError(400, "the protobuf body has a varint longer than 64 bits")


def _fields(data: bytes) -> Iterator[tuple[int, int, int, bytes]]:
    """(field number, wire type, integer, bytes) for each field of one
    message: a length-delimited field fills the bytes, every other wire
    type the integer."""
    at = 0
    while at < len(data):
        key, at = _varint(data, at)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, at = _varint(data, at)
            yield number, wire, value, b""
        elif wire == 1:
            if at + 8 > len(data):
                raise IngressError(400, "the protobuf body ends inside a 64-bit field")
            yield number, wire, int.from_bytes(data[at : at + 8], "little"), b""
            at += 8
        elif wire == 2:
            length, at = _varint(data, at)
            if at + length > len(data):
                raise IngressError(400, "the protobuf body ends inside a length-delimited field")
            yield number, wire, 0, data[at : at + length]
            at += length
        elif wire == 5:
            if at + 4 > len(data):
                raise IngressError(400, "the protobuf body ends inside a 32-bit field")
            yield number, wire, int.from_bytes(data[at : at + 4], "little"), b""
            at += 4
        else:
            raise IngressError(400, f"the protobuf body uses wire type {wire}, which OTLP does not")


def _any_value_proto(data: bytes) -> str | None:
    for number, wire, integer, raw in _fields(data):
        if number == 1 and wire == 2:
            return raw.decode("utf-8", errors="replace")
        if number == 2 and wire == 0:
            return "true" if integer else "false"
        if number == 3 and wire == 0:
            return str(integer - (1 << 64) if integer >= 1 << 63 else integer)
        if number == 4 and wire == 1:
            return repr(struct.unpack("<d", integer.to_bytes(8, "little"))[0])
        if number == 7 and wire == 2:
            return raw.hex()
        if number in (5, 6) and wire == 2:
            return "(a structured value)"
    return None


def parse_protobuf(body: bytes) -> Parsed:
    records: list[Record] = []
    rejected = 0
    for number, wire, _, resource in _fields(body):
        if number != 1 or wire != 2:
            continue
        for rnum, rwire, _, scope in _fields(resource):
            if rnum != 2 or rwire != 2:
                continue
            for snum, swire, _, item in _fields(scope):
                if snum != 2 or swire != 2:
                    continue
                severity_text: str | None = None
                severity_number: int | None = None
                text: str | None = None
                for lnum, lwire, integer, raw in _fields(item):
                    if lnum == 2 and lwire == 0:
                        severity_number = integer
                    elif lnum == 3 and lwire == 2:
                        severity_text = raw.decode("utf-8", errors="replace")
                    elif lnum == 5 and lwire == 2:
                        text = _any_value_proto(raw)
                if text is None:
                    rejected += 1
                    continue
                records.append(Record(_severity(severity_text, severity_number), text))
    return Parsed(records, rejected, "a record had no body" if rejected else None)


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def response_protobuf(parsed: Parsed) -> bytes:
    """ExportLogsServiceResponse { ExportLogsPartialSuccess partial_success = 1; }
    with { int64 rejected_log_records = 1; string error_message = 2; }.
    Empty when everything was written, as OTLP says a full success is."""
    if not parsed.rejected:
        return b""
    message = (parsed.reason or "").encode()
    inner = b"\x08" + _encode_varint(parsed.rejected)
    if message:
        inner += b"\x12" + _encode_varint(len(message)) + message
    return b"\x0a" + _encode_varint(len(inner)) + inner


# --------------------------------------------------------------------------- #
# the per-key rate
# --------------------------------------------------------------------------- #


class RecordRate:
    """At most `RECORDS_PER_MINUTE` records per key over a sliding minute.

    In memory, per agent process: a restart forgives, which is the right
    direction for a limit whose job is to keep one noisy app from filling
    the log, not to account for anything.
    """

    def __init__(self, per_minute: int = RECORDS_PER_MINUTE) -> None:
        self._per_minute = per_minute
        self._seen: dict[str, deque[tuple[float, int]]] = {}
        self._lock = threading.Lock()

    def take(self, key_id: str, count: int, *, now: float | None = None) -> int | None:
        """Record `count` records for `key_id`. None if allowed; otherwise
        the seconds until the window has room again."""
        moment = time.perf_counter() if now is None else now
        with self._lock:
            window = self._seen.setdefault(key_id, deque())
            while window and window[0][0] <= moment - 60.0:
                window.popleft()
            used = sum(n for _, n in window)
            if used + count > self._per_minute:
                if not window:
                    return 60
                return max(1, int(window[0][0] + 60.0 - moment) + 1)
            window.append((moment, count))
            return None
