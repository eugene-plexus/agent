"""Small loopback-only worker. Standard library only; no model or hub credentials."""

from __future__ import annotations

import hmac
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import folder_io

MAX_REQUEST = 65_536
MAX_REPLY = 70_000


def unavailable() -> str | None:
    if os.environ.get("EUGENE_PLEXUS_APP_ACCOUNT_KIND") not in {"windows_service", "systemd"}:
        return (
            "File helpers require a Windows service or Linux system installation "
            "with an isolated account."
        )
    if sys.platform not in {"win32", "linux"}:
        return "Safe local file operations are currently supported on Windows and Linux."
    return None


class Worker:
    def __init__(self, protected: list[Path]) -> None:
        self.protected = protected
        self.lock = threading.Lock()
        self.used: dict[str, float] = {}

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            self.used = {key: until for key, until in self.used.items() if until >= now}
            ident, expires = command.get("id"), command.get("expiresAt")
            if (
                not isinstance(ident, str)
                or not ident
                or len(ident) > 128
                or not isinstance(expires, int | float)
                or not now < expires <= now + 30
                or ident in self.used
            ):
                raise folder_io.FolderError("This file operation expired or was already used.")
            self.used[ident] = expires
            if reason := unavailable():
                raise folder_io.FolderError(reason)
            tool, args = command.get("tool"), command.get("arguments")
            if not isinstance(args, dict) or not isinstance(args.get("path"), str):
                raise folder_io.FolderError("A file operation needs a text path.")
            if tool == "inspect":
                if command.get("subject") != "operator" or set(args) != {"path"}:
                    raise folder_io.FolderError("Only the operator can register a folder.")
                path = folder_io.check_root_path(args["path"], self.protected)
                return {"path": path, "identity": folder_io.inspect(path, self.protected)}
            folder = command.get("folder")
            if (
                not isinstance(folder, dict)
                or not command.get("subject")
                or folder.get("subject") != command["subject"]
            ):
                raise folder_io.FolderError("This file operation has no matching person grant.")
            required = {"path", "text", "expectedSha256"} if tool == "write_text" else {"path"}
            if set(args) != required or tool not in {"list_directory", "read_text", "write_text"}:
                raise folder_io.FolderError(
                    "This file operation or its arguments are not supported."
                )
            if tool == "write_text":
                if folder.get("writable") is not True:
                    raise folder_io.FolderError("This folder grant is read-only. No write ran.")
                if (
                    not isinstance(args["text"], str)
                    or len(args["text"]) > 8192
                    or not isinstance(args["expectedSha256"], str)
                    or not re.fullmatch(r"(?:[a-f0-9]{64})?", args["expectedSha256"])
                ):
                    raise folder_io.FolderError(
                        "Use at most 8192 characters and a valid prior file hash."
                    )
            return folder_io.operate(folder["path"], folder["identity"], tool, args, self.protected)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    worker: Worker
    credential: str


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Never log operation bodies or file paths.

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def reply(self, code: int, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False).encode()
        if len(data) > MAX_REPLY:
            code, data = 413, b'{"status":"failed","message":"File result is too large."}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        reason = unavailable()
        self.reply(
            200 if self.path == "/healthz" and not reason else 503,
            {"ready": reason is None, "reason": reason},
        )

    def do_POST(self) -> None:
        expected = "Bearer " + self.server.credential
        if (
            self.path != "/execute"
            or not self.server.credential
            or not hmac.compare_digest(self.headers.get("Authorization", ""), expected)
        ):
            self.reply(403, {"status": "failed", "message": "File helper authentication failed."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_REQUEST:
                self.reply(413, {"status": "failed", "message": "File operation is too large."})
                return
            command = json.loads(self.rfile.read(length))
            if not isinstance(command, dict):
                raise ValueError
            result = self.server.worker.execute(command)
            self.reply(200, {"status": "done", "result": result})
        except folder_io.WriteUncertain as exc:
            self.reply(200, {"status": "uncertain", "message": str(exc)})
        except folder_io.FolderError as exc:
            self.reply(200, {"status": "failed", "message": str(exc)})
        except PermissionError:
            self.reply(
                200,
                {
                    "status": "failed",
                    "message": "The file helper's OS account cannot access this folder. "
                    "Check its permissions.",
                },
            )
        except FileNotFoundError:
            self.reply(200, {"status": "failed", "message": "This file or folder was not found."})
        except (ValueError, KeyError, TypeError, OSError):
            self.reply(
                200,
                {
                    "status": "failed",
                    "message": "The file could not be opened safely. "
                    "Check its path, permissions and file type.",
                },
            )


def main() -> None:
    data = Path(os.environ["EUGENE_PLEXUS_APP_DATA_DIR"])
    protected = [data, Path(sys.prefix), Path(__file__).parent]
    protected.extend(
        Path(p) for p in json.loads(os.environ.get("NODE_HELPER_PROTECTED_ROOTS", "[]"))
    )
    server = Server(("127.0.0.1", int(os.environ["EUGENE_PLEXUS_APP_BIND_PORT"])), Handler)
    server.credential = os.environ["EUGENE_PLEXUS_APP_ADMIN_TOKEN"]
    server.worker = Worker(protected)
    server.serve_forever()


if __name__ == "__main__":
    main()
