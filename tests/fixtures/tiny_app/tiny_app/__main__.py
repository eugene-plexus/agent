"""Record what we were handed, then answer /healthz until stopped.

The names of every `EUGENE_PLEXUS_*` variable in our environment go to
`env.json` in the data directory -- names only, so the test can assert
that no hub credential arrived without this file ever holding one.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/healthz":
            body = b'{"status":"ok"}'
            self.send_response(200)
        else:
            body = b"{}"
            self.send_response(404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return None


def main() -> None:
    data = Path(os.environ["EUGENE_PLEXUS_APP_DATA_DIR"])
    names = sorted(k for k in os.environ if k.startswith("EUGENE_PLEXUS_"))
    (data / "env.json").write_text(json.dumps(names), encoding="utf-8")
    host = os.environ.get("EUGENE_PLEXUS_APP_BIND_HOST", "127.0.0.1")
    port = int(os.environ["EUGENE_PLEXUS_APP_BIND_PORT"])
    HTTPServer((host, port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
