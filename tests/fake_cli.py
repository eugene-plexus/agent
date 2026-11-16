"""A fake `docker` or `podman` on PATH, for row 42's tests.

The fake answers each call from a list of rules (`args` prefix -> `rc`,
`out`, `err`, and optionally `sleep` seconds) and records every argv it
was given in `calls.jsonl`, so a test reads exactly what the agent ran. A
rule may `touch` a marker file, and match only `when` a marker exists or
`unless` it does: enough state for "absent until pulled".
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

_SCRIPT = """
import json, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
args = sys.argv[1:]
with open(here / "calls.jsonl", "a", encoding="utf-8") as calls:
    calls.write(json.dumps(args) + "\\n")
rules = json.loads((here / "rules.json").read_text(encoding="utf-8"))
for rule in rules:
    if args[: len(rule["args"])] == rule["args"]:
        if rule.get("when") and not (here / rule["when"]).exists():
            continue
        if rule.get("unless") and (here / rule["unless"]).exists():
            continue
        if rule.get("touch"):
            (here / rule["touch"]).write_text("", encoding="utf-8")
        if rule.get("sleep"):
            time.sleep(rule["sleep"])
        sys.stdout.write(rule.get("out", ""))
        sys.stderr.write(rule.get("err", ""))
        sys.stdout.flush()
        sys.exit(rule.get("rc", 0))
sys.exit(0)
"""


class FakeCli:
    def __init__(self, directory: Path, name: str) -> None:
        self.directory = directory
        self.name = name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}_fake.py").write_text(_SCRIPT, encoding="utf-8")
        if os.name == "nt":
            self.path = directory / f"{name}.cmd"
            self.path.write_text(
                f'@"{sys.executable}" "%~dp0{name}_fake.py" %*\r\n', encoding="utf-8"
            )
        else:
            self.path = directory / name
            self.path.write_text(
                f"#!{sys.executable}\n" + (directory / f"{name}_fake.py").read_text("utf-8"),
                encoding="utf-8",
            )
            self.path.chmod(self.path.stat().st_mode | stat.S_IEXEC)
            # The POSIX copy reads its rules beside itself, under its own name.
        self.rules([])

    def rules(self, rules: list[dict[str, Any]]) -> None:
        (self.directory / "rules.json").write_text(json.dumps(rules), encoding="utf-8")

    def calls(self) -> list[list[str]]:
        path = self.directory / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def docker_info(**overrides: Any) -> str:
    info: dict[str, Any] = {
        "ServerVersion": "27.3.1",
        "OperatingSystem": "Ubuntu 24.04.1 LTS",
        "KernelVersion": "6.8.0-45-generic",
        "Runtimes": {"io.containerd.runc.v2": {}, "runc": {}},
    }
    info.update(overrides)
    return json.dumps(info) + "\n"
