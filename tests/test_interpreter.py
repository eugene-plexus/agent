"""`sys.executable` is not an interpreter under the Windows service.

There it is pywin32's pythonservice.exe, which prints its usage and exits
when given `-m`. Strata's install ran `sys.executable -m venv` and failed
on the first service install that tried it (2026-10-09), though every
acceptance passed: they all ran the agent as python.exe. A command
needs `interpreter.command_python()`; this keeps any new reading of
`sys.executable` from landing without someone deciding it is not one.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_agent"

#: Every reading of `sys.executable` in the agent, by file, and why it is
#: not a command run under the service.
REVIEWED = {
    # The rule itself.
    "interpreter.py": 2,
    # `reexec`: only after the HTTPS entry point, which is Linux-only.
    "entrypoint_setup.py": 3,
    # A program for a firewall hint and the reach page to name, not run.
    "engines/acquisition.py": 1,
    "routes/node.py": 1,
    # The tray runs as the signed-in person, never as the service.
    "tray.py": 2,
    # Instructions printed for a person to type.
    "winservice.py": 2,
}


def _readings(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def test_every_reading_of_sys_executable_has_been_reviewed() -> None:
    found = Counter(
        {
            path.relative_to(ROOT).as_posix(): count
            for path in ROOT.rglob("*.py")
            if "_generated" not in path.parts and (count := _readings(path))
        }
    )
    assert found == Counter(REVIEWED), (
        "sys.executable is pythonservice.exe under the Windows service; to start "
        "Python use interpreter.command_python(), or review the new reading and "
        f"list it in REVIEWED. Found {dict(found)}"
    )
