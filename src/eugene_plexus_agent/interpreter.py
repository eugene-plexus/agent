"""The Python interpreter that can run commands for this agent.

Under the Windows service, `sys.executable` is pywin32's
`pythonservice.exe`: it embeds Python but cannot run `-m` commands, and
given any it prints its own usage and exits. Every place that starts a
Python command (a component, an engine's private environment) needs the
venv's own `python.exe` instead. Never an unrelated interpreter from PATH.
"""

from __future__ import annotations

import sys
from pathlib import Path


def command_python() -> str:
    """`sys.executable`, or the service venv's `python.exe` under the service.

    Raises `FileNotFoundError` naming the path when the service's venv has
    no interpreter (a broken install), rather than falling back to another.
    """
    if Path(sys.executable).name.lower() != "pythonservice.exe":
        return sys.executable
    python = Path(sys.prefix) / "Scripts" / "python.exe"
    if not python.is_file():
        raise FileNotFoundError(f"The service's Python interpreter is missing: {python}")
    return str(python)
