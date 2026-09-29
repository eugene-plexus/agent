"""What is installed on this machine, read from the code itself.

**Each component knows the commit it was built from.** Every install is
built from GitHub source archives at pinned commits, and GitHub makes an
archive with `git archive`, which stamps the commit into each package's
`_build.py` (marked `export-subst` in its `.gitattributes`). So the
version is in the code, put there by git: nothing of ours writes it, and
there is no record file to edit or to drift. It reports what was
*actually* installed -- a package that half-failed to upgrade keeps its
old commit, and a mixed install is visible as one.

The six packages live in this agent's own environment (the agent's venv
is the component runtime), so the agent reads all six. It reads each
`_build.py` as TEXT rather than importing it: importing a package runs
its `__init__`, and a version read should have no side effects.

**What keeps the install running** decides whether and how it can update
itself, and is read from evidence about this process, the way the Reach
switch reads it (`reach.describe_restart`), never from what an installer
once did.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

from ._generated.models import (
    ComponentName,
    ContainerInstall,
    InstalledComponent,
    InstalledComponentState,
    InstallMechanism,
    NodeInstall,
)

#: The seven packages, in the installers' order, by the name the installers
#: pin them under and the module each one imports as. `tool-driver` joined
#: at P8 (2026-09-29).
COMPONENTS: tuple[tuple[ComponentName, str], ...] = (
    (ComponentName.agent, "eugene_plexus_agent"),
    (ComponentName.control, "eugene_plexus_control"),
    (ComponentName.gateway, "eugene_plexus_gateway"),
    (ComponentName.inference_driver, "eugene_plexus_inference_driver"),
    (ComponentName.library, "eugene_plexus_library"),
    (ComponentName.tool_driver, "eugene_plexus_tool_driver"),
    (ComponentName.ui, "eugene_plexus_ui"),
)

_STAMP = re.compile(r'^COMMIT\s*=\s*"([^"]*)"', re.MULTILINE)
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")

#: Set by the container image. Its presence is the evidence this is a
#: container; its value is the image and tag the container was built as.
CONTAINER_IMAGE_ENV = "EUGENE_PLEXUS_CONTAINER_IMAGE"
#: Set by a template we write (the Unraid template, our Compose file), so
#: the update steps can use that platform's words. Absent otherwise: from
#: inside a container nothing reliable says what launched it.
CONTAINER_HOST_ENV = "EUGENE_PLEXUS_CONTAINER_HOST"


def read_component(module: str) -> tuple[InstalledComponentState, str | None]:
    """How one package is installed, and its commit when it has one."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return InstalledComponentState.missing, None
    stamp = Path(next(iter(spec.submodule_search_locations))) / "_build.py"
    try:
        text = stamp.read_text(encoding="utf-8")
    except OSError:
        # Installed before components recorded their commit.
        return InstalledComponentState.unrecorded, None
    found = _STAMP.search(text)
    value = found.group(1) if found else ""
    if _FULL_COMMIT.fullmatch(value):
        return InstalledComponentState.stamped, value
    if value.startswith("$Format"):
        # A git checkout: the placeholder was never substituted.
        return InstalledComponentState.development, None
    return InstalledComponentState.unrecorded, None


def installed_components() -> list[InstalledComponent]:
    out = []
    for name, module in COMPONENTS:
        state, commit = read_component(module)
        out.append(InstalledComponent(name=name, state=state, commit=commit))
    return out


def container() -> ContainerInstall | None:
    image = os.environ.get(CONTAINER_IMAGE_ENV, "").strip()
    if not image:
        return None
    host = os.environ.get(CONTAINER_HOST_ENV, "").strip().lower() or None
    return ContainerInstall(image=image, host=host)


def _systemd_scope() -> str | None:
    """`system` or `user` for a systemd unit, from this process's cgroup.

    Both kinds of unit set `INVOCATION_ID`, so that says only *a unit*.
    The cgroup path says which manager: a user unit lives under
    `user@<uid>.service`, a system unit under `system.slice`. A Linux
    system install runs as its own unprivileged account, so the effective
    uid cannot tell them apart.
    """
    if not os.environ.get("INVOCATION_ID"):
        return None
    try:
        cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    if "user@" in cgroup:
        return "user"
    if "system.slice" in cgroup:
        return "system"
    return None


def mechanism() -> InstallMechanism:
    if container() is not None:
        return InstallMechanism.container
    if sys.platform.startswith("linux"):
        scope = _systemd_scope()
        if scope == "system":
            return InstallMechanism.systemd_system
        if scope == "user":
            return InstallMechanism.systemd_user
        return InstallMechanism.none
    # Windows and macOS: the Reach switch already answers exactly this
    # question, install-scoped, and has been sabotage-checked doing it.
    from . import reach
    from ._generated.models import Mechanism

    found = reach.describe_restart().mechanism
    return {
        Mechanism.service: InstallMechanism.windows_service,
        Mechanism.logon_task: InstallMechanism.windows_task,
        Mechanism.launchd: InstallMechanism.launchd,
    }.get(found, InstallMechanism.none)


def describe(*, mechanism_now: InstallMechanism | None = None) -> NodeInstall:
    components = installed_components()
    return NodeInstall(
        components=components,
        mechanism=mechanism_now or mechanism(),
        development=any(c.state is InstalledComponentState.development for c in components),
        container=container(),
    )


def install_prefix(config_file: Path) -> Path:
    """Where the installer put this install: the folder `agent.yaml` is in.

    True of every layout the installers write (`install.ps1`'s `$Config`
    and `install.sh`'s `$CONFIG` are both `<prefix>/agent.yaml`), and of
    the container, whose `/data` is where its state lives.
    """
    return config_file.resolve().parent
