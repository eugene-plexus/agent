"""Credential boundaries for supervised children and engine probes.

This limits accidental credential inheritance; it is not an OS sandbox.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

_NAMESPACE = "EUGENE_PLEXUS_"
_CREDENTIAL_SUFFIXES = ("_MASTER_KEY", "_AUTH_SIGNING_KEY", "_AUTH_VERIFY_KEY", "_SERVICE_TOKEN")


def reserved_override(
    values: Mapping[str, object], *, component_prefix: str | None = None
) -> str | None:
    """Return a reserved variable name, never its potentially secret value."""
    for key in values:
        upper = key.upper()
        if not upper.startswith(_NAMESPACE):
            continue
        if component_prefix is None or not upper.startswith(component_prefix + "_"):
            return key
        if upper.endswith(_CREDENTIAL_SUFFIXES):
            return key
        if upper.endswith("_PASSPHRASE_FILE") and component_prefix != "EUGENE_PLEXUS_CONTROL":
            return key
    return None


def child_environment(*, component_prefix: str | None = None) -> dict[str, str]:
    """Keep host settings, plus a component's own non-credential bootstrap.

    Control's passphrase-file bootstrap is deliberate: it unlocks its own root.
    Every other child gets credentials solely from the supervisor's AuthState.
    Foreign executables, including version probes, get no Plexus namespace.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if reserved_override({key: value}, component_prefix=component_prefix) is None
    }
