"""Where another host's model directories are on this one (M11).

The library names a model by its path on the library's own host, and a
runtime declaration copies that string verbatim -- so when the engine
runs on a different machine, the agent there is handed a path from a
filesystem it does not have. `pathMappings` is the operator saying where
the same directory is mounted here: the NAS's `/models` is `Z:\\models`
on the Windows box that mounts it. This module applies that rule, and it
is the only place that does.

Three rules, each with its reason:

* **The declaration is never rewritten.** `RuntimeSpec.modelPath` stays
  the library's spelling, because the library's identity for a model
  *is* that spelling -- `GET /v1/models?path=` and admission's metadata
  basis both look a model up by it -- and a runtime that persisted the
  local path would sever that join. Resolution happens at every point
  this agent opens the file, and the answer is reported as
  `Runtime.localPath`, never stored.
* **Matching happens on the declared string, untouched.** `abspath` on
  Windows turns `/models/x.gguf` into `C:\\models\\x.gguf`, which is not
  a missing path but a different one -- the trap M3's acceptance run hit
  with `/tmp`. So nothing here normalizes the declared path before a
  rule has or has not matched. The rule's `from` decides the
  convention: a drive letter or UNC prefix means Windows rules
  (case-insensitive, either separator), a leading `/` means POSIX rules
  (case-sensitive, `/` only). The agent cannot know the library's
  operating system, and the string came from it, so its shape stands in.
* **By components, not by string prefix.** `/models2/x` is not under
  `/models`. The longest `from` wins; ties go to the first rule listed.

The resolver takes the local separator as a parameter so one test suite
exercises both directions on either platform: CI is Linux, this desk is
Windows, and the real install is a Linux library describing files for a
Windows engine.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from ._generated.models import PathMapping

log = logging.getLogger(__name__)

CONFIG_KEY = "pathMappings"

# `X:` followed by a separator or the end of the string. A bare `X:` is
# drive-relative on Windows and is accepted as a root here because an
# operator who types `D:` means the drive.
_DRIVE = re.compile(r"^[A-Za-z]:(?:[\\/]|$)")


@dataclass(frozen=True)
class PathRule:
    """One `{from, to}` entry, as the operator wrote it."""

    source: str
    target: str

    def as_mapping(self) -> PathMapping:
        """The wire shape, for `ModelLocation.mapping`."""
        return PathMapping.model_validate({"from": self.source, "to": self.target})

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}"


@dataclass(frozen=True)
class Resolution:
    """What a declared path became on this host, and by which rule."""

    declared: str
    local_path: str
    rule: PathRule | None

    @property
    def mapped(self) -> bool:
        return self.rule is not None


def is_windows_shaped(path: str) -> bool:
    """Drive letter or UNC prefix -- the string came from a Windows host."""
    return bool(_DRIVE.match(path)) or path.startswith(("\\\\", "//"))


def is_absolute_path(path: str) -> bool:
    return is_windows_shaped(path) or path.startswith("/")


def _parts(path: str, *, windows: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(comparable parts, raw parts) of `path` under one convention.

    Windows folds case and reads either separator; the raw parts keep
    the operator's spelling so a remainder is re-joined as written.
    """
    if windows:
        raw = PureWindowsPath(path).parts
        return tuple(p.casefold() for p in raw), raw
    raw = PurePosixPath(path).parts
    return raw, raw


def match(rule: PathRule, declared: str) -> tuple[str, ...] | None:
    """The components of `declared` below `rule.source`, or None.

    Component-wise: `/models2/x` is not under `/models`, and a UNC share
    or drive anchor is one component like any other.
    """
    windows = is_windows_shaped(rule.source)
    source, _ = _parts(rule.source, windows=windows)
    folded, raw = _parts(declared, windows=windows)
    if not source or len(folded) < len(source):
        return None
    if folded[: len(source)] != source:
        return None
    return raw[len(source) :]


def join_local(base: str, remainder: Sequence[str], *, sep: str) -> str:
    """`base` with `remainder` appended using this host's separator.

    The base is used as the operator spelled it, minus a trailing
    separator that is not a drive root's (`Z:\\` keeps it; `Z:\\models\\`
    loses it), so `Z:\\models` and `Z:\\models\\` produce one answer.
    """
    trimmed = base
    while len(trimmed) > 1 and trimmed[-1] in "\\/" and trimmed[-2] != ":":
        trimmed = trimmed[:-1]
    if not remainder:
        return trimmed
    if trimmed.endswith(("\\", "/")):
        return trimmed + sep.join(remainder)
    return trimmed + sep + sep.join(remainder)


def resolve_model_path(
    declared: str,
    rules: Sequence[PathRule],
    *,
    sep: str = os.sep,
    expand: Callable[[str], str] = os.path.expanduser,
) -> Resolution:
    """Where `declared` is on this host: through the longest matching
    rule, or unchanged when none matches -- which is every single-host
    install, and every path the operator typed for this machine."""
    best: tuple[int, int, PathRule, tuple[str, ...]] | None = None
    for index, rule in enumerate(rules):
        remainder = match(rule, declared)
        if remainder is None:
            continue
        depth = len(_parts(rule.source, windows=is_windows_shaped(rule.source))[0])
        if best is None or depth > best[0]:
            best = (depth, index, rule, remainder)
    if best is None:
        return Resolution(declared=declared, local_path=declared, rule=None)
    _, _, rule, remainder = best
    return Resolution(
        declared=declared,
        local_path=join_local(expand(rule.target), remainder, sep=sep),
        rule=rule,
    )


# --- the config field -------------------------------------------------------


def parse_rules(value: Any) -> list[PathRule]:
    """Rules from a config value, leniently.

    A hand-edited `agent.yaml` must not stop a spawn (degraded mode is
    the rule for every component), so a malformed entry is skipped with
    a log line rather than raised. `validate_rules` is the strict half,
    applied at PATCH time.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        log.warning("pathMappings is %s, expected a list; ignoring it", type(value).__name__)
        return []
    rules: list[PathRule] = []
    for index, item in enumerate(value):
        if (
            isinstance(item, dict)
            and isinstance(item.get("from"), str)
            and isinstance(item.get("to"), str)
            and item["from"].strip()
            and item["to"].strip()
        ):
            rules.append(PathRule(source=item["from"].strip(), target=item["to"].strip()))
        else:
            log.warning("pathMappings entry %d is malformed (%r); ignoring it", index, item)
    return rules


def rules_from_config(get_config: Callable[[str], Any] | None) -> list[PathRule]:
    """The current rules, read live so a change applies at the next use."""
    if get_config is None:
        return []
    return parse_rules(get_config(CONFIG_KEY))


def validate_rules(value: Any) -> str | None:
    """None if `value` is a well-formed mapping list, else the reason.

    Existence is deliberately not checked: the agent's rule for
    `file_path` is that an operator may point at an environment they
    are about to create, and a share about to be mounted is the same
    case. `POST /v1/config/test` is where existence is checked.
    """
    if not isinstance(value, list):
        return f"expected a list of {{from, to}} mappings, got {type(value).__name__}"
    seen: dict[tuple[str, ...], int] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return (
                f"entry {index} is {type(item).__name__}, expected an object with `from` and `to`"
            )
        extra = sorted(set(item) - {"from", "to"})
        if extra:
            return f"entry {index} has unknown key(s) {extra}; a mapping is `from` and `to` only"
        for key in ("from", "to"):
            candidate = item.get(key)
            if not isinstance(candidate, str) or not candidate.strip():
                return f"entry {index}: `{key}` must be a non-empty path"
        source, target = item["from"].strip(), item["to"].strip()
        if not is_absolute_path(source):
            return (
                f"entry {index}: `from` must be an absolute path as the other machine "
                f"spells it (`/models`, `D:\\models`, `\\\\nas\\models`), got {source!r}"
            )
        if not (is_absolute_path(target) or target.startswith("~")):
            return (
                f"entry {index}: `to` must be an absolute path on this host "
                f"(or start with `~`), got {target!r}"
            )
        identity = _parts(source, windows=is_windows_shaped(source))[0]
        if identity in seen:
            return (
                f"entry {index} duplicates entry {seen[identity]} ({source!r}); one rule per "
                f"directory -- the longest match already decides precedence"
            )
        seen[identity] = index
    return None


# --- the Test button --------------------------------------------------------


@dataclass(frozen=True)
class RuleCheck:
    """What `POST /v1/config/test` found out about one rule."""

    rule: PathRule
    target: str
    target_exists: bool
    target_is_dir: bool
    models_under: int
    reachable: int
    unreachable: list[str]
    mismatched: list[str]


def check_rules(
    rules: Sequence[PathRule],
    library_models: Sequence[dict[str, Any]] | None,
    *,
    exists: Callable[[str], bool] = os.path.exists,
    isdir: Callable[[str], bool] = os.path.isdir,
    size_of: Callable[[str], int | None] | None = None,
    sep: str = os.sep,
    expand: Callable[[str], str] = os.path.expanduser,
) -> list[RuleCheck]:
    """Stat each rule's target, and -- when the library's models are in
    hand -- resolve every model under each `from` and see whether it is
    really here and really the size the library says.

    This is the verification that launches nothing: an operator adding a
    mapping learns whether it reaches the files before saving it.
    """
    out: list[RuleCheck] = []
    for rule in rules:
        target = expand(rule.target)
        present = exists(target)
        is_dir = present and isdir(target)
        under = 0
        reachable = 0
        unreachable: list[str] = []
        mismatched: list[str] = []
        for model in library_models or []:
            path = model.get("path")
            if not isinstance(path, str):
                continue
            resolution = resolve_model_path(path, [rule], sep=sep, expand=expand)
            if resolution.rule is None:
                continue
            under += 1
            local = resolution.local_path
            if not exists(local):
                unreachable.append(local)
                continue
            reachable += 1
            expected = _library_size(model, is_file=not isdir(local))
            if size_of is not None and expected is not None:
                actual = size_of(local)
                if actual is not None and actual != expected:
                    mismatched.append(f"{local} ({actual} bytes here, {expected} in the library)")
        out.append(
            RuleCheck(
                rule=rule,
                target=target,
                target_exists=present,
                target_is_dir=is_dir,
                models_under=under,
                reachable=reachable,
                unreachable=unreachable,
                mismatched=mismatched,
            )
        )
    return out


def describe_checks(
    checks: Sequence[RuleCheck], *, library_consulted: bool
) -> tuple[bool, str, str | None]:
    """(ok, summary, error) for a `ConfigTestResult`."""
    problems: list[str] = []
    lines: list[str] = []
    for check in checks:
        if not check.target_exists:
            problems.append(f"{check.rule}: {check.target} does not exist on this host")
            continue
        if not check.target_is_dir:
            problems.append(f"{check.rule}: {check.target} is not a directory")
            continue
        if not library_consulted:
            lines.append(f"{check.rule}: {check.target} exists here")
            continue
        if check.models_under == 0:
            lines.append(
                f"{check.rule}: {check.target} exists here; the library lists no model "
                f"under {check.rule.source}"
            )
            continue
        lines.append(
            f"{check.rule}: {check.reachable} of {check.models_under} library "
            f"{'model' if check.models_under == 1 else 'models'} under "
            f"{check.rule.source} reachable at {check.target}"
            + ("; sizes match" if check.reachable and not check.mismatched else "")
        )
        if check.unreachable:
            problems.append(f"{check.rule}: not found here: " + ", ".join(check.unreachable[:5]))
        if check.mismatched:
            problems.append(
                f"{check.rule}: size differs from the library's, so the mapping may point "
                f"at a different file or the library's scan is stale: "
                + ", ".join(check.mismatched[:5])
            )
    if not checks:
        return True, "No model directory mappings configured.", None
    summary = "; ".join(lines) if lines else f"{len(checks)} mapping(s) checked."
    if not library_consulted and lines:
        summary += ". The library could not be consulted, so only the directories were checked."
    return (not problems), summary, ("; ".join(problems) if problems else None)


def _library_size(model: dict[str, Any], *, is_file: bool) -> int | None:
    """What the library says the thing at a model's path weighs: the
    weights file for a file path, the whole model for a directory."""
    if is_file:
        for entry in model.get("files") or []:
            if isinstance(entry, dict) and entry.get("role") == "weights":
                size = entry.get("sizeBytes")
                return size if isinstance(size, int) else None
        if model.get("fileCount") == 1:
            size = model.get("sizeBytes")
            return size if isinstance(size, int) else None
        return None
    size = model.get("sizeBytes")
    return size if isinstance(size, int) else None


__all__ = [
    "CONFIG_KEY",
    "PathRule",
    "Resolution",
    "RuleCheck",
    "check_rules",
    "describe_checks",
    "is_absolute_path",
    "is_windows_shaped",
    "join_local",
    "match",
    "parse_rules",
    "resolve_model_path",
    "rules_from_config",
    "validate_rules",
]
