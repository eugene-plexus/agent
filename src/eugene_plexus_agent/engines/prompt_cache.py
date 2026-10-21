"""The engine's prompt cache as a profile setting, and what a running engine says of it (PC7).

`docs/design/prompt-cache.md` and `docs/design/rows/row-09.md` are the design; Troy's call
LD-C5 (2026-10-21): the owner can resize the cache, and the size is a setting on the
profile with its default shown.

llama.cpp keeps the key/value state of recent conversations in system memory
(`--cache-ram`, MiB, 8,192 unless told otherwise), and each slot keeps a few points it can
return to (`--ctx-checkpoints`). Reuse of text found at a new place in the prompt is
`--cache-reuse`. vLLM can offload its KV cache to system memory
(`--kv-offloading-size`, GiB; `--kv-offloading-backend native`).

**Eugene always passes all three llama.cpp numbers**, so a change of upstream default can
never make a setting a profile shows wrong (settings never lie). The defaults below are the
ones of the build Eugene installs (b10948's own `--help`: 8192 MiB, 0, 32). Three rules keep
it from breaking a launch that worked before, the same ones speculation follows:

* **A flag the build does not list is left out** and the running view says so. A build whose
  `--help` could not be read is handed the flags: every build the installers ship has them.
* **`extraArgs` and `env` win.** The profile's flags come before `extraArgs` on the command
  line, so a repeated `--cache-ram` there is the one llama-server takes. A variable such as
  `LLAMA_ARG_CACHE_RAM` in the runtime's `env` is the operator's own setting and is not
  overridden by a default: nothing is passed for a setting the profile leaves unset.
* **A search model keeps no prompt cache.** An embedder or a reranker answers each input
  once; none of the three flags is passed.

What a running engine reports is **read from the command line it was started with**, never
from the profile: the profile can change after the start.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .._generated.models import PromptCacheSource, RuntimeMode, RuntimePromptCache, RuntimeSpec
from . import run_mode

GIB = 1024**3
MIB = 1024**2

RAM_KEY = "cacheRam"
REUSE_KEY = "cacheReuse"
CHECKPOINTS_KEY = "ctxCheckpoints"
KEYS = frozenset({RAM_KEY, REUSE_KEY, CHECKPOINTS_KEY})

RAM_FLAG = "--cache-ram"
REUSE_FLAG = "--cache-reuse"
CHECKPOINTS_FLAG = "--ctx-checkpoints"

#: Every spelling of each flag, as `llama-server --help` lists them (b10948).
RAM_SPELLINGS = ("-cram", RAM_FLAG)
REUSE_SPELLINGS = (REUSE_FLAG,)
CHECKPOINT_SPELLINGS = ("-ctxcp", CHECKPOINTS_FLAG, "--swa-checkpoints")

RAM_VARIABLE = "LLAMA_ARG_CACHE_RAM"
REUSE_VARIABLE = "LLAMA_ARG_CACHE_REUSE"
CHECKPOINTS_VARIABLE = "LLAMA_ARG_CTX_CHECKPOINTS"

#: llama-server's own defaults in the build Eugene installs (b10948): the numbers the
#: fields show, and the numbers every start carries.
DEFAULT_RAM_GIB = 8
DEFAULT_REUSE_TOKENS = 0
DEFAULT_CHECKPOINTS = 32

#: vLLM's setting: GiB of system memory for the KV cache it moves out of the card.
KV_OFFLOAD_KEY = "kvOffloadGb"
KV_OFFLOAD_FLAG = "--kv-offloading-size"
KV_OFFLOAD_BACKEND_FLAG = "--kv-offloading-backend"

_SEARCH_MODES = (RuntimeMode.embedding, RuntimeMode.reranking)


# --- the settings, checked ----------------------------------------------------------


def _number(flags: Mapping[str, object], key: str, default: float, *, whole: bool) -> float:
    value = flags.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{key} must be a number, not {value!r}")
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{key} must be a number, not {value!r}") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{key} must be 0 or more (0 is off), not {value!r}")
    if whole and number != int(number):
        raise ValueError(f"{key} must be a whole number, not {value!r}")
    return number


def ram_gib(flags: Mapping[str, object]) -> float:
    """The profile's "Prompt cache in system memory", in GiB; the default when unset."""
    return _number(flags, RAM_KEY, DEFAULT_RAM_GIB, whole=False)


def ram_mib(flags: Mapping[str, object]) -> int:
    """What the command line carries: the same size in MiB."""
    return round(ram_gib(flags) * 1024)


def reuse_tokens(flags: Mapping[str, object]) -> int:
    return int(_number(flags, REUSE_KEY, DEFAULT_REUSE_TOKENS, whole=True))


def checkpoints(flags: Mapping[str, object]) -> int:
    return int(_number(flags, CHECKPOINTS_KEY, DEFAULT_CHECKPOINTS, whole=True))


def kv_offload_gib(flags: Mapping[str, object]) -> float | None:
    """vLLM's offload size in GiB, or None when it is off (unset, or 0)."""
    number = _number(flags, KV_OFFLOAD_KEY, 0, whole=False)
    return number if number > 0 else None


# --- the command line -----------------------------------------------------------------


def last_value(tokens: Sequence[str], spellings: Sequence[str]) -> str | None:
    """The value of the last `--flag value` or `--flag=value` among `tokens`, the one
    llama-server's parser ends with."""
    found: str | None = None
    items = [str(t) for t in tokens]
    for index, token in enumerate(items):
        name, equals, inline = token.partition("=")
        if name not in spellings:
            continue
        if equals:
            found = inline
        elif index + 1 < len(items):
            found = items[index + 1]
    return found


def _by_hand(env: Mapping[str, str] | None, variable: str) -> str | None:
    value = (env or {}).get(variable)
    return value if value not in (None, "") else None


def argv(
    flags: Mapping[str, object],
    *,
    env: Mapping[str, str] | None,
    offered: frozenset[str] | None,
) -> tuple[list[str], list[str]]:
    """`(tokens, left_out)`: the three llama.cpp flags with their numbers, and the flags the
    build does not list (named so the caller can say so).

    `offered` is the long flags the build's own `--help` mentions; None when it could not be
    read, and then every flag is passed. A setting the profile leaves unset is not passed
    when the runtime's `env` sets it (the operator's own setting stands).
    """
    wanted = (
        (RAM_KEY, RAM_FLAG, str(ram_mib(flags)), RAM_VARIABLE),
        (REUSE_KEY, REUSE_FLAG, str(reuse_tokens(flags)), REUSE_VARIABLE),
        (CHECKPOINTS_KEY, CHECKPOINTS_FLAG, str(checkpoints(flags)), CHECKPOINTS_VARIABLE),
    )
    tokens: list[str] = []
    left_out: list[str] = []
    for key, flag, value, variable in wanted:
        if flags.get(key) is None and _by_hand(env, variable) is not None:
            continue
        if offered is not None and flag not in offered:
            left_out.append(flag)
            continue
        tokens += [flag, value]
    return tokens, left_out


# --- what a running llama.cpp says ----------------------------------------------------


def _gib_text(mib: int) -> str:
    return f"{mib / 1024:g} GiB"


def _is_search(spec: RuntimeSpec, command: Sequence[str]) -> bool:
    if command:
        embeds, reranks = run_mode.started_as(command, spec.env)
        return embeds or reranks
    return spec.mode in _SEARCH_MODES


def llama_view(spec: RuntimeSpec, command: Sequence[str]) -> RuntimePromptCache | None:
    """The prompt cache of a llama.cpp runtime, from its command line.

    `command` is the argv it was started with. Empty: nothing has started (the runtime is
    still copying its file), and the view is what the launch will pass, which is the cache the
    memory is already promised to.
    """
    if _is_search(spec, command):
        return None
    flags = spec.flags or {}
    try:
        asked = ram_mib(flags)
    except ValueError:
        # The launch refuses it and says so; there is no cache to describe.
        return None

    typed = last_value(spec.extraArgs or [], RAM_SPELLINGS)
    from_env = _by_hand(spec.env, RAM_VARIABLE) if flags.get(RAM_KEY) is None else None
    notes: list[str] = []

    if command:
        ran = last_value(command, RAM_SPELLINGS)
        if ran is None and from_env is None:
            # Left out of the command line: the build does not list the flag.
            return RuntimePromptCache(
                bytes=None,
                source=PromptCacheSource.unsupported,
                note=(
                    f"This llama.cpp build has no {RAM_FLAG}, so Prompt cache in system "
                    "memory is not passed."
                ),
            )
        value = ran if ran is not None else from_env
    else:
        value = typed if typed is not None else (from_env if from_env is not None else str(asked))

    try:
        mib = int(float(str(value)))
    except ValueError:
        return RuntimePromptCache(
            bytes=None,
            source=PromptCacheSource.extra_args,
            note=f"The prompt cache size {value!r} on the command line is not a number.",
        )

    by_hand = typed is not None or from_env is not None
    if by_hand:
        where = "extraArgs" if typed is not None else f"this runtime's environment ({RAM_VARIABLE})"
        if flags.get(RAM_KEY) is not None and typed is not None:
            notes.append(
                f"Set to {_gib_text(mib) if mib >= 0 else 'no limit'} in {where}, after the "
                f"profile's {flags[RAM_KEY]} GiB, so the profile's does not apply."
            )
        else:
            notes.append(f"Set in {where}.")
        source = PromptCacheSource.extra_args
    elif flags.get(RAM_KEY) is not None:
        source = PromptCacheSource.profile
    else:
        source = PromptCacheSource.engine_default
    if mib < 0:
        notes.append("There is no limit: the cache grows until system memory runs out.")
        size: int | None = None
    else:
        size = mib * MIB
        if mib == 0:
            source = PromptCacheSource.off
    if command and not by_hand and mib != asked:
        notes.append(
            f"Started with {_gib_text(mib)}. The profile now says {_gib_text(asked)}, which "
            "applies at the next start."
        )
    for flag, label in (
        (REUSE_FLAG, "Reuse moved text"),
        (CHECKPOINTS_FLAG, "Checkpoints per slot"),
    ):
        if command and flag not in command and not _env_sets(spec.env, flag):
            notes.append(f"This llama.cpp build has no {flag}, so {label} is not passed.")
    return RuntimePromptCache(bytes=size, source=source, note=" ".join(notes) or None)


def _env_sets(env: Mapping[str, str] | None, flag: str) -> bool:
    variable = {REUSE_FLAG: REUSE_VARIABLE, CHECKPOINTS_FLAG: CHECKPOINTS_VARIABLE}[flag]
    return _by_hand(env, variable) is not None


# --- what a running vLLM says ---------------------------------------------------------


def vllm_view(spec: RuntimeSpec, command: Sequence[str]) -> RuntimePromptCache | None:
    """vLLM's CPU KV offload, from its command line (`command` empty: what a launch will pass)."""
    flags = spec.flags or {}
    try:
        asked = kv_offload_gib(flags)
    except ValueError:
        return None
    typed = last_value(spec.extraArgs or [], (KV_OFFLOAD_FLAG,))
    if command:
        ran = last_value(command, (KV_OFFLOAD_FLAG,))
        if ran is None:
            if asked is None:
                return RuntimePromptCache(bytes=0, source=PromptCacheSource.off)
            return RuntimePromptCache(
                bytes=None,
                source=PromptCacheSource.unsupported,
                note=(
                    f"Not offered by this vLLM: it does not list {KV_OFFLOAD_FLAG}, so "
                    "Prompt cache in system memory is not passed."
                ),
            )
        value: str = ran
    else:
        value = typed if typed is not None else (str(asked) if asked is not None else "0")
    try:
        gib = float(value)
    except ValueError:
        return RuntimePromptCache(
            bytes=None,
            source=PromptCacheSource.extra_args,
            note=f"The offload size {value!r} on the command line is not a number.",
        )
    size = round(gib * GIB)
    if typed is not None:
        note = (
            "Set in extraArgs."
            if asked is None
            else (
                f"Set to {gib:g} GiB in extraArgs, after the profile's {asked:g} GiB, "
                "so the profile's does not apply."
            )
        )
        return RuntimePromptCache(bytes=size, source=PromptCacheSource.extra_args, note=note)
    if size == 0:
        return RuntimePromptCache(bytes=0, source=PromptCacheSource.off)
    return RuntimePromptCache(bytes=size, source=PromptCacheSource.profile)
