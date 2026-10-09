"""llama.cpp adapter — drives upstream `llama-server`.

We never fork or vendor llama.cpp. This knows how to start the binary the
project ships, how to read its `/health`, and which of its flags are worth
putting in front of a person.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

import httpx

from .._generated.models import (
    Accelerator,
    Arch,
    ConfigField,
    ConfigSchema,
    ConfigValueType,
    EngineKind,
    HostAccelerator,
    ModelFormat,
    ModelRequirement,
    Os,
    RuntimeCapabilities,
    RuntimeSpec,
    Secondary,
)
from ..child_env import child_environment
from . import gpu_probe
from .acquisition import (
    AcquisitionPlan,
    GitHubReleases,
    Release,
    ReleaseAsset,
    Unavailable,
)
from .base import (
    DiscoveredBinary,
    EngineAdapter,
    Loading,
    NotAnswering,
    Readiness,
    Ready,
    default_model_alias,
)
from .base import (
    probe_client as _probe_client,
)

log = logging.getLogger(__name__)

# The alias helper moved to `base` when a second engine needed it; kept
# importable from here so existing callers and tests do not have to move.
__all__ = ["LlamaCppAdapter", "default_model_alias"]

# `llama-server --version` writes to stderr, and upstream changed the
# format mid-2026:
#     version: 9846 (f708a5b2c)                          <= through ~b10000
#     version: 0.4.0-dev (build 10867, commit f3f1a8f27)  <= b10867 onward
#
# The build number is the useful half of both, and it is what the release
# tags and a managed install record. Scraping the newer line the old way
# yields `0.4.0-dev`, which answers none of the questions this field
# exists for and disagrees with the `bNNNN` beside it on the same screen.
# Try the parenthesised build number first.
_BUILD_IN_PARENS_RE = re.compile(r"\bbuild\s+(\d+)")
_VERSION_RE = re.compile(r"\b(?:version|build):\s*(\S+)")

# How long to wait on a version probe. Generous enough for a cold binary on
# a spinning disk, short enough that a wedged executable doesn't stall the
# whole engine listing.
_VERSION_TIMEOUT_SECONDS = 10.0

# Readiness probes run on the supervisor's poll cadence, so they must be
# quick and must never be the thing that blocks a poll round.
_PROBE_TIMEOUT_SECONDS = 2.0

# --- acquisition ---------------------------------------------------------

LLAMA_CPP_REPO = "ggml-org/llama.cpp"

# Builds are `bNNNN` tags. This matters more than it looks: the repository's
# `releases/latest` points at a `v0.4.0` tag that is not a server build at
# all, so resolving "newest" by asking GitHub for the latest release gets
# you something that has none of the assets you want. List and match.
_BUILD_TAG_RE = re.compile(r"^b(\d+)$")

# `llama-b10867-bin-win-cuda-13.3-x64.zip` -> variant `win-cuda-13.3-x64`.
_ASSET_RE = re.compile(r"^llama-b\d+-bin-(?P<variant>.+)\.(?:zip|tar\.gz)$")

# CUDA is a two-asset install on both platforms: the server archive
# carries no CUDA runtime, and the libraries live in a companion archive
# under the same release tag. Installing only the first produces a binary
# that dies on a missing cudart DLL (Windows) or libcudart (Linux).
#
# **The two filenames are not the same shape**, measured against b11010:
#   cudart-llama-bin-win-cuda-13.4-x64.zip            <- no build number
#   cudart-llama-b11010-bin-ubuntu-cuda-13.3-x64.tar.gz  <- has one
# The build number is optional here for exactly that reason. A pattern
# that required the Windows form matched no Linux companion at all, which
# would install a server and report success, then fail at load.
_CUDART_RE = re.compile(r"^cudart-llama-(?:b\d+-)?bin-(?P<variant>.+)\.(?:zip|tar\.gz)$")

# `win-cuda-13.3-x64` or `ubuntu-cuda-13.3-x64` -> ('13', '3'). Used to
# pick the highest published
# CUDA build the installed driver can actually load.
_CUDA_VARIANT_RE = re.compile(
    r"^(?P<platform>win|ubuntu)-cuda-(?P<major>\d+)\.(?P<minor>\d+)-(?P<arch>x64|arm64)$"
)

# `win-rocm-10.0-x64` or `ubuntu-rocm-10.0-x64` -> ('10', '0'); see `_rocm_variant`.
_ROCM_VARIANT_RE = re.compile(
    r"^(?P<platform>win|ubuntu)-rocm-(?P<major>\d+)\.(?P<minor>\d+)-(?P<arch>x64|arm64)$"
)

# How many builds back `plan_latest` looks for one that carries this host's
# assets: about a day of upstream's cadence, so an upload still in progress
# (under an hour) and a CI job failing one variant for a day both still find
# the last build that has it. A host that finds nothing in a day is told the
# newest build's own reason. **Measured 2026-10-03: 20.9 builds a day** (the
# newest hundred spanned 114.9 hours; whole days of 18, 27, 19 and 19). It
# was 8 when upstream published "several a day", which is about nine hours
# now. Re-measure if upstream's cadence changes again.
FALLBACK_BUILDS = 24


class LlamaCppAdapter(EngineAdapter):
    kind = EngineKind.llama_cpp
    binary_name = "llama-server"

    # GGUF only. A safetensors model in the library runs on vLLM
    # instead; the UI joins the two lists to decide which engine a
    # launch button offers.
    model_formats = (ModelFormat.gguf,)
    accepts = (ModelRequirement(format=ModelFormat.gguf, preference=10),)

    # llama-server answers `/health` from the moment its socket is up
    # (503 + "loading model" while the weights are read), so readiness
    # for it is entirely a network observation. Contrast `VllmAdapter`.
    answers_while_loading = True

    # --- discovery --------------------------------------------------------

    def probe_version(self, binary: Path) -> str | None:
        """Run `--version` and scrape the build number.

        llama.cpp writes this to stderr and exits non-zero on some
        builds, so neither is treated as failure — we read both streams
        and only care whether the pattern matched.
        """
        try:
            proc = subprocess.run(
                [str(binary), "--version"],
                env=child_environment(),
                capture_output=True,
                text=True,
                timeout=_VERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            log.debug("version probe for %s failed: %s", binary, e)
            return None

        for stream in (proc.stderr or "", proc.stdout or ""):
            build = _BUILD_IN_PARENS_RE.search(stream)
            if build:
                return build.group(1)
            match = _VERSION_RE.search(stream)
            if match:
                return match.group(1)
        return None

    # --- acquisition ------------------------------------------------------

    releases = GitHubReleases(LLAMA_CPP_REPO)

    def builds(self, *, force: bool = False) -> list[Release]:
        """Upstream's `bNNNN` builds that carry any asset, newest first.

        Sorted by build number rather than by publish order: the tag IS a
        monotonic counter, and trusting it beats trusting a timestamp on a
        repository that publishes several releases an hour.

        A release with no assets is skipped. Found live 2026-09-12: upstream
        published `b10931` at 14:48Z as a release object with **zero**
        assets — its CI had not uploaded yet, or never did — while `b10930`
        an hour earlier carried the usual 27. Every `b` build is marked
        prerelease, so that flag cannot be the filter; the presence of
        assets can. **It is not a sufficient filter** — see `plan_latest`.
        """
        builds = [
            (int(m.group(1)), release)
            for release in self.releases.list_releases(force=force)
            if (m := _BUILD_TAG_RE.match(release.version)) and release.assets
        ]
        builds.sort(key=lambda pair: pair[0], reverse=True)
        return [release for _, release in builds]

    def latest_release(self, *, force: bool = False) -> Release | None:
        """The newest build that carries any asset — what `latestVersion`
        reports. Not necessarily the one an install fetches: that is the
        newest build that carries **this host's** assets, `plan_latest`."""
        builds = self.builds(force=force)
        return builds[0] if builds else None

    def no_release_list_reason(self) -> str:
        """Why there is no build to offer, when the list itself came back empty.

        Two different situations, told apart by `last_failure`: GitHub never
        answered (and here is what happened instead), or it answered and
        none of the builds it listed carries a download. Until 2026-09-26
        both read *"could not reach the upstream release list. Check network
        access"*, which sent a person to a network that was working.
        """
        failure = self.releases.last_failure
        fallback = "Or set `binary` on the runtime to a llama.cpp build you already have."
        if failure is not None:
            return (
                f"could not get the list of llama.cpp builds from GitHub: "
                f"{failure.sentence()} {fallback}"
            )
        return (
            "GitHub's list of llama.cpp builds has no build with downloads in it "
            f"right now. Try again in an hour. {fallback}"
        )

    def plan_latest(
        self, host: HostAccelerator, *, force: bool = False, variant: str | None = None
    ) -> AcquisitionPlan | Unavailable:
        """The newest build this host can actually be given.

        Plans against the newest build first and, when the only obstacle
        is an asset that build does not (yet) carry, steps back to the next
        older one — at most `FALLBACK_BUILDS` deep, which is about a day of
        upstream's cadence.

        Found live 2026-09-15, forty minutes after the 13.3→13.4 finding:
        upstream published `b10991` at 23:42Z and its CI had uploaded
        **five of thirty-three** assets when the acceptance run asked —
        the `cudart` for 12.4 and no server build at all — so "the newest
        build with assets" was b10991 and every host in the install read
        *"not installable"* for as long as the upload took. A release is
        complete for a host only when it carries that host's assets; a
        count of assets says nothing about which. The b10930/b10931
        rule above was the same trap one notch cruder.

        A refusal that is about the **host** (Linux with NVIDIA, a driver
        with no CUDA version) is returned from the newest build at once:
        no older build changes what the host is. When every build in reach
        is missing the asset, the newest build's own reason is returned,
        extended with how far back was looked.
        """
        builds = self.builds(force=force)
        if not builds:
            return Unavailable(reason=self.no_release_list_reason())
        first: Unavailable | None = None
        for release in builds[:FALLBACK_BUILDS]:
            plan = self.plan_acquisition(host, release, variant=variant)
            if isinstance(plan, AcquisitionPlan):
                if release is not builds[0]:
                    log.info(
                        "llama.cpp: release %s has no %r asset yet (%d asset(s) up; "
                        "upstream's CI may still be uploading); using %s, the newest "
                        "build that has one",
                        builds[0].version,
                        plan.variant,
                        len(builds[0].assets),
                        release.version,
                    )
                return plan
            if not plan.release_bound:
                return plan
            first = first or plan
        assert first is not None
        looked = min(len(builds), FALLBACK_BUILDS)
        return Unavailable(
            reason=(
                f"{first.reason} None of the {looked - 1} build(s) before it has one either."
                if looked > 1
                else first.reason
            ),
            release_bound=True,
        )

    def plan_acquisition(
        self, host: HostAccelerator, release: Release, *, variant: str | None = None
    ) -> AcquisitionPlan | Unavailable:
        """Which assets to fetch for this host from ONE release, or why we cannot.

        Returning `Unavailable` is a real answer. Upstream publishes no
        CUDA build for Linux at all — not one that is hard to find, one
        that does not exist — so a Linux box with an NVIDIA GPU has nothing
        we can honestly install. It gets told that, rather than handed the
        Vulkan build: Vulkan runs on NVIDIA but is materially slower at
        prompt processing, and substituting it silently means the operator
        concludes the product is slow rather than that they need to build
        from source.

        `plan_latest` is the caller that chooses the release; this answers
        for the one it is handed, and marks a refusal `release_bound` when
        another release might carry what this one lacks.

        `variant` is the operator's choice over the default, and has to be
        one `alternatives` offers.
        """
        chosen = _chosen_variant(host, release, variant)
        if isinstance(chosen, Unavailable):
            return chosen

        base, plugin = split_variant(chosen)
        assets = _match_assets(release, base)
        if isinstance(assets, Unavailable):
            return assets
        plugin_assets: tuple[ReleaseAsset, ...] = ()
        if plugin is not None:
            found = _match_assets(release, plugin)
            if isinstance(found, Unavailable):
                return found
            plugin_assets = found

        return AcquisitionPlan(
            version=release.version,
            variant=chosen,
            assets=assets + plugin_assets,
            binary_name=self.binary_name,
            plugins=frozenset(a.name for a in plugin_assets),
            # The CUDA runtime goes beside the server, wherever its
            # archive put it: `libggml-cuda.so` looks for it by `$ORIGIN`.
            runtimes=frozenset(a.name for a in assets if _CUDART_RE.match(a.name)),
        )

    # --- diagnosing -------------------------------------------------------

    def explain_exit(self, return_code: int, output_tail: str) -> str | None:
        """Name the argument the engine refused, when that is why it died.

        `llama-server` prints `error: invalid argument: --x` and exits
        before it opens a socket, so every network-level surface reports
        exactly what a still-loading engine reports and the supervisor
        restarts it into the same wall. Observed live 2026-09-17: four
        identical crashes, the reason in the log the whole time, and
        `lastError` saying only that it exited.

        Worth naming the flag rather than saying "bad arguments": the
        operator did not type an argv, they ticked a field or filled a
        box, and the flag is the only thing that leads back to it.
        """
        match = _INVALID_ARGUMENT_RE.search(output_tail)
        if not match:
            return None
        argument = match.group(1).strip()
        return (
            f"llama-server rejected the argument {argument!r} and exited before "
            f"it could serve. This build does not understand it — check "
            f"`extraArgs` on this runtime, and the engine flags on its profile, "
            f"against `{self.binary_name} --help` for the installed build."
        )

    # --- launching --------------------------------------------------------

    def default_env(self, spec: RuntimeSpec, binary: DiscoveredBinary) -> dict[str, str]:
        """An empty `GGML_VK_VISIBLE_DEVICES` for a pinned runtime on a combined build.

        **A pin reaches its own backend only** (measured 2026-09-27 on the
        `+vulkan` build): `CUDA_VISIBLE_DEVICES=-1` hid every CUDA device,
        and llama.cpp loaded the model onto the same 5090 through Vulkan
        instead. So two replicas pinned to two cards would each also have
        taken every card through Vulkan. Emptying the Vulkan list keeps a
        pin meaning one card. It is a default the engine cannot be correct
        without: the runtime's own `env` or an explicit `devices` list
        still wins.
        """
        if not _combined_build(binary.path.parent):
            return {}
        if (spec.flags or {}).get("devices"):
            return {}
        env = spec.env or {}
        if any(env.get(var) is not None for var in _BACKEND_PINS):
            return {"GGML_VK_VISIBLE_DEVICES": ""}
        return {}

    def build_argv(self, spec: RuntimeSpec, binary: DiscoveredBinary, port: int) -> list[str]:
        argv = [
            str(binary.path),
            "--model",
            spec.modelPath,
            "--host",
            spec.host or "127.0.0.1",
            "--port",
            str(port),
        ]

        # The alias is what the engine reports as its model id, which is
        # what the driver reports to the gateway, which is what a client
        # asks for. Defaulting it to the filename is how "the name you
        # see is the name of the file you downloaded" actually holds.
        alias = spec.modelAlias or default_model_alias(spec.modelPath)
        argv += ["--alias", alias]

        flags = spec.flags or {}
        for field in self.flag_schema().fields:
            if field.key in _LOAD_MODE_KEYS:
                # Not a presence-only switch any more; handled together
                # below, because the two collapse into one CLI value.
                continue
            if field.key == CACHE_TYPE_KEY:
                argv += cache_type_argv(flags.get(CACHE_TYPE_KEY))
                # A quantised V cache needs flash attention; `auto` may leave
                # it off on some backends, which fails at load. One rule for
                # the cache and the setting: flash_attention_argv (agent#6).
                argv += flash_attention_argv(flags)
                continue
            if field.key == FLASH_ATTENTION_KEY:
                continue
            if field.key not in flags:
                continue
            value = flags[field.key]
            if value is None:
                continue
            cli = _FLAG_CLI_NAMES[field.key]
            if field.valueType == ConfigValueType.boolean:
                # Presence-only switches (`--cont-batching`,
                # `--no-mmproj-offload`): a false one is absent.
                if value:
                    argv.append(cli)
            else:
                argv += [cli, str(value)]

        argv += _load_mode_argv(flags, binary.path)

        # Verbatim, last, so an operator can always override something the
        # curated surface generated above.
        if spec.extraArgs:
            argv += list(spec.extraArgs)
        return argv

    # --- observing --------------------------------------------------------

    async def probe_readiness(
        self,
        base_url: str,
        *,
        established: bool = False,
        headers: dict[str, str] | None = None,
    ) -> Readiness:
        """Read `llama-server`'s `/health`.

        Upstream's contract: 503 with `{"status": "loading model"}` while
        the weights are being read, 200 with `{"status": "ok"}` once it
        can serve. Distinguishing those is the whole reason readiness is
        per-adapter rather than a generic TCP connect.
        """
        url = base_url.rstrip("/")
        try:
            response = await _probe_client().get(f"{url}/health", timeout=_PROBE_TIMEOUT_SECONDS)
        except httpx.HTTPError as e:
            return NotAnswering(detail=str(e), reached=False)

        if response.status_code == 503:
            return Loading(detail=_status_text(response) or "loading model")
        if not response.is_success:
            return NotAnswering(detail=f"/health returned {response.status_code}", reached=True)

        status = _status_text(response)
        if status and status != "ok":
            # Unknown non-ok status: treat as still coming up rather than
            # asserting readiness we can't vouch for.
            return Loading(detail=status)

        return Ready(capabilities=await self._read_capabilities(url))

    async def _read_capabilities(self, url: str) -> RuntimeCapabilities | None:
        """Read back what the engine actually resolved, from `/props`.

        Deliberately read rather than inferred from the spec: a requested
        context larger than the model or than free memory gets clamped by
        the engine, and the clamped number is the true one. Best-effort —
        a runtime that is serving but won't describe itself is still
        ready.
        """
        try:
            response = await _probe_client().get(f"{url}/props", timeout=_PROBE_TIMEOUT_SECONDS)
            if not response.is_success:
                return None
            props = response.json()
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
            log.debug("capability read from %s failed: %s", url, e)
            return None
        if not isinstance(props, dict):
            return None

        default_generation = props.get("default_generation_settings")
        context_length = None
        if isinstance(default_generation, dict):
            context_length = default_generation.get("n_ctx")
        if context_length is None:
            context_length = props.get("n_ctx")

        slots = props.get("total_slots")
        modalities = props.get("modalities")
        multimodal = None
        if isinstance(modalities, dict):
            multimodal = bool(modalities.get("vision") or modalities.get("audio"))

        return RuntimeCapabilities(
            contextLength=int(context_length) if isinstance(context_length, int) else None,
            parallelSlots=int(slots) if isinstance(slots, int) and slots >= 1 else None,
            embeddings=None,
            multimodal=multimodal,
            vision=modalities.get("vision") is True if isinstance(modalities, dict) else None,
        )

    def companion_overrides(self, spec: RuntimeSpec) -> dict[str, object]:
        """CB4: the driver pins slots exactly when the engine was started
        for it (`--no-cache-idle-slots`), from one profile flag."""
        flags = spec.flags or {}
        return {"slotPinning": True} if flags.get(SLOT_PINNING_KEY) is True else {}

    def context_pool(
        self,
        capabilities: RuntimeCapabilities,
        argv: list[str],
        env: dict[str, str] | None,
    ) -> int | None:
        """The context llama-server's slots share, from `/props` and our argv.

        **`/props` cannot say whether the pool is unified** (b11211's props
        builder has no such field), and the argv can. Measured on b11211
        with `-c 16384`: automatic slots report `n_ctx` 16384 over 4 slots
        with `kv_unified = 'true'`; `--parallel 4` reports 4096 each,
        divided; `--parallel 4 --kv-unified` reports 16384, shared. So:
        slots unset (or `-1`, auto) share `n_ctx`; an explicit count
        divides it unless `--kv-unified` is passed; `--no-kv-unified`
        divides it whatever the count. The last flag wins, as in
        llama.cpp's own parser, and an argv flag beats its environment
        variable.

        `--kv-unified-per-slot` caps each slot inside a shared pool and
        `/props` then reports the cap, not the pool, so the pool is
        unknown and this says None rather than guess.
        """
        context = capabilities.contextLength
        if not context:
            return None
        env = env or {}
        parallel: str | None = env.get("LLAMA_ARG_N_PARALLEL")
        unified: bool | None = _env_bool(env.get("LLAMA_ARG_KV_UNIFIED"))
        if env.get("LLAMA_ARG_KV_UNIFIED_PER_SLOT"):
            return None
        tokens = iter(argv)
        for token in tokens:
            name, _, inline = token.partition("=")
            if name in ("-np", "--parallel"):
                parallel = inline or next(tokens, None)
            elif name in ("-kvu", "--kv-unified"):
                unified = True
            elif name in ("-no-kvu", "--no-kv-unified"):
                unified = False
            elif name == "--kv-unified-per-slot":
                return None
        automatic = parallel is None or parallel.strip() == "-1"
        if unified is None:
            unified = automatic
        return context if unified else None

    # --- configuring ------------------------------------------------------

    def flag_schema(self) -> ConfigSchema:
        return ConfigSchema(
            component="engine:llama_cpp",
            categories=_CATEGORIES,
            fields=_FLAG_FIELDS,
        )


def _env_bool(value: str | None) -> bool | None:
    """llama.cpp's reading of a boolean `LLAMA_ARG_*` variable, or None."""
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("1", "true", "on", "enabled"):
        return True
    if lowered in ("0", "false", "off", "disabled"):
        return False
    return None


def _status_text(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(body, dict):
        status = body.get("status")
        if isinstance(status, str):
            return status
    return None


# --------------------------------------------------------------------------- #
# How the model is read off disk: `--load-mode` since about b10900, two
# separate switches before it
# --------------------------------------------------------------------------- #

# `--no-mmap` and `--mlock` were replaced upstream by one `-lm/--load-mode
# MODE`, and a build that has the new spelling **rejects the old one**:
# `error: invalid argument: --no-mmap`, exit non-zero, and the supervisor
# restarts it into the same wall. Found live 2026-09-17 on b10948, by
# ticking a field this project's own schema offers.
#
# So the spelling is **read off the binary** rather than gated on a build
# number, which is the same rule the CUDA variant matrix learned on
# 2026-09-12: upstream's own artifact is the only thing that knows what
# upstream published. `--help` is one subprocess per binary per process
# run, memoised below.
#
# The two booleans collapse into one value, and the old pair's meaning is
# preserved exactly: `--mlock` alone used to mean "mmap it AND pin it",
# because mmap was the default it did not disturb.
_LOAD_MODE_KEYS = ("noMmap", "mlock")

_LOAD_MODE_BY_FLAGS: dict[tuple[bool, bool], str] = {
    # (noMmap, mlock)
    (True, False): "none",
    (False, True): "mmap+mlock",
    (True, True): "mlock",
}

# `error: invalid argument: --no-mmap` — upstream's own wording in
# `common/arg.cpp`, printed to stderr just before a non-zero exit.
_INVALID_ARGUMENT_RE = re.compile(r"error:\s*invalid argument:\s*(\S+)")

_HELP_TIMEOUT_SECONDS = 15.0
_LONG_FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")

# Keyed by (path, mtime, size): a rebuilt or upgraded binary at the same
# path is a different binary, and the engine store writes each build to
# its own directory anyway.
_help_flags_cache: dict[tuple[str, float, int], frozenset[str]] = {}


def _supported_long_flags(binary: Path) -> frozenset[str] | None:
    """Every `--long-flag` this binary's own `--help` mentions.

    None when the binary could not be asked, which is a different answer
    from "it supports nothing" and must not be read as one: an argv is
    still built, on the legacy spelling, because refusing to launch over
    a failed `--help` would turn a cosmetic probe into an outage.
    """
    try:
        stat = binary.stat()
        key = (str(binary), stat.st_mtime, stat.st_size)
    except OSError:
        return None
    cached = _help_flags_cache.get(key)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            env=child_environment(),
            text=True,
            timeout=_HELP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("help probe for %s failed: %s", binary, e)
        return None
    found = frozenset(_LONG_FLAG_RE.findall((proc.stdout or "") + (proc.stderr or "")))
    if not found:
        return None
    _help_flags_cache[key] = found
    return found


def places_by_itself(adapter: LlamaCppAdapter, spec: RuntimeSpec, configured: str | None) -> bool:
    """Whether the llama-server this spec would run lists `--fit` in its help.

    Every build the installers ship has it, on by default, and with it an
    unset `gpuLayers` asks llama.cpp to place the model, experts or layers
    in host memory as needed (moe-aware-fit call A). Read off the build's
    own `--help`, cached per binary, so a configured old build without
    `--fit` keeps the full-offload reading. Anything that cannot be
    resolved or asked answers False, the conservative reading.
    """
    try:
        found = adapter.resolve_binary(spec, configured=configured)
    except Exception:
        return False
    flags = _supported_long_flags(found.path)
    return flags is not None and "--fit" in flags


def _load_mode_argv(flags: dict, binary: Path) -> list[str]:
    """Translate `noMmap` / `mlock` into whichever spelling this binary takes.

    Emits nothing when neither is set — the engine's own default (`auto`)
    is the right answer and naming it would be us deciding something the
    operator did not.
    """
    no_mmap = bool(flags.get("noMmap"))
    mlock = bool(flags.get("mlock"))
    if not no_mmap and not mlock:
        return []

    supported = _supported_long_flags(binary)
    if supported is None or "--load-mode" in supported:
        # Unknown binaries take the current spelling: a build old enough
        # to lack `--load-mode` predates every build this project has ever
        # installed, and guessing forward ages better than guessing back.
        mode = _LOAD_MODE_BY_FLAGS[(no_mmap, mlock)]
        if supported is None:
            log.debug("could not read --help from %s; assuming --load-mode", binary)
        return ["--load-mode", mode]

    legacy: list[str] = []
    if no_mmap and "--no-mmap" in supported:
        legacy.append("--no-mmap")
    if mlock and "--mlock" in supported:
        legacy.append("--mlock")
    if not legacy:
        # Neither spelling exists. Dropping the request is the only thing
        # left that still starts the engine, and it is said out loud
        # because a flag silently ignored is worse than one refused.
        log.warning(
            "%s understands neither --load-mode nor --no-mmap/--mlock; "
            "ignoring the noMmap/mlock flags on this runtime",
            binary,
        )
    return legacy


_CATEGORIES = {
    "memory": "Memory and context",
    "performance": "Performance",
    "sampling": "Serving",
}

#: CB4: keep each conversation in one engine slot (the driver pins it).
SLOT_PINNING_KEY = "slotPinning"

# Curated flags -> llama-server CLI names. Kept as a separate mapping from
# the schema so the UI-facing key never has to look like a CLI flag, and
# so an upstream rename touches one line.
_FLAG_CLI_NAMES: dict[str, str] = {
    "projectorPath": "--mmproj",
    "projectorOnCpu": "--no-mmproj-offload",
    "contextSize": "--ctx-size",
    "gpuLayers": "--n-gpu-layers",
    "batchSize": "--batch-size",
    "ubatchSize": "--ubatch-size",
    "threads": "--threads",
    "parallelSlots": "--parallel",
    "mainGpu": "--main-gpu",
    "splitMode": "--split-mode",
    "devices": "--device",
    "tensorSplit": "--tensor-split",
    "flashAttention": "--flash-attn",
    "continuousBatching": "--cont-batching",
    # CB4: the engine half of slot pinning; the companion's `slotPinning`
    # is the other half, set from the same flag (`companion_overrides`).
    SLOT_PINNING_KEY: "--no-cache-idle-slots",
    "memoryMargin": "--fit-target",
}

# The cache precision is one choice written as two CLI flags, because the
# flash-attention kernels of every build measured handle matching K/V pairs
# only (`FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16`, b11215,
# docs/design/profile-builder.md M7). Offering K and V separately would
# offer combinations that fall back to slow paths.
CACHE_TYPE_KEY = "cacheType"
CACHE_TYPES = ("f16", "q8_0", "q4_0")


def cache_type_argv(value: object) -> list[str]:
    """`--cache-type-k T --cache-type-v T`, or nothing for an unset value."""
    if value is None:
        return []
    if value not in CACHE_TYPES:
        raise ValueError(f"cacheType must be one of {', '.join(CACHE_TYPES)}")
    return ["--cache-type-k", str(value), "--cache-type-v", str(value)]


FLASH_ATTENTION_KEY = "flashAttention"


def flash_attention_choice(flags: dict[str, object]) -> str | None:
    """`on`, `off`, or None for "the engine decides" (agent#6).

    A profile saved before the setting had three states holds a boolean.
    True always sent `on`, so it is `on`. False always sent nothing, which
    left llama.cpp's own `auto` (on wherever supported), so it is None --
    never `off`: reading an old False as off would turn flash attention off
    in every existing profile.
    """
    value = flags.get(FLASH_ATTENTION_KEY)
    if value is True or value == "on":
        return "on"
    if value == "off":
        return "off"
    return None


def flash_attention_argv(flags: dict[str, object]) -> list[str]:
    """What `--flash-attn` says, if anything. A quantised V cache needs it
    on, whatever the profile says, so `off` there is overridden and logged."""
    choice = flash_attention_choice(flags)
    quantised = flags.get(CACHE_TYPE_KEY) not in (None, "f16")
    if quantised:
        if choice == "off":
            log.warning(
                "flash attention is set off, but an 8- or 4-bit memory precision needs it, "
                "so it is turned on for this start"
            )
        return ["--flash-attn", "on"]
    return ["--flash-attn", choice] if choice else []


_FLAG_FIELDS: list[ConfigField] = [
    ConfigField(
        key="projectorPath",
        label="Vision projector",
        description=(
            "Path on this node to the matching mmproj GGUF for this vision model. "
            "Download it from the same model repository. Image input is available "
            "only after the engine confirms that it loaded the projector."
        ),
        category="memory",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
    ConfigField(
        key="projectorOnCpu",
        label="Run vision projector on CPU",
        description="Keep image processing off the GPU. Slower, but uses less GPU memory.",
        category="performance",
        valueType=ConfigValueType.boolean,
        requiresRestart=True,
    ),
    ConfigField(
        key="contextSize",
        label="Context size",
        description=(
            "Total tokens this runtime keeps in memory, shared by its "
            "slots. The single biggest lever on memory use. With more "
            "than one slot the engine divides this between them, so each "
            "request sees a smaller window and the memory is unchanged. "
            "Leave unset to take the model's trained context — the engine "
            "clamps anything larger than the model or than available "
            "memory. A runtime reports back the window one slot gets, "
            "which is this value divided by the slot count — not a sign "
            "it ran short of memory."
        ),
        category="memory",
        valueType=ConfigValueType.integer,
        minimum=256,
        requiresRestart=True,
    ),
    ConfigField(
        key="gpuLayers",
        label="GPU layers",
        description=(
            "How many model layers to offload to the GPU. High enough to "
            "fit is the goal; too high fails at load or spills into system "
            "memory and runs slower than the CPU would. Set to 0 for "
            "CPU-only, or a large number to offload everything."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=0,
        requiresRestart=True,
    ),
    ConfigField(
        key="parallelSlots",
        label="Parallel slots",
        description=(
            "Concurrent requests this runtime serves. A number set here "
            "divides the context size between the slots rather than each "
            "getting their own, so 4 slots at 32k gives every request 8k "
            "and costs exactly what 1 slot at 32k costs — raise this and "
            "raise context together if each request needs the same room "
            "as before. Left unset, llama.cpp runs 4 slots that share one "
            "pool of the whole context instead: a request alone can use "
            "all of it, and requests running at once share it. This is "
            "also the unit of capacity the gateway divides work across."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        # Not `default=1`: an unset value rendered as one slot while
        # llama-server's `-np` defaults to -1, automatic -- 4 slots and a
        # unified pool (`tools/server/server.cpp` at b11375), which is how
        # `context_pool` reads it too (drift audit 2026-10-03).
        unsetMeans=(
            "Not set: llama.cpp decides (automatic), which is 4 slots that share one "
            "pool of the whole context, so a request alone can use all of it."
        ),
        requiresRestart=True,
    ),
    ConfigField(
        key="batchSize",
        label="Batch size",
        description=(
            "Logical batch size for prompt processing. Larger speeds up "
            "long prompts at the cost of memory during the prefill."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="ubatchSize",
        label="Micro-batch size",
        description=(
            "Physical batch size actually handed to the GPU at once. Only "
            "worth touching if you are tuning prefill throughput."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="threads",
        label="CPU threads",
        description=(
            "Threads used for CPU-side work. Defaults to the engine's own "
            "guess, which is usually right; matters most when layers are "
            "not fully offloaded."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=1,
        requiresRestart=True,
    ),
    ConfigField(
        key="flashAttention",
        label="Flash attention",
        description=(
            "The fused attention kernel: usually faster and lighter on memory "
            "where the build and card support it. Left unset, llama.cpp "
            "decides for itself (auto), which turns it on wherever it can. An "
            "8-bit or 4-bit memory precision needs it, so it is on with those "
            "whatever this says."
        ),
        category="performance",
        valueType=ConfigValueType.enum,
        enumValues=["on", "off"],
        enumLabels=["On", "Off"],
        unsetMeans="Not set: llama.cpp decides (auto), which turns it on wherever it can.",
        requiresRestart=True,
    ),
    ConfigField(
        key=CACHE_TYPE_KEY,
        label="Memory precision",
        description=(
            "How precisely the model keeps what it has read so far (its KV "
            "cache). Full is exact. 8-bit and 4-bit use about half and a "
            "quarter of the memory, leaving room for more text or more of the "
            "model on the graphics card, and change answers slightly: how much "
            "depends on the model, by as much as ninefold between two measured "
            "ones, so the profile builder measures it on yours. 8-bit and "
            "4-bit need flash attention, which is switched on with them."
        ),
        category="memory",
        valueType=ConfigValueType.enum,
        enumValues=list(CACHE_TYPES),
        enumLabels=["Full (16-bit)", "8-bit", "4-bit"],
        unsetMeans="Full precision (16-bit), llama.cpp's own default.",
        requiresRestart=True,
    ),
    ConfigField(
        key="memoryMargin",
        label="Leave this much graphics memory free (MiB)",
        description=(
            "When llama.cpp decides how much of the model goes on each graphics "
            "card, it leaves this much memory unused on every one. Raise it if "
            "you game or render on the same card while a model is loaded."
        ),
        category="memory",
        valueType=ConfigValueType.integer,
        minimum=0,
        maximum=65536,
        unsetMeans="1024 MiB on each card, llama.cpp's own default.",
        requiresRestart=True,
    ),
    ConfigField(
        key="continuousBatching",
        label="Continuous batching",
        description=(
            "Interleave requests across slots instead of serving them one "
            "after another. Wanted whenever more than one slot is in use."
        ),
        category="performance",
        valueType=ConfigValueType.boolean,
        default=True,
        requiresRestart=True,
    ),
    ConfigField(
        key=SLOT_PINNING_KEY,
        label="Keep each conversation in one slot",
        description=(
            "Give each conversation its own slot and keep it there, instead "
            "of llama.cpp's choice, which puts every conversation that starts "
            "alike (two agent sessions in one project) in one slot so each "
            "reads its history again. Only worth it when the context each "
            "slot gets holds a whole conversation: measured, about 3 points "
            "more of each prompt reused there, and up to 10 fewer where it "
            "does not, because the slots keep their histories between turns. "
            "Turns on the engine's --no-cache-idle-slots and the driver's "
            "pinning together; a turn whose slot is busy waits for it, "
            "because naming a busy slot can stall llama-server."
        ),
        category="performance",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
    ConfigField(
        key="mainGpu",
        label="Main GPU",
        description=(
            "Index of the GPU to prefer for single-GPU work. To pin a "
            "runtime to one card entirely, set CUDA_VISIBLE_DEVICES in the "
            "runtime's environment instead — that is how two replicas end "
            "up on two cards."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        minimum=0,
        requiresRestart=True,
    ),
    ConfigField(
        key="devices",
        label="GPUs to use",
        description=(
            "Which devices to spread the model across, by llama.cpp's own "
            "names, comma-separated: CUDA0, Vulkan1. Leave empty for every "
            "discrete card, which is llama.cpp's default. Naming an "
            "integrated GPU here (with a combined +vulkan build) is how to "
            "test it as overflow beside a card. A Vulkan name cannot be "
            "matched to a card Eugene measured, so such a launch is not "
            "checked against memory. Run llama-server --list-devices to see "
            "the names."
        ),
        category="performance",
        valueType=ConfigValueType.string,
        pattern=r"^\s*[A-Za-z]+\d*(\s*,\s*[A-Za-z]+\d*)*\s*$",
        requiresRestart=True,
    ),
    ConfigField(
        key="splitMode",
        label="Split across GPUs",
        description=(
            "How one model is spread across several GPUs. By layer is "
            "llama.cpp's default: each card holds whole layers and the "
            "memory adds up. By row splits each weight across the cards and "
            "can be faster on cards joined by a fast link. Tensor is "
            "upstream's experimental parallel split. One GPU only uses the "
            "Main GPU and leaves the others free. Read off llama-server "
            "b11211's own help."
        ),
        category="performance",
        valueType=ConfigValueType.enum,
        enumValues=["layer", "row", "tensor", "none"],
        enumLabels=["By layer", "By row", "Tensor (experimental)", "One GPU only"],
        requiresRestart=True,
    ),
    ConfigField(
        key="tensorSplit",
        label="Tensor split",
        description=(
            "How to divide the model across multiple GPUs, as "
            "comma-separated proportions (e.g. `0.6,0.4`). Only meaningful "
            "with more than one visible GPU."
        ),
        category="performance",
        valueType=ConfigValueType.string,
        pattern=r"^\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*$",
        requiresRestart=True,
    ),
    ConfigField(
        key="mlock",
        label="Lock model in memory",
        description=(
            "Prevent the OS from paging the model out. Keeps latency "
            "predictable; needs enough RAM to hold the whole thing and may "
            "require elevated permissions."
        ),
        category="memory",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
    ConfigField(
        key="noMmap",
        label="Disable mmap",
        description=(
            "Read the model into memory instead of mapping it from disk. "
            "Needs enough free RAM to hold the whole file. On a local "
            "disk this is slower to start; **over a network share it is "
            "usually much faster**, because a mapped read never gets the "
            "read-ahead a plain sequential read does. Measured 2026-09-17 "
            "on a 24.95 GB model over SMB on a 1 Gbps link: 10m01s "
            "mapped against 4m16s read, where the same share delivers "
            "113 MB/s to an ordinary sequential read."
        ),
        category="memory",
        valueType=ConfigValueType.boolean,
        default=False,
        requiresRestart=True,
    ),
]


def _variant_for(host: HostAccelerator, release: Release) -> str | Unavailable:
    """Map a detected host to the asset variant that fits it.

    `release` is consulted only for Windows CUDA, whose published minor
    versions are read off the release's own assets rather than a table
    (see `_cuda_variant`).
    """
    if host.os is None or host.arch is None:
        return Unavailable(
            reason=(
                "could not identify this operating system or CPU architecture, so "
                "there is no way to tell which build would run here. Point `binary` "
                "at a llama-server you trust."
            )
        )

    accelerator = host.accelerator or Accelerator.none

    if host.os is Os.macos:
        # Metal is compiled into the plain macOS build; there is no separate
        # accelerator variant to choose.
        return "macos-arm64" if host.arch is Arch.arm64 else "macos-x64"

    if host.os is Os.windows:
        if accelerator is Accelerator.cuda:
            cuda = _cuda_variant(host, release)
            if (
                isinstance(cuda, str)
                and host.secondary is Secondary.vulkan
                and gpu_probe.combinable_with_cuda(host.os.value, host.arch.value)
            ):
                # A discrete AMD or Intel card beside the NVIDIA one:
                # the CUDA build with the Vulkan backend added, so both
                # are used (HostAccelerator.secondary). **Windows x64
                # only, by the same rule detection uses** (drift audit
                # 2026-10-03): upstream publishes no `win-vulkan-arm64`,
                # so on arm64 this asked for a build that does not exist
                # and the plan failed on it.
                return f"{cuda}{VULKAN_SUFFIX}"
            return cuda
        if accelerator is Accelerator.rocm:
            return _rocm_variant(host.arch, release, "win")
        if accelerator is Accelerator.vulkan:
            # **The build every non-NVIDIA Windows GPU gets** (review
            # §6.1 #11). One asset covers AMD and Intel, needs no vendor
            # SDK, and upstream publishes it in every release.
            return f"win-vulkan-{host.arch.value}"
        if accelerator is Accelerator.sycl and host.arch is Arch.x64:
            # `sycl` is reported only when oneAPI's own `sycl-ls` saw an
            # Intel GPU. Until 2026-09-27 this branch was missing and such
            # a machine got the CPU build: the comment above said
            # `win-sycl-x64` appeared nowhere, which stopped being true
            # (b11211 publishes it) without anyone re-reading a release.
            return "win-sycl-x64"
        return f"win-cpu-{host.arch.value}"

    # Linux. **Upstream publishes CUDA builds here now**, and it did not
    # when this branch was written -- see `_cuda_variant`'s note. The
    # refusal that stood here, and the Vulkan-with-a-badge plan drawn up
    # to replace it, were both answers to a fact that has since changed.
    if accelerator is Accelerator.cuda:
        return _cuda_variant(host, release)
    if accelerator is Accelerator.rocm:
        return _rocm_variant(host.arch, release, "ubuntu")
    if accelerator is Accelerator.sycl and host.arch is Arch.x64:
        return "ubuntu-sycl-fp16-x64"
    if accelerator is Accelerator.vulkan:
        # An AMD card without ROCm, an Intel card without oneAPI, an
        # NVIDIA card on the Mesa driver (2026-09-27). Before, the first
        # got a CPU build and the second a SYCL build it could not load.
        return f"ubuntu-vulkan-{host.arch.value}"
    return f"ubuntu-{host.arch.value}" if host.arch is Arch.arm64 else "ubuntu-x64"


def _rocm_variant(arch: Arch, release: Release, platform: str) -> str | Unavailable:
    """The ROCm build `release` publishes for this platform, by its own name.

    **Read off the asset names, like the CUDA versions** (drift audit
    2026-10-03). This was the constant `rocm-10.0`, while upstream bakes
    the version its CI was given into the name (`ROCM_VERSION_SHORT` in
    `release.yml`), so the day it moved, every AMD host would have read
    "not installable" for a build that was there. Upstream publishes one
    ROCm version per platform; if it ever publishes several, the newest is
    taken -- compared as numbers, so 10.0 is above 9.4 -- and a host whose
    own ROCm is older may not load it, which the expert's `variant` menu
    answers.
    """
    found = [
        ((int(m.group("major")), int(m.group("minor"))), variant)
        for variant in _published_variants(release)
        if (m := _ROCM_VARIANT_RE.match(variant))
        and m.group("platform") == platform
        and m.group("arch") == arch.value
    ]
    if found:
        return max(found)[1]
    platform_name = "Windows" if platform == "win" else "Linux"
    return Unavailable(
        reason=(
            f"release {release.version} publishes no {platform_name} ROCm build for "
            f"{arch.value}. Published variants: "
            f"{', '.join(_published_variants(release)) or '(none)'}."
        ),
        # A release mid-upload looks like this; the one before it may not.
        release_bound=True,
    )


# The prefixes a server build for each OS carries. **Linux has two**:
# `ubuntu-` for the generic builds and `linux-` for
# `linux-arm64-snapdragon` (OpenCL on Adreno plus Hexagon, built in
# Qualcomm's own toolchain container), which the expert's menu never
# offered because it listed `ubuntu-` alone (drift audit 2026-10-03).
# `android-` is not Linux for this purpose.
_PLATFORM_PREFIXES: dict[Os, tuple[str, ...]] = {
    Os.windows: ("win-",),
    Os.linux: ("ubuntu-", "linux-"),
    Os.macos: ("macos-",),
}

_BACKEND_PINS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def _combined_build(directory: Path) -> bool:
    """A build carrying the Vulkan backend beside a CUDA or HIP one."""
    names = {p.name.lower() for p in directory.glob("*ggml-*")}
    vulkan = {"ggml-vulkan.dll", "libggml-vulkan.so"} & names
    other = {"ggml-cuda.dll", "libggml-cuda.so", "ggml-hip.dll", "libggml-hip.so"} & names
    return bool(vulkan and other)


#: A build with the Vulkan backend from the same release added to it.
VULKAN_SUFFIX = "+vulkan"


def split_variant(variant: str) -> tuple[str, str | None]:
    """`win-cuda-13.4-x64+vulkan` -> (`win-cuda-13.4-x64`, `win-vulkan-x64`)."""
    if not variant.endswith(VULKAN_SUFFIX):
        return variant, None
    base = variant[: -len(VULKAN_SUFFIX)]
    arch = base.rsplit("-", 1)[-1]
    return base, f"win-vulkan-{arch}"


def alternatives(host: HostAccelerator, release: Release) -> list[str]:
    """Every server build `release` publishes for this OS and CPU.

    The expert's menu (`EngineInstallRequest.variant`). Read off the
    release rather than a table, because upstream adds a family every few
    months: `win-sycl-x64`, `win-openvino-*` and `win-opencl-adreno-arm64`
    all arrived without this code hearing about it.
    """
    if host.os is None or host.arch is None:
        return []
    prefixes = _PLATFORM_PREFIXES[host.os]
    # The architecture is a dash-separated word of the name, usually the
    # last one -- `linux-arm64-snapdragon` puts it second.
    offered = [
        v
        for v in _published_variants(release)
        if v.startswith(prefixes) and host.arch.value in v.split("-")
    ]
    vulkan = f"win-vulkan-{host.arch.value}"
    if host.os is Os.windows and vulkan in offered:
        # Each CUDA build with the Vulkan backend added: the default for
        # a second vendor's card beside NVIDIA, and the expert's way to
        # try an integrated GPU as overflow (with the `devices` flag).
        offered += [f"{v}{VULKAN_SUFFIX}" for v in offered if v.startswith("win-cuda-")]
    return sorted(offered)


def _chosen_variant(
    host: HostAccelerator, release: Release, requested: str | None
) -> str | Unavailable:
    """The default build, or the one the operator asked for if it is here."""
    if requested is None:
        return _variant_for(host, release)
    offered = alternatives(host, release)
    if requested in offered:
        return requested
    platform_name = host.os.value if host.os is not None else "this operating system"
    arch_name = host.arch.value if host.arch is not None else "this CPU"
    prefixes = _PLATFORM_PREFIXES.get(host.os) if host.os is not None else None
    fits_host = bool(prefixes and requested.startswith(prefixes) and host.arch is not None)
    return Unavailable(
        reason=(
            f"{requested!r} is not a build release {release.version} publishes for "
            f"{platform_name} on {arch_name}. It publishes: {', '.join(offered) or '(none)'}."
        ),
        # A build this machine could run, missing from this release, may be
        # in the one before it (a release mid-upload); anything else will
        # not be in any release.
        release_bound=fits_host,
    )


def _cuda_variant(host: HostAccelerator, release: Release) -> str | Unavailable:
    """The CUDA build this driver can load, from what the release publishes.

    Four rules, in order, all NVIDIA's rather than ours:

    0. **Only builds that carry code for the card** (2026-09-23). From
       CUDA 13 the toolkit no longer compiles for Maxwell, Pascal or
       Volta, and upstream's `ggml-cuda/CMakeLists.txt` adds
       `50-virtual 61-virtual 70-virtual` only below 13 -- so a card
       below 7.5 cannot run a 13.x build, and the last driver branch
       that supports one (580) reports CUDA 13.0. Before this rule a
       Pascal card on that driver was handed exactly the build with no
       kernels for it, which fails at model load naming nothing we did.
       The LOWEST card decides (`host._probe_compute_capability`). An
       unknown capability filters nothing, which is the old behaviour.

    1. **Prefer a build for the driver's own CUDA minor or an older one**
       within the same major -- the highest such minor. A 12.4 build on a
       12.8 driver is the ordinary case and needs no caveat.
    2. **Otherwise take the lowest published minor above the driver's,
       still within the same major -- when the card has finished code
       in it.** CUDA's minor-version compatibility guarantees that an
       application built with any 13.x toolkit runs on any driver of the
       13.x family, minus PTX JIT for newer PTX and minus APIs the older
       driver lacks. **The PTX exclusion bites** (corrected 2026-10-03;
       this said it did not): upstream's builds carry finished code only
       for 8.6, 8.9, 12.0 and 12.1 and PTX for the rest
       (`_FINISHED_CODE`), so a Turing, A100 or H100 on a 13.0 driver
       would be handed a 13.4 build it cannot load. Such a card takes
       rule 3 instead, and with no older major is refused with the fix.
       An unknown capability takes the newer minor, as before. **Verified live
       2026-09-15:** the day upstream stopped publishing `win-cuda-13.3`
       and shipped `win-cuda-13.4-x64` instead (every build from b10983
       on), the b10990 13.4 build loaded a 1.8 GB Q8 on an RTX 5090
       whose driver reports 13.3, offloaded to the GPU, and served a
       completion, `model loaded` at 1.6 s. Before this rule that host
       read *"not installable"* -- the second time in four days a host
       with a perfectly good GPU was refused for a reason that was about
       upstream's publishing and not about the host. The choice is
       logged, because a build newer than the driver is a fact worth
       having in the log if the engine ever does fail to start.

    3. **Otherwise the newest build from an OLDER major.** NVIDIA's
       drivers are backward compatible and the cudart companion ships
       the build's own runtime, so a 12.x build loads on a 13.x driver.
       This is how a Pascal card on a 580 driver gets the 12.8 build.
       Until 2026-09-23 a different major was refused *either way*,
       which also told the owner of a 13.x driver facing a 12.x-only
       release to *update* a driver that was already newer.

    **A NEWER major is never crossed**: a 13.x build needs a 13.x
    driver, and the refusal says so and names the fix.

    The candidate minors come from **the release's asset names**, not a
    table. The table this replaced was written from b10867 and went stale
    the moment upstream moved from 13.3 to 13.4; a stale table cannot
    fail loudly when its defect is an entry it lacks.

    **Both platforms, since 2026-09-16.** This was Windows-only because
    Linux+NVIDIA was refused outright: *"llama.cpp publishes no prebuilt
    CUDA build for Linux"*, a claim re-verified against upstream on
    2026-09-11 and **false five days later**. b11010 publishes
    `ubuntu-cuda-12.8-x64`, `ubuntu-cuda-13.3-x64` and
    `ubuntu-cuda-13.3-arm64`, with cudart companions for each. The same
    two rules pick among them, because they are NVIDIA's rules and not
    ours. Found by the first acceptance run that ever walked the
    hobbyist golden path on Linux with an NVIDIA card, which is also the
    only reason anyone looked again.
    """
    arch = host.arch or Arch.x64
    platform = "win" if host.os is Os.windows else "ubuntu"
    # The asset prefix is what we match on; the name is what a person reads.
    platform_name = "Windows" if host.os is Os.windows else "Linux"
    published = _published_cuda_variants(release, arch, platform)
    if not published:
        return Unavailable(
            reason=(
                f"release {release.version} publishes no {platform_name} CUDA build for "
                f"{arch.value}. Published variants: "
                f"{', '.join(_published_variants(release)) or '(none)'}."
            ),
            release_bound=True,
        )

    driver = host.acceleratorVersion
    if driver is None:
        # An NVIDIA card whose driver would not report a version. Refusing
        # is better than guessing: install the wrong major and the binary
        # fails at load with a message about the driver, not about us.
        return Unavailable(
            reason=(
                "an NVIDIA GPU is present but `nvidia-smi` did not report a CUDA "
                "version, so there is no safe way to choose between the published "
                "CUDA builds. Update the driver, or set `binary` on the runtime by "
                "hand."
            )
        )

    try:
        major, _, minor = driver.partition(".")
        driver_major, driver_minor = int(major), int(minor or 0)
    except ValueError:
        return Unavailable(reason=f"could not read the reported CUDA version {driver!r}.")

    offered = ", ".join(sorted({f"{mj}.{mn}" for mj, mn, _ in published}))

    # Rule 0, the card: drop every build that carries no code for it.
    capability = _parse_capability(host.computeCapability)
    runnable = [
        t for t in published if capability is None or capability >= _lowest_capability(t[0])
    ]
    if not runnable:
        assert capability is not None  # nothing is filtered without one
        cc = host.computeCapability
        if capability < _lowest_capability(12):
            # Below 5.0 nothing any release publishes will ever run, so
            # stepping back through older builds is pointless.
            return Unavailable(
                reason=(
                    f"this machine's NVIDIA card has compute capability {cc}, and no "
                    f"published llama.cpp CUDA build carries code for a card older than "
                    f"5.0. Set `binary` on the runtime to a build compiled for this card."
                ),
            )
        return Unavailable(
            reason=(
                f"this machine's NVIDIA card has compute capability {cc}, and release "
                f"{release.version} publishes {platform_name} CUDA builds only for "
                f"{offered}. CUDA 13 builds carry no code for a card below 7.5, and no "
                f"CUDA 12 build is published. Set `binary` on the runtime to a build "
                f"compiled for this card."
            ),
            release_bound=True,
        )

    same_major = [t for t in runnable if t[0] == driver_major]
    if same_major:
        fitting = [t for t in same_major if t[1] <= driver_minor]
        if fitting:
            return max(fitting, key=lambda t: (t[0], t[1]))[2]
    else:
        # Rule 3, an older major: NVIDIA's drivers are backward
        # compatible, and the cudart companion ships the build's own
        # runtime, so a 12.x build loads on a 13.x driver.
        older = [t for t in runnable if t[0] < driver_major]
        if older:
            chosen = max(older, key=lambda t: (t[0], t[1]))
            # The driver's major was published and filtered out, or never
            # published: `runnable` holds none of it either way.
            why = (
                f"this card's compute capability {host.computeCapability} is below "
                f"what CUDA {driver_major} builds carry code for"
                if any(t[0] == driver_major for t in published)
                else f"release {release.version} publishes no CUDA {driver_major} build"
            )
            log.info(
                "llama.cpp: taking the CUDA %s.%s build on a driver that supports %s, "
                "because %s. A driver runs builds from older CUDA majors.",
                chosen[0],
                chosen[1],
                driver,
                why,
            )
            return chosen[2]
        # Release-bound: a release whose 12.x asset has not been uploaded
        # yet looks, to a 12.x driver, like one that publishes 13.x only.
        return Unavailable(
            reason=(
                f"this driver supports CUDA up to {driver}, and release {release.version} "
                f"publishes {platform_name} CUDA builds only for {offered}. A CUDA build "
                f"needs a driver at least as new as its own major version. Update the "
                f"NVIDIA driver, or set `binary` on the runtime to a build you compiled."
            ),
            release_bound=True,
        )

    newer = min(same_major, key=lambda t: (t[0], t[1]))
    # Rule 2's condition. **An unknown capability keeps the behaviour this
    # had before the condition existed** (take the newer minor): with no
    # `compute_cap` there is nothing to decide on, and refusing would turn
    # every such host away for a fact we could not read. A machine with
    # several cards is decided by its LOWEST (the contract's
    # `computeCapability`), so a lowest card with finished code beside a
    # higher one without it -- a 4090 (8.9) beside an H100 (9.0) -- is not
    # seen here and still gets the newer minor.
    if capability is None or has_finished_code(capability, (newer[0], newer[1])):
        log.info(
            "llama.cpp: release %s publishes no CUDA %s build at or below this driver's "
            "%s; taking the %s.%s build under CUDA minor-version compatibility (same "
            "major, newer minor)%s. If the engine fails to start, this is the first "
            "thing to know.",
            release.version,
            driver_major,
            driver,
            newer[0],
            newer[1],
            ", which carries finished code for this card"
            if capability is not None
            else "; the card's compute capability is unknown, so whether that build "
            "carries finished code for it could not be checked",
        )
        return newer[2]

    # The card has only PTX in the newer build, and a driver cannot compile
    # PTX from a toolkit newer than itself -- the one exclusion of
    # minor-version compatibility. The older major's build runs instead:
    # its PTX is older than the driver.
    cc = host.computeCapability
    older = [t for t in runnable if t[0] < driver_major]
    if older:
        chosen = max(older, key=lambda t: (t[0], t[1]))
        log.info(
            "llama.cpp: taking the CUDA %s.%s build on a driver that supports %s: the "
            "release's CUDA %s build is %s.%s, newer than the driver, and has only PTX "
            "for this card (compute capability %s), which this driver cannot compile. "
            "A driver runs builds from older CUDA majors.",
            chosen[0],
            chosen[1],
            driver,
            driver_major,
            newer[0],
            newer[1],
            cc,
        )
        return chosen[2]
    return Unavailable(
        reason=(
            f"this machine's NVIDIA card (compute capability {cc}) has no finished code in "
            f"release {release.version}'s CUDA {newer[0]}.{newer[1]} build, only PTX, and a "
            f"driver that supports CUDA {driver} cannot compile PTX from a newer CUDA. No "
            f"older CUDA build is published to fall back on. Update the NVIDIA driver to "
            f"one that supports CUDA {newer[0]}.{newer[1]} or newer, or set `binary` on the "
            f"runtime to a build compiled for this card."
        ),
        # A release whose older-major asset is still uploading looks like
        # this too, and the one before it may carry it.
        release_bound=True,
    )


def _published_variants(release: Release) -> list[str]:
    """Every variant the release carries a server build for, sorted."""
    return sorted(
        m.group("variant") for asset in release.assets if (m := _ASSET_RE.match(asset.name))
    )


def _published_cuda_variants(
    release: Release, arch: Arch, platform: str
) -> list[tuple[int, int, str]]:
    """`(major, minor, variant)` for each CUDA server build this host could take."""
    out: set[tuple[int, int, str]] = set()
    for variant in _published_variants(release):
        m = _CUDA_VARIANT_RE.match(variant)
        if m is None or m.group("arch") != arch.value or m.group("platform") != platform:
            continue
        out.add((int(m.group("major")), int(m.group("minor")), variant))
    return sorted(out)


# The oldest card a build from each CUDA toolkit major carries code for.
# **A table, and deliberately one**: unlike upstream's asset names these
# are NVIDIA's support decisions, made once per toolkit major and never
# revised within it -- CUDA 12 dropped Kepler, CUDA 13 dropped Maxwell,
# Pascal and Volta -- and upstream's `ggml-cuda/CMakeLists.txt` mirrors
# them exactly (`50-virtual 61-virtual 70-virtual` only below 13,
# `75-virtual` upward always). Checked 2026-09-23 against that file.
_CUDA_LOWEST_CAPABILITY: dict[int, tuple[int, int]] = {12: (5, 0), 13: (7, 5)}


def _lowest_capability(major: int) -> tuple[int, int]:
    """The oldest compute capability a CUDA `major` build can run on.

    **A major newer than the table assumes the newest floor we know**,
    because a toolkit only ever drops architectures. That is a guess for
    exactly one case -- a future CUDA 14 that also drops Turing -- and
    the day upstream publishes a 14.x build is the day to re-read its
    CMakeLists and add the row. An older major has no floor we need:
    nothing older than 12 is published.
    """
    if major in _CUDA_LOWEST_CAPABILITY:
        return _CUDA_LOWEST_CAPABILITY[major]
    newest = max(_CUDA_LOWEST_CAPABILITY)
    if major > newest:
        return _CUDA_LOWEST_CAPABILITY[newest]
    return (0, 0)


# The architectures upstream's published CUDA builds carry FINISHED code
# (SASS) for, with the toolkit version that adds each. Upstream's release
# workflow passes no `CMAKE_CUDA_ARCHITECTURES` ("use the broad default arch
# set ... so the release binary covers many GPUs"), so the default list in
# `ggml/src/ggml-cuda/CMakeLists.txt` is what ships. Read at b11375
# (2026-10-03):
#
#     75-virtual 80-virtual 86-real          always
#     89-real 90-virtual                     CUDA >= 11.8
#     120a-real                              CUDA >= 12.8
#     121a-real                              CUDA >= 12.9
#     50-virtual 61-virtual 70-virtual       CUDA < 13
#
# `-virtual` is PTX, compiled by the DRIVER at load -- and a driver cannot
# compile PTX from a toolkit newer than itself, which is the one thing
# NVIDIA's minor-version compatibility excludes. So these rows are what
# decide whether a newer minor than the driver's is safe for a card.
#
# Each row: (architecture, arch-specific, toolkit that adds it). A plain
# `-real` cubin for X.Y runs on X.Z for any Z >= Y; an `a` (arch-specific)
# one runs on X.Y alone. Re-read the CMakeLists when upstream publishes a
# new CUDA major, as for `_CUDA_LOWEST_CAPABILITY` above.
_FINISHED_CODE: tuple[tuple[tuple[int, int], bool, tuple[int, int]], ...] = (
    ((8, 6), False, (0, 0)),
    ((8, 9), False, (11, 8)),
    ((12, 0), True, (12, 8)),
    ((12, 1), True, (12, 9)),
)


def has_finished_code(capability: tuple[int, int], build: tuple[int, int]) -> bool:
    """Whether a published CUDA `build` (major, minor) carries finished
    machine code for a card of `capability`, rather than only PTX."""
    for arch, specific, since in _FINISHED_CODE:
        if build < since:
            continue
        if specific:
            if capability == arch:
                return True
        elif capability[0] == arch[0] and capability[1] >= arch[1]:
            return True
    return False


def _parse_capability(value: str | None) -> tuple[int, int] | None:
    """`"6.1"` -> `(6, 1)`, compared as a tuple so `12.0` sorts above `8.6`."""
    if not value:
        return None
    major, _, minor = value.partition(".")
    try:
        return int(major), int(minor or 0)
    except ValueError:
        return None


def _match_assets(release: Release, variant: str) -> tuple[ReleaseAsset, ...] | Unavailable:
    """Find the asset(s) for a variant in a release.

    Loud on failure, by design. Upstream's asset naming is not a contract —
    `linux-` became `ubuntu-`, ROCm and OpenVINO versions are baked into
    filenames — so when nothing matches, say what was wanted and list what
    was there. Falling back to a near-miss installs the wrong
    accelerator's build and reports success.
    """
    main = next(
        (
            asset
            for asset in release.assets
            if (m := _ASSET_RE.match(asset.name)) and m.group("variant") == variant
        ),
        None,
    )
    if main is None:
        available = sorted(
            m.group("variant") for asset in release.assets if (m := _ASSET_RE.match(asset.name))
        )
        return Unavailable(
            reason=(
                f"release {release.version} has no asset for {variant!r}. "
                f"Published variants: {', '.join(available) or '(none)'}."
            ),
            release_bound=True,
        )

    assets = [main]

    # A CUDA build needs its companion runtime archive, on either platform.
    if variant.startswith(("win-cuda-", "ubuntu-cuda-")):
        cudart = next(
            (
                asset
                for asset in release.assets
                if (m := _CUDART_RE.match(asset.name)) and m.group("variant") == variant
            ),
            None,
        )
        if cudart is None:
            return Unavailable(
                reason=(
                    f"release {release.version} has the {variant!r} server build but not "
                    f"its cudart companion archive. Installing one without the other "
                    f"produces a binary that cannot start."
                ),
                release_bound=True,
            )
        assets.append(cudart)

    return tuple(assets)
