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
    Os,
    RuntimeCapabilities,
    RuntimeSpec,
)
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

# Windows CUDA is a two-asset install: the server zip carries no CUDA
# runtime, and the DLLs live in a companion archive whose filename has no
# build number even though it sits under the same release tag. Installing
# only the first produces a binary that dies on a missing cudart DLL.
_CUDART_RE = re.compile(r"^cudart-llama-bin-(?P<variant>.+)\.(?:zip|tar\.gz)$")

# `win-cuda-13.3-x64` -> ('13', '3'). Used to pick the highest published
# CUDA build the installed driver can actually load.
_CUDA_VARIANT_RE = re.compile(r"^win-cuda-(?P<major>\d+)\.(?P<minor>\d+)-(?P<arch>x64|arm64)$")

# How many builds back `plan_latest` looks for one that carries this host's
# assets. Upstream publishes several a day and an incomplete upload lasts
# under an hour, so eight is about a day; a host that finds nothing in a
# day is told the newest build's own reason.
FALLBACK_BUILDS = 8


class LlamaCppAdapter(EngineAdapter):
    kind = EngineKind.llama_cpp
    binary_name = "llama-server"

    # GGUF only. A safetensors model in the library runs on vLLM
    # instead; the UI joins the two lists to decide which engine a
    # launch button offers.
    model_formats = (ModelFormat.gguf,)

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

    def plan_latest(
        self, host: HostAccelerator, *, force: bool = False
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
            return Unavailable(
                reason=(
                    "could not reach the upstream release list. Check network access, or "
                    "set `binary` on the runtime to a build you already have."
                )
            )
        first: Unavailable | None = None
        for release in builds[:FALLBACK_BUILDS]:
            plan = self.plan_acquisition(host, release)
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
        self, host: HostAccelerator, release: Release
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
        """
        variant = _variant_for(host, release)
        if isinstance(variant, Unavailable):
            return variant

        assets = _match_assets(release, variant)
        if isinstance(assets, Unavailable):
            return assets

        return AcquisitionPlan(
            version=release.version,
            variant=variant,
            assets=assets,
            binary_name=self.binary_name,
        )

    # --- launching --------------------------------------------------------

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
            if field.key not in flags:
                continue
            value = flags[field.key]
            if value is None:
                continue
            cli = _FLAG_CLI_NAMES[field.key]
            if field.valueType == ConfigValueType.boolean:
                # llama-server's booleans are presence-only switches;
                # passing `--flash-attn false` is not a thing.
                if value:
                    argv.append(cli)
            else:
                argv += [cli, str(value)]

        # Verbatim, last, so an operator can always override something the
        # curated surface generated above.
        if spec.extraArgs:
            argv += list(spec.extraArgs)
        return argv

    # --- observing --------------------------------------------------------

    async def probe_readiness(self, base_url: str) -> Readiness:
        """Read `llama-server`'s `/health`.

        Upstream's contract: 503 with `{"status": "loading model"}` while
        the weights are being read, 200 with `{"status": "ok"}` once it
        can serve. Distinguishing those is the whole reason readiness is
        per-adapter rather than a generic TCP connect.
        """
        url = base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{url}/health")
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
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{url}/props")
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
        )

    # --- configuring ------------------------------------------------------

    def flag_schema(self) -> ConfigSchema:
        return ConfigSchema(
            component="engine:llama_cpp",
            categories=_CATEGORIES,
            fields=_FLAG_FIELDS,
        )


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


_CATEGORIES = {
    "memory": "Memory and context",
    "performance": "Performance",
    "sampling": "Serving",
}

# Curated flags -> llama-server CLI names. Kept as a separate mapping from
# the schema so the UI-facing key never has to look like a CLI flag, and
# so an upstream rename touches one line.
_FLAG_CLI_NAMES: dict[str, str] = {
    "contextSize": "--ctx-size",
    "gpuLayers": "--n-gpu-layers",
    "batchSize": "--batch-size",
    "ubatchSize": "--ubatch-size",
    "threads": "--threads",
    "parallelSlots": "--parallel",
    "mainGpu": "--main-gpu",
    "tensorSplit": "--tensor-split",
    "flashAttention": "--flash-attn",
    "mlock": "--mlock",
    "noMmap": "--no-mmap",
    "continuousBatching": "--cont-batching",
}

_FLAG_FIELDS: list[ConfigField] = [
    ConfigField(
        key="contextSize",
        label="Context size",
        description=(
            "Maximum tokens the model can attend to, per slot. The single "
            "biggest lever on memory use. Leave unset to take the model's "
            "trained context — the engine clamps anything larger than the "
            "model or than available memory, and the clamped value is what "
            "the runtime reports back."
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
            "Concurrent requests this runtime serves. Context size is "
            "per-slot, so 4 slots at 8k needs roughly the memory of 1 slot "
            "at 32k — raising this without lowering context is the usual "
            "way to run out of VRAM. This is also the unit of capacity the "
            "gateway divides work across."
        ),
        category="performance",
        valueType=ConfigValueType.integer,
        default=1,
        minimum=1,
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
            "Enable the fused attention kernel. Usually faster and lighter "
            "on memory where the build and hardware support it; harmless to "
            "leave off if you are unsure."
        ),
        category="performance",
        valueType=ConfigValueType.boolean,
        default=False,
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
            "Slower to start and needs more RAM; occasionally necessary on "
            "network filesystems."
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
            return _cuda_variant(host, release)
        if accelerator is Accelerator.rocm:
            return f"win-rocm-10.0-{host.arch.value}"
        return f"win-cpu-{host.arch.value}"

    # Linux.
    if accelerator is Accelerator.cuda:
        return Unavailable(
            reason=(
                "llama.cpp publishes no prebuilt CUDA build for Linux, so there is "
                "nothing to install for an NVIDIA GPU here. Build llama.cpp from "
                "source with GGML_CUDA=ON, or use the official CUDA container, then "
                "set `binary` on the runtime to the llama-server you built. A Vulkan "
                "build would install cleanly and run on this card, but it is "
                "materially slower at prompt processing and we will not substitute "
                "it for CUDA without being asked."
            )
        )
    if accelerator is Accelerator.rocm:
        return f"ubuntu-rocm-10.0-{host.arch.value}"
    if accelerator is Accelerator.sycl:
        return "ubuntu-sycl-fp16-x64"
    return f"ubuntu-{host.arch.value}" if host.arch is Arch.arm64 else "ubuntu-x64"


def _cuda_variant(host: HostAccelerator, release: Release) -> str | Unavailable:
    """The Windows CUDA build this driver can load, from what the release publishes.

    Two rules, in order, both NVIDIA's rather than ours:

    1. **Prefer a build for the driver's own CUDA minor or an older one**
       within the same major -- the highest such minor. A 12.4 build on a
       12.8 driver is the ordinary case and needs no caveat.
    2. **Otherwise take the lowest published minor above the driver's,
       still within the same major.** CUDA's minor-version compatibility
       guarantees that an application built with any 13.x toolkit runs
       on any driver of the 13.x family, minus PTX JIT for newer PTX and
       minus APIs the older driver lacks; llama.cpp ships SASS for the
       shipping architectures, so neither bites. **Verified live
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

    **A different major is never crossed**, either way: a 13.x build
    needs a 13.x driver, and the refusal says so and names the fix.

    The candidate minors come from **the release's asset names**, not a
    table. The table this replaced was written from b10867 and went stale
    the moment upstream moved from 13.3 to 13.4; a stale table cannot
    fail loudly when its defect is an entry it lacks.
    """
    arch = host.arch or Arch.x64
    published = _published_cuda_variants(release, arch)
    if not published:
        return Unavailable(
            reason=(
                f"release {release.version} publishes no Windows CUDA build for "
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

    same_major = [t for t in published if t[0] == driver_major]
    if not same_major:
        offered = ", ".join(sorted({f"{mj}.{mn}" for mj, mn, _ in published}))
        # Release-bound: a release whose 12.x asset has not been uploaded
        # yet looks, to a 12.x driver, like one that publishes 13.x only.
        return Unavailable(
            reason=(
                f"this driver supports CUDA up to {driver}, and release {release.version} "
                f"publishes Windows CUDA builds only for {offered}. A CUDA build runs "
                f"only on a driver of the same major version. Update the NVIDIA "
                f"driver, or set `binary` on the runtime to a build you compiled."
            ),
            release_bound=True,
        )

    fitting = [t for t in same_major if t[1] <= driver_minor]
    if fitting:
        return max(fitting, key=lambda t: (t[0], t[1]))[2]

    newer = min(same_major, key=lambda t: (t[0], t[1]))
    log.info(
        "llama.cpp: release %s publishes no CUDA %s build at or below this driver's "
        "%s; taking the %s.%s build under CUDA minor-version compatibility (same "
        "major, newer minor). If the engine fails to start, this is the first "
        "thing to know.",
        release.version,
        driver_major,
        driver,
        newer[0],
        newer[1],
    )
    return newer[2]


def _published_variants(release: Release) -> list[str]:
    """Every variant the release carries a server build for, sorted."""
    return sorted(
        m.group("variant") for asset in release.assets if (m := _ASSET_RE.match(asset.name))
    )


def _published_cuda_variants(release: Release, arch: Arch) -> list[tuple[int, int, str]]:
    """`(major, minor, variant)` for each Windows CUDA server build in the release."""
    out: set[tuple[int, int, str]] = set()
    for variant in _published_variants(release):
        m = _CUDA_VARIANT_RE.match(variant)
        if m is None or m.group("arch") != arch.value:
            continue
        out.add((int(m.group("major")), int(m.group("minor")), variant))
    return sorted(out)


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

    # Windows CUDA needs the companion runtime archive.
    if variant.startswith("win-cuda-"):
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
