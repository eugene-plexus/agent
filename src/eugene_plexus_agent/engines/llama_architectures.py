"""Which GGUF architectures a llama.cpp build loads (library-sources-and-engines.md, LS2).

llama.cpp loads a GGUF only if its build names the file's
`general.architecture`, and the names live in upstream's
`src/llama-arch.cpp` (`{ LLM_ARCH_QWEN3, "qwen3" },`). The build's binary
cannot be asked, so the adapter declares the list as data (`accepts`) and
the Library judges against it.

Two lists, because the build that matters is the one installed, and
Eugene installs upstream's newest rather than a pinned tag, and a borrowed
or updated build can be anything:

* **The installed build's own list**, read from upstream's source at that
  build's tag the first time `/v1/engines` sees the build, in the
  background, and kept beside the engine's builds by tag. With it, an
  architecture the build does not name is a plain *no*.
* **The list Eugene ships** (`llama_cpp_architectures.json`, refreshed by
  `scripts/llama-cpp-architectures.py`), for llama.cpp not installed, and
  for an installed build until its own list is in hand (offline, GitHub
  down, a build with no tag). Upstream adds architectures and does not
  drop them, so a shipped name `runs`; any other is *may run*, because
  only the build can say.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .._http import egress_ssl_context

log = logging.getLogger(__name__)

SOURCE_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/src/llama-arch.cpp"
SHIPPED_FILE = Path(__file__).with_name("llama_cpp_architectures.json")

#: A list shorter than this is not llama-arch.cpp's (a moved file, an error
#: page): every build since 2024 names well over a hundred.
MIN_NAMES = 20
FETCH_TIMEOUT_SECONDS = 15.0
#: After a failed read, how long before the same tag is tried again.
RETRY_SECONDS = 3600.0

_NAME_RE = re.compile(r'\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}')
_TAG_RE = re.compile(r"^b?(\d+)$")


def parse(source: str) -> list[str]:
    """`{ LLM_ARCH_QWEN3, "qwen3" },` -> `qwen3`, sorted, once each.

    Leaves out `LLM_ARCH_UNKNOWN`'s `(unknown)`, which names no file."""
    return sorted({n for n in _NAME_RE.findall(source) if not n.startswith("(")})


def build_tag(version: str | None) -> str | None:
    """The upstream tag of a build: `b10948` as Eugene records it, or the
    bare `10948` a borrowed build's `--version` prints. None otherwise."""
    match = _TAG_RE.match((version or "").strip())
    return f"b{match.group(1)}" if match else None


@dataclass(frozen=True)
class BuildList:
    tag: str
    names: tuple[str, ...]


def shipped() -> BuildList:
    data = json.loads(SHIPPED_FILE.read_text(encoding="utf-8"))
    return BuildList(tag=str(data["tag"]), names=tuple(data["architectures"]))


def _fetch_source(tag: str) -> str:
    # Egress: upstream's own source on GitHub, with the user's proxy and the
    # shared SSL context, as the release list is read.
    request = urllib.request.Request(
        SOURCE_URL.format(tag=tag), headers={"User-Agent": "eugene-plexus-agent"}
    )
    with urllib.request.urlopen(
        request, timeout=FETCH_TIMEOUT_SECONDS, context=egress_ssl_context()
    ) as response:
        return str(response.read().decode("utf-8"))


class InstalledLists:
    """Each build's own list, by tag, kept on disk under `directory`.

    `get` never waits on the network: a list not kept yet is read in a
    background thread, one tag at a time, and `get` answers None until it
    lands, so `/v1/engines` stays as fast as it was.
    """

    def __init__(
        self,
        directory: Path,
        *,
        fetch: Callable[[str], str] | None = None,
        background: bool = True,
    ) -> None:
        self._dir = directory
        # Looked up when called, so a test can stand in for GitHub.
        self._fetch = fetch or (lambda tag: _fetch_source(tag))
        self._background = background
        self._lock = threading.Lock()
        self._busy: set[str] = set()
        self._failed: dict[str, float] = {}

    def path(self, tag: str) -> Path:
        return self._dir / f"{tag}.json"

    def get(self, tag: str) -> BuildList | None:
        kept = self._read(tag)
        if kept is not None:
            return kept
        with self._lock:
            failed = self._failed.get(tag)
            if tag in self._busy or (
                failed is not None and time.perf_counter() - failed < RETRY_SECONDS
            ):
                return None
            self._busy.add(tag)
        if self._background:
            threading.Thread(target=self._learn, args=(tag,), daemon=True).start()
            return None
        self._learn(tag)
        return self._read(tag)

    def _read(self, tag: str) -> BuildList | None:
        try:
            data = json.loads(self.path(tag).read_text(encoding="utf-8"))
            names = tuple(str(n) for n in data["architectures"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return BuildList(tag=tag, names=names) if len(names) >= MIN_NAMES else None

    def _learn(self, tag: str) -> None:
        try:
            names = parse(self._fetch(tag))
            if len(names) < MIN_NAMES:
                raise ValueError(f"only {len(names)} architectures in llama-arch.cpp at {tag}")
            self._dir.mkdir(parents=True, exist_ok=True)
            partial = self.path(tag).with_suffix(".partial")
            partial.write_text(
                json.dumps({"tag": tag, "architectures": names}, indent=1), encoding="utf-8"
            )
            partial.replace(self.path(tag))
            log.info("llama.cpp %s loads %d architectures", tag, len(names))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            log.warning("could not read llama.cpp %s's architecture list: %s", tag, exc)
            with self._lock:
                self._failed[tag] = time.perf_counter()
        finally:
            with self._lock:
                self._busy.discard(tag)
