"""This machine's engines as Prometheus series (row 12).

`GET /metrics` on an agent reads each running engine's own `/metrics` when
the scrape arrives, in parallel, two seconds each, and answers two kinds of
series, every one labelled `node`, `runtime`, `model` and `engine`:

* **Eugene's own** `eugene_engine_*`, one set of names for every engine, so a
  dashboard does not care whether llama.cpp or vLLM answered.
* **The engine's own**, passed through under upstream's names.

**A counter the engine lacks has no series, never a 0**: a 0 says "nothing
was drafted", and an engine that does not draft said nothing at all. An engine
that does not answer makes `eugene_engine_scrape_ok` 0 and nothing else, so a
dashboard shows the cause (the scrape failed) and not an engine at rest.

Prometheus already reports a machine that is down, so each machine serves its
own engines and the gateway gathers nothing (call MET-ENGINE, Choice 1).

No prompt or answer is in an engine's counters, and none is read.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import httpx

from .engines import speculation

#: How long one engine is given to answer a scrape. A slow engine costs its own
#: series, never the scrape.
SCRAPE_TIMEOUT_SECONDS = 2.0

log = logging.getLogger(__name__)

#: The last cause each runtime's read failed with, so a cause is logged when it
#: changes and not at every scrape.
_LAST_CAUSE: dict[tuple[str, str], str] = {}

#: The most bytes read from one engine. An engine's counters are a few tens of
#: KiB; a body past this is not read further.
MAX_BODY_BYTES = 4 * 1024 * 1024

LABELS = ("node", "runtime", "model", "engine")

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?[ \t]+(\S+)(?:[ \t]+\S+)?[ \t]*$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_META = re.compile(r"^#[ \t]+(HELP|TYPE)[ \t]+([a-zA-Z_:][a-zA-Z0-9_:]*)[ \t]*(.*)$")


@dataclass(frozen=True)
class Target:
    """One running engine to read."""

    node: str
    runtime: str
    model: str
    engine: str
    url: str


@dataclass
class Sample:
    name: str
    labels: list[tuple[str, str]]
    value: str

    def number(self) -> float | None:
        try:
            parsed = float(self.value)
        except ValueError:
            return None
        return parsed if parsed == parsed else None  # NaN is no number


@dataclass
class Parsed:
    samples: list[Sample] = field(default_factory=list)
    #: `name -> (help, type)` as the engine said them.
    meta: dict[str, dict[str, str]] = field(default_factory=dict)


def parse(text: str) -> Parsed:
    """The samples and the HELP/TYPE lines of Prometheus text. A line that is
    not one of them is dropped: an engine's odd line never fails a scrape."""
    out = Parsed()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            found = _META.match(line)
            if found:
                kind, name, rest = found.groups()
                out.meta.setdefault(name, {})[kind] = rest
            continue
        found = _SAMPLE.match(line)
        if not found:
            continue
        name, labels, value = found.groups()
        pairs = _LABEL.findall(labels) if labels else []
        try:
            float(value)
        except ValueError:
            continue
        out.samples.append(Sample(name, list(pairs), value))
    return out


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(pairs: Iterable[tuple[str, str]]) -> str:
    return ",".join(f'{key}="{_escape(value)}"' for key, value in pairs)


# --- the names, per engine ---------------------------------------------------


@dataclass(frozen=True)
class _Series:
    name: str
    kind: str
    help: str
    #: How samples of one upstream name with different labels are joined.
    join: Callable[[Sequence[float]], float]


_SERIES = {
    "requests_waiting": _Series(
        "eugene_engine_requests_waiting", "gauge", "Requests waiting in the engine.", sum
    ),
    "requests_running": _Series(
        "eugene_engine_requests_running", "gauge", "Requests the engine is running.", sum
    ),
    "kv_cache_usage_ratio": _Series(
        "eugene_engine_kv_cache_usage_ratio",
        "gauge",
        "Share of the engine's KV cache in use, 0 to 1.",
        max,
    ),
    "prompt_tokens_total": _Series(
        "eugene_engine_prompt_tokens_total", "counter", "Prompt tokens the engine read.", sum
    ),
    "generated_tokens_total": _Series(
        "eugene_engine_generated_tokens_total", "counter", "Tokens the engine generated.", sum
    ),
    "draft_tokens_total": _Series(
        "eugene_engine_draft_tokens_total",
        "counter",
        "Tokens drafted by speculative decoding.",
        sum,
    ),
    "draft_accepted_tokens_total": _Series(
        "eugene_engine_draft_accepted_tokens_total",
        "counter",
        "Drafted tokens the engine accepted.",
        sum,
    ),
}

#: Each engine's own name for what `_SERIES` calls one thing. Read from
#: llama-server's `/metrics` (`--metrics`) and vLLM's default endpoint. An
#: engine that is not listed here has no mapping, and is not scraped.
_UPSTREAM: dict[str, dict[str, str]] = {
    "llama_cpp": {
        "requests_waiting": "llamacpp:requests_deferred",
        "requests_running": "llamacpp:requests_processing",
        "kv_cache_usage_ratio": "llamacpp:kv_cache_usage_ratio",
        "prompt_tokens_total": "llamacpp:prompt_tokens_total",
        "generated_tokens_total": "llamacpp:tokens_predicted_total",
        "draft_tokens_total": speculation.DRAFTED_METRIC,
        "draft_accepted_tokens_total": speculation.ACCEPTED_METRIC,
    },
    "vllm": {
        "requests_waiting": "vllm:num_requests_waiting",
        "requests_running": "vllm:num_requests_running",
        "kv_cache_usage_ratio": "vllm:kv_cache_usage_perc",
        "prompt_tokens_total": "vllm:prompt_tokens_total",
        "generated_tokens_total": "vllm:generation_tokens_total",
        "draft_tokens_total": speculation.VLLM_DRAFTED_METRIC,
        "draft_accepted_tokens_total": speculation.VLLM_ACCEPTED_METRIC,
    },
}


def scrapes(engine: str) -> bool:
    """Whether an engine of this kind has a metrics mapping here."""
    return engine in _UPSTREAM


@dataclass
class _Read:
    target: Target
    parsed: Parsed | None
    #: Why the read failed, for the log and never for a label.
    error: str | None = None


async def _read(client: httpx.AsyncClient, target: Target, seconds: float) -> _Read:
    try:
        async with asyncio.timeout(seconds):
            response = await client.get(f"{target.url.rstrip('/')}/metrics", timeout=seconds)
            if not response.is_success:
                return _Read(target, None, f"HTTP {response.status_code}")
            body = response.text[:MAX_BODY_BYTES]
    except (httpx.HTTPError, TimeoutError) as exc:
        return _Read(target, None, type(exc).__name__)
    return _Read(target, parse(body))


async def gather(
    client: httpx.AsyncClient, targets: Sequence[Target], *, seconds: float = SCRAPE_TIMEOUT_SECONDS
) -> list[_Read]:
    """Every target's `/metrics`, in parallel, `seconds` seconds each."""
    reads = list(await asyncio.gather(*(_read(client, t, seconds) for t in targets)))
    note_causes(reads)
    return reads


def note_causes(reads: Sequence[_Read]) -> None:
    """Say why an engine's `/metrics` could not be read, once for each change:
    `eugene_engine_scrape_ok 0` is the symptom, and this is the observed cause."""
    for read in reads:
        who = (read.target.node, read.target.runtime)
        if read.error is None:
            if _LAST_CAUSE.pop(who, None) is not None:
                log.info("engine %r: /metrics answers again", read.target.runtime)
        elif _LAST_CAUSE.get(who) != read.error:
            _LAST_CAUSE[who] = read.error
            log.warning(
                "engine %r on %s: /metrics could not be read (%s) at %s",
                read.target.runtime,
                read.target.node,
                read.error,
                read.target.url,
            )


def render(reads: Sequence[_Read]) -> str:
    """Prometheus text for what was read: Eugene's series first, then each
    engine's own under its own names."""
    lines: list[str] = []

    own: dict[str, list[str]] = {}
    for key in _SERIES:
        own[key] = []
    scrape_ok: list[str] = []
    passthrough: dict[str, list[str]] = {}
    meta: dict[str, dict[str, str]] = {}

    for read in reads:
        target = read.target
        identity = [
            ("node", target.node),
            ("runtime", target.runtime),
            ("model", target.model),
            ("engine", target.engine),
        ]
        labels = _render_labels(identity)
        scrape_ok.append(f"eugene_engine_scrape_ok{{{labels}}} {1 if read.parsed else 0}")
        if read.parsed is None:
            continue
        names = _UPSTREAM.get(target.engine, {})
        for key, upstream in names.items():
            values = [
                v
                for s in read.parsed.samples
                if s.name == upstream and (v := s.number()) is not None and math.isfinite(v)
            ]
            if not values:
                continue
            joined = _SERIES[key].join(values)
            own[key].append(f"{_SERIES[key].name}{{{labels}}} {_number(joined)}")
        taken = {key for key, _ in identity}
        for sample in read.parsed.samples:
            kept = [(f"exported_{k}" if k in taken else k, v) for k, v in sample.labels] + identity
            passthrough.setdefault(sample.name, []).append(
                f"{sample.name}{{{_render_labels(kept)}}} {sample.value}"
            )
            if sample.name in read.parsed.meta:
                meta.setdefault(sample.name, read.parsed.meta[sample.name])

    for key, series in _SERIES.items():
        if not own[key]:
            continue
        lines.append(f"# HELP {series.name} {series.help}")
        lines.append(f"# TYPE {series.name} {series.kind}")
        lines.extend(own[key])
    lines.append("# HELP eugene_engine_scrape_ok 1 when the engine answered its own /metrics.")
    lines.append("# TYPE eugene_engine_scrape_ok gauge")
    lines.extend(scrape_ok)
    for name, samples in passthrough.items():
        said = meta.get(name, {})
        if "HELP" in said:
            lines.append(f"# HELP {name} {said['HELP']}")
        if "TYPE" in said:
            lines.append(f"# TYPE {name} {said['TYPE']}")
        lines.extend(samples)
    return "\n".join(lines) + "\n"


def _number(value: float) -> str:
    return str(int(value)) if value == int(value) and abs(value) < 1e15 else repr(value)
