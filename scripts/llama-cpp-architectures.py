"""Refresh the llama.cpp architecture list the agent ships (LS2).

    python scripts/llama-cpp-architectures.py b11530

Reads upstream's `src/llama-arch.cpp` at that tag and rewrites
`src/eugene_plexus_agent/engines/llama_cpp_architectures.json`. The list
is what the agent declares for llama.cpp when it is not installed, and
for an installed build until that build's own list has been read
(`engines/llama_architectures.py`). Take a recent build's tag: upstream
adds architectures and does not drop them.
"""

from __future__ import annotations

import json
import sys

from eugene_plexus_agent.engines import llama_architectures


def main(tag: str) -> None:
    resolved = llama_architectures.build_tag(tag)
    if resolved is None:
        raise SystemExit(f"not a llama.cpp build tag: {tag!r} (expected e.g. b11530)")
    names = llama_architectures.parse(llama_architectures._fetch_source(resolved))
    if len(names) < llama_architectures.MIN_NAMES:
        raise SystemExit(f"only {len(names)} architectures read at {resolved}; not writing")
    llama_architectures.SHIPPED_FILE.write_text(
        json.dumps({"tag": resolved, "architectures": names}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"{resolved}: {len(names)} architectures -> {llama_architectures.SHIPPED_FILE}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
