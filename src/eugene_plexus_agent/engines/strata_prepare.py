"""Strata prepares a GGUF on its own list: upstream's setup, run for a person (LS5).

library-sources-and-engines.md §6.6 (calls B43-B56). Read off `setup.py` at
the commit the adapter pins: with `--gguf-dir` and `--data-dir` it checks
the PC, installs its Python packages into the interpreter running it, takes
the engine already in its own `engine/` folder, builds the pack and
tokenizer from the GGUF, fetches the MTP draft layer (~5 GB, once per data
folder) and writes `strata-<tag>.json` and a start script into its own
folder. With `--yes` it asks nothing and takes its own recommendation for
every question; Eugene reimplements none of them (B45).

What this adds around it: a check that the install brought setup's
preparation tools (B47 reversed in LS7: they come with the engine), a fresh settings
folder per run so setup never moves a person's own Strata files (B45), the
engine-files marker (B44), and afterwards the configuration moved into the
data folder with its paths relative (B48), checked the way a launch checks
it, before the library lists it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .._generated.models import SupportedModel
from ..child_env import child_environment
from ..preparation import (
    PreparationError,
    PreparationResult,
    Progress,
    folder_bytes,
    run_process,
)
from ..prepared_facts import facts_fields
from ..supervisor import SpawnPlanError
from .base import PreparedInspectError
from .strata import (
    OWNED_KEYS,
    PATH_ARGS,
    SETUP_KEYS,
    VERSION,
    inspect_config,
    prepared_config,
    runtime_lib_dirs,
)
from .strata_install import LLAMA_CPP_COMMIT, tools_installed
from .strata_models import SETUP_CONTEXTS, SetupChoice, choice_for_file, disk_needed

RECIPE = "strata-prepare"
#: Strata's data folder at the top of the Library folder (B43): setup's own
#: default name for it, `Strata-data`.
DATA_FOLDER = "Strata-data"
#: The library's marker for an engine's own folder (B44). The library scans
#: for the same name (`prepared.ENGINE_FILES_MARKER`).
ENGINE_FILES_MARKER = ".eugene-engine-files"
MARKER_TEXT = (
    "Strata's own files, made by Eugene Plexus with Strata's setup.\n"
    "The Library lists the *.eugene-prepared.json files in this folder and reads nothing else\n"
    "here. Deleting a model's .eugene-prepared.json removes it from the Library.\n"
)

#: setup.py `CONTEXTS`: the sizes it offers.
CONTEXTS = SETUP_CONTEXTS

_STEP = re.compile(r"^=== Step (\d+): (.+?) ===$")
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def shards_of(first: Path) -> list[Path]:
    """Every shard of a split GGUF, named from its first, as setup and the
    engine find them; the file alone when it is not split."""
    found = _SHARD.search(first.name)
    if not found:
        return [first]
    total = int(found.group(2))
    stem = first.name[: found.start()]
    return [first.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


def free_bytes(path: Path) -> int | None:
    """Free space on the drive `path` will be on (its nearest existing folder)."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            try:
                return shutil.disk_usage(candidate).free
            except OSError:
                return None
    return None


def main_gpu() -> int | None:
    """On a node with several NVIDIA cards, the one setup would take alone:
    the most memory, then the lowest number, as `nvidia-smi` numbers them
    (B49). None with one card, or when `nvidia-smi` cannot say."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    cards: list[tuple[int, int]] = []
    for line in done.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            cards.append((int(parts[0]), int(parts[1])))
    if len(cards) < 2:
        return None
    # setup.py `choose_gpus`: (-round(vram_gb), index).
    return min(cards, key=lambda c: (-round(c[1] / 1024), c[0]))[0]


def plan(
    *,
    root: Path,
    gguf: Path,
    data_dir: Path,
    source_path: str,
    context: int | None,
    ram_bytes: int | None,
    gpu: int | None = None,
) -> StrataPreparation:
    """Check what can be checked before anything starts (B51), or say why not.

    `root` is the Strata build's source folder (`setup.py` and `serve/`),
    `gguf` the first shard as this node reaches it, `data_dir` the Strata
    data folder as this node reaches it, `source_path` the GGUF as the
    library spells it."""
    found = choice_for_file(str(gguf))
    if found is None:
        raise PreparationError(
            f"Strata prepares only the files on its own list, and {gguf.name} is not one of "
            "them (its setup refuses any other GGUF)."
        )
    supported, choice = found
    if context is not None and context not in CONTEXTS:
        raise PreparationError(
            f"Strata's setup prepares a context of {', '.join(str(c) for c in CONTEXTS)} "
            f"tokens, not {context}."
        )
    setup = root / "setup.py"
    python = venv_python(root)
    if not setup.is_file() or not python.is_file():
        raise PreparationError(
            f"This Strata install has no setup or Python environment ({setup}); reinstall Strata."
        )
    missing = [s.name for s in shards_of(gguf) if not s.is_file()]
    if missing:
        raise PreparationError(
            f"Strata needs every file of {gguf.name} beside it in {gguf.parent}; missing: "
            + ", ".join(missing)
        )
    need = disk_needed(choice, ram_bytes)
    free = free_bytes(data_dir)
    if free is not None and free < need:
        raise PreparationError(
            f"Strata's setup needs about {need / 1e9:.0f} GB free beside the model, by its own "
            f"rule, and {data_dir.anchor or data_dir} has {free / 1e9:.0f} GB free. Make room, "
            "then prepare it again."
        )
    return StrataPreparation(
        root=root,
        gguf=gguf,
        data_dir=data_dir,
        choice=choice,
        supported=supported,
        context=context,
        gpu=gpu,
        source_path=source_path,
        bytes_needed=need,
    )


def venv_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def check_tools(root: Path) -> None:
    """The preparation tools came with this install (LS7, B47 reversed), or
    the preparation stops naming the fix: no fetch halfway through."""
    if not tools_installed(root):
        raise PreparationError(
            "Strata on this node was installed before its preparation tools came with it "
            f"(llama.cpp's gguf-py at {LLAMA_CPP_COMMIT[:7]} and setup's packages). Reinstall "
            "Strata from Backends: uninstall it, then install it again; prepared models in "
            "your Library folders are kept."
        )


@dataclass
class StrataPreparation:
    """One run of upstream's setup for one choice on its list."""

    root: Path
    gguf: Path
    data_dir: Path
    choice: SetupChoice
    supported: SupportedModel
    context: int | None
    gpu: int | None
    source_path: str
    bytes_needed: int | None
    _failure: list[str] = field(default_factory=list)
    _pending: str = ""

    @property
    def output(self) -> Path:
        return self.data_dir

    @property
    def config_name(self) -> str:
        return f"strata-{self.choice.tag}.json"

    def argv(self) -> list[str]:
        out = [
            str(venv_python(self.root)),
            str(self.root / "setup.py"),
            "--family",
            self.choice.family,
            "--model",
            self.choice.model,
            "--gguf-dir",
            str(self.gguf.parent),
            "--data-dir",
            str(self.data_dir),
            "--vision",
            "no",
            "--experimental-speed-projection",
            "off",
            "--no-browser",
            "--no-start",
            "--yes",
        ]
        if self.context is not None:
            out += ["--context", str(self.context)]
        if self.gpu is not None:
            out += ["--gpu", str(self.gpu)]
        return out

    def environment(self, settings: Path) -> dict[str, str]:
        """Setup's settings file in a folder of its own for this run (B45);
        its output as UTF-8, unbuffered; pip without the agent's config."""
        env = child_environment()
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_REQUIRE_VIRTUALENV"):
            env.pop(name, None)
        env.update(
            {
                "APPDATA": str(settings),
                "XDG_CONFIG_HOME": str(settings),
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INPUT": "1",
                "PATH": os.pathsep.join([*runtime_lib_dirs(self.root), env.get("PATH", "")]),
            }
        )
        return env

    def _produced(self) -> list[Path]:
        """What setup writes into the engine's own folder for this choice."""
        tag = self.choice.tag
        config = self.root / self.config_name
        return [
            config,
            config.with_name(config.name + ".bak"),
            config.with_name(config.name + ".tmp"),
            self.root / f"run-{tag}.bat",
            self.root / f"run-{tag}.sh",
        ]

    def run(self, progress: Progress) -> PreparationResult:
        check_tools(self.root)
        progress.check_cancelled()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        marker = self.data_dir / ENGINE_FILES_MARKER
        if not marker.is_file():
            marker.write_text(MARKER_TEXT, encoding="utf-8", newline="\n")
        for leftover in self._produced():
            leftover.unlink(missing_ok=True)
        baseline = folder_bytes(self.data_dir)

        def measure() -> None:
            progress.bytes_written = max(0, folder_bytes(self.data_dir) - baseline)

        progress.step = "Starting Strata's setup"
        with tempfile.TemporaryDirectory(
            prefix="eugene-strata-setup-", ignore_cleanup_errors=True
        ) as settings:
            code = run_process(
                self.argv(),
                cwd=self.root,
                env=self.environment(Path(settings)),
                log_path=self.data_dir / f"strata-{self.choice.tag}.setup.log",
                progress=progress,
                on_output=lambda text: self._read(text, progress),
                measure=measure,
            )
        self._read("\n", progress)
        if code != 0:
            raise PreparationError(self._stopped(code, progress))
        progress.step = "Making it a Library model"
        progress.message = None
        entry = self._adopt_config()
        try:
            facts = facts_fields(inspect_config(entry), folder=entry.parent)
        except PreparedInspectError as exc:
            raise PreparationError(f"Strata's setup left the model incomplete: {exc}") from exc
        return PreparationResult(
            entry=entry,
            name=self.choice.model_name,
            recipe=RECIPE,
            recipe_version=VERSION,
            source={
                "path": self.source_path,
                "repoId": self.supported.source.repoId,
                "file": self.supported.source.file,
                "revision": self.supported.source.revision,
            },
            facts=facts,
        )

    # --- setup's output ---------------------------------------------------

    def _read(self, text: str, progress: Progress) -> None:
        """Setup's lines: its steps in its own words, its warnings (`[!]`),
        its failure (`[X]` and the hint under it), and its last line (B56)."""
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        for raw in lines:
            # A Windows line ends "\r\n"; a progress bar redraws after a bare "\r".
            line = raw.rstrip("\r").split("\r")[-1].rstrip()
            text_only = line.strip()
            if not text_only:
                continue
            step = _STEP.match(text_only)
            if step:
                progress.step = f"Step {step.group(1)} of 7: {step.group(2)}"
                continue
            if text_only.startswith("[!]"):
                progress.warn(text_only[3:].strip())
            elif text_only.startswith("[X]"):
                self._failure = [text_only[3:].strip()]
                continue
            elif self._failure and len(self._failure) == 1 and line.startswith("       "):
                self._failure.append(text_only)
                continue
            progress.message = text_only[:300]

    def _stopped(self, code: int, progress: Progress) -> str:
        if self._failure:
            return "Strata's setup stopped: " + " ".join(self._failure)
        last = progress.message or "it printed nothing"
        log = self.data_dir / f"strata-{self.choice.tag}.setup.log"
        return f"Strata's setup exited {code}: {last} (its whole output: {log})"

    # --- afterwards -------------------------------------------------------

    def _relative(self, value: str) -> str:
        """A path setup wrote, relative to the data folder when it can be (on
        the same drive), so the folder travels (B43); as written otherwise."""
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        try:
            return Path(os.path.relpath(path, self.data_dir)).as_posix()
        except ValueError:
            return str(path)

    def _adopt_config(self) -> Path:
        """Setup's configuration, moved into the data folder (B48), and
        checked the way a launch checks it before the library lists it."""
        produced = self.root / self.config_name
        try:
            cfg = json.loads(produced.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            raise PreparationError(
                f"Strata's setup finished without writing {self.config_name}: {exc}"
            ) from exc
        except ValueError as exc:
            raise PreparationError(
                f"Strata's setup wrote {self.config_name} unreadably: {exc}"
            ) from exc
        if not isinstance(cfg, dict) or not isinstance(cfg.get("args"), list):
            raise PreparationError(f"Strata's setup wrote {self.config_name} without its args")
        args = [str(a) for a in cfg["args"]]
        pack = self.data_dir / "packs" / self.choice.tag
        for i in range(len(args) - 1):
            if args[i] == "--expert-profile":
                profile = Path(args[i + 1])
                if profile.is_file() and profile.is_relative_to(self.root):
                    # In the engine's own folder: copied, so a reinstall keeps it.
                    pack.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(profile, pack / profile.name)
                    args[i + 1] = str(pack / profile.name)
        for i in range(len(args) - 1):
            if args[i] in PATH_ARGS:
                args[i + 1] = self._relative(args[i + 1])
        cfg["args"] = args
        if isinstance(cfg.get("tokenizer"), str):
            cfg["tokenizer"] = self._relative(cfg["tokenizer"])
        # Node-neutral (Troy, LS7): this file lives in the Library and any
        # node may launch it, so nothing of the node that prepared it stays.
        # A launch writes its own exe, log, libraries, port and name.
        for key in (*OWNED_KEYS, *SETUP_KEYS, "open_browser"):
            cfg.pop(key, None)
        target = self.data_dir / self.config_name
        partial = target.with_name(target.name + ".partial")
        partial.write_text(json.dumps(cfg, indent=1), encoding="utf-8", newline="\n")
        os.replace(partial, target)
        for leftover in self._produced():
            leftover.unlink(missing_ok=True)
        try:
            prepared_config(target, alias=self.choice.model_name, root=self.root)
        except SpawnPlanError as exc:
            raise PreparationError(
                f"Strata's setup wrote a configuration Eugene cannot launch: {exc}"
            ) from exc
        return target
