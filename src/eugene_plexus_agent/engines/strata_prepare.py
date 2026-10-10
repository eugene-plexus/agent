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

Where it runs (LS10, §6.13): in a folder of this node's that stands for the
Library folder, its layout the same (B101), so setup writes nothing in the
Library folder and the configuration's relative paths are the Library's. The
GGUF stays where it is: its shards appear there as links (B102), which setup
reads and leaves its `.done` marks beside. What the model is made of is then
sent to the library, which writes it into the Library folder itself.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
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
    files_of,
    inspect_config,
    named_paths,
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
#: A copy of a shard, when it can be neither linked nor used in place.
_COPY_CHUNK = 16 * 1024 * 1024


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


def _under(path: Path, folder: Path) -> bool:
    """`path` at or below `folder`, compared as the OS compares names."""
    try:
        inside = os.path.relpath(path, folder)
    except ValueError:  # another drive
        return False
    return not (inside == ".." or inside.startswith(".." + os.sep) or os.path.isabs(inside))


def _place(make: Callable[[Path, Path], None], source: Path, link: Path) -> None:
    """`link` stands for `source` (B102): kept when it already does (a link
    to it, or a whole copy of it from an earlier preparation), made by
    `make` otherwise."""
    if link.is_symlink():
        try:
            if os.path.samefile(link, source):
                return
        except OSError:
            pass
        link.unlink()
    elif link.is_file():
        if link.stat().st_size == source.stat().st_size:
            return
        link.unlink()
    make(source, link)


def _symlink(source: Path, link: Path) -> None:
    os.symlink(source, link)


def _hardlink(source: Path, link: Path) -> None:
    os.link(source, link)


def _junction(view: Path, folder: Path) -> bool:
    """On Windows, `view` made a junction to the local folder `folder`, which
    needs no privilege a symlink needs (never to a share: Windows refuses)."""
    if sys.platform != "win32" or str(folder).startswith("\\\\"):
        return False
    import _winapi

    if os.path.isjunction(view):
        if os.path.samefile(view, folder):
            return True
        os.rmdir(view)
    try:
        if view.is_dir():
            view.rmdir()  # empty, or not a candidate
        _winapi.CreateJunction(str(folder), str(view))
    except OSError:
        return False
    return True


def _writable(folder: Path) -> bool:
    """Whether this node can make a file in `folder` (setup's marks)."""
    probe = folder / f".eugene-write-check-{uuid.uuid4().hex}"
    try:
        with open(probe, "x"):
            pass
    except OSError:
        return False
    probe.unlink(missing_ok=True)
    return True


def _copy(source: Path, target: Path, progress: Progress) -> None:
    """A shard copied whole, or not under its own name at all."""
    size = source.stat().st_size
    if target.is_file() and not target.is_symlink() and target.stat().st_size == size:
        progress.bytes_written = (progress.bytes_written or 0) + size
        return
    partial = target.with_name(target.name + ".partial")
    with open(source, "rb") as reading, open(partial, "wb") as writing:
        while block := reading.read(_COPY_CHUNK):
            progress.check_cancelled()
            writing.write(block)
            progress.bytes_written = (progress.bytes_written or 0) + len(block)
    target.unlink(missing_ok=True)
    os.replace(partial, target)


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
    folder: Path,
    work: Path,
    source_path: str,
    context: int | None,
    ram_bytes: int | None,
    gpu: int | None = None,
) -> StrataPreparation:
    """Check what can be checked before anything starts (B51), or say why not.

    `root` is the Strata build's source folder (`setup.py` and `serve/`),
    `gguf` the first shard and `folder` the Library folder holding it, both
    as this node reaches them, `work` this node's folder standing for that
    Library folder (LS10), `source_path` the GGUF as the library spells it."""
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
    try:
        inside = Path(os.path.relpath(gguf.parent, folder))
    except ValueError:  # another drive
        inside = Path("..")
    if inside.parts[:1] == ("..",) or inside.is_absolute():
        raise PreparationError(f"{gguf} is not inside the Library folder {folder}.")
    need = disk_needed(choice, ram_bytes)
    free = free_bytes(work)
    if free is not None and free < need:
        raise PreparationError(
            f"Strata's setup needs about {need / 1e9:.0f} GB free on this node, by its own "
            f"rule, where it prepares the model ({work.anchor or work}), and that drive has "
            f"{free / 1e9:.0f} GB free. Make room, then prepare it again."
        )
    return StrataPreparation(
        root=root,
        gguf=gguf,
        folder=folder,
        work=work,
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
    #: The Library folder holding the GGUF, as this node reaches it.
    folder: Path
    #: This node's folder standing for it (LS10, B101).
    work: Path
    choice: SetupChoice
    supported: SupportedModel
    context: int | None
    gpu: int | None
    source_path: str
    bytes_needed: int | None
    _failure: list[str] = field(default_factory=list)
    _pending: str = ""
    #: The folder setup is given as the GGUF's (B102).
    _gguf_dir: Path | None = None
    #: Shards copied because no link could be made: removed after listing.
    _copies: list[Path] = field(default_factory=list)

    @property
    def data_dir(self) -> Path:
        """`Strata-data`, in this node's folder (B43, B101)."""
        return self.work / DATA_FOLDER

    @property
    def view(self) -> Path:
        """Where the GGUF's shards appear in this node's folder: their place
        in the Library folder."""
        return self.work / os.path.relpath(self.gguf.parent, self.folder)

    @property
    def output(self) -> Path:
        return self.data_dir

    @property
    def log_path(self) -> Path:
        return self.data_dir / f"strata-{self.choice.tag}.setup.log"

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
            str(self._gguf_dir or self.view),
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
        self._gguf_dir = self._sources(progress)
        baseline = folder_bytes(self.data_dir)

        def measure() -> None:
            progress.bytes_written = max(0, folder_bytes(self.data_dir) - baseline)

        progress.step = "Starting Strata's setup"
        progress.bytes_written = None
        progress.log_file = self.log_path
        progress.log_name = self.log_path.relative_to(self.work).as_posix()
        with tempfile.TemporaryDirectory(
            prefix="eugene-strata-setup-", ignore_cleanup_errors=True
        ) as settings:
            code = run_process(
                self.argv(),
                cwd=self.root,
                env=self.environment(Path(settings)),
                log_path=self.log_path,
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
        files = self._made(entry)
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
            root=self.work,
            files=files,
            # B107: the MTP helper (once per data folder, ~7 GB with setup's
            # intermediates) stays for the next preparation on this node.
            discard=(
                *(f for f in files if not f.is_relative_to(self.data_dir / "mtp")),
                *self._copies,
            ),
        )

    # --- the GGUF, read where it is (B102) --------------------------------

    def _sources(self, progress: Progress) -> Path:
        """This node's view of the GGUF's folder, given to setup as it: each
        shard a link to the Library's file (B102), a symbolic link, or a
        hard link on the same drive; on Windows without either, the folder
        joined to the Library's when that is on a local drive this node can
        write (setup's marks then go beside the shards); copies otherwise,
        said as they go, and removed once the library lists the model."""
        progress.step = "Finding the model's files"
        shards = shards_of(self.gguf)
        view = self.view
        if os.path.isjunction(view) and _junction(view, self.gguf.parent):
            return view
        view.mkdir(parents=True, exist_ok=True)
        why: OSError | None = None
        for make in (_symlink, _hardlink):
            try:
                for shard in shards:
                    _place(make, shard, view / shard.name)
                return view
            except OSError as exc:
                why = why or exc
        if _writable(self.gguf.parent) and _junction(view, self.gguf.parent):
            return view
        view.mkdir(parents=True, exist_ok=True)
        total = sum(s.stat().st_size for s in shards)
        reason = (why.strerror or str(why)) if why is not None else "unknown"
        progress.step = (
            f"Copying {self.gguf.name} to this node: it cannot link to the Library's file "
            f"here ({reason})"
        )
        progress.bytes_needed, progress.bytes_written = total, 0
        for shard in shards:
            _copy(shard, view / shard.name, progress)
            self._copies.append(view / shard.name)
        progress.bytes_needed = self.bytes_needed
        return view

    def _made(self, entry: Path) -> tuple[Path, ...]:
        """What the library must hold for the model (B103): the configuration,
        every file it names that setup made (the GGUF is the Library's own),
        and setup's log."""
        cfg = json.loads(entry.read_text(encoding="utf-8"))
        found: list[Path] = [entry]
        for _flag, path in named_paths(entry, cfg):
            found += files_of(path) or []
        found.append(self.log_path)
        out: list[Path] = []
        seen: set[str] = set()
        for path in found:
            key = os.path.normcase(os.path.abspath(path))
            if key in seen or path.is_symlink() or not path.is_relative_to(self.data_dir):
                continue
            seen.add(key)
            out.append(path)
        return tuple(out)

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
        # Where its whole output is, the run worker says: the library's copy
        # of the log when it could be sent (B106).
        last = progress.message or "it printed nothing"
        return f"Strata's setup exited {code}: {last}"

    # --- afterwards -------------------------------------------------------

    def _relative(self, value: str) -> str:
        """A path setup wrote, relative to the data folder, so the folder
        travels (B43): its place in this node's folder is its place in the
        Library folder (B101)."""
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        if not _under(path, self.work):
            raise PreparationError(
                f"Strata's setup named {value}, which is not among the model's files in the "
                f"Library folder {self.folder}: Eugene cannot list a model made of it."
            )
        return Path(os.path.relpath(path, self.data_dir)).as_posix()

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
