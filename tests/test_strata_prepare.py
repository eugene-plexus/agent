"""LS5: Strata prepares a GGUF on its list, with upstream's setup (design §6.6).

A stand-in `setup.py` takes upstream's arguments and writes what upstream's
writes, where it writes it: the pack, tokenizer and MTP helper into the data
folder, `strata-<tag>.json` (absolute paths) and a start script into its own
folder, `.done` marks beside the shards. No real Strata, model or GPU.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

from eugene_plexus_agent.engines import strata_prepare
from eugene_plexus_agent.engines.strata_models import (
    SETUP_CHOICES,
    SUPPORTED_MODELS,
    choice_for_file,
    disk_needed,
    supported_here,
)
from eugene_plexus_agent.preparation import (
    PreparationCancelled,
    PreparationError,
    PreparationJobs,
    Progress,
)

FIRST = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
SECOND = "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"

FAKE_SETUP = r"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
for flag in ("--family", "--model", "--gguf-dir", "--data-dir", "--context", "--gpu", "--vision",
             "--experimental-speed-projection"):
    ap.add_argument(flag)
for flag in ("--no-browser", "--no-start", "--yes"):
    ap.add_argument(flag, action="store_true")
a = ap.parse_args()
data, gguf = Path(a.data_dir), Path(a.gguf_dir)
(data / "fake-setup-call.json").write_text(json.dumps(
    {"argv": sys.argv[1:], "appdata": os.environ.get("APPDATA"),
     "xdg": os.environ.get("XDG_CONFIG_HOME"), "utf8": os.environ.get("PYTHONUTF8")}))
appdata = Path(os.environ["APPDATA"]) / "Strata"
appdata.mkdir(parents=True, exist_ok=True)
(appdata / "settings.json").write_text(json.dumps({"data_dir": str(data)}))
print("=== Step 1: checking your PC ===", flush=True)
print("  [!]  Windows' page file is 1.0 GB", flush=True)
if os.environ.get("FAKE_SETUP_SLEEP"):
    (data / "fake-setup.pid").write_text(str(os.getpid()))
    time.sleep(60)
if os.environ.get("FAKE_SETUP_FAIL"):
    print("\n  [X]  RAM: 16 GB - the smallest model (the Coder) needs about 32 GB", flush=True)
    print("       --model NAME --yes installs one anyway", flush=True)
    print("\nSetup stopped. Fix the item above and run it again.", flush=True)
    sys.exit(1)
print("=== Step 6: preparing the model for Strata ===", flush=True)
print("  writing pack 10%\r  writing pack 100%", flush=True)
tag = ("" if a.family == "qwen" else a.family + "-") + a.model
tag = tag.lower()
pack = data / "packs" / tag
(pack / "tokenizer").mkdir(parents=True, exist_ok=True)
for name in ("vocab.json", "merges.txt", "token_type.json"):
    (pack / "tokenizer" / name).write_text("{}")
(pack / "native_experts.txt").write_text("x" * 1000)
rt = data / "mtp" / "rt"
rt.mkdir(parents=True, exist_ok=True)
(rt / "experts.bin").write_bytes(b"\0" * 5000)
(data / "mtp" / "mtp-q2_0.gguf").write_bytes(b"GGUF")
shards = sorted(gguf.glob("*-of-*.gguf"))
for s in shards:
    s.with_name(s.name + ".done").write_text("whole")
args = ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(shards[-1]),
        "--expert-profile", str(ROOT / "data" / "expert-profile.bin"), "--expert-cache", "auto",
        "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
        "--max-context", str(a.context or 65536), "--kv", "int8"]
cfg = {"exe": str(ROOT / "engine" / "strata.exe"), "args": args, "cwd": str(ROOT),
       "tokenizer": str(pack / "tokenizer"), "model_name": "qwen3.8-flash-next-" + a.model.lower(),
       "log": str(ROOT / f"strata-{tag}.log"), "lib_dirs": [], "port": 8080}
if os.environ.get("FAKE_SETUP_VISION"):
    cfg["vision"] = {"exe": "strata-vision.exe"}
(ROOT / f"strata-{tag}.json").write_text(json.dumps(cfg))
(ROOT / f"run-{tag}.bat").write_text("@echo off")
print("All set.", flush=True)
"""


@pytest.fixture
def build(tmp_path, monkeypatch):
    """A Strata build (setup.py beside serve/ and engine/), its preparation
    tools already present, and both shards of IQ2_XS in a Library folder."""
    root = tmp_path / "engines" / "strata" / "v0.1.39" / "source" / "Strata-x"
    (root / "serve").mkdir(parents=True)
    (root / "serve" / "server.py").write_text("")
    native = root / "engine" / ("strata.exe" if os.name == "nt" else "strata")
    native.parent.mkdir()
    native.write_text("")
    (root / "data").mkdir()
    (root / "data" / "expert-profile.bin").write_bytes(b"\1" * 64)
    (root / "setup.py").write_text(FAKE_SETUP, encoding="utf-8")
    llama = root / "third_party" / "llama.cpp"
    (llama / "ggml").mkdir(parents=True)
    (llama / "ggml" / "CMakeLists.txt").write_text("")
    (llama / "gguf-py").mkdir()
    monkeypatch.setattr(strata_prepare, "venv_python", lambda _root: Path(sys.executable))
    folder = tmp_path / "models"
    repo = folder / "ISTA-DASLab" / "Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    repo.mkdir(parents=True)
    (repo / FIRST).write_bytes(b"GGUF first")
    (repo / SECOND).write_bytes(b"GGUF second")
    return root, folder, repo / FIRST


def _plan(build, **kw):
    root, folder, gguf = build
    args = {
        "root": root,
        "gguf": gguf,
        "data_dir": folder / "Strata-data",
        "source_path": "/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/" + FIRST,
        "context": None,
        "ram_bytes": 64 * 2**30,
    }
    args.update(kw)
    return strata_prepare.plan(**args)


# --- the list, as setup names and sizes it --------------------------------------


def test_every_choice_on_the_list_is_named_as_setup_names_it():
    assert set(SETUP_CHOICES) == {m.id for m in SUPPORTED_MODELS}
    for model in SUPPORTED_MODELS:
        choice = SETUP_CHOICES[model.id]
        # setup's tag is its family's prefix and the size: the list's id.
        assert choice.tag == model.id.lower()
        assert choice.model == model.quantization
    assert SETUP_CHOICES["IQ2_XS"].model_name == "qwen3.8-flash-next-iq2_xs"
    assert SETUP_CHOICES["coder-IQ1_M"].tag == "coder-iq1_m"
    assert SETUP_CHOICES["unsloth-UD-IQ4_XS"].model_name == "qwen3.8-flash-next-unsloth-ud-iq4_xs"


@pytest.mark.parametrize(
    ("choice", "ram_gib", "gb"),
    [
        ("IQ2_XS", 64, 8),  # its experts and 10 GB fit
        ("IQ2_XS", 32, 8 + 36.5),  # the low-RAM mode writes them into one file
        ("unsloth-UD-Q4_K_XL", 32, 8),  # a RAM budget: never that file
        ("Q2_0", 128, 48),  # repacked for AVX-512, counted always
        ("Q2_0", 16, 48),  # and then no low-RAM file, as in setup
        ("IQ3_S", None, 8 + 51.3),  # RAM unknown: the larger answer
    ],
)
def test_disk_is_setups_own_rule(choice, ram_gib, gb):
    ram = None if ram_gib is None else ram_gib * 2**30
    assert disk_needed(SETUP_CHOICES[choice], ram) == int(gb * 1e9)


def test_the_node_reports_its_own_disk_and_keeps_the_rest():
    here = supported_here(64 * 2**30)
    assert [m.id for m in here] == [m.id for m in SUPPORTED_MODELS]
    for mine, theirs in zip(here, SUPPORTED_MODELS, strict=True):
        assert mine.source == theirs.source
        assert mine.preparation.recipe == "strata-prepare"
        assert mine.preparation.diskBytes >= 8_000_000_000
    # The shared declaration is untouched.
    assert all(m.preparation.diskBytes is None for m in SUPPORTED_MODELS)


def test_a_file_is_on_the_list_by_its_name_alone():
    assert choice_for_file("D:\\models\\x\\" + FIRST.lower())[1].model == "IQ2_XS"
    assert choice_for_file("/models/" + FIRST)[0].id == "IQ2_XS"
    assert choice_for_file("/models/" + SECOND) is None  # a second shard is not a choice
    assert choice_for_file("/models/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00002.gguf") is None


# --- before anything starts -----------------------------------------------------


def test_off_the_list_is_refused_naming_the_list(build):
    _root, _folder, gguf = build
    other = gguf.with_name("Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00002.gguf")
    other.write_bytes(b"GGUF")
    with pytest.raises(PreparationError, match="only the files on its own list"):
        _plan(build, gguf=other)


def test_a_missing_shard_is_named(build):
    _root, _folder, gguf = build
    gguf.with_name(SECOND).unlink()
    with pytest.raises(PreparationError, match=SECOND):
        _plan(build)


def test_a_context_setup_does_not_offer_is_refused(build):
    with pytest.raises(PreparationError, match="not 50000"):
        _plan(build, context=50000)


def test_too_little_disk_names_both_numbers(build, monkeypatch):
    monkeypatch.setattr(strata_prepare, "free_bytes", lambda _p: 3_000_000_000)
    with pytest.raises(PreparationError) as refused:
        _plan(build)
    assert "about 8 GB free" in str(refused.value) and "has 3 GB free" in str(refused.value)


def test_an_install_without_setup_says_reinstall(build):
    root, _folder, _gguf = build
    (root / "setup.py").unlink()
    with pytest.raises(PreparationError, match="reinstall Strata"):
        _plan(build)


def test_the_arguments_are_setups_non_interactive_ones(build):
    recipe = _plan(build, context=32768, gpu=1)
    argv = recipe.argv()
    assert argv[1].endswith("setup.py")
    pairs = dict(zip(argv[2::2], argv[3::2], strict=False))
    assert pairs["--family"] == "qwen" and pairs["--model"] == "IQ2_XS"
    assert Path(pairs["--gguf-dir"]) == build[2].parent
    assert Path(pairs["--data-dir"]) == build[1] / "Strata-data"
    assert "--yes" in argv and "--no-start" in argv and "--no-browser" in argv
    assert argv[argv.index("--vision") + 1] == "no"
    assert argv[argv.index("--experimental-speed-projection") + 1] == "off"
    assert argv[argv.index("--context") + 1] == "32768"
    assert argv[argv.index("--gpu") + 1] == "1"
    assert "--context" not in _plan(build).argv()


# --- the run --------------------------------------------------------------------


def test_a_preparation_makes_a_launchable_model_in_strata_data(build, monkeypatch):
    root, folder, gguf = build
    monkeypatch.setenv("APPDATA", str(folder / "persons-own-appdata"))
    progress = Progress()
    result = _plan(build).run(progress)
    data = folder / "Strata-data"
    assert result.entry == data / "strata-iq2_xs.json"
    assert result.name == "qwen3.8-flash-next-iq2_xs"
    assert result.recipe == "strata-prepare" and result.recipe_version == "v0.1.39"
    assert result.source["repoId"] == "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    assert result.source["path"].endswith(FIRST)
    # B48: moved out of the engine's folder, paths relative, cwd dropped,
    # the expert profile copied beside the pack, the start script gone.
    cfg = json.loads(result.entry.read_text(encoding="utf-8"))
    assert "cwd" not in cfg
    # LS7: node-neutral, since any node may launch it from the Library.
    assert not {"exe", "log", "lib_dirs", "port", "model_name", "open_browser"} & set(cfg)
    args = cfg["args"]
    assert args[args.index("--pack") + 1] == "packs/iq2_xs"
    assert (
        args[args.index("--native") + 1]
        == f"../ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/{FIRST}"
    )
    assert args[args.index("--expert-profile") + 1] == "packs/iq2_xs/expert-profile.bin"
    assert (data / "packs" / "iq2_xs" / "expert-profile.bin").read_bytes() == b"\1" * 64
    assert cfg["tokenizer"] == "packs/iq2_xs/tokenizer"
    assert not list(root.glob("strata-*.json")) and not list(root.glob("run-*.bat"))
    # B44: the marker is there; B46: the GGUF is unchanged, setup's marks beside it.
    assert (data / ".eugene-engine-files").is_file()
    assert gguf.read_bytes() == b"GGUF first"
    assert gguf.with_name(FIRST + ".done").is_file()
    # B45: setup ran with its own settings folder, never the person's.
    call = json.loads((data / "fake-setup-call.json").read_text())
    assert call["appdata"] != str(folder / "persons-own-appdata")
    assert call["appdata"] == call["xdg"] and call["utf8"] == "1"
    assert not (folder / "persons-own-appdata").exists()
    assert not Path(call["appdata"]).exists()  # gone with the run
    # B56: its steps in its own words, its warnings kept, what it wrote measured.
    assert progress.warnings == ["Windows' page file is 1.0 GB"]
    assert progress.bytes_written and progress.bytes_written >= 6000
    assert (data / "strata-iq2_xs.setup.log").is_file()
    # LS7b: what the configuration says of the model, for its provenance.
    facts = result.facts
    assert facts["title"] == "Qwen3.8-Flash-Next IQ2_XS"
    assert facts["architecture"] == "qwen4exp" and facts["quantization"] == "IQ2_XS"
    assert facts["mode"]
    files = {f["path"]: f for f in facts["files"]}
    assert files["strata-iq2_xs.json"]["sizeBytes"] == result.entry.stat().st_size
    assert files["packs/iq2_xs/expert-profile.bin"]["sizeBytes"] == 64
    assert not any(FIRST in path for path in files)  # the source model's own


def test_setups_failure_is_the_cause(build, monkeypatch):
    monkeypatch.setenv("FAKE_SETUP_FAIL", "1")
    with pytest.raises(PreparationError) as stopped:
        _plan(build).run(Progress())
    assert str(stopped.value) == (
        "Strata's setup stopped: RAM: 16 GB - the smallest model (the Coder) needs about 32 GB "
        "--model NAME --yes installs one anyway"
    )


def test_a_configuration_eugene_cannot_launch_is_not_listed(build, monkeypatch):
    monkeypatch.setenv("FAKE_SETUP_VISION", "1")
    with pytest.raises(PreparationError, match=r"cannot launch: .*vision"):
        _plan(build).run(Progress())


def test_cancel_stops_setup_and_what_it_started(build, monkeypatch):
    monkeypatch.setenv("FAKE_SETUP_SLEEP", "1")
    recipe = _plan(build)
    progress = Progress()
    failure: list[BaseException] = []

    def run() -> None:
        try:
            recipe.run(progress)
        except BaseException as exc:
            failure.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    pid_file = build[1] / "Strata-data" / "fake-setup.pid"
    for _ in range(300):
        if pid_file.is_file() and pid_file.read_text():
            break
        time.sleep(0.05)
    pid = int(pid_file.read_text())
    progress.cancelled.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert isinstance(failure[0], PreparationCancelled)
    assert not _alive(pid)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import subprocess

        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# --- the preparation tools (B47) ------------------------------------------------


def _archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        top = "llama.cpp-3cf0325/"
        zf.writestr(top + "ggml/CMakeLists.txt", "cmake")
        zf.writestr(top + "gguf-py/gguf/__init__.py", "")
        deep = "examples/" + "very-long-folder-name/" * 12 + "file.plist"
        zf.writestr(top + deep, "never unpacked")
        zf.writestr(top + "tools/ui/x.svelte", "never unpacked")
    return buffer.getvalue()


def test_the_install_unpacks_only_what_setup_reads(tmp_path):
    """The preparation tools come with Strata's install (LS7, B47 reversed),
    unpacked only as far as setup reads them: Windows' 260-character limit."""
    from eugene_plexus_agent.engines import strata_install

    archive = tmp_path / "llama.cpp.zip"
    archive.write_bytes(_archive())
    root = tmp_path / "root"
    root.mkdir()
    assert not strata_install.tools_installed(root)
    llama = root / "third_party" / "llama.cpp"
    strata_install.unpack_llama_parts(archive, llama)
    assert (llama / "ggml" / "CMakeLists.txt").read_text() == "cmake"
    assert (llama / "gguf-py" / "gguf" / "__init__.py").is_file()
    assert not (llama / "examples").exists() and not (llama / "tools").exists()
    assert not (root / "third_party" / ".eugene-llama").exists()
    assert strata_install.tools_installed(root)


def test_the_install_downloads_the_tools_with_the_engine():
    """setup.py's LLAMA_CPP_COMMIT at the pinned Strata commit, in the plan."""
    from eugene_plexus_agent._generated.models import HostAccelerator
    from eugene_plexus_agent.engines import strata_install

    assert strata_install.LLAMA_CPP_COMMIT in strata_install.LLAMA_CPP.url
    host = HostAccelerator.model_validate(
        {"os": "windows", "arch": "x64", "accelerator": "cuda", "acceleratorVersion": "13.0"}
    )
    plan = strata_install.plan(host, None, None)
    assert strata_install.LLAMA_CPP in plan.assets  # type: ignore[union-attr]


def test_a_preparation_without_the_tools_names_the_reinstall(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(PreparationError, match="Reinstall Strata"):
        strata_prepare.check_tools(root)


# --- jobs (B52, B53) ------------------------------------------------------------


class _Recipe:
    def __init__(self, gate: threading.Event, *, fail: str | None = None) -> None:
        self.gate, self.fail, self.ran = gate, fail, False
        self.output, self.bytes_needed = Path("."), 10

    def run(self, progress):
        self.ran = True
        while not self.gate.wait(0.01):
            progress.check_cancelled()
        if self.fail:
            raise PreparationError(self.fail)
        return "made"


def _wait(progress: Progress) -> None:
    for _ in range(500):
        if progress.finished:
            return
        time.sleep(0.01)


def test_one_preparation_at_a_time_and_the_next_waits():
    jobs = PreparationJobs()
    first_gate, second_gate = threading.Event(), threading.Event()
    first, second = _Recipe(first_gate), _Recipe(second_gate)
    a = jobs.start("a", first, engine="strata")
    b = jobs.start("b", second, engine="strata")
    assert a.state == "running"
    assert b.state == "waiting" and "another preparation" in (b.message or "")
    assert jobs.busy_for("strata") and not jobs.busy_for("llama_cpp")
    first_gate.set()
    _wait(a)
    assert a.state == "done"
    assert jobs.poll("b").state == "running"  # its turn, at the next poll
    second_gate.set()
    _wait(b)
    jobs.forget("a")
    jobs.forget("b")
    assert not jobs.busy_for("strata")


def test_a_cancelled_operation_stops_its_job_and_a_waiting_one_never_runs():
    jobs = PreparationJobs()
    gate = threading.Event()
    running, waiting = _Recipe(gate), _Recipe(threading.Event())
    a = jobs.start("a", running, engine="strata")
    jobs.start("b", waiting, engine="strata")
    jobs.cancel_except(set())
    _wait(a)
    assert a.state == "cancelled"
    assert not waiting.ran
    jobs.cancel_except(set())
    assert jobs.poll("a") is None and jobs.poll("b") is None


def test_a_recipe_failure_is_the_jobs_error():
    jobs = PreparationJobs()
    gate = threading.Event()
    gate.set()
    progress = jobs.start("a", _Recipe(gate, fail="setup stopped: no disk"), engine="strata")
    _wait(progress)
    assert progress.state == "failed" and progress.error == "setup stopped: no disk"
    assert progress.snapshot()["state"] == "failed"


# --- one GPU (B49) and the contexts it offers (B50) ----------------------------


@pytest.mark.parametrize(
    ("listing", "picked"),
    [
        ("0, 32607\n", None),  # one card: setup's own default, no --gpu
        ("0, 12288\n1, 32607\n", 1),  # the most memory
        ("0, 24576\n1, 24564\n", 0),  # the same, rounded: the lower number
        ("", None),  # nvidia-smi said nothing
    ],
)
def test_on_several_cards_the_one_with_the_most_memory(monkeypatch, listing, picked):
    import subprocess as sp

    monkeypatch.setattr(
        strata_prepare.subprocess,
        "run",
        lambda *a, **k: sp.CompletedProcess(a, 0, stdout=listing, stderr=""),
    )
    assert strata_prepare.main_gpu() == picked


def test_the_contexts_offered_are_setups_own():
    here = supported_here(64 * 2**30)
    for model in here:
        assert [c.root for c in model.preparation.contexts] == list(strata_prepare.CONTEXTS)
    assert strata_prepare.CONTEXTS == (8192, 32768, 65536, 131072, 262144, 393216, 524288)
