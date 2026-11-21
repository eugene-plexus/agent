"""agent#53: which engine flags an operations token may set.

`operations.refuse_owner_fields` decides from what each engine adapter
declares about a flag, not from the spelling of its key. The first test
walks every registered engine's flags: a flag that is not classified fails
it, so a new engine flag cannot reach an operations token unreviewed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from eugene_plexus_agent import operations, tokens
from eugene_plexus_agent._generated.models import EngineKind, RuntimeSpec
from eugene_plexus_agent.engines import ADAPTERS


def _declared() -> list[tuple[str, Any]]:
    return [
        (kind.value, field)
        for kind, adapter in ADAPTERS.items()
        for field in adapter.flag_schema().fields
    ]


def test_every_engine_flag_is_classified_for_operations_tokens() -> None:
    """A flag is a path (inside a Library folder for an operations token),
    the owner's, or on the reviewed list of flags an operations token may
    set. The key's ending alone never classifies a flag."""
    unclassified = [
        f"{engine}.{field.key}"
        for engine, field in _declared()
        if operations.flag_class(engine, field) is None
    ]
    assert unclassified == [], (
        "classify these engine flags in operations.py (OWNER_FLAGS, PATH_FLAG_KEYS "
        f"or OPERATIONS_FLAGS): {unclassified}"
    )


def test_a_safe_key_on_another_engine_is_not_classified() -> None:
    """Review: the lists are per engine, so an engine that adds a flag under a
    key another engine has (vLLM's `--chat-template` takes a file, MLX's takes
    text) fails the walk until someone classifies it for that engine."""
    from eugene_plexus_agent._generated.models import ConfigField, ConfigValueType

    field = ConfigField(
        key="chatTemplate", label="x", category="model", valueType=ConfigValueType.string
    )
    assert operations.flag_class("mlx", field) == "operations"
    assert operations.flag_class("vllm", field) is None
    assert operations.flag_class(EngineKind.vllm, field) is None


def test_the_reviewed_lists_name_only_declared_flags_and_no_flag_twice() -> None:
    declared = {(engine, field.key) for engine, field in _declared()}
    lists = {
        "OWNER_FLAGS": set(operations.OWNER_FLAGS),
        "PATH_FLAG_KEYS": set(operations.PATH_FLAG_KEYS),
        "OPERATIONS_FLAGS": set(operations.OPERATIONS_FLAGS),
    }
    for name, keys in lists.items():
        assert keys <= declared, f"{name} names flags no engine declares: {keys - declared}"
    names = list(lists)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            both = lists[first] & lists[second]
            assert not both, f"{both} is in both {first} and {second}"
    contained = set(operations.CONTAINED_FLAGS)
    assert contained <= declared, f"CONTAINED_FLAGS names undeclared flags: {contained}"


@pytest.mark.parametrize(
    ("engine", "key", "kind"),
    [
        ("mlx", "draftModel", "path"),
        ("mlx", "adapterPath", "path"),
        ("mlx", "trustRemoteCode", "owner"),
        ("mlx", "chatTemplate", "operations"),
        ("vllm", "trustRemoteCode", "owner"),
        ("vllm", "tokenizer", "path"),
        ("vllm", "draftModelPath", "path"),
        ("llama_cpp", "chatTemplateFile", "path"),
        ("llama_cpp", "projectorPath", "path"),
        ("llama_cpp", "draftModelPath", "path"),
        ("kev", "fallback", "path"),
    ],
)
def test_the_flags_that_reach_past_running_a_model(engine: str, key: str, kind: str) -> None:
    field = next(f for e, f in _declared() if e == engine and f.key == key)
    assert operations.flag_class(engine, field) == kind


# ------------------------------------------------------------- the check alone


def _claims(typ: str = tokens.TYP_OPERATIONS) -> tokens.Claims:
    return tokens.Claims(
        typ=typ,
        iss="control",
        sub="p-dana",
        aud=("node:gpu-a",),
        iat=0,
        exp=2**31,
        jti="j",
        ops=(tokens.POWER_OPERATE_MODELS,),
        name="Dana",
    )


def _request(claims: tokens.Claims | None) -> Any:
    state = SimpleNamespace()
    if claims is not None:
        setattr(state, operations.CLAIMS_KEY, claims)
    return SimpleNamespace(state=state)


@pytest.fixture
def library(tmp_path: Path) -> tuple[Any, Path]:
    from eugene_plexus_agent.library_folders import FolderRecord, LibraryFolderCache

    lib = tmp_path / "lib"
    (lib / "Qwen3-8B-4bit").mkdir(parents=True)
    (lib / "Qwen3-0.6B-4bit").mkdir()
    cache = LibraryFolderCache(tmp_path / "library_folders.json")
    cache.update([FolderRecord(path=str(lib))], library_url="http://library.test")
    return cache, lib


def _spec(lib: Path, engine: str = "mlx", **flags: Any) -> RuntimeSpec:
    return RuntimeSpec(
        name="q",
        engine=EngineKind(engine),
        modelPath=str(lib / "Qwen3-8B-4bit"),
        flags=flags or None,
    )


def _refused(spec: RuntimeSpec, folders: Any, previous: RuntimeSpec | None = None) -> str | None:
    try:
        operations.refuse_owner_fields(_request(_claims()), spec, previous, folders=folders)
    except HTTPException as exc:
        assert exc.status_code == 403
        return str(exc.detail["detail"])  # type: ignore[index]
    return None


@pytest.mark.parametrize(
    "value",
    ["mlx-community/Qwen3-0.6B-4bit", "/elsewhere/draft", "../Qwen3-0.6B-4bit"],
    ids=["hub-id", "outside", "relative"],
)
def test_an_mlx_draft_model_outside_the_library_is_the_owners(
    library: tuple[Any, Path], value: str
) -> None:
    folders, lib = library
    said = _refused(_spec(lib, draftModel=value), folders)
    assert said is not None and "(flags.draftModel)" in said
    assert "outside the Library's folders" in said


def test_an_mlx_draft_model_inside_the_library_is_allowed(library: tuple[Any, Path]) -> None:
    folders, lib = library
    assert _refused(_spec(lib, draftModel=str(lib / "Qwen3-0.6B-4bit")), folders) is None


@pytest.mark.parametrize("engine", ["mlx", "vllm"])
def test_turning_on_code_that_comes_with_a_model_is_the_owners(
    library: tuple[Any, Path], engine: str
) -> None:
    folders, lib = library
    said = _refused(_spec(lib, engine, trustRemoteCode=True), folders)
    assert said is not None and "(flags.trustRemoteCode)" in said
    assert "running code that comes with a model" in said
    assert _refused(_spec(lib, engine, trustRemoteCode=False), folders) is None


def test_values_the_declaration_already_had_are_left_alone(library: tuple[Any, Path]) -> None:
    folders, lib = library
    before = _spec(lib, trustRemoteCode=True, draftModel="mlx-community/Qwen3-0.6B-4bit")
    kept = _spec(
        lib,
        trustRemoteCode=True,
        draftModel="mlx-community/Qwen3-0.6B-4bit",
        prefillStepSize=1024,
    )
    assert _refused(kept, folders, before) is None
    # Turning the code off needs nobody.
    assert _refused(_spec(lib, draftModel="mlx-community/Qwen3-0.6B-4bit"), folders, before) is None
    changed = _spec(lib, trustRemoteCode=True, draftModel="mlx-community/Other")
    said = _refused(changed, folders, before)
    assert said is not None and "(flags.draftModel)" in said


def test_a_kept_trust_does_not_carry_over_to_another_model(library: tuple[Any, Path]) -> None:
    """The owner turned code on for one model; a different model's code is
    the owner's to allow, even when the new model is in the Library."""
    folders, lib = library
    before = _spec(lib, "vllm", trustRemoteCode=True)
    other = _spec(lib, "vllm", trustRemoteCode=True).model_copy(
        update={"modelPath": str(lib / "Qwen3-0.6B-4bit")}
    )
    said = _refused(other, folders, before)
    assert said is not None and "(flags.trustRemoteCode)" in said
    tokenizer = _spec(lib, "vllm", trustRemoteCode=True, tokenizer=str(lib / "Qwen3-0.6B-4bit"))
    said = _refused(tokenizer, folders, before)
    assert said is not None and "(flags.trustRemoteCode)" in said


def test_a_flag_this_engine_does_not_declare_is_left_to_the_spec_check(
    library: tuple[Any, Path],
) -> None:
    """Review: an undeclared flag (a typo) is not an owner's setting. The spec
    check that follows refuses it with the same 400 the owner gets, which
    names only flags `GET /v1/engines` already shows an operations token."""
    folders, lib = library
    assert _refused(_spec(lib, "llama_cpp", contextSze=4096), folders) is None


def test_an_engine_change_needs_the_owners_fields_again(library: tuple[Any, Path]) -> None:
    """Review: a changed engine is a changed model. What the owner set for one
    engine (code that comes with a model, the environment, extra arguments)
    does not carry over to another."""
    folders, lib = library
    before = _spec(lib, "vllm", trustRemoteCode=True)
    said = _refused(_spec(lib, "mlx", trustRemoteCode=True), folders, before)
    assert said is not None and "(flags.trustRemoteCode)" in said
    assert "another engine" in said
    with_env = _spec(lib, "vllm").model_copy(update={"env": {"HF_TOKEN": "x"}})
    moved = _spec(lib, "mlx").model_copy(update={"env": {"HF_TOKEN": "x"}})
    said = _refused(moved, folders, with_env)
    assert said is not None and "(env)" in said and "another engine" in said
    # Without the owner's fields, the engine change is the operator's.
    assert _refused(_spec(lib, "mlx"), folders, with_env) is None


@pytest.mark.parametrize(
    ("engine", "flags", "field"),
    [
        ("mlx", {"promptCacheBytes": "--trust-remote-code"}, "flags.promptCacheBytes"),
        ("mlx", {"chatTemplate": " -x"}, "flags.chatTemplate"),
        ("llama_cpp", {"tensorSplit": "-1,2"}, "flags.tensorSplit"),
        ("strata", {"gpus": ["0", "--x"]}, "flags.gpus"),
    ],
)
def test_a_value_that_starts_with_a_dash_is_the_owners(
    library: tuple[Any, Path], engine: str, flags: dict[str, Any], field: str
) -> None:
    folders, lib = library
    said = _refused(_spec(lib, engine, **flags), folders)
    assert said is not None and f"({field})" in said and 'starts with "-"' in said


def test_a_negative_number_written_as_text_is_not_a_dash_value(
    library: tuple[Any, Path],
) -> None:
    folders, lib = library
    assert _refused(_spec(lib, "llama_cpp", tensorSplit="-1"), folders) is None
    assert _refused(_spec(lib, "llama_cpp", mainGpu=-1), folders) is None


def test_the_name_ending_still_catches_a_path(library: tuple[Any, Path]) -> None:
    """The second net: a key that ends in Path is checked as one whatever its
    declaration says."""
    folders, lib = library
    said = _refused(_spec(lib, "llama_cpp", someOtherPath="/etc/passwd"), folders)
    assert said is not None and "(flags.someOtherPath)" in said


def test_the_owner_is_never_checked(library: tuple[Any, Path]) -> None:
    folders, lib = library
    spec = _spec(
        lib,
        trustRemoteCode=True,
        draftModel="mlx-community/Qwen3-0.6B-4bit",
        adapterPath="/elsewhere/adapter",
        notAFlag="x",
    )
    # `operation(...)` keeps claims on the request only for an operations
    # token; the owner's request carries none, and nothing is refused.
    operations.refuse_owner_fields(_request(None), spec, None, folders=None)
    operations.refuse_owner_fields(_request(None), spec, _spec(lib), folders=folders)


@pytest.mark.parametrize(
    "after",
    [{"source": "binary"}, {}],
    ids=["to-binary", "source-removed"],
)
def test_a_kept_trust_does_not_follow_the_model_out_of_the_container(
    library: tuple[Any, Path], after: dict[str, Any]
) -> None:
    """Review of library#22: the container gives a model's code only the
    model's folder, read only; vLLM on the host runs it in the agent's
    account. An unset `source` may mean the host (`unsetMeans`), so removing
    it leaves the container too."""
    folders, lib = library
    before = _spec(lib, "vllm", trustRemoteCode=True, source="container")
    said = _refused(_spec(lib, "vllm", trustRemoteCode=True, **after), folders, before)
    assert said is not None and "(flags.trustRemoteCode)" in said
    assert "outside the container" in said
    # Without the owner's flag, the move is the operator's.
    assert _refused(_spec(lib, "vllm", **after), folders, before) is None


@pytest.mark.parametrize(
    ("first", "then"),
    [
        ({"source": "container"}, {"source": "container", "maxModelLen": 8192}),
        ({}, {"source": "container"}),
        ({"source": "binary"}, {}),
        ({"source": "binary"}, {"source": "container"}),
    ],
    ids=["same", "unset-to-container", "binary-to-unset", "binary-to-container"],
)
def test_a_kept_trust_stays_where_the_code_reaches_no_further(
    library: tuple[Any, Path], first: dict[str, Any], then: dict[str, Any]
) -> None:
    """The owner allowed the code where it could already run on the host (an
    unset `source` or `binary`); the container reaches less, so it stays."""
    folders, lib = library
    before = _spec(lib, "vllm", trustRemoteCode=True, **first)
    assert _refused(_spec(lib, "vllm", trustRemoteCode=True, **then), folders, before) is None
