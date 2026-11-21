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
        if operations.flag_class(field) is None
    ]
    assert unclassified == [], (
        "classify these engine flags in operations.py (OWNER_FLAGS, PATH_FLAG_KEYS "
        f"or OPERATIONS_FLAGS): {unclassified}"
    )


def test_the_reviewed_lists_name_only_declared_flags_and_no_flag_twice() -> None:
    declared = {field.key for _, field in _declared()}
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
    assert operations.flag_class(field) == kind


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


def test_a_flag_no_engine_declares_is_the_owners(library: tuple[Any, Path]) -> None:
    folders, lib = library
    said = _refused(_spec(lib, notAFlag="x"), folders)
    assert said is not None and "(flags.notAFlag)" in said


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
