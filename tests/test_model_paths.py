"""Where another host's model directories are on this one (M11).

Pure tests over the resolver, in both directions on whichever platform
runs them: the `to` side's shape decides the separator, so CI's Linux
and the Windows desk both exercise a POSIX library describing files for
a Windows engine and the reverse -- which is how CI caught the first
version joining `Z:\\models` with `/`.
"""

from __future__ import annotations

from eugene_plexus_agent.model_paths import (
    PathRule,
    check_rules,
    describe_checks,
    is_windows_shaped,
    join_local,
    match,
    parse_rules,
    resolve_model_path,
    validate_rules,
)

NAS = PathRule(source="/models", target="Z:\\models")
WIN = PathRule(source="D:\\models", target="/mnt/d-models")
UNC = PathRule(source="\\\\nas\\models", target="/mnt/nas")


def _expand(path: str) -> str:
    return path.replace("~", "/home/troy", 1) if path.startswith("~") else path


# --- shape ------------------------------------------------------------------


def test_the_from_side_says_which_convention_it_came_from() -> None:
    assert is_windows_shaped("D:\\models")
    assert is_windows_shaped("d:/models")
    assert is_windows_shaped("D:")
    assert is_windows_shaped("\\\\nas\\models")
    assert is_windows_shaped("//nas/models")
    assert not is_windows_shaped("/models")
    assert not is_windows_shaped("models")


# --- the direction the first real install needs -----------------------------


def test_a_posix_library_path_opens_as_a_windows_path() -> None:
    result = resolve_model_path("/models/lmstudio-community/Qwen3-GGUF/Qwen3-Q4_K_M.gguf", [NAS])
    assert result.local_path == "Z:\\models\\lmstudio-community\\Qwen3-GGUF\\Qwen3-Q4_K_M.gguf"
    assert result.rule is NAS
    assert result.declared == "/models/lmstudio-community/Qwen3-GGUF/Qwen3-Q4_K_M.gguf"


def test_a_windows_library_path_opens_as_a_posix_path() -> None:
    result = resolve_model_path("D:\\models\\qwen\\q.gguf", [WIN])
    assert result.local_path == "/mnt/d-models/qwen/q.gguf"


def test_windows_rules_read_either_separator_and_any_case() -> None:
    assert resolve_model_path("d:/Models/Qwen/q.gguf", [WIN]).local_path == (
        "/mnt/d-models/Qwen/q.gguf"
    )
    assert resolve_model_path("D:\\MODELS\\q.gguf", [WIN]).rule is WIN


def test_a_unc_share_is_one_anchor() -> None:
    result = resolve_model_path("\\\\nas\\models\\a\\b.gguf", [UNC])
    assert result.local_path == "/mnt/nas/a/b.gguf"
    # A different share on the same server is not under it.
    assert resolve_model_path("\\\\nas\\other\\b.gguf", [UNC]).rule is None


def test_posix_rules_are_case_sensitive_and_slash_only() -> None:
    assert resolve_model_path("/Models/x.gguf", [NAS]).rule is None
    # A backslash is a legal filename character on POSIX, so this is one
    # component and stays one.
    result = resolve_model_path("/models/a\\b.gguf", [NAS])
    assert result.local_path == "Z:\\models\\a\\b.gguf"


# --- matching ---------------------------------------------------------------


def test_matching_is_by_component_not_by_string_prefix() -> None:
    assert match(NAS, "/models2/x.gguf") is None
    assert match(NAS, "/models/x.gguf") == ("x.gguf",)
    assert match(NAS, "/models") == ()


def test_the_longest_match_wins_whatever_the_order() -> None:
    broad = PathRule(source="/models", target="Z:\\models")
    narrow = PathRule(source="/models/big", target="Y:\\big")
    for rules in ([broad, narrow], [narrow, broad]):
        result = resolve_model_path("/models/big/x.gguf", rules)
        assert result.rule is narrow
        assert result.local_path == "Y:\\big\\x.gguf"
    assert resolve_model_path("/models/small/x.gguf", [narrow, broad]).rule is broad


def test_a_tie_goes_to_the_first_rule_listed() -> None:
    first = PathRule(source="/models", target="Z:\\models")
    second = PathRule(source="/models", target="Y:\\models")
    assert resolve_model_path("/models/x", [first, second]).rule is first


def test_no_match_leaves_the_path_exactly_as_given() -> None:
    result = resolve_model_path("C:\\Users\\troyc\\models\\q.gguf", [NAS, WIN])
    assert result.rule is None
    assert result.local_path == "C:\\Users\\troyc\\models\\q.gguf"
    assert not result.mapped
    assert resolve_model_path("/x", []).local_path == "/x"


def test_the_declared_path_is_never_normalized_before_matching() -> None:
    """`abspath` on Windows would turn `/models/x` into `C:\\models\\x`
    and then nothing could match it. The rule matches the string."""
    assert resolve_model_path("/models/x.gguf", [NAS]).rule is NAS


# --- joining ----------------------------------------------------------------


def test_trailing_separators_on_either_side_change_nothing() -> None:
    rule = PathRule(source="/models/", target="Z:\\models\\")
    assert resolve_model_path("/models/x.gguf", [rule]).local_path == ("Z:\\models\\x.gguf")
    assert join_local("Z:\\models\\", ["a"]) == "Z:\\models\\a"
    assert join_local("/mnt/models/", ["a"]) == "/mnt/models/a"


def test_a_drive_root_keeps_its_separator() -> None:
    assert join_local("Z:\\", ["a", "b"]) == "Z:\\a\\b"
    assert join_local("/", ["a", "b"]) == "/a/b"
    assert join_local("Z:\\", []) == "Z:\\"


def test_the_separator_follows_the_target_not_the_host() -> None:
    """A `to` is a path on the host that wrote it. CI's Linux joined
    `Z:\\models` with `/` in the first version; the shape decides now."""
    assert join_local("Z:\\models", ["a", "b.gguf"]) == "Z:\\models\\a\\b.gguf"
    assert join_local("\\\\nas\\models", ["b.gguf"]) == "\\\\nas\\models\\b.gguf"
    assert join_local("/mnt/models", ["a", "b.gguf"]) == "/mnt/models/a/b.gguf"
    # An explicit override still wins, for a caller that knows better.
    assert join_local("/mnt/models", ["a"], sep="\\") == "/mnt/models\\a"


def test_the_to_side_is_used_as_spelled_and_tilde_expands() -> None:
    rule = PathRule(source="/models", target="~/models")
    result = resolve_model_path("/models/x.gguf", [rule], expand=_expand)
    assert result.local_path == "/home/troy/models/x.gguf"
    # Forward slashes in a Windows target are the operator's choice.
    win = PathRule(source="/models", target="Z:/models")
    assert resolve_model_path("/models/x.gguf", [win]).local_path == "Z:/models\\x.gguf"


def test_the_mapping_reports_itself_on_the_wire() -> None:
    wire = NAS.as_mapping().model_dump(by_alias=True)
    assert wire == {"from": "/models", "to": "Z:\\models"}
    assert str(NAS) == "/models -> Z:\\models"


# --- the config field -------------------------------------------------------


def test_validation_accepts_well_formed_mappings() -> None:
    assert (
        validate_rules(
            [
                {"from": "/models", "to": "Z:\\models"},
                {"from": "D:\\models", "to": "/mnt/d"},
                {"from": "\\\\nas\\models", "to": "~/nas"},
            ]
        )
        is None
    )
    assert validate_rules([]) is None


def test_validation_names_what_is_wrong() -> None:
    assert "expected a list" in (validate_rules({"from": "/a", "to": "/b"}) or "")
    assert "entry 0 is str" in (validate_rules(["/models"]) or "")
    assert "unknown key" in (validate_rules([{"from": "/a", "to": "/b", "too": "/c"}]) or "")
    assert "`to` must be a non-empty path" in (validate_rules([{"from": "/a", "to": " "}]) or "")
    assert "`from` must be an absolute path" in (
        validate_rules([{"from": "models", "to": "/b"}]) or ""
    )
    assert "`to` must be an absolute path" in (
        validate_rules([{"from": "/models", "to": "models"}]) or ""
    )


def test_validation_refuses_two_rules_for_one_directory() -> None:
    message = validate_rules(
        [{"from": "D:\\models", "to": "/a"}, {"from": "d:/models/", "to": "/b"}]
    )
    assert message is not None and "duplicates entry 0" in message
    # POSIX spellings that differ in case are two directories.
    assert (
        validate_rules([{"from": "/models", "to": "/a"}, {"from": "/Models", "to": "/b"}]) is None
    )


def test_parsing_is_lenient_because_a_spawn_must_not_die_on_a_hand_edit() -> None:
    rules = parse_rules(
        [{"from": " /models ", "to": "Z:\\models"}, "junk", {"from": "/x"}, None, {"to": "/y"}]
    )
    assert rules == [PathRule(source="/models", target="Z:\\models")]
    assert parse_rules(None) == []
    assert parse_rules("not a list") == []


# --- the Test button --------------------------------------------------------


def test_the_check_stats_targets_and_walks_the_librarys_models() -> None:
    present = {"Z:\\models", "Z:\\models\\a.gguf", "Z:\\models\\sub"}
    sizes = {"Z:\\models\\a.gguf": 100}
    library = [
        {"path": "/models/a.gguf", "sizeBytes": 100, "fileCount": 1},
        {"path": "/models/b.gguf", "sizeBytes": 200, "fileCount": 1},
        {"path": "/elsewhere/c.gguf", "sizeBytes": 300},
    ]
    [check] = check_rules(
        [NAS],
        library,
        exists=lambda p: p in present,
        isdir=lambda p: p in {"Z:\\models", "Z:\\models\\sub"},
        size_of=lambda p: sizes.get(p),
    )
    assert check.target_exists and check.target_is_dir
    assert check.models_under == 2
    assert check.reachable == 1
    assert check.unreachable == ["Z:\\models\\b.gguf"]
    assert check.mismatched == []

    ok, summary, error = describe_checks([check], library_consulted=True)
    assert ok is False
    assert "1 of 2 library models under /models reachable at Z:\\models" in summary
    assert error is not None and "not found here: Z:\\models\\b.gguf" in error


def test_a_size_that_disagrees_with_the_library_is_named() -> None:
    library = [{"path": "/models/a.gguf", "sizeBytes": 100, "fileCount": 1}]
    [check] = check_rules(
        [NAS],
        library,
        exists=lambda p: True,
        isdir=lambda p: not p.endswith(".gguf"),
        size_of=lambda p: 99,
    )
    assert check.mismatched == ["Z:\\models\\a.gguf (99 bytes here, 100 in the library)"]
    ok, _, error = describe_checks([check], library_consulted=True)
    assert ok is False
    assert error is not None and "size differs" in error


def test_a_missing_target_is_the_problem_before_anything_else() -> None:
    [check] = check_rules([NAS], None, exists=lambda p: False, isdir=lambda p: False)
    ok, _summary, error = describe_checks([check], library_consulted=False)
    assert ok is False
    assert error == "/models -> Z:\\models: Z:\\models does not exist on this host"


def test_without_the_library_only_the_directories_are_checked() -> None:
    [check] = check_rules([NAS], None, exists=lambda p: True, isdir=lambda p: True)
    ok, summary, error = describe_checks([check], library_consulted=False)
    assert ok is True and error is None
    assert "Z:\\models exists here" in summary
    assert "could not be consulted" in summary


def test_no_rules_is_not_a_failure() -> None:
    assert describe_checks([], library_consulted=True) == (
        True,
        "No model directory mappings configured.",
        None,
    )
