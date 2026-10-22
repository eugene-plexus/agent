"""Stable card identity and temporary recovery, without starting an engine."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from eugene_plexus_agent._generated.models import ComputeDevice, RuntimeSpec
from eugene_plexus_agent.admission import LibraryFit, check_admission, target_devices
from eugene_plexus_agent.engines.devices import DeviceSnapshot
from eugene_plexus_agent.runtimes import _RuntimePlanner


def card(index, uuid):
    return ComputeDevice(
        kind="cuda",
        index=index,
        name=f"Card {uuid}",
        uuid=uuid,
        memoryTotalBytes=24 * 1024**3,
        memoryFreeBytes=20 * 1024**3,
    )


def snapshot(*cards):
    return DeviceSnapshot(tuple(cards), (), 96 * 1024**3, 64 * 1024**3, datetime.now(UTC))


def spec(**over):
    return RuntimeSpec.model_validate(
        {"name": "model", "engine": "llama_cpp", "modelPath": "/model.gguf", "port": 19043, **over}
    )


def test_uuid_pins_measure_only_the_selected_card():
    a, b = card(0, "GPU-a"), card(1, "GPU-b")
    assert target_devices(spec(env={"CUDA_VISIBLE_DEVICES": "GPU-b"}), snapshot(a, b)) == [b]


def test_index_pins_launch_by_uuid_after_renumbering():
    from eugene_plexus_agent.card_launch import select_cards

    a, b = card(0, "GPU-a"), card(1, "GPU-b")
    saved = spec(env={"CUDA_VISIBLE_DEVICES": "1"})
    first = select_cards(saved, snapshot(a, b))
    assert first.spec.env["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    second = select_cards(saved, snapshot(card(0, "GPU-b")), expected=first.expected)
    assert second.spec.env["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert second.missing == []
    assert saved.env == {"CUDA_VISIBLE_DEVICES": "1"}


async def test_a_missing_pinned_card_refits_on_the_cards_left():
    from eugene_plexus_agent.card_launch import refit_on_cards_left, select_cards

    a, b = card(0, "GPU-a"), card(1, "GPU-b")
    saved = spec(flags={"gpuLayers": 99, "contextSize": 32768, "tensorSplit": "1,3", "mainGpu": 0})
    before = saved.model_dump_json()
    launch = select_cards(saved, snapshot(card(0, "GPU-b")), expected=[a, b])
    library = AsyncMock()
    library.fit.side_effect = [
        LibraryFit(25 * 1024**3, "tight", 32768, 16384),
        LibraryFit(19 * 1024**3, "fits", 16384, 16384),
    ]

    async def measure(candidate):
        return await check_admission(
            candidate,
            snapshot=snapshot(card(0, "GPU-b")),
            library=library,
            running=[],
            exists=lambda _: True,
            size_of=lambda _: 10 * 1024**3,
        )

    launch, answer = await refit_on_cards_left(launch, measure)
    assert answer.fit.value == "fits"
    assert launch.spec.flags["contextSize"] == 16384
    assert launch.spec.flags["tensorSplit"] == "1"
    assert launch.spec.flags["mainGpu"] == 0
    assert launch.report.missing[0].uuid == "GPU-a"
    assert any("32768" in c.root and "16384" in c.root for c in launch.report.changes)
    assert library.fit.call_args.kwargs["gpu_count"] == 1
    assert saved.model_dump_json() == before
    restored = select_cards(saved, snapshot(a, b), expected=launch.expected)
    assert restored.report is None
    assert restored.spec.flags["contextSize"] == 32768


def test_do_not_start_names_the_missing_card():
    from eugene_plexus_agent.card_launch import select_cards

    with pytest.raises(ValueError, match=r"Card GPU-a.*was #0.*set not to start"):
        select_cards(
            spec(flags={"missingCard": "do-not-start"}),
            snapshot(card(0, "GPU-b")),
            expected=[card(0, "GPU-a"), card(1, "GPU-b")],
        )


def test_unknown_identity_keeps_index_fallback():
    from eugene_plexus_agent.card_launch import select_cards

    unknown = card(0, "GPU-a").model_copy(update={"uuid": None})
    launch = select_cards(spec(env={"CUDA_VISIBLE_DEVICES": "0"}), snapshot(unknown))
    assert launch.spec.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert launch.report is None


async def test_a_crash_respawn_runs_the_card_check(monkeypatch):
    from eugene_plexus_agent.engines import LlamaCppAdapter
    from eugene_plexus_agent.supervisor import SpawnPlan

    planner = _RuntimePlanner(spec(), LlamaCppAdapter(), logging.getLogger(__name__))
    check = AsyncMock(return_value=None)
    planner.before_start = check
    monkeypatch.setattr(planner, "plan", lambda: SpawnPlan(["fixture"], {}))
    await planner.plan_async()
    await planner.plan_async()
    assert check.await_count == 2


async def test_launch_policy_precedes_card_checks_and_engine_probes(monkeypatch):
    from eugene_plexus_agent.engines import LlamaCppAdapter
    from eugene_plexus_agent.supervisor import SpawnPlan, SpawnPlanError

    planner = _RuntimePlanner(
        spec(extraArgs=["--arbitrary"]), LlamaCppAdapter(), logging.getLogger(__name__)
    )
    check = AsyncMock(return_value=None)
    planner.before_start = check
    monkeypatch.setattr(planner, "plan", lambda: SpawnPlan(["fixture"], {}))
    with pytest.raises(SpawnPlanError, match="allowUnrestrictedEngineLaunch"):
        await planner.plan_async()
    check.assert_not_awaited()


def test_observations_survive_boot_without_changing_profile(tmp_path: Path):
    from eugene_plexus_agent.card_launch import DeviceMemory, select_cards

    memory = DeviceMemory(tmp_path / "runtime-devices.json")
    launch = select_cards(spec(), snapshot(card(0, "GPU-a"), card(1, "GPU-b")))
    memory.record(spec(), launch)
    reloaded = DeviceMemory(memory.path)
    assert [d.uuid for d in reloaded.expected(spec())] == ["GPU-a", "GPU-b"]


def test_context_changes_keep_the_card_uuid_binding():
    from eugene_plexus_agent.card_launch import DeviceMemory, select_cards

    saved = spec(flags={"contextSize": 32768}, env={"CUDA_VISIBLE_DEVICES": "1"})
    memory = DeviceMemory()
    memory.record(saved, select_cards(saved, snapshot(card(0, "GPU-a"), card(1, "GPU-b"))))
    changed = saved.model_copy(update={"flags": {"contextSize": 16384}})
    launch = select_cards(changed, snapshot(card(0, "GPU-b")), expected=memory.expected(changed))
    assert launch.spec.env["CUDA_VISIBLE_DEVICES"] == "GPU-b"


def test_tensor_parallel_drops_to_a_valid_size(tmp_path: Path):
    from eugene_plexus_agent.engines.vllm import refit_parallelism

    (tmp_path / "config.json").write_text('{"num_attention_heads": 32}')
    flags, changes = refit_parallelism({"tensorParallelSize": 4}, 3, str(tmp_path))
    assert flags["tensorParallelSize"] == 2
    assert changes


def test_missing_card_setting_is_in_both_gui_schemas():
    from eugene_plexus_agent.engines import LlamaCppAdapter, VllmAdapter

    for adapter in (LlamaCppAdapter(), VllmAdapter()):
        field = next(f for f in adapter.flag_schema().fields if f.key == "missingCard")
        assert field.default == "start-on-cards-left"
        assert field.enumValues == ["start-on-cards-left", "do-not-start"]


def test_runtime_reports_the_effective_launch_settings():
    from eugene_plexus_agent.card_launch import select_cards
    from eugene_plexus_agent.engines import LlamaCppAdapter
    from eugene_plexus_agent.runtimes import RuntimeSupervisor
    from eugene_plexus_agent.supervisor import ProcessState, SupervisedProcess

    saved = spec(flags={"contextSize": 32768})
    supervisor = RuntimeSupervisor()
    planner = _RuntimePlanner(saved, LlamaCppAdapter(), logging.getLogger(__name__))
    launch = select_cards(
        saved, snapshot(card(0, "GPU-b")), expected=[card(0, "GPU-a"), card(1, "GPU-b")]
    )
    launch.spec = launch.spec.model_copy(update={"flags": {"contextSize": 16384}})
    planner.card_launch = launch
    process = SupervisedProcess(planner, logging.getLogger(__name__))
    process.state = ProcessState.starting
    supervisor._processes[saved.name] = process
    supervisor._planners[saved.name] = planner
    supervisor.device_memory.record(saved, launch)
    runtime = supervisor.compose(saved)
    assert runtime.flags["contextSize"] == 16384
    assert runtime.env["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert runtime.startedOnFewerCards.missing[0].uuid == "GPU-a"
    assert saved.flags["contextSize"] == 32768


def test_an_unreadable_inventory_is_not_a_missing_card():
    from eugene_plexus_agent.card_launch import select_cards

    unavailable = detect_unavailable_snapshot()
    launch = select_cards(
        spec(flags={"missingCard": "do-not-start"}), unavailable, expected=[card(0, "GPU-a")]
    )
    assert launch.report is None


async def test_an_unreadable_inventory_does_not_refuse_on_a_cpu_budget():
    from eugene_plexus_agent.card_launch import refit_on_cards_left, select_cards

    unreadable = detect_unavailable_snapshot()
    launch = select_cards(
        spec(env={"CUDA_VISIBLE_DEVICES": "GPU-a"}), unreadable, expected=[card(0, "GPU-a")]
    )

    async def measure(candidate):
        return await check_admission(
            candidate,
            snapshot=unreadable,
            library=None,
            running=[],
            exists=lambda _: True,
            size_of=lambda _: 1024**3,
        )

    launch, answer = await refit_on_cards_left(launch, measure)
    assert answer.decision.value == "admit"
    assert answer.fit.value == "unknown"
    assert launch.spec.env["CUDA_VISIBLE_DEVICES"] == "GPU-a"


def detect_unavailable_snapshot():
    from eugene_plexus_agent.engines.devices import detect_devices

    return detect_devices(run=lambda _: None, memory=lambda: (1, 1), os_adapters=lambda: [])


def test_cuda_device_names_follow_the_uuid_visibility_order():
    a, b = card(0, "GPU-a"), card(1, "GPU-b")
    selected = target_devices(
        spec(env={"CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a"}, flags={"devices": "CUDA0"}),
        snapshot(a, b),
    )
    assert selected == [b]


def test_main_gpu_pin_selects_that_card_when_split_is_disabled():
    from eugene_plexus_agent.card_launch import select_cards

    a = card(0, "GPU-a").model_copy(update={"memoryFreeBytes": 22 * 1024**3})
    b = card(1, "GPU-b")
    launch = select_cards(spec(flags={"mainGpu": 1, "splitMode": "none"}), snapshot(a, b))
    assert launch.devices == [b]
    assert launch.spec.env["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert launch.spec.flags["mainGpu"] == 0


@pytest.mark.parametrize("selector", ["", "-1", "MIG-GPU-a/1/0", "GPU-unknown"])
def test_untested_or_disabled_cuda_selections_are_preserved(selector):
    from eugene_plexus_agent.card_launch import select_cards

    saved = spec(env={"CUDA_VISIBLE_DEVICES": selector})
    launch = select_cards(saved, snapshot(card(0, "GPU-a")))
    assert launch.spec.env["CUDA_VISIBLE_DEVICES"] == selector
    assert launch.report is None


async def test_missing_card_can_use_ram_without_editing_the_profile():
    from eugene_plexus_agent.card_launch import refit_on_cards_left, select_cards

    saved = spec(flags={"gpuLayers": 99, "contextSize": 32768})
    left = snapshot(card(0, "GPU-b"))
    launch = select_cards(saved, left, expected=[card(0, "GPU-a"), card(1, "GPU-b")])
    library = AsyncMock()
    library.fit.return_value = LibraryFit(30 * 1024**3, "split", 32768, 0)

    async def measure(candidate):
        return await check_admission(
            candidate,
            snapshot=left,
            library=library,
            running=[],
            exists=lambda _: True,
            size_of=lambda _: 12 * 1024**3,
            engine_places=True,
        )

    launch, answer = await refit_on_cards_left(launch, measure)
    assert answer.decision.value == "admit"
    assert "gpuLayers" not in launch.spec.flags
    assert any("system memory" in c for c in launch.changes)
    assert saved.flags["gpuLayers"] == 99


async def test_measured_no_fit_names_the_numbers():
    from eugene_plexus_agent.card_launch import refit_on_cards_left, select_cards

    launch = select_cards(
        spec(flags={"reuseRepeatedText": False}),
        snapshot(card(0, "GPU-b")),
        expected=[card(0, "GPU-a"), card(1, "GPU-b")],
    )
    library = AsyncMock()
    library.fit.return_value = LibraryFit(200 * 1024**3, "no", 32768)

    async def measure(candidate):
        return await check_admission(
            candidate,
            snapshot=snapshot(card(0, "GPU-b")),
            library=library,
            running=[],
            exists=lambda _: True,
            size_of=lambda _: 12 * 1024**3,
        )

    with pytest.raises(ValueError, match=r"200.0 GiB.*20.0 GiB"):
        await refit_on_cards_left(launch, measure)


@pytest.mark.parametrize("entry", ["null", '"broken"', '{"last":null,"expected":null}'])
def test_damaged_observations_do_not_block_a_start(tmp_path, entry):
    from eugene_plexus_agent.card_launch import DeviceMemory

    path = tmp_path / "runtime-devices.json"
    path.write_text('{"model":' + entry + "}")
    memory = DeviceMemory(path)
    assert memory.expected(spec()) == []
    assert memory.observed(spec()) == ([], None)
