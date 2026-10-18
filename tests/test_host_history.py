"""Host history: the store, its rollups, the probes, the endpoint.

Specs `docs/design/host-history.md`. Probes run against fakes here; the
real NVML/PDH read was measured on the development box (see
`host_probes.py`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_agent import host_history, host_probes
from eugene_plexus_agent._generated.models import (
    ComputeDevice,
    ComputeDeviceKind,
    HostHistoryStep,
    HostHistoryStepChoice,
    HostMetric,
    HostUnit,
)
from eugene_plexus_agent.engines import gpu_probe
from eugene_plexus_agent.host_history import HostHistoryStore, HostSampler, read_settings
from eugene_plexus_agent.host_probes import (
    Folder,
    HostProbe,
    NvmlReading,
    Reading,
    RuntimeCounts,
    WindowsGpuCounters,
)

from .conftest import fake_devices

# 2026-10-18 12:00:00 UTC, on the hour.
T0 = 1_792_324_800
DEFAULTS = read_settings(lambda key: None)


def _gpu_used(value: float | None, reason: str | None = None) -> Reading:
    return Reading(
        key="gpu:cuda:0:gpuMemoryUsed",
        metric=HostMetric.gpuMemoryUsed,
        unit=HostUnit.bytes,
        label="Fake GPU 0 memory used",
        value=value,
        unavailable=reason,
        device_kind=ComputeDeviceKind.cuda,
        device_index=0,
        device_name="Fake GPU 0",
    )


def _temp(value: float | None, reason: str | None = None) -> Reading:
    return Reading(
        key="gpu:cuda:0:gpuTemperature",
        metric=HostMetric.gpuTemperature,
        unit=HostUnit.celsius,
        label="Fake GPU 0 temperature",
        value=value,
        unavailable=reason,
        device_kind=ComputeDeviceKind.cuda,
        device_index=0,
    )


@pytest.fixture
def store(tmp_path: Path) -> HostHistoryStore:
    s = HostHistoryStore(tmp_path / host_history.FILE_NAME)
    assert s.open()
    return s


def _series(history: Any, key: str) -> Any:
    return next(s for s in history.series if s.key == key)


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


def test_raw_window_has_every_bucket_and_shows_a_gap(store: HostHistoryStore) -> None:
    # Samples every 10 s for the first two minutes, then nothing for a minute.
    for i in range(12):
        store.record(T0 + i * 10, 10, [_gpu_used(1000.0 + i)])
    history = store.query(
        now=T0 + 180, settings=DEFAULTS, start=T0, end=T0 + 180, step=HostHistoryStepChoice.raw
    )
    assert history.step is HostHistoryStep.raw
    assert history.stepSeconds == 10
    assert len(history.buckets) == 18
    assert [b.samples for b in history.buckets] == [1] * 12 + [0] * 6
    used = _series(history, "gpu:cuda:0:gpuMemoryUsed")
    assert used.mean[:3] == [1000.0, 1001.0, 1002.0]
    assert used.mean[12:] == [None] * 6
    assert used.latest == 1011.0
    assert history.latestAt == datetime.fromtimestamp(T0 + 110, tz=UTC)


def test_minute_rollup_is_min_mean_max_and_counts_samples(store: HostHistoryStore) -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]  # one sample missed in the minute
    for i, v in enumerate(values):
        store.record(T0 + i * 10, 10, [_gpu_used(v)])
    history = store.query(
        now=T0 + 120, settings=DEFAULTS, start=T0, end=T0 + 120, step=HostHistoryStepChoice.minute
    )
    first = history.buckets[0]
    assert (first.samples, first.expectedSamples) == (5, 6)  # partial
    assert history.buckets[1].samples == 0  # gap
    used = _series(history, "gpu:cuda:0:gpuMemoryUsed")
    assert (used.min[0], used.mean[0], used.max[0]) == (10.0, 30.0, 50.0)


def test_hour_rollup_weights_minutes_by_their_samples(store: HostHistoryStore) -> None:
    # Minute 0: six samples of 10. Minute 1: two samples of 40.
    for i in range(6):
        store.record(T0 + i * 10, 10, [_gpu_used(10.0)])
    for i in range(2):
        store.record(T0 + 60 + i * 10, 10, [_gpu_used(40.0)])
    history = store.query(
        now=T0 + 3600, settings=DEFAULTS, start=T0, end=T0 + 3600, step=HostHistoryStepChoice.hour
    )
    assert history.buckets[0].samples == 8
    assert history.buckets[0].expectedSamples == 360
    used = _series(history, "gpu:cuda:0:gpuMemoryUsed")
    assert used.mean[0] == 18.0  # (6*10 + 2*40) / 8, not (10+40)/2


def test_a_series_this_host_cannot_read_is_null_with_its_reason(store: HostHistoryStore) -> None:
    reason = "Windows publishes no temperature or power for this card"
    for i in range(3):
        store.record(T0 + i * 10, 10, [_gpu_used(5.0), _temp(None, reason)])
    history = store.query(now=T0 + 30, settings=DEFAULTS, start=T0, end=T0 + 30)
    temp = _series(history, "gpu:cuda:0:gpuTemperature")
    assert temp.unavailable == reason
    assert temp.mean == [None, None, None]
    assert temp.latest is None


def test_a_reading_that_stops_shows_null_not_zero(store: HostHistoryStore) -> None:
    store.record(T0, 10, [_temp(40.0)])
    store.record(T0 + 10, 10, [_temp(None, "NVML: GPU is lost")])
    history = store.query(now=T0 + 20, settings=DEFAULTS, start=T0, end=T0 + 20)
    temp = _series(history, "gpu:cuda:0:gpuTemperature")
    assert temp.mean == [40.0, None]
    assert temp.unavailable == "NVML: GPU is lost"


def test_metric_filter(store: HostHistoryStore) -> None:
    store.record(T0, 10, [_gpu_used(5.0), _temp(30.0)])
    history = store.query(
        now=T0 + 10, settings=DEFAULTS, start=T0, end=T0 + 10, metrics=[HostMetric.gpuTemperature]
    )
    assert [s.metric for s in history.series] == [HostMetric.gpuTemperature]


def test_a_gone_series_shows_only_while_the_window_holds_its_data(
    store: HostHistoryStore,
) -> None:
    store.record(T0, 10, [_gpu_used(5.0), _temp(30.0)])
    store.record(T0 + 10, 10, [_gpu_used(6.0)])  # the card's sensor series is gone
    later = store.query(now=T0 + 20, settings=DEFAULTS, start=T0 + 10, end=T0 + 20)
    assert [s.key for s in later.series] == ["gpu:cuda:0:gpuMemoryUsed"]
    earlier = store.query(now=T0 + 20, settings=DEFAULTS, start=T0, end=T0 + 20)
    assert {s.key for s in earlier.series} == {
        "gpu:cuda:0:gpuMemoryUsed",
        "gpu:cuda:0:gpuTemperature",
    }


def test_auto_step_follows_the_window() -> None:
    now = float(T0)

    def pick(seconds: float) -> HostHistoryStep:
        step, _ = host_history.choose_step(
            now - seconds, now, now, HostHistoryStepChoice.auto, DEFAULTS
        )
        return step

    assert pick(3600) is HostHistoryStep.raw
    assert pick(6 * 3600) is HostHistoryStep.minute
    assert pick(24 * 3600) is HostHistoryStep.minute
    assert pick(7 * 86400) is HostHistoryStep.hour
    assert pick(30 * 86400) is HostHistoryStep.hour


def test_window_refusals(store: HostHistoryStore) -> None:
    with pytest.raises(host_history.HistoryWindowError):
        store.query(now=T0, settings=DEFAULTS, start=T0, end=T0)
    with pytest.raises(host_history.HistoryWindowError, match="more than 5000"):
        store.query(
            now=T0, settings=DEFAULTS, start=T0 - 86400, end=T0, step=HostHistoryStepChoice.raw
        )


def test_retention_prunes_each_resolution_and_marks_truncation(store: HostHistoryStore) -> None:
    store.record(T0, 10, [_gpu_used(1.0)])
    later = T0 + 7 * 3600
    store.record(later, 10, [_gpu_used(2.0)])
    store.prune(later, DEFAULTS.retention)
    raw = store.query(
        now=later, settings=DEFAULTS, start=T0, end=later + 10, step=HostHistoryStepChoice.minute
    )
    # Minutes are kept for 7 days, so the first sample survives there.
    assert _series(raw, "gpu:cuda:0:gpuMemoryUsed").mean[0] == 1.0
    just_raw = store.query(
        now=later, settings=DEFAULTS, start=later - 600, end=later + 10,
        step=HostHistoryStepChoice.raw,
    )  # fmt: skip
    assert just_raw.recordingSince == datetime.fromtimestamp(later, tz=UTC)
    old = store.query(
        now=later, settings=DEFAULTS, start=T0, end=T0 + 600, step=HostHistoryStepChoice.raw
    )
    assert old.truncated is True


def test_clear_deletes_everything(store: HostHistoryStore) -> None:
    store.record(T0, 10, [_gpu_used(1.0)])
    store.clear()
    history = store.query(now=T0 + 10, settings=DEFAULTS, start=T0, end=T0 + 10)
    assert history.series == []
    assert history.latestAt is None
    store.record(T0 + 10, 10, [_gpu_used(2.0)])
    assert store.query(now=T0 + 20, settings=DEFAULTS, start=T0, end=T0 + 20).series


def test_store_that_cannot_open_names_the_cause(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    s = HostHistoryStore(blocker / host_history.FILE_NAME)
    assert s.open() is False
    assert s.error is not None and str(blocker) in s.error


def test_settings_are_clamped() -> None:
    values = {"hostHistorySampleSeconds": 0, "hostHistoryEnabled": False, "hostHistoryRawHours": 2}
    settings = read_settings(values.get)
    assert settings.sample_seconds == host_history.MIN_SAMPLE_SECONDS
    assert settings.enabled is False
    assert settings.retention.raw_seconds == 7200


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #


class _FixedProbe:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self) -> list[Reading]:
        self.calls += 1
        return [_gpu_used(float(self.calls))]

    def close(self) -> None:
        pass


def test_sampler_aligns_to_the_grid_and_skips_when_off(store: HostHistoryStore) -> None:
    config: dict[str, Any] = {}
    probe = _FixedProbe()
    clock = [T0 + 13.7]
    sampler = HostSampler(store, lambda: probe, config.get, clock=lambda: clock[0])  # type: ignore[arg-type,return-value]
    sampler.tick()
    history = store.query(now=T0 + 20, settings=DEFAULTS, start=T0, end=T0 + 20)
    assert history.latestAt == datetime.fromtimestamp(T0 + 10, tz=UTC)
    config["hostHistoryEnabled"] = False
    clock[0] = T0 + 21
    sampler.tick()
    assert probe.calls == 1


# --------------------------------------------------------------------------- #
# Probe parsing
# --------------------------------------------------------------------------- #


def test_engine_utilization_is_the_busiest_engine_summed_over_processes() -> None:
    luid = "luid_0x00000000_0x0000d46e"
    values = {
        f"pid_10_{luid}_phys_0_eng_0_engtype_3D": 30.0,
        f"pid_11_{luid}_phys_0_eng_0_engtype_3D": 25.0,
        f"pid_12_{luid}_phys_0_eng_1_engtype_Compute_0": 70.0,
        f"pid_13_{luid}_phys_0_eng_1_engtype_Compute_0": 60.0,  # 130: clamped
        "pid_14_luid_0x00000000_0x0000e96d_phys_0_eng_0_engtype_3D": 5.0,
    }
    found = host_probes.parse_engine_utilization(values)
    assert found[luid] == 100.0
    assert found["luid_0x00000000_0x0000e96d"] == 5.0


def test_process_memory_by_adapter_and_pid() -> None:
    luid = "luid_0x00000000_0x0000d46e"
    found = host_probes.parse_process_memory(
        {f"pid_10_{luid}_phys_0": 100, f"pid_11_{luid}_phys_0": 7}, {f"pid_10_{luid}_phys_0": 5}
    )
    assert found[luid] == {10: (100, 5), 11: (7, 0)}


def test_fdinfo_parses_both_kernel_spellings() -> None:
    old = "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t7\ndrm-memory-vram:\t2048 KiB\n"
    new = "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t8\ndrm-resident-vram:\t1 MiB\n"
    assert host_probes.parse_fdinfo(old) == ("amdgpu", "0000:03:00.0", "7", 2048 * 1024, 0)
    assert host_probes.parse_fdinfo(new)[3] == 1024 * 1024


def test_amdgpu_process_memory_counts_each_client_once(tmp_path: Path) -> None:
    fdinfo = tmp_path / "42" / "fdinfo"
    fdinfo.mkdir(parents=True)
    text = "drm-driver:\tamdgpu\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t7\ndrm-memory-vram:\t1024 KiB\n"
    (fdinfo / "5").write_text(text)
    (fdinfo / "6").write_text(text)  # the same client on a second fd
    (fdinfo / "7").write_text(text.replace("0000:03:00.0", "0000:04:00.0"))  # another card
    assert (
        host_probes.amdgpu_process_memory([42, 999], "0000:03:00.0", shared=False, proc=tmp_path)
        == 1024 * 1024
    )


def test_cards_match_adapters_by_name_in_order() -> None:
    def device(index: int, name: str) -> ComputeDevice:
        return ComputeDevice(kind=ComputeDeviceKind.vulkan, index=index, name=name)

    def adapter(name: str, key: str) -> gpu_probe.Adapter:
        return gpu_probe.Adapter(
            name=name, vendor="amd", integrated=False, dedicated_bytes=1, shared_bytes=None, key=key
        )

    matched = host_probes.match_adapters(
        [device(0, "Twin"), device(1, "Twin"), device(2, "Other")],
        [adapter("Twin", "a"), adapter("twin", "b")],
    )
    assert [m.adapter.key if m.adapter else None for m in matched] == ["a", "b", None]


# --------------------------------------------------------------------------- #
# The probe on fake platforms
# --------------------------------------------------------------------------- #


class _FakeNvml:
    loaded = True
    error = None

    def __init__(self, processes: dict[int, int] | str) -> None:
        self._processes = processes

    def read(self, index: int) -> NvmlReading:
        return NvmlReading(
            memory_used=8 * 1024**3,
            memory_total=32 * 1024**3,
            utilization=55.0,
            temperature=61.0,
            power=312.5,
        )

    def process_memory(self, index: int) -> dict[int, int] | str:
        return self._processes

    def close(self) -> None:
        pass


def _values(readings: list[Reading]) -> dict[str, Reading]:
    return {r.key: r for r in readings}


def test_linux_nvidia_reads_nvml_and_eugenes_share_by_process(tmp_path: Path) -> None:
    probe = HostProbe(
        devices=lambda: fake_devices(),
        adapters=lambda: [],
        runtimes=lambda: RuntimeCounts(running=1, loading=0, pids=(4242,)),
        folders=lambda: [Folder("Engines", str(tmp_path))],
        nvml=lambda: _FakeNvml({4242: 6 * 1024**3, 7: 1024**3}),  # type: ignore[arg-type,return-value]
        platform="linux",
    )
    got = _values(probe.sample())
    assert got["gpu:cuda:0:gpuMemoryUsed"].value == 8 * 1024**3
    assert got["gpu:cuda:0:gpuMemoryEugene"].value == 6 * 1024**3
    assert got["gpu:cuda:0:gpuUtilization"].value == 55.0
    assert got["gpu:cuda:0:gpuTemperature"].value == 61.0
    assert got["gpu:cuda:0:gpuPower"].value == 312.5
    assert got["runtimesRunning"].value == 1.0
    disks = [r for r in got.values() if r.metric is HostMetric.diskFree]
    assert len(disks) == 1 and disks[0].value is not None
    assert disks[0].holds == (f"Engines: {tmp_path}",)
    probe.close()


def test_a_driver_without_per_process_memory_says_so() -> None:
    reason = "this driver does not report each process's memory"
    probe = HostProbe(
        devices=lambda: fake_devices(),
        adapters=lambda: [],
        runtimes=lambda: RuntimeCounts(0, 0),
        folders=lambda: [],
        nvml=lambda: _FakeNvml(reason),  # type: ignore[arg-type,return-value]
        platform="linux",
    )
    eugene = _values(probe.sample())["gpu:cuda:0:gpuMemoryEugene"]
    assert eugene.value is None and eugene.unavailable == reason
    probe.close()


class _FakePdh:
    def __init__(self, counters: WindowsGpuCounters) -> None:
        self._counters = counters

    def collect(self) -> WindowsGpuCounters:
        return self._counters

    def close(self) -> None:
        pass


def test_windows_card_without_nvml_uses_pdh_and_names_missing_sensors() -> None:
    luid = "luid_0x00000000_0x0000d46e"
    devices = fake_devices()
    vulkan = ComputeDevice(
        kind=ComputeDeviceKind.vulkan, index=0, name="Radeon", memoryTotalBytes=16 * 1024**3
    )
    snapshot = type(devices)(
        devices=(vulkan, *devices.devices[1:]),
        warnings=(),
        ram_total_bytes=devices.ram_total_bytes,
        ram_available_bytes=devices.ram_available_bytes,
        detected_at=devices.detected_at,
    )
    adapter = gpu_probe.Adapter(
        name="Radeon", vendor="amd", integrated=False, dedicated_bytes=16 * 1024**3,
        shared_bytes=None, key=luid,
    )  # fmt: skip
    counters = WindowsGpuCounters(
        utilization={luid: 42.0},
        adapter_dedicated={luid: 4 * 1024**3},
        adapter_shared={luid: 0},
        processes={luid: {4242: (3 * 1024**3, 0), 99: (1024**3, 0)}},
    )
    probe = HostProbe(
        devices=lambda: snapshot,
        adapters=lambda: [adapter],
        runtimes=lambda: RuntimeCounts(1, 0, (4242,)),
        folders=lambda: [],
        windows=lambda: _FakePdh(counters),  # type: ignore[arg-type,return-value]
        platform="win32",
    )
    got = _values(probe.sample())
    assert got["gpu:vulkan:0:gpuMemoryUsed"].value == 4 * 1024**3
    assert got["gpu:vulkan:0:gpuMemoryEugene"].value == 3 * 1024**3
    assert got["gpu:vulkan:0:gpuUtilization"].value == 42.0
    temp = got["gpu:vulkan:0:gpuTemperature"]
    assert temp.value is None and temp.unavailable and "Windows publishes no" in temp.unavailable
    probe.close()


# --------------------------------------------------------------------------- #
# The endpoint and the settings
# --------------------------------------------------------------------------- #


def _seed(app: FastAPI) -> HostHistoryStore:
    s: HostHistoryStore = app.state.host_history
    assert s.open()
    now = int(datetime.now(UTC).timestamp()) // 10 * 10
    for i in range(3):
        s.record(now - 30 + i * 10, 10, [_gpu_used(100.0 + i), _temp(None, "no sensor")])
    return s


def test_endpoint_returns_the_window(authed_client: TestClient, app: FastAPI) -> None:
    _seed(app)
    response = authed_client.get("/v1/node/history", params={"metric": ["gpuMemoryUsed"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["step"] == "raw" and body["enabled"] is True and "from" in body
    [used] = body["series"]
    assert used["metric"] == "gpuMemoryUsed"
    assert used["latest"] == 102.0
    # Nulls inside the arrays survive `exclude_none`.
    assert None in used["mean"] and len(used["mean"]) == len(body["buckets"])


def test_endpoint_refuses_a_backwards_window(authed_client: TestClient, app: FastAPI) -> None:
    response = authed_client.get(
        "/v1/node/history",
        params={"from": "2026-10-18T12:00:00Z", "to": "2026-10-18T11:00:00Z"},
    )
    assert response.status_code == 400


def test_endpoint_needs_a_sign_in(client: TestClient) -> None:
    assert client.get("/v1/node/history").status_code == 401


def test_delete_empties_the_store(authed_client: TestClient, app: FastAPI) -> None:
    _seed(app)
    assert authed_client.delete("/v1/node/history").status_code == 204
    assert authed_client.get("/v1/node/history").json()["series"] == []


def test_settings_are_in_the_schema_with_bounds(authed_client: TestClient) -> None:
    fields = {f["key"]: f for f in authed_client.get("/v1/config/schema").json()["fields"]}
    assert fields["hostHistoryEnabled"]["default"] is True
    assert fields["hostHistoryRawHours"]["status"]["level"] == "info"
    refused = authed_client.patch("/v1/config", json={"hostHistorySampleSeconds": 1}).json()
    assert refused["rejected"][0]["key"] == "hostHistorySampleSeconds"
    applied = authed_client.patch("/v1/config", json={"hostHistorySampleSeconds": 30}).json()
    assert "hostHistorySampleSeconds" in applied["applied"]
