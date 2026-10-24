"""Recorded inventory shapes for Strata tests; never probe hardware."""

from dataclasses import replace

from eugene_plexus_agent._generated.models import ComputeDevice

from .conftest import fake_devices


def cuda(gib: int, *, index: int) -> ComputeDevice:
    return ComputeDevice(
        kind="cuda", index=index, memoryTotalBytes=gib * 2**30, memoryFreeBytes=gib * 2**30
    )


def snap(*cards: ComputeDevice):
    return replace(fake_devices(), devices=cards)
