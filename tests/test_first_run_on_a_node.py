"""An enrolled node is onboarded, and must say so.

Reported from a real two-machine install: signing in on the worker
bounced the operator into the first-run wizard. The node was fully
enrolled -- `Amish_Station`, epoch 1, `controlUrl` set -- and
`agent.yaml` still carried `firstRunComplete: false`.

**Two notions of "set up" had drifted apart.** `should_seed` already
treats `enrolled` as decisive and refuses to seed a control plane onto a
node, which is correct and invisible. But `firstRunComplete` was written
by exactly one thing -- the web wizard -- so an enrolled node kept
`false`, and the UI reads that flag alone. Completing that wizard on a
worker would have raised a SECOND install on a machine already belonging
to one.
"""

from __future__ import annotations

from typing import Any

from eugene_plexus_agent import default_topology
from eugene_plexus_agent.state import AgentState


def _state(tmp_path: Any) -> AgentState:
    state = AgentState(tmp_path / "agent.yaml")
    state.load()
    return state


def test_a_fresh_install_has_not_completed_first_run(tmp_path: Any) -> None:
    """The baseline, so the test below is not asserting a constant."""
    assert _state(tmp_path).get_config("firstRunComplete") is False


def test_marking_onboarded_records_it(tmp_path: Any) -> None:
    state = _state(tmp_path)

    assert default_topology.mark_onboarded(state) is True
    assert state.get_config("firstRunComplete") is True


def test_marking_onboarded_is_idempotent(tmp_path: Any) -> None:
    """It runs on every boot of every enrolled node, so it has to be
    free the second time -- and must report that it changed nothing, or
    the boot log would announce a repair on every restart."""
    state = _state(tmp_path)
    default_topology.mark_onboarded(state)

    assert default_topology.mark_onboarded(state) is False


def test_it_survives_a_reload_from_disk(tmp_path: Any) -> None:
    """Written through the ordinary patch path, so it persists. Poking
    the in-memory config would have passed every assertion above and
    left the operator seeing the wizard again after a restart."""
    path = tmp_path / "agent.yaml"
    first = AgentState(path)
    first.load()
    default_topology.mark_onboarded(first)

    second = AgentState(path)
    second.load()
    assert second.get_config("firstRunComplete") is True


def test_it_changes_nothing_else(tmp_path: Any) -> None:
    """`apply_config_patch` dumps the whole request, so a patch model
    that declared fields would reset every absent one to its default.
    `ConfigUpdateRequest` declares none and allows extras, which is what
    makes a one-key patch safe -- asserted rather than assumed, because
    the blast radius is the operator's entire agent config.

    (`uiTheme`'s enum is `light`/`dark`/`auto` -- a v0.2 field that
    matches none of the UI's real themes and that nothing reads. A first
    draft of this test used "modern" and was rejected by validation,
    which is at least proof the patch path validates.)"""
    state = _state(tmp_path)
    state.apply_config_patch(_patch(uiTheme="dark", uiFontSize="large"))

    default_topology.mark_onboarded(state)

    assert state.get_config("uiTheme") == "dark"
    assert state.get_config("uiFontSize") == "large"


def test_an_enrolled_node_does_not_get_a_control_plane(tmp_path: Any) -> None:
    """The half that already worked, pinned here beside the half that
    did not: the two decisions must keep agreeing about what 'set up'
    means."""
    state = _state(tmp_path)

    assert default_topology.should_seed(state, enrolled=True) is False
    assert default_topology.should_seed(state, enrolled=False) is True


def test_marking_onboarded_also_stops_the_seeder(tmp_path: Any) -> None:
    """A node that has been marked onboarded must not later seed itself
    a control plane if its enrollment is read as absent for a moment --
    which is what `should_seed`'s own `firstRunComplete` check is for."""
    state = _state(tmp_path)
    default_topology.mark_onboarded(state)

    assert default_topology.should_seed(state, enrolled=False) is False


def _patch(**kwargs: Any) -> Any:
    from eugene_plexus_agent._generated.common_models import ConfigUpdateRequest

    return ConfigUpdateRequest(**kwargs)
