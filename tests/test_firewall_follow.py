"""This install's own firewall rule follows its listeners (Troy, 2026-10-10).

Strata's driver on Amish_Station listened on 8093 and the rule the person
had allowed named 8079 and 8091, so the NAS's gateway could not reach it.
"""

from __future__ import annotations

from eugene_plexus_agent.firewall_follow import follow_once, wanted_ports
from eugene_plexus_agent.reach import Listener

LISTENING = [
    Listener(process="agent", port=8079, bind_host="0.0.0.0"),
    Listener(process="inference-driver", port=8093, bind_host="0.0.0.0"),
    Listener(process="library", port=8082, bind_host="127.0.0.1"),
    Listener(process="gateway", port=8080, bind_host=None),
    Listener(process="app:workbench", port=8101, bind_host="192.168.16.75"),
]


class _Firewall:
    def __init__(self, ports, *, elevated=True, works=True):
        self.ports, self._elevated, self.works, self.sets = ports, elevated, works, []

    def elevated(self):
        return self._elevated

    def rule_ports(self):
        return self.ports

    def set_rule_ports(self, ports):
        self.sets.append(ports)
        if self.works:
            self.ports = ports
            return True, f"Eugene Plexus now allows TCP {','.join(map(str, ports))}."
        return False, "Set-NetFirewallRule failed: access denied"


def test_only_what_listens_off_loopback():
    assert wanted_ports(LISTENING) == (8079, 8093, 8101)


def test_the_rule_the_person_allowed_follows_what_listens():
    firewall = _Firewall((8079, 8091))
    assert (
        follow_once(LISTENING, firewall=firewall) == "Eugene Plexus now allows TCP 8079,8093,8101."
    )
    assert firewall.sets == [(8079, 8093, 8101)]
    # Once it does, nothing more is done.
    assert follow_once(LISTENING, firewall=firewall) is None
    assert len(firewall.sets) == 1


def test_no_rule_is_made_that_the_person_did_not_allow():
    firewall = _Firewall(None)
    assert follow_once(LISTENING, firewall=firewall) is None
    assert firewall.sets == []


def test_unelevated_it_changes_nothing():
    firewall = _Firewall((8079,), elevated=False)
    assert follow_once(LISTENING, firewall=firewall) is None
    assert firewall.sets == []


def test_with_nothing_off_loopback_the_rule_is_left_alone():
    firewall = _Firewall((8079,))
    quiet = [Listener(process="agent", port=8079, bind_host="127.0.0.1")]
    assert follow_once(quiet, firewall=firewall) is None
    assert firewall.sets == []


def test_a_failure_is_said_and_tried_again_next_time():
    firewall = _Firewall((8079,), works=False)
    assert "access denied" in (follow_once(LISTENING, firewall=firewall) or "")
    assert "access denied" in (follow_once(LISTENING, firewall=firewall) or "")
    assert len(firewall.sets) == 2


def test_off_windows_nothing_is_read():
    assert follow_once(LISTENING, platform="linux") is None
    assert follow_once(LISTENING, platform="darwin") is None
