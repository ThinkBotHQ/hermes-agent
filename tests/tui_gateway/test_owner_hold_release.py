"""The owner-input hold always releases, and an owner message is delivered or visibly failed.

Live 2026-09-30 (session 20260924_200208_c68a80): an owner forward queued at 00:14:31 behind a host turn
that never ended on its own result; peer rows 1108 and 1114 were refused "owner input pending or Stop hold
active" until 01:03:46 and 01:05:32. These are behaviour contracts for the hold, not for that one path.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from types import SimpleNamespace

import pytest

try:
    from tui_gateway import owner_hold
except ImportError:  # pragma: no cover - only on the pre-fix tree (red-before proof): no bound exists at all
    owner_hold = None
from tui_gateway import server
from tui_gateway import session_mailbox as mailbox

BOUND = 0.2


def _session(sdk=None, **extra):
    agent = SimpleNamespace(api_mode="claude_agent_sdk", _claude_sdk_session=sdk) if sdk else SimpleNamespace()
    return {
        "agent": agent, "session_key": "owner-hold", "history": [],
        "history_lock": threading.RLock(), "history_version": 0, "running": False,
        "transport": None, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", **extra,
    }


class _StuckWokenSdk:
    """An unclaimed CLI turn that never reports its boundary (its ResultMessage never arrives)."""

    def __init__(self):
        self.active = True
        self.interrupts = 0
        self.callback = None
        self.peer_sent = []

    def woken_turn_active(self):
        return self.active

    def interrupt_woken_turn(self):
        self.interrupts += 1
        return True

    def set_idle_boundary_callback(self, callback):
        self.callback = callback


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def bounded(monkeypatch):
    if owner_hold is not None:
        monkeypatch.setattr(owner_hold, "TIMEOUT_S", BOUND)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _s, **_kw: contextlib.nullcontext())
    monkeypatch.setattr(server, "_start_session_work", lambda target, **_kw: target())
    monkeypatch.setattr(mailbox, "schedule_drain", lambda *_a, **_kw: None)
    yield
    server._sessions.pop("owner-hold", None)


def test_stop_hold_with_nothing_to_deliver_releases_after_the_bound_with_a_log_line(bounded, caplog):
    session = _session()
    server._sessions["owner-hold"] = session
    server._interrupt_session_turn("owner-hold", session, hold_auto_started=True)
    assert mailbox._mailbox_auto_blocked(session), "Stop holds the auto-started chain at first"
    time.sleep(BOUND + 0.05)
    with caplog.at_level(logging.WARNING, logger="tui_gateway.owner_hold"):
        assert mailbox._mailbox_auto_blocked(session) is False
    assert session["_owner_stop_hold"] is False
    assert any("Stop hold" in r.getMessage() and "released" in r.getMessage() for r in caplog.records)


def test_owner_hold_set_by_an_unstamped_path_still_expires(bounded):
    session = _session(_owner_stop_hold=True)
    assert mailbox._mailbox_auto_blocked(session)
    time.sleep(BOUND + 0.05)
    assert mailbox._mailbox_auto_blocked(session) is False


def test_owner_submit_that_never_returns_stops_holding_after_the_bound(bounded):
    session = _session()
    owner_hold.owner_submit_started(session)
    assert mailbox._mailbox_auto_blocked(session)
    time.sleep(BOUND + 0.05)
    assert mailbox._mailbox_auto_blocked(session) is False
    owner_hold.owner_submit_finished(session)
    assert "_owner_submit_waiting" not in session and "_owner_submit_waiting_since" not in session


def test_owner_message_parked_behind_woken_turn_stays_queued(bounded, monkeypatch):
    """The recovery bound reports queue status without deleting the owner's words."""
    monkeypatch.setattr(owner_hold, "RECOVERY_TIMEOUT_S", BOUND)
    sdk = _StuckWokenSdk()
    session = _session(sdk)
    fired, emitted = [], []
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda _r, _sid, _s, text, **_kw: fired.append(text))
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr(mailbox, "drain_session", lambda *_a, **_kw: 0)
    server._sessions["owner-hold"] = session
    response = server.handle_request({"id": "r", "method": "prompt.submit", "params": {
        "session_id": "owner-hold", "text": "> Q5: b, re-arm on a fresh ledger"}})
    assert response["result"]["status"] == "queued"
    assert fired == [] and mailbox._mailbox_auto_blocked(session)
    assert _wait_for(lambda: any(event == "status.update" for event, _sid, _p in emitted))
    assert fired == []
    assert not any(event == "error" for event, _sid, _p in emitted)
    assert [p["kind"] for event, _sid, p in emitted if event == "status.update"] == ["owner_queued"]
    assert session["queued_prompt"]["text"] == "> Q5: b, re-arm on a fresh ledger"
    assert mailbox._mailbox_auto_blocked(session) is True


def test_owner_message_queued_behind_a_stalled_turn_keeps_peers_behind_it_then_runs_first(bounded, monkeypatch, caplog):
    """R2-P1-3: past the bound the owner message is overdue, but peers are still never admitted ahead of
    it (the bound changes how it is delivered, never who goes first)."""
    fired = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda _r, _sid, _s, text, **_kw: fired.append(text))
    session = _session(running=True)
    with session["history_lock"]:
        server._enqueue_prompt(session, "owner answer", None, display_kind="owner_forward")
        server._enqueue_prompt(session, "peer envelope", None, display_kind="peer_message")
    assert mailbox._mailbox_auto_blocked(session)
    time.sleep(BOUND + 0.05)
    with caplog.at_level(logging.WARNING, logger="tui_gateway.owner_hold"):
        assert mailbox._mailbox_auto_blocked(session) is True, "an overdue owner message still goes first"
    assert any("stays held behind it" in r.getMessage() for r in caplog.records)
    # The stalled turn finally ends: the owner message is still delivered, and ahead of the peer.
    session["running"] = False
    assert server._drain_queued_prompt("r", "owner-hold", session)
    assert fired == ["owner answer"]


def test_stop_reports_a_queued_owner_message_as_not_delivered(bounded, monkeypatch):
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    session = _session(running=True)
    with session["history_lock"]:
        server._enqueue_prompt(session, "please also check CI", None)
        server._enqueue_prompt(session, "peer envelope", None, display_kind="peer_message")
    server._interrupt_session_turn("owner-hold", session, hold_auto_started=True)
    assert session["queued_prompt"] is None and not session.get("queued_prompts")
    errors = [payload["message"] for event, sid, payload in emitted if event == "error" and sid == "owner-hold"]
    assert len(errors) == 1, "exactly the owner message is reported; the peer row stays durable in its mailbox"
    assert "please also check CI" in errors[0] and "Send it again" in errors[0]


def test_peer_mail_queued_under_the_hold_drains_after_release(bounded, monkeypatch, tmp_path):
    from hermes_state import SessionDB

    session = _session()
    db = SessionDB(tmp_path / "state.db")
    db.create_session("owner-hold", "desktop")
    db.create_session("sender", "desktop")
    submitted = []
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(mailbox, "_submit", lambda sid, text, **_kw: submitted.append(text) or {"result": {}})
    server._sessions["owner-hold"] = session
    try:
        server._interrupt_session_turn("owner-hold", session, hold_auto_started=True)
        sent = mailbox.send_message(target="owner-hold", body="peer while held", from_session_id="sender")
        assert sent["status"] == "queued"
        assert mailbox.drain_session("owner-hold") == 0
        assert db.peer_mailbox_get(sent["message_id"])["status"] == "queued"
        time.sleep(BOUND + 0.05)
        # Either the release backstop already drained it or the next drain does; both deliver exactly once.
        mailbox.drain_session("owner-hold")
        assert _wait_for(lambda: db.peer_mailbox_get(sent["message_id"])["status"] == "delivered")
        assert len(submitted) == 1
        assert submitted and "peer while held" in submitted[0]
    finally:
        db.close()


def test_the_backstop_drains_peer_mail_when_the_stop_hold_times_out(bounded, monkeypatch):
    drained = []
    monkeypatch.setattr(mailbox, "drain_session", lambda key, *_a, **_kw: drained.append(key) or 0)
    session = _session()
    server._sessions["owner-hold"] = session
    server._interrupt_session_turn("owner-hold", session, hold_auto_started=True)
    assert _wait_for(lambda: drained == ["owner-hold"]), "the release must drain the durable peer queue"
    assert session["_owner_stop_hold"] is False
