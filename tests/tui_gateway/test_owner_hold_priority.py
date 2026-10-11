"""R2-P1-3: an expired owner hold never hands peers priority over the queued owner message.

While an owner message is queued, peer mail is never admitted ahead of it, expired or not: the bound
only changes HOW the owner message is delivered (the backstop), and recovery claims the session for
the owner atomically before peers drain. A queue discard (Stop, failure, delivery) resets the hold
clock, so a fresh owner message always gets a fresh deadline.
"""

from __future__ import annotations

import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

from tui_gateway import owner_hold
from tui_gateway import server
from tui_gateway import session_mailbox as mailbox

BOUND = 0.2


def _session(**extra):
    return {
        "agent": SimpleNamespace(), "session_key": "owner-prio", "history": [],
        "history_lock": threading.RLock(), "history_version": 0, "running": False,
        "transport": None, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", **extra,
    }


@pytest.fixture
def bounded(monkeypatch):
    monkeypatch.setattr(owner_hold, "TIMEOUT_S", BOUND)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _s, **_kw: contextlib.nullcontext())
    monkeypatch.setattr(server, "_start_session_work", lambda target, **_kw: target())
    drains: list = []
    monkeypatch.setattr(mailbox, "schedule_drain", lambda *a, **_kw: drains.append(a))
    yield drains
    server._sessions.pop("owner-prio", None)


def test_expired_owner_hold_never_admits_a_concurrent_peer_ahead_of_the_owner(bounded, monkeypatch) -> None:
    order: list = []
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda _r, _sid, _s, text, **_kw: order.append(("owner", text)))
    session = _session(running=True)
    with session["history_lock"]:
        server._enqueue_prompt(session, "owner answer", None, display_kind="owner_forward")
    time.sleep(BOUND + 0.05)

    # The bound passed while the turn ran: the owner message is overdue, peers are STILL held.
    assert mailbox._mailbox_auto_blocked(session) is True
    assert bounded == [], "no release drain may be scheduled while the owner message is queued"

    # The turn releases. A mailbox drain wins the race to the idle session before the owner drain runs.
    session["running"] = False
    if mailbox._native_peer_idle(session, None):
        order.append(("peer", "native"))
    assert mailbox._mailbox_auto_blocked(session) is True

    assert server._drain_queued_prompt("r", "owner-prio", session)
    assert order == [("owner", "owner answer")]
    # The owner turn holds the session now; peers still wait for its end, not for the clock.
    assert mailbox._native_peer_idle(session, None) is False


def test_backstop_claims_the_session_for_the_owner_before_peers_drain(bounded, monkeypatch) -> None:
    order: list = []
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda _r, _sid, _s, text, **_kw: order.append(("owner", text, _s.get("running"))))
    monkeypatch.setattr(mailbox, "drain_session", lambda key, *_a, **_kw: order.append(("peers", key)) or 0)
    session = _session(running=True)
    server._sessions["owner-prio"] = session
    with session["history_lock"]:
        server._enqueue_prompt(session, "owner answer", None, display_kind="owner_forward")
        owner_hold.cancel_backstop(session)  # this test drives the backstop itself
    time.sleep(BOUND + 0.05)
    session["running"] = False  # the turn ended without its post-turn drain

    owner_hold.run_backstop(session)

    assert order and order[0] == ("owner", "owner answer", True), order
    assert not any(item[0] == "peers" for item in order), "peers drain after the owner turn, never before"


def test_stop_resets_the_owner_hold_clock_for_the_next_owner_message(bounded) -> None:
    session = _session(running=True)
    with session["history_lock"]:
        server._enqueue_prompt(session, "first", None)
    time.sleep(BOUND + 0.05)
    assert owner_hold.overdue_owner_entry(session) is True

    server._interrupt_session_turn("owner-prio", session, hold_auto_started=True)
    assert "_owner_queued_since" not in session, "Stop discarded the queue: its hold clock goes with it"

    before = time.time()
    with session["history_lock"]:
        server._enqueue_prompt(session, "second", None)
    assert float(session["_owner_queued_since"]) >= before
    assert owner_hold.overdue_owner_entry(session) is False
    with session["history_lock"]:
        assert owner_hold.owner_pending(session) is True


def test_a_queue_discarded_without_the_helper_still_restarts_the_clock(bounded) -> None:
    """Defence in depth: any path that bumps the queue generation (compress re-anchor, …) also
    restarts the clock, even if it never calls the reset helper."""
    session = _session(running=True)
    with session["history_lock"]:
        server._enqueue_prompt(session, "first", None)
    time.sleep(BOUND + 0.05)
    with session["history_lock"]:
        session["queued_prompt"] = None
        session.pop("queued_prompts", None)
        session["_queued_prompt_generation"] = int(session.get("_queued_prompt_generation", 0)) + 1
        server._enqueue_prompt(session, "second", None)
    assert owner_hold.overdue_owner_entry(session) is False
