"""R2-P1-4: elapsed time is never a turn boundary for the SDK stream.

An owner message queued behind an unclaimed CLI turn (a woken/injected burst) is dispatched only after
a real boundary. When the bound passes, the backstop nudges the transport and reports the queued status
once. The hold keeps the message until delivery or an explicit Stop discards it; the prompt never reaches
query() on the still-active stream (that would fold it into the injected turn mid-loop, reader at
agent/transports/claude_agent_sdk_session_turn.py ``_is_own_prompt_echo``).
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
RECOVERY = 0.3
SID = "owner-sdk"


class _WokenSdk:
    """An unclaimed CLI turn. ``ends_on_interrupt=n``: the n-th interrupt makes it reach its terminal
    result shortly after (the reader then reports the idle boundary); ``None``: it never ends."""

    def __init__(self, ends_on_interrupt=None):
        self.active = True
        self.interrupts = 0
        self.callback = None
        self.ends_on_interrupt = ends_on_interrupt

    def woken_turn_active(self):
        return self.active

    def interrupt_woken_turn(self):
        self.interrupts += 1
        if self.ends_on_interrupt is not None and self.interrupts >= self.ends_on_interrupt and self.active:
            def end():
                time.sleep(0.05)
                self.active = False
                if self.callback is not None:
                    self.callback()
            threading.Thread(target=end, daemon=True).start()
        return True

    def set_idle_boundary_callback(self, callback):
        self.callback = callback


def _session(sdk):
    return {
        "agent": SimpleNamespace(api_mode="claude_agent_sdk", _claude_sdk_session=sdk),
        "session_key": SID, "history": [], "history_lock": threading.RLock(), "history_version": 0,
        "running": False, "transport": None, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
    }


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(owner_hold, "TIMEOUT_S", BOUND)
    monkeypatch.setattr(owner_hold, "RECOVERY_TIMEOUT_S", RECOVERY, raising=False)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _s, **_kw: contextlib.nullcontext())
    monkeypatch.setattr(server, "_start_session_work", lambda target, **_kw: target())
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a: None)
    monkeypatch.setattr(mailbox, "schedule_drain", lambda *_a, **_kw: None)
    monkeypatch.setattr(mailbox, "drain_session", lambda *_a, **_kw: 0)
    h = SimpleNamespace(fired=[], emitted=[], sdk=None)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: h.emitted.append((event, sid, payload)))

    def run_prompt_submit(_rid, _sid, _session, prompt, **_kw):
        # What reaches query(): record whether the SDK stream was still inside the unclaimed turn.
        h.fired.append((prompt, h.sdk.woken_turn_active()))

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt_submit)
    yield h
    server._sessions.pop(SID, None)


def _submit(session, h, sdk, text):
    server._sessions[SID] = session
    h.sdk = sdk
    response = server.handle_request({"id": "r", "method": "prompt.submit",
                                      "params": {"session_id": SID, "text": text}})
    assert response["result"]["status"] == "queued"


def _errors(h):
    return [payload["message"] for event, sid, payload in h.emitted
            if event == "error" and sid == SID and isinstance(payload, dict)]


def test_idle_owner_prompt_dispatches_immediately(harness) -> None:
    sdk = _WokenSdk()
    sdk.active = False
    session = _session(sdk)
    server._sessions[SID] = session
    harness.sdk = sdk
    response = server.handle_request({"id": "idle", "method": "prompt.submit",
                                      "params": {"session_id": SID, "text": "owner reply"}})
    assert response["result"]["status"] == "streaming"
    assert harness.fired == [("owner reply", False)]


def test_owner_prompt_remains_queued_when_sdk_turn_has_no_result(harness) -> None:
    sdk = _WokenSdk(ends_on_interrupt=None)
    session = _session(sdk)
    _submit(session, harness, sdk, "ship the hotfix after CI")

    assert _wait_for(lambda: any(event == "status.update" for event, _sid, _payload in harness.emitted))
    time.sleep(BOUND + RECOVERY + 0.1)  # the recovery retry still must not delete or re-report it
    assert harness.fired == [], "the owner prompt was passed to query() on the still-active stream"
    assert _errors(harness) == []
    statuses = [payload for event, _sid, payload in harness.emitted if event == "status.update"]
    assert len(statuses) == 1 and statuses[0]["kind"] == "owner_queued"
    assert "queued behind a background turn" in statuses[0]["text"]
    assert sdk.interrupts >= 2, "the bound must first try explicit recovery (interrupt, then wait)"
    assert session["queued_prompt"]["text"] == "ship the hotfix after CI"
    assert mailbox._mailbox_auto_blocked(session) is True
    sdk.active = False
    assert sdk.callback is not None
    sdk.callback()
    assert harness.fired == [("ship the hotfix after CI", False)]


def test_resent_owner_messages_remain_in_order_and_deliver_once(harness) -> None:
    sdk = _WokenSdk()
    session = _session(sdk)
    _submit(session, harness, sdk, "first")
    _submit(session, harness, sdk, "second")
    # Plain text uses the existing lossless consecutive-user coalescing rule.
    assert session["queued_prompt"]["text"] == "first\n\nsecond"
    assert not session.get("queued_prompts")
    sdk.active = False
    assert sdk.callback is not None
    sdk.callback()
    assert harness.fired == [("first\n\nsecond", False)]
    sdk.callback()
    assert harness.fired == [("first\n\nsecond", False)]
    assert session.get("queued_prompt") is None


def test_owner_prompt_is_delivered_at_the_boundary_that_follows_the_recovery_interrupt(harness) -> None:
    sdk = _WokenSdk(ends_on_interrupt=2)  # the submit-time interrupt does not take; the recovery one does
    session = _session(sdk)
    _submit(session, harness, sdk, "and bump the version")

    assert _wait_for(lambda: harness.fired), "the owner message was never delivered"
    time.sleep(RECOVERY + 0.1)
    assert harness.fired == [("and bump the version", False)], "delivered only after the terminal result"
    assert sdk.interrupts == 2
    assert _errors(harness) == []
    assert "_owner_recovery_since" not in session
