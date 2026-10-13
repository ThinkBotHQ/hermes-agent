"""Native /steer on the claude-agent-sdk lane.

Background: the SDK owns tool execution, so Hermes' tool-batch drain points are
never reached on this lane and a /steer stranded in ``_pending_steer`` until the
turn finalizer handed it back — which the gateway only redelivers when nothing
else is queued. Steers were silently lost.

The fix routes an in-flight steer through ``ClaudeSDKClient.query()``, the SDK's
own streaming-input contract. These tests pin the two properties that would
regress silently:

  1. EXACTLY-ONCE — an accepted native steer must not ALSO be stashed, or the
     model sees it twice (once injected, once redelivered as the next turn).
  2. FALL-BACK — when there is no live turn to steer into, the native path must
     decline so the ordinary stash still runs. Declining must not swallow.
"""

from __future__ import annotations

import asyncio
import threading
import types


# --------------------------------------------------------------------------
# run_agent.steer() routing
# --------------------------------------------------------------------------

def _agent(api_mode="claude_agent_sdk", sdk_session=None):
    return types.SimpleNamespace(
        api_mode=api_mode,
        _claude_sdk_session=sdk_session,
        _pending_steer=None,
        _pending_steer_lock=None,   # exercises the no-lock stub branch
    )


def _session(result):
    """Fake transport whose steer() returns `result` or raises if it's an Exception."""
    calls: list[str] = []

    def _steer(text):
        calls.append(text)
        if isinstance(result, Exception):
            raise result
        return result

    return types.SimpleNamespace(steer=_steer, calls=calls)


def test_accepted_native_steer_is_not_also_stashed():
    """Exactly-once: native delivery suppresses the pending-steer stash."""
    from run_agent import AIAgent

    sess = _session(True)
    agent = _agent(sdk_session=sess)

    assert AIAgent.steer(agent, "  turn left  ") is True
    assert sess.calls == ["turn left"], "text should reach the transport stripped"
    assert agent._pending_steer is None, (
        "an accepted native steer must NOT also be stashed — the turn finalizer "
        "would redeliver it as a second user turn"
    )


def test_declined_native_steer_falls_back_to_stash():
    """No live turn -> transport declines -> ordinary stash still happens."""
    from run_agent import AIAgent

    sess = _session(False)
    agent = _agent(sdk_session=sess)

    assert AIAgent.steer(agent, "turn right") is True
    assert sess.calls == ["turn right"]
    assert agent._pending_steer == "turn right", "declined steer must not be lost"


def test_native_steer_exception_falls_back_to_stash():
    """A raising transport must degrade to the stash, never drop the text."""
    from run_agent import AIAgent

    sess = _session(RuntimeError("client gone"))
    agent = _agent(sdk_session=sess)

    assert AIAgent.steer(agent, "still important") is True
    assert agent._pending_steer == "still important"


def test_non_sdk_lane_never_consults_the_transport():
    """Other lanes keep the old behaviour byte-for-byte."""
    from run_agent import AIAgent

    sess = _session(True)
    agent = _agent(api_mode="chat_completions", sdk_session=sess)

    assert AIAgent.steer(agent, "hello") is True
    assert sess.calls == [], "non-SDK lanes must not reach the SDK transport"
    assert agent._pending_steer == "hello"


def test_empty_steer_rejected_before_any_routing():
    from run_agent import AIAgent

    sess = _session(True)
    agent = _agent(sdk_session=sess)

    assert AIAgent.steer(agent, "   ") is False
    assert sess.calls == []
    assert agent._pending_steer is None


# --------------------------------------------------------------------------
# transport-level steer()
# --------------------------------------------------------------------------

def _transport(turn_inbox, client=True, loop=True):
    from agent.transports import claude_agent_sdk_session as mod

    queried: list[object] = []
    fake_client = types.SimpleNamespace(query=lambda t: queried.append(t))
    stub = types.SimpleNamespace(
        _turn_inbox=turn_inbox,
        _turn_callback_lock=threading.RLock(),
        _turn_claim_requested=False,
        _rename_claim_requested=False,
        _unsolicited_burst_open=False,
        _native_peer_in_flight=False,
        _closed=False,
        _retiring=False,
        _stream_ended=None,
        _client=fake_client if client else None,
        _loop=object() if loop else None,
        _interrupt_commit_lock=threading.Lock(),
        _pending_steer_results=0,
        _terminal_result_committed=False,
        is_live=lambda: True,
    )
    stub._native_peer_since = 0.0
    stub._native_peer_msg_id = None
    stub._native_peer_seen = False
    stub._native_peer_gen = 0
    stub._native_peer_written = False
    stub.idle_notifications = []
    stub._notify_idle_boundary = lambda: stub.idle_notifications.append(True)
    for name in ("native_peer_idle", "woken_turn_active", "_native_peer_pending_locked",
                 "_clear_native_peer_locked", "_expire_native_peer"):
        setattr(stub, name, types.MethodType(getattr(mod.ClaudeAgentSdkSession, name), stub))
    return mod, stub, queried


def test_transport_declines_when_no_turn_is_in_flight(monkeypatch):
    """Without a claimed turn there is nothing to steer INTO.

    Sending anyway would open an unclaimed turn whose output the reader routes
    to the unsolicited path — a reply appearing from nowhere.
    """
    mod, stub, queried = _transport(turn_inbox=None)
    called = []
    monkeypatch.setattr(
        mod.asyncio, "run_coroutine_threadsafe",
        lambda *a, **kw: called.append(a), raising=False,
    )

    assert mod.ClaudeAgentSdkSession.steer(stub, "mid-turn note") is False
    assert queried == [] and called == []


def test_transport_schedules_query_on_a_live_turn(monkeypatch):
    """With a turn claimed, the steer is scheduled onto the session loop."""
    mod, stub, queried = _transport(turn_inbox=object())
    scheduled = []

    class _Fut:
        def add_done_callback(self, cb):
            scheduled.append(cb)

    monkeypatch.setattr(
        mod.asyncio, "run_coroutine_threadsafe",
        lambda coro, loop: _Fut(), raising=False,
    )

    assert mod.ClaudeAgentSdkSession.steer(stub, "  actually, stop  ") is True
    assert len(queried) == 1
    async def collect(stream):
        return [message async for message in stream]

    assert asyncio.run(collect(queried[0])) == [{
        "type": "user",
        "message": {"role": "user", "content": "actually, stop"},
        "parent_tool_use_id": None,
        "origin": {"kind": "human"},
    }]
    assert len(scheduled) == 1, "future must carry a done-callback so the "
    "exception is retrieved, not logged at teardown"


def test_peer_message_uses_sdk_stream_with_peer_origin_when_idle(monkeypatch):
    mod, stub, queried = _transport(turn_inbox=None)
    scheduled = []

    class _Fut:
        def add_done_callback(self, cb):
            scheduled.append(cb)
        def result(self, timeout=None):
            return None

    monkeypatch.setattr(
        mod.asyncio, "run_coroutine_threadsafe", lambda coro, loop: _Fut(), raising=False,
    )
    origin = {"kind": "peer", "subkind": "peer-send-message", "msg_id": "7", "body": "hello"}

    assert mod.ClaudeAgentSdkSession.send_peer_message(stub, " hello ", origin) is True

    async def collect(stream):
        return [message async for message in stream]

    assert asyncio.run(collect(queried[0])) == [{
        "type": "user", "message": {"role": "user", "content": "hello"},
        "parent_tool_use_id": None, "origin": origin,
    }]
    assert len(scheduled) == 1
    assert mod.ClaudeAgentSdkSession.send_peer_message(stub, "later", origin) is False
    assert len(queried) == 1, "a second native peer waits for the first result"


def test_peer_message_declines_while_a_turn_is_in_flight(monkeypatch):
    mod, stub, queried = _transport(turn_inbox=object())
    scheduled = []

    class _Fut:
        def add_done_callback(self, cb):
            scheduled.append(cb)
        def result(self, timeout=None):
            return None

    monkeypatch.setattr(
        mod.asyncio, "run_coroutine_threadsafe", lambda *a, **kw: _Fut(), raising=False,
    )
    assert mod.ClaudeAgentSdkSession.send_peer_message(stub, "hello", {"kind": "peer"}) is False
    assert queried == [] and scheduled == []


def test_peer_message_declines_when_scheduled_query_fails(monkeypatch):
    mod, stub, queried = _transport(turn_inbox=None)

    class _Fut:
        def add_done_callback(self, cb):
            pass
        def result(self, timeout=None):
            raise RuntimeError("SDK query failed")
        def cancel(self):
            return True

    monkeypatch.setattr(
        mod.asyncio, "run_coroutine_threadsafe", lambda *a, **kw: _Fut(), raising=False,
    )
    assert mod.ClaudeAgentSdkSession.send_peer_message(
        stub, "hello", {"kind": "peer", "msg_id": "row-1"}
    ) is False


class _SettlingFuture:
    """A run_coroutine_threadsafe future the test settles by hand."""

    def __init__(self, first_wait_times_out=True):
        self._callbacks, self._exc, self._done, self._cancelled = [], None, False, False
        self.first_wait_times_out = first_wait_times_out
        self.waits = []

    def add_done_callback(self, cb):
        self._callbacks.append(cb)
        if self._done:
            cb(self)

    def settle(self, exc=None):
        self._done, self._exc = True, exc
        for cb in self._callbacks:
            cb(self)

    def result(self, timeout=None):
        self.waits.append(timeout)
        if not self._done:
            raise TimeoutError("still writing")
        if self._exc:
            raise self._exc
        return None

    def cancel(self):
        if self._done:
            return False
        self._cancelled = True
        self.settle()
        return True

    def cancelled(self):
        return self._cancelled

    def exception(self):
        return self._exc


def test_slow_write_that_completes_is_delivered_once(monkeypatch):
    """A write still going after 5 s is waited on, not reported early either way."""
    mod, stub, _queried = _transport(turn_inbox=None)
    future = _SettlingFuture()
    original_result = future.result

    def result(timeout=None):
        if len(future.waits) == 1:  # the second wait sees the write land
            future.settle()
        return original_result(timeout)

    future.result = result
    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", lambda *a, **kw: future, raising=False)
    monkeypatch.setattr(mod.threading, "Timer", lambda *a, **kw: types.SimpleNamespace(daemon=True, start=lambda: None))

    assert mod.ClaudeAgentSdkSession.send_peer_message(stub, "hello", {"kind": "peer", "msg_id": "row-2"}) is True
    assert future.waits == [5.0, mod._NATIVE_PEER_WRITE_MAX_SECONDS - 5.0]
    assert stub._native_peer_written is True and stub.woken_turn_active() is True


def test_write_stuck_past_the_limit_is_cancelled_and_not_delivered(monkeypatch):
    """Codex recheck P1: a write reported as delivered and failing later would lose the row."""
    mod, stub, _queried = _transport(turn_inbox=None)
    future = _SettlingFuture()
    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", lambda *a, **kw: future, raising=False)

    assert mod.ClaudeAgentSdkSession.send_peer_message(stub, "hello", {"kind": "peer", "msg_id": "row-2"}) is False
    assert future.cancelled() is True
    assert stub._native_peer_in_flight is False
    assert stub.native_peer_idle() is True


def test_unconfirmed_write_never_expires_into_a_second_admission(monkeypatch):
    """Codex recheck P1: the grace can't release a claim whose write is still pending."""
    mod, stub, _queried = _transport(turn_inbox=None)
    stub._native_peer_in_flight = True
    stub._native_peer_written = False
    stub._native_peer_since = mod.time.monotonic() - mod._NATIVE_PEER_ADMIT_GRACE_SECONDS - 60
    assert stub.native_peer_idle() is False
    assert stub._native_peer_in_flight is True


def test_late_failure_of_an_old_write_leaves_the_new_claim(monkeypatch):
    mod, stub, _queried = _transport(turn_inbox=None)
    old = _SettlingFuture()
    old_cbs = []
    old.add_done_callback = lambda cb: old_cbs.append(cb)
    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", lambda *a, **kw: old, raising=False)
    old.result = lambda timeout=None: None  # the scheduling wait returns; settlement comes later
    monkeypatch.setattr(mod.threading, "Timer", lambda *a, **kw: types.SimpleNamespace(daemon=True, start=lambda: None))
    assert mod.ClaudeAgentSdkSession.send_peer_message(stub, "one", {"kind": "peer", "msg_id": "row-1"}) is True

    # The CLI finished that peer turn; a second row is admitted under a new generation.
    stub._native_peer_in_flight = False
    new = _SettlingFuture(first_wait_times_out=False)
    new.settle()
    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", lambda *a, **kw: new, raising=False)
    assert mod.ClaudeAgentSdkSession.send_peer_message(stub, "two", {"kind": "peer", "msg_id": "row-3"}) is True

    old._done, old._exc = True, RuntimeError("late failure")
    for cb in old_cbs:
        cb(old)
    assert stub._native_peer_in_flight is True
    assert stub._native_peer_msg_id == "row-3"


def test_grace_expiry_wakes_queued_work(monkeypatch):
    """Codex recheck P1: expiry must notify the idle boundary so a queued owner prompt drains."""
    mod, stub, _queried = _transport(turn_inbox=None)
    stub._native_peer_gen = 4
    stub._native_peer_in_flight = True
    stub._native_peer_written = True
    stub._native_peer_since = mod.time.monotonic() - mod._NATIVE_PEER_ADMIT_GRACE_SECONDS - 1

    stub._expire_native_peer(3)  # a stale timer does nothing
    assert stub._native_peer_in_flight is True and stub.idle_notifications == []

    stub._expire_native_peer(4)
    assert stub._native_peer_in_flight is False
    assert stub.idle_notifications == [True]


def test_peer_claim_expires_when_the_cli_never_starts_a_turn(monkeypatch):
    """A confirmed input that never became a CLI turn cannot hold owner input back forever."""
    mod, stub, _queried = _transport(turn_inbox=None)
    stub._native_peer_in_flight = True
    stub._native_peer_written = True
    stub._native_peer_since = mod.time.monotonic()
    assert stub.woken_turn_active() is True
    assert stub.native_peer_idle() is False

    stub._native_peer_since -= mod._NATIVE_PEER_ADMIT_GRACE_SECONDS + 1
    assert stub.woken_turn_active() is False
    assert stub.native_peer_idle() is True
    assert stub._native_peer_in_flight is False


def test_open_burst_keeps_the_peer_claim_past_the_grace(monkeypatch):
    mod, stub, _queried = _transport(turn_inbox=None)
    stub._native_peer_in_flight = True
    stub._native_peer_written = True
    stub._native_peer_since = mod.time.monotonic() - mod._NATIVE_PEER_ADMIT_GRACE_SECONDS - 1
    stub._unsolicited_burst_open = True
    assert stub.woken_turn_active() is True
    assert stub._native_peer_in_flight is True


def test_peer_message_declines_when_cli_stream_has_ended(monkeypatch):
    mod, stub, queried = _transport(turn_inbox=None)
    stub._stream_ended = object()
    stub.is_live = lambda: stub._stream_ended is None

    class _Fut:
        def add_done_callback(self, _cb):
            pass
        def result(self, timeout=None):
            return None

    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", lambda *_a, **_kw: _Fut())

    assert mod.ClaudeAgentSdkSession.send_peer_message(
        stub, "hello", {"kind": "peer", "msg_id": "dead-cli"}
    ) is False
    assert queried == []


def test_session_liveness_rejects_a_dead_cli_child(monkeypatch):
    import threading

    mod, stub, _queried = _transport(turn_inbox=None)
    stub._turn_callback_lock = threading.RLock()
    stub._closed = False
    stub._retiring = False
    stub._stream_ended = None
    stub._loop = types.SimpleNamespace(is_closed=lambda: False, is_running=lambda: True)
    monkeypatch.setattr(mod, "_sdk_child_pid", lambda _client: 123)
    monkeypatch.setattr(mod, "_own_sdk_child_process", lambda _pid: None)

    assert mod.ClaudeAgentSdkSession.is_live(stub) is False


def test_transport_declines_when_client_or_loop_missing(monkeypatch):
    mod, stub, queried = _transport(turn_inbox=object(), client=False)
    assert mod.ClaudeAgentSdkSession.steer(stub, "note") is False

    mod, stub, queried = _transport(turn_inbox=object(), loop=False)
    assert mod.ClaudeAgentSdkSession.steer(stub, "note") is False
    assert queried == []


def test_transport_declines_empty_text():
    mod, stub, queried = _transport(turn_inbox=object())
    assert mod.ClaudeAgentSdkSession.steer(stub, "") is False
    assert mod.ClaudeAgentSdkSession.steer(stub, "   \n ") is False
    assert queried == []


def test_mid_turn_steer_result_stays_owned_by_live_turn():
    """A human-origin result from query(steer) is not a background result."""
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
    from tests.agent.claude_sdk_fakes import (
        ResultMessage,
        StreamEvent,
        _FakeClient,
    )

    def result(text, uuid, origin=None):
        message = ResultMessage(result=text, uuid=uuid)
        message.origin = origin
        return message

    class SteerClient(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                if len(self.queried) == 1:
                    self._pending.append(StreamEvent(event={
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "working"},
                    }))
                    self._pending.append(result("original", "original-result"))
                else:
                    self._pending.append(result(
                        "steered", "steer-result", {"kind": "human"},
                    ))
            else:
                payload = [message async for message in prompt]
                self.queried.append(payload)
                self._pending.append(result(
                    "steered", "steer-result", {"kind": "human"},
                ))

    holder = {}
    delivered = []
    steer_sent = False

    def factory(options=None):
        holder["client"] = SteerClient(options=options)
        return holder["client"]

    def on_delta(_text):
        nonlocal steer_sent
        if not steer_sent:
            steer_sent = True
            assert session.steer("correct course") is True

    session = ClaudeAgentSdkSession(
        cwd="/tmp",
        model="claude-opus-4-8",
        client_factory=factory,
        on_stream_delta=on_delta,
        on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items)),
    )
    try:
        turn = session.run_turn("initial")
    finally:
        session.close()

    assert holder["client"].queried[0] == "initial"
    assert holder["client"].queried[1] == [{
        "type": "user",
        "message": {"role": "user", "content": "correct course"},
        "parent_tool_use_id": None,
        "origin": {"kind": "human"},
    }], "steers must use the SDK message stream with human origin"
    assert turn.final_text == "steered", "steer result must close the live host turn"
    assert turn.turn_id == "steer-result", "live accounting must use the steer result once"
    assert delivered == [], "steer result must not use unsolicited delivery"
    assert session._unsolicited_results == 0


def test_two_steers_keep_foreground_ownership_until_both_results():
    """Every accepted steer result belongs to the same host turn."""
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
    from tests.agent.claude_sdk_fakes import ResultMessage, StreamEvent, _FakeClient

    def result(text, uuid, origin=None):
        message = ResultMessage(result=text, uuid=uuid)
        message.origin = origin
        return message

    class TwoSteerClient(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                if len([item for item in self.queried if isinstance(item, str)]) == 1:
                    self._pending.append(StreamEvent(event={
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "working"},
                    }))
                    self._pending.append(result("original", "original-result"))
                elif not any(isinstance(item, list) for item in self.queried):
                    steer_number = len([
                        item for item in self.queried if isinstance(item, str)
                    ]) - 1
                    self._pending.append(result(
                        f"steer-{steer_number}", f"steer-{steer_number}-result",
                        {"kind": "human"},
                    ))
                else:
                    self._pending.append(result("next host answer", "next-result"))
                return
            payload = [message async for message in prompt]
            self.queried.append(payload)
            steer_number = len([item for item in self.queried if isinstance(item, list)])
            self._pending.append(result(
                f"steer-{steer_number}", f"steer-{steer_number}-result",
                {"kind": "human"},
            ))

    holder = {}
    sent = 0

    def factory(options=None):
        holder["client"] = TwoSteerClient(options=options)
        return holder["client"]

    def on_delta(_text):
        nonlocal sent
        if sent == 0:
            sent = 1
            assert session.steer("correction-1") is True
            sent = 2
            assert session.steer("correction-2") is True

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=factory,
        on_stream_delta=on_delta,
    )
    try:
        turn = session.run_turn("initial")
        next_turn = session.run_turn("next host question")
    finally:
        session.close()

    assert turn.final_text == "steer-2", (
        "the foreground turn must remain open until the second steer result"
    )
    assert turn.turn_id == "steer-2-result"
    assert next_turn.final_text == "next host answer", (
        "the following host turn must receive its own result"
    )


def test_steer_results_are_durable_before_turn_release_without_end_duplicates(tmp_path):
    """Each settled live steer reaches state.db before the foreground turn ends."""
    from types import SimpleNamespace

    from agent.claude_sdk_runtime_continuity import _persist_turn
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, StreamEvent, TextBlock, _FakeClient,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("steer-durability", source="test")
    db.append_message("steer-durability", "user", "initial")
    messages = db.get_messages_as_conversation("steer-durability")
    agent = SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False,
        session_id="steer-durability", _session_persist_lock=None,
        _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None,
        _persist_user_message_override=None, _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)

    def on_steer_settled(steer_text, projected_messages, steer_index=None):
        assert session._turn_inbox is not None, "the host turn must still own the stream at persistence"
        if steer_text:
            messages.append({"role": "user", "content": steer_text})
        messages.extend(projected_messages)
        assert agent._flush_messages_to_session_db(messages) is True
        if steer_text:
            saved = db.get_messages("steer-durability")
            assert any(row["role"] == "assistant" and row["content"] == f"answer to {steer_text}" for row in saved)

    def result(text, uuid, origin=None):
        message = ResultMessage(result=text, uuid=uuid)
        message.origin = origin
        return message

    class DurableTwoSteerClient(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                self._pending.append(StreamEvent(event={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "working"},
                }))
                self._pending.append(AssistantMessage(content=[TextBlock("initial progress")]))
                self._pending.append(result("initial settled", "initial-result"))
                return
            payload = [message async for message in prompt]
            self.queried.append(payload)
            text = payload[0]["message"]["content"]
            self._pending.append(AssistantMessage(content=[TextBlock(f"answer to {text}")]))
            self._pending.append(result(f"answer to {text}", f"{text}-result", {"kind": "human"}))

    holder = {}
    sent = 0

    def factory(options=None):
        holder["client"] = DurableTwoSteerClient(options=options)
        return holder["client"]

    def on_delta(_text):
        nonlocal sent
        if sent == 0:
            sent = 1
            assert session.steer("correction-1") is True
            assert session.steer("correction-2") is True

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=factory,
        on_stream_delta=on_delta,
    )
    session._on_steer_settled = on_steer_settled
    try:
        turn = session.run_turn("initial")
        assert turn.turn_id == "correction-2-result"
        _persist_turn(agent, SimpleNamespace(
            turn=turn, messages=messages, failover_reason=None,
            turn_session_cwd=None,
        ))
    finally:
        session.close()

    saved = db.get_messages("steer-durability")
    assert [(row["role"], row["content"]) for row in saved if row["role"] in {"user", "assistant"}] == [
        ("user", "initial"),
        ("assistant", "initial progress"),
        ("user", "correction-1"),
        ("assistant", "answer to correction-1"),
        ("user", "correction-2"),
        ("assistant", "answer to correction-2"),
    ]
    assert all(message.get(_DB_PERSISTED_MARKER) for message in messages if message.get("role") in {"user", "assistant"})


def test_steer_admitted_during_result_projection_remains_foreground_owned(monkeypatch):
    """A steer admitted in the projection/commit gap stays in this turn."""
    from agent.transports import claude_agent_sdk_session_turn as turn_module
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
    from tests.agent.claude_sdk_fakes import ResultMessage, _FakeClient

    def result(text, uuid, origin=None):
        message = ResultMessage(result=text, uuid=uuid)
        message.origin = origin
        return message

    class BarrierClient(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                self._pending.append(result("original", "original-result"))
            else:
                self.queried.append([message async for message in prompt])
                self._pending.append(result(
                    "barrier steer", "barrier-result", {"kind": "human"},
                ))

    holder = {}

    def factory(options=None):
        holder["client"] = BarrierClient(options=options)
        return holder["client"]

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=factory,
    )
    original_project = turn_module.ClaudeSdkTurnMixin._project_message_step
    admitted = False

    def project(projector, watch, message, out):
        nonlocal admitted
        projection = original_project(projector, watch, message, out)
        if type(message).__name__ == "ResultMessage" and not admitted:
            admitted = True
            assert session.steer("during projection") is True
        return projection

    monkeypatch.setattr(
        turn_module.ClaudeSdkTurnMixin,
        "_project_message_step",
        staticmethod(project),
    )
    try:
        turn = session.run_turn("initial")
    finally:
        session.close()

    assert turn.final_text == "barrier steer", (
        "a steer admitted during projection must stay owned by the live turn"
    )
    assert turn.turn_id == "barrier-result"


def test_steer_after_terminal_result_is_not_claimed_by_next_turn():
    """A steer racing result acceptance falls back instead of opening a stray query."""
    mod, stub, queried = _transport(turn_inbox=object())
    stub._terminal_result_committed = True

    called = []
    stub._client.query = lambda text: called.append(text)
    assert mod.ClaudeAgentSdkSession.steer(stub, "late correction") is False
    assert queried == [] and called == []


def test_late_human_result_is_not_classified_as_unsolicited():
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
    from tests.agent.claude_sdk_fakes import ResultMessage

    delivered = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp",
        on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items)),
    )
    message = ResultMessage(result="steered", uuid="late-steer")
    message.origin = {"kind": "human"}
    session._unsolicited_text.append("stale steer text")

    session._handle_unsolicited(message)

    assert delivered == []
    assert session._unsolicited_results == 0
    assert session._unsolicited_text == []


# --- owner message typed mid-turn is written where it was sent ------------------------------------


def _steer_order_fakes():
    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, StreamEvent, TextBlock, _EOS, _FakeClient,
    )

    def result(text, uuid, origin=None):
        message = ResultMessage(result=text, uuid=uuid)
        message.origin = origin
        return message

    def delta(text):
        return StreamEvent(event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}})

    return AssistantMessage, TextBlock, _EOS, _FakeClient, result, delta


def test_folded_steer_is_written_between_output_before_and_after_it():
    """The CLI folded the steer into the running turn (one human-origin result for the whole turn): the owner's
    row sits after the output that had streamed when it was typed and before the reply that answers it."""
    from agent.claude_sdk_runtime_continuity import _persist_steer_boundary
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    AssistantMessage, TextBlock, _EOS, _FakeClient, result, delta = _steer_order_fakes()

    class FoldedClient(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                self._pending.append(AssistantMessage(content=[TextBlock("pre-steer output")]))
                self._pending.append(delta("typing"))
                return
            self.queried.append([message async for message in prompt])
            self._pending.append(AssistantMessage(content=[TextBlock("reply to the steer")]))
            self._pending.append(result("reply to the steer", "folded-result", {"kind": "human"}))

    sent = False

    def on_delta(_text):
        nonlocal sent
        if not sent:
            sent = True
            assert session.steer("and what do i need to do?") is True

    calls = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=lambda options=None: FoldedClient(options=options),
        on_stream_delta=on_delta,
    )
    session._on_steer_settled = lambda text, settled, index=None: calls.append((text, list(settled), index))
    try:
        session.run_turn("initial")
    finally:
        session.close()

    assert len(calls) == 1
    text, settled, index = calls[0]
    assert text == "and what do i need to do?"
    assert [m["content"] for m in settled if m["role"] == "assistant"] == ["pre-steer output", "reply to the steer"]
    assert index == 1, "the steer belongs after the one message that had streamed, not before the whole turn"

    transcript = []
    _persist_steer_boundary(types.SimpleNamespace(_session_db=None), transcript, text, settled, index)
    assert [(m["role"], m["content"]) for m in transcript] == [
        ("assistant", "pre-steer output"),
        ("user", "and what do i need to do?"),
        ("assistant", "reply to the steer"),
    ]


def test_steer_the_turn_never_settles_is_still_written_before_the_late_reply():
    """The turn released before the steer's own result (the answer surfaces later as unsolicited output): the
    owner's words used to be dropped, leaving only the reply. They are written at the point they were typed."""
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    AssistantMessage, TextBlock, _EOS, _FakeClient, result, delta = _steer_order_fakes()

    class UnsettledClient(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                self._pending.append(AssistantMessage(content=[TextBlock("pre-steer output")]))
                self._pending.append(delta("typing"))
                self._pending.append(AssistantMessage(content=[TextBlock("more output")]))
                return
            self.queried.append([message async for message in prompt])
            self._pending.append(_EOS)  # the stream ends before any result for the steer

    sent = False

    def on_delta(_text):
        nonlocal sent
        if not sent:
            sent = True
            assert session.steer("and what do i need to do?") is True

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=lambda options=None: UnsettledClient(options=options),
        on_stream_delta=on_delta,
    )
    try:
        turn = session.run_turn("initial")
    finally:
        session.close()

    rows = [(m["role"], m["content"]) for m in turn.projected_messages if m["role"] in {"user", "assistant"}]
    assert rows == [
        ("assistant", "pre-steer output"),
        ("user", "and what do i need to do?"),
        ("assistant", "more output"),
    ]
    assert session._pending_steer_marks == []


def _assert_steer_outside_tool_group(rows):
    """call -> result adjacent, steer after them, and the real repair pass keeps both."""
    from agent.agent_runtime_helpers import repair_message_sequence

    roles = [(m["role"], m.get("content") if m["role"] == "user" else None) for m in rows]
    call = next(i for i, m in enumerate(rows) if m["role"] == "assistant" and m.get("tool_calls"))
    assert rows[call + 1]["role"] == "tool", f"steer split the call from its result: {roles}"
    steer = next(i for i, m in enumerate(rows) if m["role"] == "user" and m["content"] == "and what do i need to do?")
    assert steer > call + 1
    restored = [dict(m) for m in rows]
    repair_message_sequence(None, restored)
    assert any(m["role"] == "assistant" and m.get("tool_calls") for m in restored)
    assert any(m["role"] == "tool" for m in restored), "repair dropped the completed tool result"


def _tool_group_fakes():
    from tests.agent.claude_sdk_fakes import ToolResultBlock, ToolUseBlock, UserMessage

    return ToolResultBlock, ToolUseBlock, UserMessage


def test_release_never_writes_a_steer_between_a_tool_call_and_its_result():
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    AssistantMessage, TextBlock, _EOS, _FakeClient, result, delta = _steer_order_fakes()
    ToolResultBlock, ToolUseBlock, UserMessage = _tool_group_fakes()

    class Client(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                self._pending.append(AssistantMessage(content=[ToolUseBlock(id="tx", name="Bash", input={})]))
                self._pending.append(delta("typing"))
                self._pending.append(UserMessage(content=[ToolResultBlock(tool_use_id="tx", content="done")]))
                return
            self.queried.append([message async for message in prompt])
            self._pending.append(_EOS)

    sent = False

    def on_delta(_text):
        nonlocal sent
        if not sent:
            sent = True
            assert session.steer("and what do i need to do?") is True

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=lambda options=None: Client(options=options),
        on_stream_delta=on_delta,
    )
    try:
        turn = session.run_turn("initial")
    finally:
        session.close()
    _assert_steer_outside_tool_group(list(turn.projected_messages))


def test_folded_steer_never_lands_between_a_tool_call_and_its_result():
    from agent.claude_sdk_runtime_continuity import _persist_steer_boundary
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    AssistantMessage, TextBlock, _EOS, _FakeClient, result, delta = _steer_order_fakes()
    ToolResultBlock, ToolUseBlock, UserMessage = _tool_group_fakes()

    class Client(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                self._pending.append(AssistantMessage(content=[ToolUseBlock(id="tx", name="Bash", input={})]))
                self._pending.append(delta("typing"))
                self._pending.append(UserMessage(content=[ToolResultBlock(tool_use_id="tx", content="done")]))
                return
            self.queried.append([message async for message in prompt])
            self._pending.append(AssistantMessage(content=[TextBlock("reply")]))
            self._pending.append(result("reply", "folded", {"kind": "human"}))

    sent = False

    def on_delta(_text):
        nonlocal sent
        if not sent:
            sent = True
            assert session.steer("and what do i need to do?") is True

    calls = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=lambda options=None: Client(options=options),
        on_stream_delta=on_delta,
    )
    session._on_steer_settled = lambda text, settled, index=None: calls.append((text, list(settled), index))
    try:
        session.run_turn("initial")
    finally:
        session.close()
    text, settled, index = calls[0]
    transcript = []
    _persist_steer_boundary(types.SimpleNamespace(_session_db=None), transcript, text, settled, index)
    _assert_steer_outside_tool_group(transcript)
    assert [m["role"] for m in transcript][-2:] == ["user", "assistant"]


# --- a steer the CLI folds into the running turn settles with that turn's result ------------------
#
# Live 2026-10-13: the owner typed mid-turn, the CLI took the steer in right after a tool result (its
# echo appeared INSIDE the running turn) and ended the whole turn with ONE result that was not
# human-origin. The host turn waited for a second, human-origin result that never came.


def _fold_fakes():
    from tests.agent.claude_sdk_fakes import ToolResultBlock, ToolUseBlock, UserMessage

    AssistantMessage, TextBlock, _EOS, _FakeClient, result, delta = _steer_order_fakes()

    def echo(text, uuid=None, origin="human"):
        """The CLI's replay of one stdin user message, emitted when it consumes that message."""
        message = UserMessage(content=text)
        message.origin = {"kind": origin} if origin else None
        message.uuid = uuid
        return message

    def tool_call():
        return AssistantMessage(content=[ToolUseBlock(id="tx", name="Bash", input={})])

    def tool_result():
        return UserMessage(content=[ToolResultBlock(tool_use_id="tx", content="done")])

    def say(text):
        return AssistantMessage(content=[TextBlock(text)])

    return _FakeClient, result, delta, echo, tool_call, tool_result, say


def _scripted_session(first, steers, on_delta_steers=(), **kwargs):
    """A session whose fake CLI answers the prompt with ``first`` and the n-th steer with ``steers[n]``.

    ``on_delta_steers`` are sent, in order, on the first stream delta (the owner typing mid-turn)."""
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    _FakeClient = _fold_fakes()[0]
    holder = {"steer_results": []}

    class Client(_FakeClient):
        async def query(self, prompt):
            if isinstance(prompt, str):
                self.queried.append(prompt)
                turn = len([item for item in self.queried if isinstance(item, str)])
                self._pending.extend(first if turn == 1 else holder.get("later", {}).get(prompt, ()))
                return
            self.queried.append([message async for message in prompt])
            number = len([item for item in self.queried if isinstance(item, list)])
            self._pending.extend(steers[number - 1] if number <= len(steers) else ())

    def factory(options=None):
        holder["client"] = Client(options=options)
        return holder["client"]

    sent = False

    def on_delta(_text):
        nonlocal sent
        if not sent:
            sent = True
            for text in on_delta_steers:
                holder["steer_results"].append(session.steer(text))

    calls = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=factory, on_stream_delta=on_delta, **kwargs,
    )
    session._on_steer_settled = lambda text, settled, index=None: calls.append(
        (text, list(settled), index, session._pending_steer_results)
    )
    return session, holder, calls


def _bounded_turn(session, prompt="initial"):
    """Run one turn; a turn that never finishes trips the idle limit instead of hanging the suite."""
    return session.run_turn(prompt, turn_timeout=1.5, watch_poll_interval=0.05, abort_grace=0.2)


def _transcript(calls):
    from agent.claude_sdk_runtime_continuity import _persist_steer_boundary

    rows = []
    for text, settled, index, _pending in calls:
        _persist_steer_boundary(types.SimpleNamespace(_session_db=None), rows, text, settled, index)
    return rows


def _shape(rows):
    return [
        (row["role"], "call" if row.get("tool_calls") else row.get("content"))
        for row in rows if row["role"] in {"user", "assistant", "tool"}
    ]


def test_folded_steer_settles_with_the_running_turns_result():
    """T1, the incident: the steer's echo was seen inside the running turn, so that turn's single
    (not human-origin) result is the steer's result too and ends the host turn."""
    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    session, holder, calls = _scripted_session(
        first=[echo("initial", "prompt-echo", origin=None), tool_call(), delta("typing")],
        steers=[[tool_result(), echo("x", "echo-x"), say("folded answer"),
                 result("folded answer", "turn-result")]],
        on_delta_steers=["x"],
    )
    try:
        turn = _bounded_turn(session)
    finally:
        session.close()

    assert holder["steer_results"] == [True]
    assert turn.error is None, f"the host turn never finished: {turn.error}"
    assert turn.final_text == "folded answer"
    assert turn.turn_id == "turn-result"
    assert turn.interrupted is False
    assert [(text, pending) for text, _rows, _index, pending in calls] == [("x", 0)], (
        "the folded steer is settled once, by the turn's own result, and nothing stays pending"
    )
    assert _shape(_transcript(calls)) == [
        ("assistant", "call"), ("tool", "done"), ("user", "x"), ("assistant", "folded answer"),
    ]
    assert [m for m in turn.projected_messages if m["role"] == "user"] == [], (
        "the steer row is written once, at its boundary, never again at the turn's end"
    )


def test_steer_after_a_folded_result_finishes_on_its_own_human_result(monkeypatch):
    """T2, the incident's second step: with the folded steer settled the count is back to zero, so a
    later steer the idle CLI runs as its OWN turn ends the host turn on its human-origin result."""
    from agent.transports import claude_agent_sdk_session_turn as turn_module

    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    session, holder, calls = _scripted_session(
        first=[echo("initial", "prompt-echo", origin=None), tool_call(), delta("typing")],
        steers=[
            [tool_result(), echo("x", "echo-x"), say("folded answer"), result("folded answer", "turn-result")],
            [echo("y", "echo-y"), say("answer y"), result("answer y", "y-result", {"kind": "human"})],
        ],
        on_delta_steers=["x"],
    )
    original_project = turn_module.ClaudeSdkTurnMixin._project_message_step
    later = []

    def project(projector, watch, message, out):
        projection = original_project(projector, watch, message, out)
        if type(message).__name__ == "ResultMessage" and not later:
            # The CLI is idle (its turn just ended) and the host turn has not released yet.
            later.append(session.steer("y"))
        return projection

    monkeypatch.setattr(turn_module.ClaudeSdkTurnMixin, "_project_message_step", staticmethod(project))
    try:
        turn = _bounded_turn(session)
    finally:
        session.close()

    assert later == [True], "the folded turn's result must reach projection as the turn's own result"
    assert turn.error is None, f"the host turn never finished: {turn.error}"
    assert turn.final_text == "answer y"
    assert turn.turn_id == "y-result"
    assert [(text, pending) for text, _rows, _index, pending in calls] == [("x", 0), ("y", 0)]


def test_two_steers_first_folded_second_its_own_turn_finish_in_order():
    """T3: the first steer is folded into the running turn, the second runs as its own CLI turn. The
    host turn ends on the second's result; rows and answers keep the order they were sent in, the same
    order two separately answered steers already have (prompt output, steer 1, answer 1, steer 2, answer 2)."""
    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    session, holder, calls = _scripted_session(
        first=[echo("initial", "prompt-echo", origin=None), tool_call(), delta("typing")],
        steers=[
            [tool_result(), echo("x", "echo-x"), say("answer x"), result("answer x", "turn-result")],
            [echo("y", "echo-y"), say("answer y"), result("answer y", "y-result", {"kind": "human"})],
        ],
        on_delta_steers=["x", "y"],
    )
    try:
        turn = _bounded_turn(session)
    finally:
        session.close()

    assert holder["steer_results"] == [True, True]
    assert turn.error is None, f"the host turn never finished: {turn.error}"
    assert turn.final_text == "answer y", "the host turn stays open until the second steer's result"
    assert turn.turn_id == "y-result"
    assert [(text, pending) for text, _rows, _index, pending in calls] == [("x", 1), ("y", 0)], (
        "each result settles the steer it answered: the folded one first, then the one run alone"
    )
    assert _shape(_transcript(calls)) == [
        ("assistant", "call"), ("tool", "done"), ("user", "x"), ("assistant", "answer x"),
        ("user", "y"), ("assistant", "answer y"),
    ]


def test_steer_not_yet_consumed_keeps_the_turn_open_until_its_own_result():
    """T4: the running turn's result arrives BEFORE the steer's echo, so the steer is still queued in
    the CLI: the host turn stays open and the steer's own human-origin result finishes it."""
    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    session, holder, calls = _scripted_session(
        first=[echo("initial", "prompt-echo", origin=None), delta("typing"), say("original answer"),
               result("original answer", "turn-result")],
        steers=[[echo("y", "echo-y"), say("answer y"), result("answer y", "y-result", {"kind": "human"})]],
        on_delta_steers=["y"],
    )
    try:
        turn = _bounded_turn(session)
    finally:
        session.close()

    assert turn.error is None
    assert turn.final_text == "answer y"
    assert turn.turn_id == "y-result"
    assert [(text, pending) for text, _rows, _index, pending in calls] == [(None, 1), ("y", 0)], (
        "the first result settles no steer and leaves the turn open"
    )
    assert _shape(_transcript(calls)) == [
        ("assistant", "original answer"), ("user", "y"), ("assistant", "answer y"),
    ]


def test_two_steers_folded_into_one_turn_settle_with_its_single_result():
    """Both steers were consumed at the same tool boundary: one result settles both, and their text is
    one user row (never two user rows in a row)."""
    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    session, holder, calls = _scripted_session(
        first=[echo("initial", "prompt-echo", origin=None), tool_call(), delta("typing")],
        steers=[[tool_result(), echo("x", "echo-x"), echo("y", "echo-y"), say("answer both"),
                 result("answer both", "turn-result")], []],
        on_delta_steers=["x", "y"],
    )
    try:
        turn = _bounded_turn(session)
    finally:
        session.close()

    assert turn.error is None, f"the host turn never finished: {turn.error}"
    assert turn.final_text == "answer both"
    assert turn.turn_id == "turn-result"
    assert [pending for _text, _rows, _index, pending in calls] == [0]
    assert _shape(_transcript(calls)) == [
        ("assistant", "call"), ("tool", "done"), ("user", "x\ny"), ("assistant", "answer both"),
    ]


def test_prompt_echo_with_the_steers_text_is_not_the_steers_echo():
    """The CLI may stamp its replay of the host prompt human too. A steer with the same words, written
    before that replay arrived, is still queued: the prompt's echo must not settle it."""
    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    session, holder, calls = _scripted_session(
        first=[delta("typing"), echo("continue", "prompt-echo"), say("original answer"),
               result("original answer", "turn-result")],
        steers=[[echo("continue", "echo-steer"), say("steer answer"),
                 result("steer answer", "steer-result", {"kind": "human"})]],
        on_delta_steers=["continue"],
    )
    try:
        turn = _bounded_turn(session, "continue")
    finally:
        session.close()

    assert turn.error is None
    assert turn.final_text == "steer answer"
    assert turn.turn_id == "steer-result"


def test_steer_folded_into_an_injected_turn_settles_with_that_turns_result():
    """The steer was still queued when the host's turn ended; the CLI then ran an injected turn and
    consumed the steer inside it. That turn's one result (injected origin) answers the steer."""
    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    delivered = []
    session, holder, calls = _scripted_session(
        first=[echo("initial", "prompt-echo", origin=None), delta("typing"), say("original answer"),
               result("original answer", "turn-result")],
        steers=[[echo("a background task finished", "wake", origin="task-notification"), tool_call(),
                 tool_result(), echo("x", "echo-x"), say("did both"),
                 result("did both", "injected-result", {"kind": "task-notification"})]],
        on_delta_steers=["x"],
        on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items)),
    )
    try:
        turn = _bounded_turn(session)
    finally:
        session.close()

    assert turn.error is None, f"the host turn never finished: {turn.error}"
    assert turn.final_text == "did both"
    assert turn.turn_id == "injected-result"
    assert delivered == [], "the result that answered the steer is not a background delivery"
    assert [(text, pending) for text, _rows, _index, pending in calls] == [(None, 1), ("x", 0)]


class _SteerWaitClock:
    """Injectable clock: reports when the turn starts waiting on steer results and lets a test jump it."""

    def __init__(self):
        self.now = 0.0
        self.waiting = threading.Event()

    def __call__(self):
        self.waiting.set()
        return self.now


def _turn_in_thread(session, prompt="initial"):
    box = {}

    def run():
        box["turn"] = session.run_turn(prompt, turn_timeout=30.0, watch_poll_interval=0.05, abort_grace=0.2)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def _idle_steer_wait_session(monkeypatch, steers=((),), **kwargs):
    """A host turn whose CLI turn has ended while one steer's result is still awaited."""
    from agent.transports import claude_agent_sdk_session_turn as turn_module

    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    monkeypatch.setattr(turn_module, "_STEER_WAIT_POLL_SECONDS", 0.01, raising=False)
    session, holder, calls = _scripted_session(
        first=[echo("initial", "prompt-echo", origin=None), delta("typing"), say("original answer"),
               result("original answer", "turn-result")],
        steers=list(steers),
        on_delta_steers=["never answered"],
        **kwargs,
    )
    holder["later"] = {"next": [echo("next", "next-echo", origin=None), say("next answer"),
                                result("next answer", "next-result")]}
    clock = session._steer_wait_clock = _SteerWaitClock()
    return session, holder, calls, clock


def _await_open_result(calls, clock):
    """Block until the turn's own result was read with the steer still unanswered (the stuck state)."""
    import time

    deadline = time.monotonic() + 5.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [(text, pending) for text, _rows, _index, pending in calls] == [(None, 1)]
    clock.waiting.wait(1.0)  # the wait reads its start time before a test moves the clock


def test_stop_releases_a_turn_waiting_on_steer_results_with_an_idle_cli(monkeypatch):
    """T5: the CLI's last message was a result and nothing was written since, so the interrupt can
    produce no result. Stop must end the host turn, and the next prompt must run normally."""
    session, holder, calls, clock = _idle_steer_wait_session(monkeypatch)
    thread, box = _turn_in_thread(session)
    try:
        _await_open_result(calls, clock)
        session.request_interrupt()
        clock.now = 5.0
        thread.join(3.0)
        released = not thread.is_alive()
        assert released, "Stop did not release a host turn that waits on an idle CLI"
        turn = box["turn"]
        assert turn.interrupted is True
        assert turn.error is None
        assert turn.should_retire is False
        assert holder["client"].interrupted is True, "the CLI is still told to stop"
        assert ("user", "never answered") in _shape(turn.projected_messages), (
            "the owner's steer keeps its row even though no result answered it"
        )
        assert session._pending_steer_results == 0 and session._turn_inbox is None
        following = session.run_turn("next", turn_timeout=3.0, watch_poll_interval=0.05, abort_grace=0.2)
        assert following.error is None and following.interrupted is False
        assert following.final_text == "next answer"
    finally:
        session.close()
        thread.join(3.0)


def test_stop_waits_for_a_steer_turn_the_cli_is_just_starting(monkeypatch):
    """Stop pressed in the instant between a turn's result and the queued steer's own turn: the CLI is
    not idle, so the host turn keeps reading until that turn's result instead of orphaning it."""
    _FakeClient, result, delta, echo, tool_call, tool_result, say = _fold_fakes()
    session, holder, calls, clock = _idle_steer_wait_session(monkeypatch)
    thread, box = _turn_in_thread(session)
    try:
        _await_open_result(calls, clock)
        session.request_interrupt()
        holder["client"].feed(
            echo("never answered", "late-echo"), say("late answer"),
            result("late answer", "late-result", {"kind": "human"}),
        )
        thread.join(3.0)
        assert not thread.is_alive()
        turn = box["turn"]
        assert turn.interrupted is True
        assert turn.turn_id == "late-result", "the steer turn's result was read by the turn that owns it"
        assert session._unsolicited_results == 0
    finally:
        session.close()
        thread.join(3.0)


def test_quiet_cli_after_a_result_finishes_the_turn_with_one_warning(monkeypatch, caplog):
    """T6, the safety net: a result was read, the CLI then produced nothing for the quiet limit and the
    turn is open only for a steer result. Finish with the answer it has and say so once."""
    import logging

    from agent.transports import claude_agent_sdk_session_turn as turn_module

    session, holder, calls, clock = _idle_steer_wait_session(monkeypatch, hermes_session_id="sess-quiet")
    thread, box = _turn_in_thread(session)
    try:
        with caplog.at_level(logging.WARNING, logger="agent.transports.claude_agent_sdk_session"):
            _await_open_result(calls, clock)
            assert turn_module._STEER_WAIT_QUIET_SECONDS == 60.0
            clock.now = 59.0
            thread.join(0.3)
            assert thread.is_alive(), "the safety net must not fire before its limit"
            clock.now = 61.0
            thread.join(3.0)
            assert not thread.is_alive(), "the safety net did not finish the turn"
        turn = box["turn"]
        assert turn.error is None
        assert turn.interrupted is False
        assert turn.final_text == "original answer"
        assert turn.turn_id == "turn-result"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "steer" in r.getMessage()]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        text = warnings[0].getMessage()
        assert "sess-quiet" in text and "pending=1" in text
        assert "never answered" not in caplog.text, "no message content in logs"
        assert ("user", "never answered") in _shape(turn.projected_messages)
    finally:
        session.close()
        thread.join(3.0)
