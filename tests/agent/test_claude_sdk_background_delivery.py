"""Background result delivery wiring — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""


import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from tests.agent.claude_sdk_fakes import (
    _make_turn,
    _make_agent,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestBackgroundDeliveryWiring:
    """Runtime glue: the session's delivery callback enqueues an
    sdk_background_result event for the gateway watcher's direct outbound
    send, config-gated."""

    def _spy_kwargs(self, monkeypatch):

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input, **kw):
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(
            "agent.transports.claude_agent_sdk_session.ClaudeAgentSdkSession",
            SpySession,
        )
        return captured

    def test_flag_on_wires_callback_and_queue_event(self, monkeypatch):
        import hermes_cli.config as cfg
        from tools.process_registry import process_registry

        captured = self._spy_kwargs(monkeypatch)
        events = []

        class _FakeQueue:
            def put(self, evt):
                events.append(evt)

        monkeypatch.setattr(process_registry, "completion_queue", _FakeQueue())
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: "gw-key-7"
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        # Opt-in flag (upstream-conservative default is OFF).
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": True}}
            },
            raising=False,
        )

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_id = "sess-bg-1"
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        callback = captured.get("on_unsolicited_result")
        assert callback is not None, "flag defaults ON — callback must be wired"
        assert captured.get("on_unsolicited_start") is not None
        callback(
            ["Research landed — writing up.", "background answer text"],
            [{"kind": "peer_in", "text": "incoming", "uuid": "peer-1"}],
        )
        assert len(events) == 1
        evt = events[0]
        # Direct-outbound event: the payload burst rides UNJOINED (each text
        # becomes its own outbound message) and no model-facing directive is
        # prepended — on a direct send it would leak to the user.
        assert evt["type"] == "sdk_background_result"
        assert evt["payloads"] == [
            "Research landed — writing up.", "background answer text",
        ]
        assert evt["items"] == [{"kind": "peer_in", "text": "incoming", "uuid": "peer-1"}]
        assert not any("[USER IS WAITING" in p for p in evt["payloads"])
        assert evt["session_key"] == "gw-key-7"
        assert evt["parent_session_id"] == "sess-bg-1"
        assert "delegation_id" not in evt

    def test_bg_parent_resolved_at_delivery_time_after_rotation(
        self, monkeypatch,
    ):
        # P0.g: the SDK session outlives hermes session rotations. The old
        # code snapshotted parent_session_id/session_key at SDK-session
        # CREATION, so a completion firing after rotation carried the dead
        # parent — the gateway classified it permanently gone and dropped
        # it. The callback must resolve the parent AT DELIVERY TIME, with
        # the creation-time snapshot only as a fallback for the SDK-loop
        # thread where the session-key contextvar is unset.
        import hermes_cli.config as cfg
        from tools.process_registry import process_registry

        captured = self._spy_kwargs(monkeypatch)
        events = []

        class _FakeQueue:
            def put(self, evt):
                events.append(evt)

        monkeypatch.setattr(process_registry, "completion_queue", _FakeQueue())
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: "gw-key-7"
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": True}}
            },
            raising=False,
        )

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_id = "sess-before"
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        callback = captured.get("on_unsolicited_result")
        assert callback is not None

        # Hermes rotates the session between turns; the completion fires on
        # the SDK loop thread where the contextvar reads empty.
        agent.session_id = "sess-after-rotation"
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: ""
        )
        callback(["late background report"])
        assert len(events) == 1
        assert events[0]["parent_session_id"] == "sess-after-rotation"
        # Empty live key -> creation-time snapshot fallback keeps the route.
        assert events[0]["session_key"] == "gw-key-7"

        # A live, non-empty contextvar read wins over the snapshot.
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: "gw-key-LIVE"
        )
        callback(["second late report"])
        assert len(events) == 2
        assert events[1]["session_key"] == "gw-key-LIVE"
        assert events[1]["parent_session_id"] == "sess-after-rotation"

    def test_flag_off_leaves_callback_unwired(self, monkeypatch):
        import hermes_cli.config as cfg

        captured = self._spy_kwargs(monkeypatch)
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": False}}
            },
            raising=False,
        )
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert captured.get("on_unsolicited_result") is None
        # No delivery path retires a CLI-injected turn marker here, so none may be written:
        # an uncleared marker would replay a finished peer/task turn after the next restart.
        assert captured.get("on_unsolicited_start") is None


def test_unsolicited_peer_turn_projects_ordered_items_once():
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class ToolUseBlock:
        def __init__(self, tool_id, name, input):
            self.id, self.name, self.input = tool_id, name, input

    class ToolResultBlock:
        def __init__(self, tool_id, content, is_error=False):
            self.tool_use_id, self.content, self.is_error = tool_id, content, is_error

    class UserMessage:
        def __init__(self, content, *, uuid=None, origin=None):
            self.content, self.uuid, self.origin = content, uuid, origin
            self.parent_tool_use_id = None

    class AssistantMessage:
        def __init__(self, content, *, uuid=None):
            self.content, self.uuid = content, uuid
            self.parent_tool_use_id = None

    class ResultMessage:
        def __init__(self, result, uuid):
            self.result, self.uuid = result, uuid

    delivered = []
    started = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp", on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items)),
        on_unsolicited_start=started.append,
    )
    session._handle_unsolicited(UserMessage(
        "ignored envelope text", uuid="peer-in-1",
        origin={
            "kind": "peer", "from": "peer-id", "name": "Peer Name",
            "fromSession": "peer-session", "body": "incoming body",
        },
    ))
    session._handle_unsolicited(AssistantMessage([
        ToolUseBlock("tool-1", "SendMessage", {"to": "peer-id", "message": "outgoing body"}),
    ]))
    session._handle_unsolicited(UserMessage([
        ToolResultBlock("tool-1", [{"type": "text", "text": "sent"}]),
    ], uuid="tool-result-1"))
    session._handle_unsolicited(AssistantMessage([TextBlock("final body")]))
    session._handle_unsolicited(ResultMessage("final body", "result-1"))
    session._handle_unsolicited(ResultMessage("final body", "result-1"))

    assert len(delivered) == 1
    assert started == ["incoming body"]
    texts, items = delivered[0]
    assert texts == ["final body"]
    assert items == [
        {
            "kind": "lifecycle", "event": "woken",
            "source": "peer", "by": "Peer Name", "uuid": "peer-in-1",
        },
        {
                "kind": "peer_in", "text": "incoming body", "from": "peer-id",
                "name": "Peer Name", "from_session": "peer-session",
                "msg_id": "peer-in-1", "uuid": "peer-in-1",
        },
        {
            "kind": "tool", "tool_use_id": "tool-1", "name": "SendMessage",
            "args": {"to": "peer-id", "message": "outgoing body"},
            "result": "sent", "is_error": False,
        },
        {"kind": "peer_out", "text": "outgoing body", "to": "peer-id", "tool_use_id": "tool-1"},
        {"kind": "text", "text": "final body"},
    ]


def test_gateway_unsolicited_start_sink_records_the_live_session_marker(monkeypatch):
    from types import SimpleNamespace
    from tui_gateway import server
    from agent.claude_sdk_runtime_session import _gateway_unsolicited_start_sink

    session = {"session_key": "stored-key", "profile_home": "/profile/home"}
    monkeypatch.setattr(server, "_sessions", {"runtime-sid": session})
    recorded = []
    monkeypatch.setattr(server, "_record_turn_marker", lambda *args: recorded.append(args))
    _gateway_unsolicited_start_sink(SimpleNamespace(_tui_gateway_runtime_sid="runtime-sid"), "peer task")
    assert recorded == [(session, "peer task")]


def test_terminal_result_text_is_trailing_item_when_it_differs_from_buffered_text():
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content):
            self.content, self.uuid, self.parent_tool_use_id = content, "assistant-2", None

    class ResultMessage:
        def __init__(self, result):
            self.result, self.uuid = result, "result-2"

    delivered = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp", on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items))
    )
    session._handle_unsolicited(AssistantMessage([TextBlock("buffered answer")]))
    session._handle_unsolicited(ResultMessage("authoritative terminal answer"))

    assert delivered[0][0] == ["buffered answer", "authoritative terminal answer"]
    assert delivered[0][1][-1] == {"kind": "text", "text": "authoritative terminal answer"}


@pytest.mark.parametrize("terminal", ["a\nb", " a  \n b "])
def test_terminal_result_does_not_repeat_joined_text_blocks(terminal):
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
    from tests.agent.claude_sdk_fakes import AssistantMessage, TextBlock, ResultMessage

    delivered = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp", on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items))
    )
    session._handle_unsolicited(AssistantMessage(content=[TextBlock("a"), TextBlock("b")]))
    session._handle_unsolicited(ResultMessage(result=terminal))
    assert delivered[0][1] == [{"kind": "text", "text": "a"}, {"kind": "text", "text": "b"}]
    assert delivered[0][0] == ["a\nb"]


def test_scheduled_task_wake_is_the_first_unsolicited_item():
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    class UserMessage:
        def __init__(self):
            self.content = "scheduled task details"
            self.uuid = "wake-1"
            self.origin = {
                "kind": "task-notification",
                "subkind": "scheduled-trigger",
                "description": "morning report",
            }
            self.parent_tool_use_id = None

    delivered = []
    started = []
    session = ClaudeAgentSdkSession(
        cwd="/tmp", on_unsolicited_result=lambda texts, items=None: delivered.append(items),
        on_unsolicited_start=started.append,
    )
    session._handle_unsolicited(UserMessage())

    assert delivered == []
    assert started == ["scheduled task details"]
    assert session._unsolicited_items == [{
        "kind": "lifecycle", "event": "woken",
        "source": "task-notification/scheduled-trigger",
        "by": "morning report", "uuid": "wake-1",
    }]


def test_task_notification_without_subkind_wakes_the_burst():
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    class UserMessage:
        content = "generic task notification"
        uuid = "wake-generic"
        origin = {"kind": "task-notification", "description": "a completed task"}
        parent_tool_use_id = None

    session = ClaudeAgentSdkSession(cwd="/tmp", on_unsolicited_result=lambda *_: None)
    session._handle_unsolicited(UserMessage())

    assert session._unsolicited_items[0]["source"] == "task-notification"


def test_suppressed_own_answer_clears_all_projection_buffers():
    from agent.transports.claude_agent_sdk_session_turn import _clear_unsolicited_projection

    session = type("Session", (), {
        "_unsolicited_text": ["answer"],
        "_unsolicited_items": [{"kind": "tool"}],
        "_unsolicited_tool_items": {"tool-1": {"kind": "tool"}},
        "_unsolicited_seen": {"stale-1"},
    })()
    _clear_unsolicited_projection(session)
    assert session._unsolicited_text == []
    assert session._unsolicited_items == []
    assert session._unsolicited_tool_items == {}
    assert session._unsolicited_seen == set()


def test_stream_end_does_not_duplicate_runtime_child_exit_emission():
    from agent.transports.claude_agent_sdk_session_turn import ClaudeSdkTurnMixin, _StreamEnd, _claim_child_exit_emission
    from types import SimpleNamespace

    events = []
    session = SimpleNamespace(
        _on_unsolicited_result=lambda texts, items: events.append((texts, items)),
        _unsolicited_stream_end_emitted=False,
    )
    ClaudeSdkTurnMixin._handle_unsolicited(session, _StreamEnd("eof"))
    assert events == []
    # The runtime's retirement path remains the sole claimant/emitter.
    assert _claim_child_exit_emission(session) is True


def test_stale_projection_discard_clears_seen_ids():
    from agent.transports.claude_agent_sdk_session_turn import _clear_unsolicited_projection

    session = type("Session", (), {
        "_unsolicited_text": [],
        "_unsolicited_items": [{"kind": "lifecycle", "event": "woken"}],
        "_unsolicited_tool_items": {"tool-1": {"kind": "tool"}},
        "_unsolicited_seen": {"wake-1", "tool-1"},
    })()
    _clear_unsolicited_projection(session)
    assert session._unsolicited_items == []
    assert session._unsolicited_tool_items == {}
    assert session._unsolicited_seen == set()


def test_peer_burst_that_straddles_admission_never_answers_the_host_turn():
    """H-2 (holistic review): peer turn starts idle, host prompt claims the stream,
    peer turn finishes, then the host's own turn runs. The host must get its own
    answer and the peer answer must go to the background (display) path."""
    import time as _time

    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, TextBlock, UserMessage, _make_session,
    )

    delivered = []
    session, holder = _make_session(
        script=[AssistantMessage(content=[TextBlock("host answer")]),
                ResultMessage(result="host answer", uuid="host-1")],
        on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items)),
    )
    peer_origin = {
        "kind": "peer", "from": "uds:/tmp/x.sock", "name": "hermes:other",
        "body": "ping", "msg_id": "mailbox-row-42",
    }
    try:
        session.ensure_started()
        client = holder["client"]
        peer_in = UserMessage(content="ping")
        peer_in.origin = peer_origin
        peer_in.uuid = "peer-in-1"
        client.feed(peer_in)
        deadline = _time.time() + 2
        while not session._unsolicited_burst_open and _time.time() < deadline:
            _time.sleep(0.01)
        assert session._unsolicited_burst_open

        real_query = client.query

        async def query_after_peer_finishes(text):
            # The CLI finishes the injected turn before it starts the host's.
            peer_result = ResultMessage(result="peer answer", uuid="peer-res-1")
            peer_result.origin = peer_origin
            client.feed(AssistantMessage(content=[TextBlock("peer answer")]), peer_result)
            await real_query(text)

        client.query = query_after_peer_finishes
        turn = session.run_turn("host question", turn_timeout=10.0)
    finally:
        session.close()

    assert turn.final_text == "host answer"
    assert all("peer answer" not in str(m.get("content")) for m in turn.projected_messages)
    assert delivered, "the peer burst must be delivered on the background path"
    texts, items = delivered[-1]
    assert "peer answer" in " ".join(texts)
    assert any(item.get("kind") == "peer_in" for item in (items or []))
    assert session._unsolicited_burst_open is False


def test_idle_subagent_assistant_does_not_open_woken_turn():
    import time as _time
    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, StreamEvent, TextBlock, UserMessage, _make_session,
    )

    session, holder = _make_session()
    try:
        session.ensure_started()
        child_user = UserMessage(content="child task notification")
        child_user.parent_tool_use_id = "toolu-parent"
        child_user.origin = {"kind": "task-notification"}
        holder["client"].feed(
            AssistantMessage(content=[TextBlock("child progress")], parent_tool_use_id="toolu-parent"),
            child_user,
            StreamEvent(parent_tool_use_id="toolu-parent"),
        )
        deadline = _time.time() + 2
        while holder["client"]._pending and _time.time() < deadline:
            _time.sleep(0.01)
        assert session.woken_turn_active() is False
    finally:
        session.close()


def test_top_level_unsolicited_result_closes_woken_turn_once():
    import time as _time
    from tests.agent.claude_sdk_fakes import AssistantMessage, ResultMessage, TextBlock, _make_session

    boundaries = []
    session, holder = _make_session()
    session.set_idle_boundary_callback(lambda: boundaries.append(True))
    try:
        session.ensure_started()
        holder["client"].feed(AssistantMessage(content=[TextBlock("top-level work")]))
        deadline = _time.time() + 2
        while not session.woken_turn_active() and _time.time() < deadline:
            _time.sleep(0.01)
        assert session.woken_turn_active() is True
        holder["client"].feed(ResultMessage(result="done", uuid="woken-result"))
        while session.woken_turn_active() and _time.time() < deadline:
            _time.sleep(0.01)
        assert session.woken_turn_active() is False
        assert boundaries == [True]
    finally:
        session.close()


def test_top_level_unsolicited_quiet_interval_fires_boundary_without_result():
    import time as _time
    from agent.transports import claude_agent_sdk_session as sdk_module
    from tests.agent.claude_sdk_fakes import AssistantMessage, TextBlock, _make_session

    now = [100.0]
    boundaries = []
    session, holder = _make_session()
    session._unsolicited_clock = lambda: now[0]
    session.set_idle_boundary_callback(lambda: boundaries.append(True))
    try:
        session.ensure_started()
        holder["client"].feed(AssistantMessage(content=[TextBlock("top-level work")]))
        deadline = _time.time() + 2
        while not session.woken_turn_active() and _time.time() < deadline:
            _time.sleep(0.01)
        assert session.woken_turn_active() is True
        now[0] += sdk_module.UNSOLICITED_QUIET_S + 0.1
        session._expire_unsolicited_burst()
        assert session.woken_turn_active() is False
        assert boundaries == [True]
    finally:
        session.close()


def test_only_top_level_activity_extends_unsolicited_quiet_window():
    import time as _time
    from agent.transports import claude_agent_sdk_session as sdk_module
    from tests.agent.claude_sdk_fakes import AssistantMessage, TextBlock, _make_session

    now = [100.0]
    session, holder = _make_session()
    session._unsolicited_clock = lambda: now[0]
    try:
        session.ensure_started()
        holder["client"].feed(AssistantMessage(content=[TextBlock("top-level work")]))
        deadline = _time.time() + 2
        while not session.woken_turn_active() and _time.time() < deadline:
            _time.sleep(0.01)
        assert session.woken_turn_active()
        now[0] += sdk_module.UNSOLICITED_QUIET_S - 1
        holder["client"].feed(AssistantMessage(content=[TextBlock("more work")]))
        while session._unsolicited_last_top_level_at != now[0] and _time.time() < deadline:
            _time.sleep(0.01)
        assert session._unsolicited_last_top_level_at == now[0]
        now[0] += 2
        assert session.woken_turn_active()
        holder["client"].feed(AssistantMessage(
            content=[TextBlock("child noise")], parent_tool_use_id="toolu-parent"))
        while holder["client"]._pending and _time.time() < deadline:
            _time.sleep(0.01)
        now[0] += sdk_module.UNSOLICITED_QUIET_S
        session._expire_unsolicited_burst()
        assert session.woken_turn_active() is False
    finally:
        session.close()


def _gateway_owner_for_sdk(monkeypatch, sdk):
    import contextlib
    import threading
    from types import SimpleNamespace
    from tui_gateway import server, session_mailbox

    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a: None)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda _s, **_kw: contextlib.nullcontext())
    monkeypatch.setattr(server, "_start_session_work", lambda target, **_kw: target())
    monkeypatch.setattr(session_mailbox, "schedule_drain", lambda *_a, **_kw: None)
    monkeypatch.setattr(session_mailbox, "drain_session", lambda *_a, **_kw: 0)
    fired = []
    monkeypatch.setattr(server, "_run_prompt_submit", lambda _r, _sid, _s, text, **_kw: fired.append(text))
    gateway_session = {
        "agent": SimpleNamespace(api_mode="claude_agent_sdk", _claude_sdk_session=sdk),
        "session_key": "sdk-owner-boundary", "history": [], "history_lock": threading.RLock(),
        "history_version": 0, "running": False, "transport": None, "attached_images": [],
        "image_counter": 0, "cols": 80, "slash_worker": None, "show_reasoning": False,
        "tool_progress_mode": "all",
    }
    monkeypatch.setitem(server._sessions, "sdk-owner-boundary", gateway_session)
    return server, gateway_session, fired


def test_idle_subagent_output_allows_immediate_owner_dispatch(monkeypatch):
    import time as _time
    from tests.agent.claude_sdk_fakes import AssistantMessage, TextBlock, _make_session

    sdk, holder = _make_session()
    server, gateway_session, fired = _gateway_owner_for_sdk(monkeypatch, sdk)
    try:
        sdk.ensure_started()
        holder["client"].feed(AssistantMessage(
            content=[TextBlock("child progress")], parent_tool_use_id="toolu-parent"))
        deadline = _time.time() + 2
        while holder["client"]._pending and _time.time() < deadline:
            _time.sleep(0.01)
        response = server.handle_request({"id": "owner", "method": "prompt.submit",
                                          "params": {"session_id": "sdk-owner-boundary", "text": "owner next"}})
        assert response["result"]["status"] == "streaming"
        assert fired == ["owner next"]
        assert gateway_session.get("queued_prompt") is None
    finally:
        sdk.close()


def test_quiet_sdk_boundary_drains_queued_owner_prompt_once(monkeypatch):
    import time as _time
    from agent.transports import claude_agent_sdk_session as sdk_module
    from tests.agent.claude_sdk_fakes import AssistantMessage, TextBlock, _make_session

    now = [100.0]
    sdk, holder = _make_session()
    sdk._unsolicited_clock = lambda: now[0]
    server, gateway_session, fired = _gateway_owner_for_sdk(monkeypatch, sdk)
    try:
        sdk.ensure_started()
        holder["client"].feed(AssistantMessage(content=[TextBlock("top-level work")]))
        deadline = _time.time() + 2
        while not sdk.woken_turn_active() and _time.time() < deadline:
            _time.sleep(0.01)
        assert sdk.woken_turn_active()
        response = server.handle_request({"id": "owner", "method": "prompt.submit",
                                          "params": {"session_id": "sdk-owner-boundary", "text": "owner next"}})
        assert response["result"]["status"] == "queued"
        assert gateway_session["queued_prompt"]["text"] == "owner next"
        now[0] += sdk_module.UNSOLICITED_QUIET_S + 0.1
        sdk._expire_unsolicited_burst()
        assert fired == ["owner next"]
        sdk._expire_unsolicited_burst()
        assert fired == ["owner next"]
    finally:
        sdk.close()


def test_dead_cli_releases_queued_owner_and_restarts_on_drain(monkeypatch, caplog):
    import logging
    import threading
    import time as _time
    from functools import partial

    from agent import claude_sdk_runtime_session as runtime_session
    from tui_gateway import tool_progress
    from tests.agent.claude_sdk_fakes import (
        _EOS, AssistantMessage, ResultMessage, TextBlock, _make_session,
    )

    sdk, holder = _make_session()
    server, gateway_session, fired = _gateway_owner_for_sdk(monkeypatch, sdk)
    agent = _make_agent()
    agent.api_mode = "claude_agent_sdk"
    agent._claude_sdk_session = sdk
    gateway_session["agent"] = agent
    fresh_clients = []
    results = []
    workers = []
    subagent_events = []

    def create_fresh(_agent, **_kwargs):
        fresh, fresh_holder = _make_session(script=[ResultMessage(result="new answer")])
        fresh_clients.append(fresh_holder)
        _agent._claude_sdk_session = fresh
        return fresh

    def dispatch(_rid, _sid, _session, text, **_kwargs):
        fired.append(text)
        results.append(run_claude_agent_sdk_turn(
            agent, user_message=text, original_user_message=text,
            messages=[{"role": "user", "content": text}], effective_task_id="t",
        ))

    monkeypatch.setattr(runtime_session, "_create_session", create_fresh)
    monkeypatch.setattr(server, "_run_prompt_submit", dispatch)
    monkeypatch.setattr(tool_progress, "_emit", lambda event, sid, payload: subagent_events.append((event, sid, payload)),
                        raising=False)
    def start_work(target, **_kwargs):
        worker = threading.Thread(target=target, daemon=True)
        workers.append(worker)
        worker.start()

    monkeypatch.setattr(server, "_start_session_work", start_work)
    caplog.set_level(logging.INFO)
    try:
        sdk.ensure_started()
        dead_client = holder["client"]
        dead_client.feed(AssistantMessage(content=[TextBlock("partial background work")]))
        deadline = _time.time() + 3
        while not sdk.woken_turn_active() and _time.time() < deadline:
            _time.sleep(0.01)
        assert sdk.woken_turn_active()
        response = server.handle_request({"id": "owner", "method": "prompt.submit",
                                          "params": {"session_id": "sdk-owner-boundary", "text": "owner next"}})
        assert response["result"]["status"] == "queued"
        sdk._sdk_task_records = {"child-1": {"goal": "background child", "parent_tool_id": None}}
        agent.tool_progress_callback = lambda event, name, preview, args, **kw: tool_progress._on_tool_progress(
            "sdk-owner-boundary", event, name, preview, args, **kw
        )
        sdk._on_subagent_event = partial(runtime_session._on_sdk_subagent_event, agent)
        dead_client.feed(_EOS)
        while sdk._stream_ended is None and _time.time() < deadline:
            _time.sleep(0.01)
        assert sdk._stream_ended is not None
        while not results and _time.time() < deadline:
            _time.sleep(0.01)
        for worker in workers:
            worker.join(timeout=1)
        assert sdk.woken_turn_active() is False
        assert sdk._unsolicited_text == []
        assert sdk._unsolicited_items == []
        assert sdk._unsolicited_tool_items == {}
        assert sdk._unsolicited_seen == set()
        assert sdk._native_peer_written is False
        assert sdk._turn_inbox is None
        assert sdk._turn_claim_requested is False
        assert sdk._turn_claims is None
        assert sdk._rename_claim_requested is False
        assert sdk._pending_rename_ack is None
        assert sdk._deferred_rename is None
        assert sdk._resume_backlog is None
        assert sdk._sdk_task_records == {}
        assert [(event, sid, payload["status"]) for event, sid, payload in subagent_events] == [
            ("subagent.complete", "sdk-owner-boundary", "interrupted")
        ]
        assert sum("unsolicited burst closed" in record.message and "reason=stream-ended" in record.message
                   for record in caplog.records) == 1
        assert fired == ["owner next"]
        assert len(fresh_clients) == 1
        assert fresh_clients[0]["client"].queried == ["owner next"]
        assert results[0]["final_response"] == "new answer"
        assert gateway_session.get("queued_prompt") is None
    finally:
        sdk.close()
        if agent._claude_sdk_session is not sdk:
            agent._claude_sdk_session.close()


def test_reader_start_error_reaches_idle_boundary():
    import time as _time
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
    from tests.agent.claude_sdk_fakes import _FakeClient

    class BrokenClient(_FakeClient):
        def receive_messages(self):
            raise RuntimeError("transport failed")

    boundaries = []
    sdk = ClaudeAgentSdkSession(cwd="/tmp", client_factory=lambda **_kw: BrokenClient())
    sdk.set_idle_boundary_callback(lambda: boundaries.append(True))
    try:
        sdk.ensure_started()
        deadline = _time.time() + 2
        while sdk._stream_ended is None and _time.time() < deadline:
            _time.sleep(0.01)
        assert sdk._stream_ended is not None
        assert sdk._stream_ended.error == "transport failed"
        assert boundaries == [True]
    finally:
        sdk.close()


def test_quiet_timer_closes_without_another_reader_message(monkeypatch, caplog):
    import logging
    import time as _time
    from agent.transports import claude_agent_sdk_session as sdk_module
    from tests.agent.claude_sdk_fakes import AssistantMessage, TextBlock, _make_session

    now = [100.0]
    scheduled = []

    class Timer:
        def __init__(self, interval, callback):
            self.interval, self.callback = interval, callback
            self.daemon = False

        def start(self):
            scheduled.append(self)

        def cancel(self):
            pass

    sdk, holder = _make_session()
    sdk._unsolicited_clock = lambda: now[0]
    boundaries = []
    sdk.set_idle_boundary_callback(lambda: boundaries.append(True))
    try:
        caplog.set_level(logging.INFO)
        sdk.ensure_started()
        monkeypatch.setattr(sdk_module.threading, "Timer", Timer)
        holder["client"].feed(AssistantMessage(content=[TextBlock("background")]))
        deadline = _time.time() + 2
        while not scheduled and _time.time() < deadline:
            _time.sleep(0.01)
        assert len(scheduled) == 1
        assert scheduled[0].interval == sdk_module.UNSOLICITED_QUIET_S
        now[0] += sdk_module.UNSOLICITED_QUIET_S + 0.1
        scheduled[0].callback()
        assert sdk.woken_turn_active() is False
        assert boundaries == [True]
        assert sum("unsolicited burst closed" in record.message and "reason=quiet-interval" in record.message
                   for record in caplog.records) == 1
    finally:
        sdk.close()


def test_unsolicited_burst_logs_open_and_result_close_with_session_tag(caplog):
    import logging
    import time as _time
    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, ToolUseBlock, _make_session,
    )

    sdk, holder = _make_session(hermes_session_id="owner-log-id")
    try:
        sdk.ensure_started()
        with caplog.at_level(logging.INFO):
            holder["client"].feed(AssistantMessage(content=[
                ToolUseBlock(id="tool-1", name="Agent", input={"description": "private"}),
            ]))
            deadline = _time.time() + 2
            while not sdk.woken_turn_active() and _time.time() < deadline:
                _time.sleep(0.01)
            holder["client"].feed(ResultMessage(result="done", uuid="result-1"))
            while sdk.woken_turn_active() and _time.time() < deadline:
                _time.sleep(0.01)
        lines = [record.message for record in caplog.records if "unsolicited" in record.message]
        assert any("owner-log-id" in line and "opened" in line and "AssistantMessage" in line
                   and "Agent" in line and "top_level=True" in line for line in lines)
        assert any("owner-log-id" in line and "closed" in line and "result" in line for line in lines)
        assert all("private" not in line for line in lines)
    finally:
        sdk.close()


def test_late_result_after_quiet_boundary_cannot_answer_owner_turn():
    import time as _time
    from agent.transports import claude_agent_sdk_session as sdk_module
    from tests.agent.claude_sdk_fakes import AssistantMessage, ResultMessage, TextBlock, _make_session

    now = [100.0]
    sdk, holder = _make_session(script=[
        AssistantMessage(content=[TextBlock("owner answer")]),
        ResultMessage(result="owner answer", uuid="owner-result"),
    ])
    sdk._unsolicited_clock = lambda: now[0]
    try:
        sdk.ensure_started()
        client = holder["client"]
        client.feed(AssistantMessage(content=[TextBlock("old burst")]))
        deadline = _time.time() + 2
        while not sdk.woken_turn_active() and _time.time() < deadline:
            _time.sleep(0.01)
        assert sdk.woken_turn_active()
        now[0] += sdk_module.UNSOLICITED_QUIET_S + 0.1
        sdk._expire_unsolicited_burst()
        real_query = client.query

        async def query_after_late_result(text):
            client.feed(ResultMessage(result="late burst result", uuid="late-result"))
            await real_query(text)

        client.query = query_after_late_result
        turn = sdk.run_turn("owner prompt", turn_timeout=5.0)
        assert turn.final_text == "owner answer"
    finally:
        sdk.close()


def test_host_prompt_folded_into_a_peer_turn_answers_the_host_turn(caplog):
    """Live 2026-09-24 (session ce3d96): a peer turn was running when the owner's prompt
    arrived. The CLI folded the queued prompt into that turn at a tool boundary (it replays
    the prompt mid-turn, proven with a live probe) and answered both with ONE ResultMessage
    that keeps the peer origin. Routing that result to the background lane left the host turn
    waiting out its 600 s timeout, the answer landed late as a background row, and the runtime
    was retired. After a mid-burst replay, the result is the host turn's answer."""
    import time as _time
    import logging

    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, TextBlock, UserMessage, _make_session,
    )

    peer_origin = {
        "kind": "peer", "from": "uds:/tmp/x.sock", "name": "hermes:other",
        "body": "ping", "msg_id": "mailbox-row-42",
    }
    folded = ResultMessage(result="answer to both", uuid="folded-res-1")
    folded.origin = peer_origin
    delivered = []
    # The fake replays the host prompt on query(), like --replay-user-messages; the script
    # is the rest of the (folded) injected turn.
    session, holder = _make_session(
        script=[AssistantMessage(content=[TextBlock("answer to both")]), folded],
        on_unsolicited_result=lambda texts, items=None: delivered.append((texts, items)),
    )
    try:
        caplog.set_level(logging.INFO)
        session.ensure_started()
        peer_in = UserMessage(content="ping")
        peer_in.origin = peer_origin
        peer_in.uuid = "peer-in-1"
        holder["client"].feed(peer_in)
        deadline = _time.time() + 2
        while not session._unsolicited_burst_open and _time.time() < deadline:
            _time.sleep(0.01)
        assert session._unsolicited_burst_open
        started = _time.time()
        turn = session.run_turn("host question", turn_timeout=5.0)
        elapsed = _time.time() - started
    finally:
        session.close()

    assert turn.error is None
    assert turn.final_text == "answer to both"
    assert elapsed < 4.0, "the host turn must not wait out its timeout"
    assert sum("unsolicited burst closed" in record.message and "reason=host-prompt" in record.message
               for record in caplog.records) == 1
    assert all("answer to both" not in " ".join(texts) for texts, _items in delivered)
    # The peer message that started the folded turn still gets its card.
    assert any(item.get("kind") == "peer_in" and item.get("uuid") == "peer-in-1"
               for _texts, items in delivered for item in (items or []))
    assert any(item.get("msg_id") == "mailbox-row-42"
               for _texts, items in delivered for item in (items or []))


def test_peer_message_folded_during_host_tool_round_keeps_display_order():
    """A peer row arriving during a host turn belongs between its tool card and answer."""
    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock,
        UserMessage, _make_session,
    )

    events = []
    peer_origin = {
        "kind": "peer", "from": "uds:/tmp/peer.sock", "name": "hermes:peer",
        "body": "please check this",
    }
    peer = UserMessage(content="please check this")
    peer.origin = peer_origin
    peer.uuid = "peer-folded-1"
    session, holder = _make_session(
        script=[],
        on_tool_use=lambda *_args: events.append(("tool", None)),
        on_unsolicited_result=lambda texts, items=None, _delivery_id=None: events.append(
            ("peer", items[0] if items else None)
        ),
    )
    try:
        session.ensure_started()
        client = holder["client"]

        async def query_with_folded_peer(text):
            client.queried.append(text)
            client.feed(
                UserMessage(content=str(text)),
                AssistantMessage(content=[ToolUseBlock(id="tool-1", name="Bash", input={"command": "true"})]),
                UserMessage(content=[ToolResultBlock(tool_use_id="tool-1", content="done")]),
                peer,
                AssistantMessage(content=[TextBlock("host reply")]),
                ResultMessage(result="host reply", uuid="host-peer-fold-1"),
            )

        client.query = query_with_folded_peer
        turn = session.run_turn("host prompt", turn_timeout=5, post_tool_quiet_timeout=0.0)
        events.append(("reply", turn.final_text))
    finally:
        session.close()

    assert [event for event, _value in events] == ["tool", "peer", "reply"]
    assert events[1][1]["kind"] == "peer_in"
    assert events[1][1]["uuid"] == "peer-folded-1"
    assert events[1][1]["text"] == "please check this"
    assert events[2][1] == "host reply"


def test_background_result_before_callback_wiring_is_replayed_when_wired():
    """A fast CLI result survives the short gap before the turn installs delivery hooks."""
    from tests.agent.claude_sdk_fakes import AssistantMessage, ResultMessage, TextBlock
    from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession

    delivered = []
    session = ClaudeAgentSdkSession(cwd="/tmp")
    assistant = AssistantMessage(content=[TextBlock("early result")])
    result = ResultMessage(result="early result", uuid="early-result-1")
    try:
        session._handle_unsolicited(assistant)
        session._handle_unsolicited(result)
        assert delivered == []
        assert len(session._pending_unsolicited_deliveries) == 1
        session.set_unsolicited_result_callback(
            lambda texts, items=None, delivery_id=None: delivered.append(
                (texts, items, delivery_id)
            )
        )
    finally:
        session.close()

    assert len(delivered) == 1
    assert delivered[0][0] == ["early result"]
    assert any(item.get("text") == "early result" for item in delivered[0][1])


@pytest.mark.parametrize("origin_kind", ["peer", "task-notification"])
def test_injected_final_is_delivered_without_interim_duplicate(origin_kind):
    """Peer and task-woken finals use background delivery, never interim text."""
    import time as _time

    from tests.agent.claude_sdk_fakes import AssistantMessage, ResultMessage, TextBlock, UserMessage, _make_session

    interim = []
    finals = []
    origin = {
        "kind": origin_kind, "from": "uds:/tmp/peer.sock", "name": "hermes:peer", "body": "ping"
    }
    wake = UserMessage(content="ping")
    wake.origin = origin
    wake.uuid = "peer-wake-no-duplicate"
    session, holder = _make_session(
        script=[],
        on_interim_assistant=interim.append,
        on_unsolicited_result=lambda texts, items=None, delivery_id=None: finals.append(list(texts)),
    )
    try:
        session.ensure_started()
        holder["client"].feed(
            wake,
            AssistantMessage(content=[TextBlock("peer final answer")]),
            ResultMessage(result="peer final answer", uuid="peer-final-no-duplicate"),
        )
        deadline = _time.time() + 3
        while not finals and _time.time() < deadline:
            _time.sleep(0.01)
    finally:
        session.close()

    assert finals == [["peer final answer"]]
    assert interim == []


def test_diverted_injected_result_does_not_flush_host_final_paragraph():
    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, TextBlock, _make_session,
    )

    interim, background = [], []
    injected = ResultMessage(result="peer result", uuid="peer-result")
    injected.origin = {"kind": "peer", "from": "peer-session", "body": "ping"}
    session, _holder = _make_session(
        script=[
            AssistantMessage(content=[TextBlock("held host paragraph")]),
            injected,
            ResultMessage(result="held host paragraph", uuid="host-result"),
        ],
        on_interim_assistant=interim.append,
        on_unsolicited_result=lambda texts, items=None, delivery_id=None: background.append(texts),
    )
    try:
        turn = session.run_turn("host question", turn_timeout=5)
    finally:
        session.close()

    assert turn.final_text == "held host paragraph"
    assert background == [["peer result"]]
    assert interim == []


def test_stream_end_flushes_held_assistant_as_partial_once():
    from tests.agent.claude_sdk_fakes import AssistantMessage, TextBlock, _make_session

    interim = []
    session, _holder = _make_session(
        script=[AssistantMessage(content=[TextBlock("partial host paragraph")])],
        on_interim_assistant=interim.append,
    )
    try:
        turn = session.run_turn("host question", turn_timeout=5)
    finally:
        session.close()

    assert turn.error is not None
    assert interim == ["partial host paragraph"]


def test_multi_step_turn_relays_intermediate_prose_but_not_final_answer():
    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, _make_session,
        UserMessage,
    )

    interim = []
    session, holder = _make_session(
        script=[
            AssistantMessage(content=[TextBlock("checking the workspace")]),
            AssistantMessage(content=[TextBlock("running a check"), ToolUseBlock(
                id="interim-tool", name="Bash", input={"command": "true"}
            )]),
            UserMessage(content=[ToolResultBlock(tool_use_id="interim-tool", content="done")]),
            AssistantMessage(content=[TextBlock("final answer")]),
            ResultMessage(result="final answer", uuid="multi-step-final"),
        ],
        on_interim_assistant=interim.append,
    )
    try:
        turn = session.run_turn("check", turn_timeout=5, post_tool_quiet_timeout=0.0)
    finally:
        session.close()

    assert turn.final_text == "final answer"
    assert interim == ["checking the workspace", "running a check"]


def test_bash_background_task_is_not_a_subagent_and_agent_task_is_not_a_process():
    """Owner screenshot 09-16: Agent tasks showed under both Subagents and Background."""
    from agent.transports.claude_sdk_background_tasks import classify_sdk_task

    class Started:
        def __init__(self, task_type, tool_use_id="t1"):
            self.task_type = task_type
            self.tool_use_id = tool_use_id

    assert classify_sdk_task(Started("local_bash")) == "shell"
    assert classify_sdk_task(Started("local_agent")) == "agent"
    assert classify_sdk_task(Started("remote_agent")) == "agent"
    assert classify_sdk_task(Started("dream")) == "other"
    assert classify_sdk_task(Started(None), agent_tool_ids={"t1"}) == "agent"
    assert classify_sdk_task(Started(None)) == "unknown"


def test_injected_origin_result_in_the_host_inbox_is_never_the_answer():
    from tests.agent.claude_sdk_fakes import AssistantMessage, ResultMessage, TextBlock, _make_session

    stray = ResultMessage(result="peer answer", uuid="peer-res-2")
    stray.origin = {"kind": "peer", "from": "uds:/tmp/y.sock", "body": "hi"}
    delivered = []
    session, _holder = _make_session(
        script=[stray, AssistantMessage(content=[TextBlock("host answer")]),
                ResultMessage(result="host answer", uuid="host-2")],
        on_unsolicited_result=lambda texts, items=None: delivered.append(texts),
    )
    try:
        turn = session.run_turn("host question", turn_timeout=10.0)
    finally:
        session.close()
    assert turn.final_text == "host answer"


def test_a_woken_turn_closes_the_tool_cards_it_opens():
    """Owner screenshots 09-20/21: after a peer-woken turn, every tool it ran (bash, sendmessage,
    plan_contract_write) showed 'Result unavailable'. The unsolicited path opened a card per tool
    call but only the foreground loop ever closed one, so they were all sealed at settle."""
    import time as _time

    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage,
        _make_session,
    )

    opened, closed = [], []
    session, holder = _make_session(
        script=[],
        on_unsolicited_result=lambda texts, items=None: None,
        on_tool_use=lambda tool_id, name, args: opened.append((tool_id, name)),
        on_tool_result=lambda tool_id, name, args, result, **kwargs: closed.append((tool_id, name)),
    )
    peer_origin = {"kind": "peer", "from": "uds:/tmp/x.sock", "name": "hermes:audit", "body": "ping"}
    try:
        session.ensure_started()
        client = holder["client"]
        woke = UserMessage(content="ping")
        woke.origin = peer_origin
        client.feed(
            woke,
            AssistantMessage(content=[ToolUseBlock(id="tool-bash", name="Bash", input={"command": "ls"})]),
            UserMessage(content=[ToolResultBlock(tool_use_id="tool-bash", content="total 0")]),
            AssistantMessage(content=[ToolUseBlock(id="tool-msg", name="SendMessage", input={"to": "x"})]),
            UserMessage(content=[ToolResultBlock(tool_use_id="tool-msg", content="sent")]),
            AssistantMessage(content=[TextBlock("replied")]),
            ResultMessage(result="replied", uuid="woken-1"),
        )
        deadline = _time.time() + 3
        while len(closed) < 2 and _time.time() < deadline:
            _time.sleep(0.01)
    finally:
        session.close()

    assert [name for _id, name in opened] == ["Bash", "SendMessage"]
    assert sorted(closed) == sorted(opened), "every card a woken turn opens must be closed with its result"


def test_retained_unsolicited_deliveries_are_bounded_without_a_callback():
    """Background delivery off = no result callback is ever wired; retaining every peer/task
    delivery "until the callback exists" would grow without bound on a long-lived session."""
    from agent.transports import claude_agent_sdk_session_turn as turn_mod
    from tests.agent.claude_sdk_fakes import _make_session

    session, _holder = _make_session(script=[])
    try:
        assert session._on_unsolicited_result is None
        for index in range(turn_mod._MAX_RETAINED_UNSOLICITED + 9):
            session._deliver_or_buffer_unsolicited([f"answer {index}"], [], f"deliv-{index}")
        pending = session._pending_unsolicited_deliveries
        assert len(pending) == turn_mod._MAX_RETAINED_UNSOLICITED
        assert pending[-1][2] == f"deliv-{turn_mod._MAX_RETAINED_UNSOLICITED + 8}"
    finally:
        session.close()


def test_session_deduplication_id_sets_keep_only_the_newest_entries():
    from agent.transports import claude_agent_sdk_session_turn as turn_mod
    from tests.agent.claude_sdk_fakes import ResultMessage, UserMessage, _make_session

    session, _holder = _make_session(script=[], on_unsolicited_result=lambda *_args: None)
    try:
        max_ids = 512
        count = max_ids + 8
        for index in range(count):
            session._handle_unsolicited(
                ResultMessage(result=f"background {index}", uuid=f"result-{index}")
            )
            peer = UserMessage(content=f"peer {index}")
            peer.uuid = f"peer-{index}"
            session._deliver_host_peer_item(peer, {"kind": "peer", "body": f"peer {index}"})

        assert len(session._unsolicited_delivered) == max_ids
        assert len(session._host_peer_seen) == max_ids
        assert "result-0" not in session._unsolicited_delivered
        assert f"result-{count - 1}" in session._unsolicited_delivered
        assert "peer-0" not in session._host_peer_seen
        assert f"peer-{count - 1}" in session._host_peer_seen
    finally:
        session.close()


def test_unsolicited_result_releases_the_native_peer_claim_without_a_recognised_echo():
    """Review P0 (b9): the claim must not depend on parsing the peer echo.

    A peer turn whose UserMessage lost its origin still ends with a ResultMessage;
    that result has to free the CLI for owner input and the next mailbox row."""
    import time as _time

    from tests.agent.claude_sdk_fakes import (
        AssistantMessage, ResultMessage, TextBlock, UserMessage, _make_session,
    )

    session, holder = _make_session(script=[], on_unsolicited_result=lambda *a, **k: None)
    try:
        session.ensure_started()
        with session._turn_callback_lock:
            session._native_peer_in_flight = True
            session._native_peer_msg_id = "row-9"
            session._native_peer_since = _time.monotonic()
        client = holder["client"]
        echo = UserMessage(content="wrapped peer text")  # origin stripped by the CLI
        echo.uuid = "peer-in-9"
        result = ResultMessage(result="done", uuid="peer-res-9")
        client.feed(echo, AssistantMessage(content=[TextBlock("done")]), result)
        deadline = _time.time() + 2
        while session._native_peer_in_flight and _time.time() < deadline:
            _time.sleep(0.01)
        assert session._native_peer_in_flight is False
        assert session.woken_turn_active() is False
        assert session.native_peer_idle() is True
    finally:
        session.close()
