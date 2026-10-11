"""Session adapter for the claude-agent-sdk runtime.

Owns one Claude Agent SDK client per Hermes session — the structural twin of
``codex_app_server_session.py``, with the Codex JSON-RPC subprocess replaced
by Anthropic's official ``claude-agent-sdk`` (which manages the Claude Code
CLI subprocess, its agent loop, and — critically — **subscription OAuth**:
``CLAUDE_CODE_OAUTH_TOKEN`` / the ``~/.claude`` credential store by default;
known metered sources and Extra Usage fail closed unless the operator opts in).
See GitHub issue #25267.

Lifecycle:
    session = ClaudeAgentSdkSession(cwd="/home/x/proj", model="claude-opus-4-8")
    session.ensure_started()                       # loop thread + SDK connect
    result = session.run_turn(user_input="hello")  # blocks until ResultMessage
    session.close()                                # disconnect + stop loop

Threading model: the SDK is async-first, but AIAgent.run_conversation() is
synchronous (the same constraint that made CodexAppServerClient thread-based).
The adapter owns a dedicated background thread running one asyncio event loop
for the whole session lifetime; every SDK coroutine is marshaled onto it with
``asyncio.run_coroutine_threadsafe`` and awaited with a timeout, so the SDK
client keeps stable loop affinity and ``run_turn`` stays blocking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import contextlib
import threading
import time
import uuid
from typing import Any, Callable, Optional

from agent.transports.claude_agent_sdk_session_availability import (
    _safe_sdk_error_text,
    check_claude_sdk_available,
)
from agent.transports.claude_agent_sdk_session_sanitize import (
    safe_sdk_tool_presentation_from_canonical,
    validate_canonical_sdk_request_serialization,
)
from agent.transports.claude_agent_sdk_session_child import (
    ClaudeSdkChildProcessMixin,
    SdkShuttingDownError,
    _SDK_DISCONNECT_TIMEOUT_S,
    _admit_sdk_session,
    _force_kill_sdk_child,
    _forget_sdk_session,
    _own_sdk_child_process,
    _sdk_child_pid,
    sdk_shutdown_begun,
)
from agent.transports.claude_agent_sdk_session_input import (
    _coerce_turn_input,
    _sdk_image_content_block,
    _sdk_user_message_stream,
)
from agent.transports.claude_agent_sdk_session_config import (
    _HERMES_TO_SDK_PERMISSION_MODE,
    _build_hermes_tools_mcp_config,
    _configured_cli_path,
    _configured_plugins,
    _configured_session_name_template,
    render_sdk_session_name,
    _configured_hybrid_exclude,
    _configured_max_buffer_size,
    _configured_max_turns,
    _configured_permission_mode,
    _configured_setting_sources,
    _configured_auto_mode_settings,
    _http_mcp_entries_from_config,
    _is_subscription_oauth_token,
    _mcp_registry_tool_specs,
    _provider_config,
    _provider_flag,
    _sdk_env_overrides,
)
from agent.transports.claude_agent_sdk_session_watchdog import (
    _StreamEnd,
    _TurnWatch,
    _swallow_interrupt_result,
    _swallow_steer_result,
)
from agent.transports.claude_agent_sdk_session_billing import (
    ClaudeSdkBillingMixin,
)
from agent.transports.claude_agent_sdk_session_compaction import (
    ClaudeSdkCompactionMixin,
)
from agent.transports.claude_agent_sdk_session_hooks import (
    ClaudeSdkHooksMixin,
)
from agent.transports.claude_agent_sdk_session_notify import (
    ClaudeSdkNotifyMixin,
)
from agent.transports.claude_agent_sdk_session_permissions import (
    ClaudeSdkPermissionsMixin,
)
from agent.transports.claude_agent_sdk_session_turn import (
    ClaudeSdkTurnMixin,
)
from agent.transports import claude_sdk_peer_name_lease as _peer_lease

logger = logging.getLogger(__name__)
_RENAME_ACK_TIMEOUT_SECONDS = 10.0
_NATIVE_PEER_QUERY_TIMEOUT_SECONDS = 5.0
_NATIVE_PEER_ADMIT_GRACE_SECONDS = 120.0
_NATIVE_PEER_WRITE_MAX_SECONDS = 60.0
UNSOLICITED_QUIET_S = 120.0


class _RenameClaimDeferred(RuntimeError):
    """A rename claim reached the reader while an injected burst was open."""


def _run_disconnect_without_loop(disconnect_coro: Any) -> bool:
    """Run late cleanup without letting an uncooperative coroutine pin teardown."""
    error: list[BaseException] = []

    def run() -> None:
        async def bounded_disconnect() -> Any:
            return await asyncio.wait_for(
                disconnect_coro, timeout=_SDK_DISCONNECT_TIMEOUT_S
            )

        try:
            asyncio.run(bounded_disconnect())
        except BaseException as exc:
            error.append(exc)

    worker = threading.Thread(
        target=run, name="claude-sdk-late-disconnect", daemon=True
    )
    worker.start()
    worker.join(timeout=_SDK_DISCONNECT_TIMEOUT_S)
    if worker.is_alive():
        logger.warning(
            "claude-agent-sdk late disconnect exceeded %.1fs; abandoning daemon cleanup",
            _SDK_DISCONNECT_TIMEOUT_S,
        )
        return False
    if error:
        raise error[0]
    return True


class _TurnAdmissionReservation:
    __slots__ = ("loop", "released")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.released = False


# The names other packages import through this facade (the class, the runtime's and the
# auxiliary client's config/input entry points, the approval gateway's validators).
__all__ = [
    "ClaudeAgentSdkSession",
    "_coerce_turn_input",
    "_provider_config",
    "_provider_flag",
    "_sdk_env_overrides",
    "_sdk_image_content_block",
    "_sdk_user_message_stream",
    "safe_sdk_tool_presentation_from_canonical",
    "validate_canonical_sdk_request_serialization",
]


def _cli_model_id(model):
    """The Claude Code CLI takes API ids; catalogs spell some versions with dots (``claude-fable-5.1``)."""
    if not model:
        return model
    from agent.anthropic_message_convert import normalize_model_name

    return normalize_model_name(str(model))



def _forget_steer_mark(owner: Any, text: str) -> None:
    """Drop the newest position mark for *text* (caller holds the commit lock)."""
    marks = getattr(owner, "_pending_steer_marks", [])
    for index in range(len(marks) - 1, -1, -1):
        if marks[index].get("text") == text:
            del marks[index]
            return


class ClaudeAgentSdkSession(
    ClaudeSdkTurnMixin,
    ClaudeSdkPermissionsMixin,
    ClaudeSdkNotifyMixin,
    ClaudeSdkCompactionMixin,
    ClaudeSdkHooksMixin,
    ClaudeSdkBillingMixin,
    ClaudeSdkChildProcessMixin,
):
    """One SDK client per Hermes session, lifetime owned by AIAgent.

    Not thread-safe from the caller's side — one caller drives it at a time,
    matching AIAgent.run_conversation(). Internally owns a loop thread."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        approval_bypass_provider: Optional[Callable[[], bool]] = None,
        on_tool_started: Optional[Callable[[str, str, dict], None]] = None,
        on_tool_use: Optional[Callable[[str, str, dict], None]] = None,
        on_tool_result: Optional[Callable[[str, str, dict, str], None]] = None,
        max_budget_usd: Optional[float] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        include_hermes_tools: bool = True,
        hermes_session_id: Optional[str] = None,
        hermes_lineage: Optional[list[str]] = None,
        task_list_id: Optional[str] = None,
        task_env: Optional[dict[str, str]] = None,
        session_name: str = "",
        resume_session_id: Optional[str] = None,
        on_stream_delta: Optional[Callable[[str], None]] = None,
        on_interim_assistant: Optional[Callable[[str], None]] = None,
        on_tool_iteration: Optional[Callable[[], None]] = None,
        on_unsolicited_result: Optional[Callable[[list[str], Optional[list[dict]]], None]] = None,
        on_unsolicited_start: Optional[Callable[[str], None]] = None,
        on_unsolicited_header: Optional[Callable[[str, list[dict]], None]] = None,
        resume_interrupted_turn: bool = False,
        on_compaction: Optional[Callable[[str], None]] = None,
        on_compact_boundary: Optional[Callable[[str], None]] = None,
        # Hybrid MCP bridge (ported from PR #56413): the explicit config
        # opt-in plus both `agent` and `tools` activate in-process servers
        # exposing the full Hermes registry (including proxified third-party
        # MCPs). Missing either input or the opt-in preserves fcava's original
        # stdio-only behavior.
        agent: Optional[Any] = None,
        tools: Optional[list[dict]] = None,
        generation: Optional[str] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._model = model
        requested_permission_mode = (
            permission_mode
            or _configured_permission_mode()
            or _HERMES_TO_SDK_PERMISSION_MODE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "default",
            )
        )
        self._sdk_approval_bypass_requested = (
            requested_permission_mode == "bypassPermissions"
        )
        # The SDK auto-approves before can_use_tool in bypassPermissions mode.
        # Keep Hermes in callback-capable mode and carry the bypass intent into
        # this mandatory session wrapper, where canonical validation and
        # immutable hardline/sudo/user-deny floors run before every selected
        # gateway, CLI, ACP, plugin, or custom callback.
        self._permission_mode = (
            "default"
            if self._sdk_approval_bypass_requested
            else requested_permission_mode
        )
        self._system_prompt_append = system_prompt_append
        self._approval_callback = approval_callback
        self._approval_bypass_provider = approval_bypass_provider
        self._on_tool_started = on_tool_started
        # Stable-id tool CARD hooks (desktop/TUI tool rows), the codex-runtime
        # pattern: ``on_tool_use(tool_use_id, name, args)`` when a ToolUseBlock
        # issues, ``on_tool_result(tool_use_id, name, args, result)`` when its
        # ToolResultBlock echoes back. ``on_tool_started`` is only the progress
        # breadcrumb, which the gateway drops when a name is present — so
        # without these the desktop showed NO tool activity on this lane.
        self._on_tool_use = on_tool_use
        self._on_tool_result = on_tool_result
        self._open_tool_cards: dict[str, tuple[str, dict]] = {}
        self._on_compaction = on_compaction
        self._on_compact_boundary = on_compact_boundary
        self._max_budget_usd = max_budget_usd
        # Snapshot the canonical Hermes iteration cap once. The SDK options
        # are session-scoped and must remain byte-identical across resumes.
        self._max_turns = _configured_max_turns()
        # Frozen with the client: turn payloads are budgeted against the reader this CLI actually has.
        self._max_buffer_size = _configured_max_buffer_size()
        # Auto-mode classifier rules (agent.claude_agent_sdk.auto_mode),
        # resolved and serialized ONCE: a later ~/.claude/settings.json or
        # config.yaml edit must not change a live session's options
        # (prompt-cache safe); it lands on the next session.
        self._auto_mode_settings = _configured_auto_mode_settings()
        self._client_factory = client_factory  # test seam
        self._include_hermes_tools = include_hermes_tools
        # Hermes-side session id, exported to the CLI subprocess and hermes-tools MCP
        # subprocess so the CLI environment and session_search shim always carry THIS
        # session's id (and never a sibling session's ambient id).
        if hermes_session_id is None:
            try:
                from gateway.session_context import _SESSION_ID, _UNSET
                val = _SESSION_ID.get()
                if val is not _UNSET and val:
                    hermes_session_id = str(val)
            except Exception:
                pass
        self._hermes_session_id = hermes_session_id
        self._hermes_lineage = [str(x) for x in (hermes_lineage or []) if x][-16:]
        self._task_list_id = task_list_id
        self._task_env = dict(task_env) if task_env is not None else None
        # Peer-addressable name for the spawned CLI session (see
        # _configured_session_name_template). "" keeps the CLI's own naming.
        self._session_name = (session_name or "").strip()
        # Name of an in-flight /rename whose ack must be swallowed (never delivered as a
        # background result). Cleared when the ack arrives. (cntrl carry)
        self._pending_rename_ack: Optional[str] = None
        # A rename refused because a turn owned the stream; applied when that turn releases
        # it. CLI-originated turns never reach the Hermes turn boundary, so waiting for the
        # next Hermes turn left peers seeing the old name indefinitely. (cntrl carry)
        self._deferred_rename: Optional[str] = None
        # Peer names this session object holds a single-owner lease on (see
        # claude_sdk_peer_name_lease); two while a live /rename is unacknowledged.
        # A refused claim spawns the CLI without --name rather than duplicating
        # another live registrant's name (2026-09-25 orphan incident).
        self._peer_leases: set[str] = set()
        self._peer_name_refused = False
        self._peer_lease_owner_cache: Optional[_peer_lease.LeaseOwner] = None
        # SDK-side session id to resume (#25267 continuity). Verified live:
        # resume restores the model context and keeps the SAME session id; a
        # stale id fails the session start (the caller retires + retries
        # fresh).
        self._resume_session_id = resume_session_id
        self._planned_session_id = resume_session_id or str(uuid.uuid4())
        # Display-only partial-text consumer (W4 streaming). Deltas never
        # enter the projected transcript; the gateway's stream consumer
        # handles rate limiting and the already_sent final-send dedup.
        self._turn_callback_lock = threading.RLock()
        self._on_stream_delta = on_stream_delta
        # Hybrid MCP bridge inputs (see param docstring above).
        self._agent = agent
        self._tools = tools
        # Completed assistant prose that accompanies a tool call is a true
        # interim status, not final-answer text. It follows the gateway's
        # existing commentary callback so platforms can render it separately.
        # These callbacks are refreshed before EVERY run_turn: a session may
        # safely span several Hermes turns, while visibility must stay scoped
        # to the current one.
        self._on_interim_assistant = on_interim_assistant
        self._on_tool_iteration = on_tool_iteration
        self._instance_generation: str = str(generation) if generation else f"sdk_gen_{uuid.uuid4().hex[:12]}"
        self._generation: str = self._instance_generation

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._session_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        # Serializes pre-terminal interrupt admission with ResultMessage
        # acceptance.  The boolean remains true through ownership release so
        # a late /stop cannot still interrupt the persistent SDK client after
        # the turn has committed.
        self._interrupt_commit_lock = threading.Lock()
        self._terminal_result_committed = False
        self._post_terminal_interrupt_pending = False
        self._closed = False
        self._retiring = False
        # Startup and admission reservations close the check-to-use windows
        # without holding _turn_callback_lock across blocking work. A close
        # waits on the corresponding events before teardown can publish or
        # destroy resources owned by the reservation.
        self._starting = False
        self._startup_done = threading.Event()
        self._startup_done.set()
        self._admission_inflight = 0
        self._admission_reservations: set[_TurnAdmissionReservation] = set()
        self._admission_done = threading.Event()
        self._admission_done.set()
        self._turn_admission_reservation: Optional[_TurnAdmissionReservation] = None
        self._pending_turn_admission_future: Any = None
        # A close that has passed its bounded startup wait owns these objects.
        # Keep the identity markers until startup has signalled completion so a
        # late starter cannot disconnect or stop a resource twice.
        self._close_owned_client: Any = None
        self._close_owned_loop: Any = None
        self._close_owned_loop_thread: Any = None
        self._close_owned_reader_task: Any = None
        # Activity evidence for the in-flight turn (None between turns).
        self._turn_watch: Optional[_TurnWatch] = None
        # Snapshot the streaming posture once: the quiet-watchdog default is
        # derived from it, and a mid-session config edit must not flip the
        # watchdog's semantics while the live client still has the old
        # include_partial_messages option.
        self._streaming = _provider_flag("streaming")
        # Stream ownership (the protocol lives in
        # claude_agent_sdk_session_turn: `_reader_loop` arbitrates, `_consume_turn`
        # claims and releases). The reader task is the ONLY consumer of the SDK
        # stream; `_turn_inbox` is non-None exactly while a turn is in flight,
        # which is what makes "unsolicited" decidable.
        self._reader_task: Any = None
        self._turn_inbox: Any = None
        # Number of SDK queries submitted by ``steer()`` whose terminal result
        # still belongs to the foreground Hermes turn.  The SDK multiplexes
        # these results on the same stream and marks them with human origin.
        self._pending_steer_results = 0
        self._pending_steer_inputs: list[str] = []
        # One mark per pending steer, index-aligned with ``_pending_steer_inputs``:
        # ``pos`` is how many projected messages the turn had settled when the
        # steer was typed (stamped by the turn loop on the next stream message),
        # so the owner's row lands where it was sent, not before or after the
        # whole turn's output.
        self._pending_steer_marks: list[dict[str, Any]] = []
        # The foreground coroutine owns this acknowledgement while waiting
        # for the reader to publish its claim.  Shutdown resolves it even if
        # the reader cancellation has already dequeued the claim.
        self._turn_claim_ack: Any = None
        self._turn_claims: Any = None
        self._turn_claim_requested = False
        self._rename_claim_requested = False
        self._unsolicited_results = 0
        self._stream_ended: Optional[_StreamEnd] = None
        # Delivery half of the stream-ownership fix (dasbrow-hermes-coder#2):
        # a finished background Agent task's answer is captured here and
        # handed to the callback — it must never enter a turn's result, but
        # dropping it entirely left completed work silently undelivered
        # (observed live 2026-07-29: answers sat in the CLI session until the
        # operator poked). No callback wired = the historical drop semantics.
        self._on_unsolicited_result = on_unsolicited_result
        self._pending_unsolicited_deliveries: list[tuple[list[str], list[dict], Optional[str]]] = []
        self._on_unsolicited_start = on_unsolicited_start
        self._on_unsolicited_header = on_unsolicited_header
        # D62 L2: spawn with CLAUDE_CODE_RESUME_INTERRUPTED_TURN=1 (see _build_options).
        self._resume_interrupted_turn = bool(resume_interrupted_turn)
        self._unsolicited_delivery_id: Optional[str] = None
        self._unsolicited_header_emitted = False
        self._unsolicited_start_notified = False
        self._unsolicited_text: list[str] = []
        # True while a CLI-injected turn (peer message, task notification) is
        # in flight; the reader keeps routing it to the background path even
        # after a host turn claims the stream (see _reader_loop).
        self._unsolicited_burst_open = False
        self._unsolicited_quiet_expired = False
        self._replay_user_messages = True
        self._unsolicited_last_top_level_at: Optional[float] = None
        self._unsolicited_outstanding_tools: set[str] = set()
        self._unsolicited_quiet_timer: Optional[threading.Timer] = None
        self._unsolicited_clock = time.monotonic
        self._native_peer_in_flight = False
        self._native_peer_msg_id: Optional[str] = None
        self._native_peer_seen = False
        self._native_peer_since = 0.0
        # Each claim gets a generation: a late callback from an older write can never
        # clear a newer claim. The admission grace only runs once the write is confirmed.
        self._native_peer_gen = 0
        self._native_peer_written = False
        self._idle_boundary_callback: Optional[Callable[[], None]] = None
        self._unsolicited_items: list[dict] = []
        self._unsolicited_tool_items: dict[str, dict] = {}
        self._unsolicited_seen: set[str] = set()
        self._unsolicited_delivered: set[str] = set()
        # One immutable billing posture per child. The startup environment,
        # post-start evidence guard, and accounting label must never disagree
        # because config.yaml changed while a long-lived SDK session was live.
        self._allow_metered = _provider_flag("allow_metered_key")
        self._billing_evidence: dict[str, Any] = {}
        # ``slash_commands`` from the CLI's system/init message: built-ins,
        # bundled skills and every user-invocable plugin skill (``plugin:name``)
        # the spawned CLI will expand when a prompt is ``/<name> [args]``. The
        # live half of agent.claude_sdk_slash — Hermes' slash dispatchers use it
        # to forward plugin skills instead of printing "Unknown command".
        self.slash_commands: list[str] = []
        self._billing_guard_error: Optional[str] = None

    # ---------- lifecycle ----------

    def live_cli_session_id(self) -> Optional[str]:
        """#60 T-6: the Claude session id of the CONNECTED CLI (the id it announced on its stream, which
        is the ``session_id`` its hooks read from stdin), or None when no CLI is connected, it is
        retiring/closed, or it has not announced an id yet. Never read from state.db."""
        with self._turn_callback_lock:
            if self._retiring or self._closed or self._client is None:
                return None
            sid = self._session_id
        return sid if isinstance(sid, str) and sid and sid != "pending" else None

    def planned_cli_session_id(self) -> str:
        """Pre-assigned Claude session id (--session-id UUID, or the resumed id)."""
        return self._planned_session_id

    def _admission_retired(self) -> bool:
        """Return whether this session must reject a new SDK turn."""
        with self._turn_callback_lock:
            return bool(self._retiring or self._closed)

    def _reserve_turn_admission(
        self, reservation: Optional[_TurnAdmissionReservation] = None
    ) -> Optional[_TurnAdmissionReservation]:
        """Register a caller-owned token for the check-to-schedule gap.

        The normal caller creates ``reservation`` before entering this helper so
        an interrupt after insertion still leaves the caller with a token it can
        release. The no-argument form remains for narrow legacy/test seams.
        """
        with self._turn_callback_lock:
            if self._retiring or self._closed or self._loop is None:
                return None
            if reservation is None:
                reservation = _TurnAdmissionReservation(self._loop)
            reservations = self._admission_reservations
            try:
                self._admission_done.clear()
                reservations.add(reservation)
            finally:
                # The token set is the source of truth. Repair the event/count
                # pair even if an interrupt lands during either mutation.
                self._admission_inflight = len(reservations)
                if not reservations:
                    self._admission_done.set()
            return reservation

    @staticmethod
    def _new_turn_admission_reservation(
        loop: asyncio.AbstractEventLoop,
    ) -> _TurnAdmissionReservation:
        """Create an unregistered token for the caller-owned admission path."""
        return _TurnAdmissionReservation(loop)

    def _release_turn_admission(
        self, reservation: Optional[_TurnAdmissionReservation]
    ) -> None:
        if reservation is None:
            return
        with self._turn_callback_lock:
            reservations = self._admission_reservations
            if reservation.released and reservation not in reservations:
                if not reservations:
                    self._admission_done.set()
                return
            try:
                reservations.discard(reservation)
            finally:
                self._admission_inflight = len(reservations)
                reservation.released = reservation not in reservations
            if not reservations:
                self._admission_done.set()

    def _startup_is_retired(self) -> bool:
        with self._turn_callback_lock:
            return bool(self._retiring or self._closed)

    def _claim_startup_resource(self, attr: str, resource: Any) -> bool:
        """Claim a late startup resource unless close already owns it."""
        if resource is None:
            return False
        owned_attr = f"_close_owned{attr}"
        with self._turn_callback_lock:
            if getattr(self, owned_attr, None) is resource:
                return False
            current = getattr(self, attr, None)
            if current is resource:
                setattr(self, attr, None)
            return True

    def _cleanup_startup_resources(
        self,
        *,
        client: Any = None,
        loop: Any = None,
        loop_thread: Any = None,
        reader_task: Any = None,
    ) -> None:
        """Reap resources published after close() completed its cleanup."""
        claimed_client = self._claim_startup_resource("_client", client)
        claimed_loop = self._claim_startup_resource("_loop", loop)
        claimed_thread = self._claim_startup_resource("_loop_thread", loop_thread)
        claimed_reader = self._claim_startup_resource("_reader_task", reader_task)
        if claimed_reader and reader_task is not None and loop is not None:
            try:
                loop.call_soon_threadsafe(reader_task.cancel)
            except Exception:
                pass
        if claimed_reader and reader_task is not None and loop is not None:
            async def wait_reader() -> None:
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass

            try:
                future = asyncio.run_coroutine_threadsafe(wait_reader(), loop)
                future.result(timeout=min(1.0, _SDK_DISCONNECT_TIMEOUT_S))
            except Exception:
                pass
        if claimed_client and client is not None:
            pid = _sdk_child_pid(client)
            child_process = _own_sdk_child_process(pid) if pid else None
            disconnect_future = None
            try:
                disconnect_coro = client.disconnect()
                if loop is not None and loop.is_running():
                    disconnect_future = asyncio.run_coroutine_threadsafe(
                        disconnect_coro, loop
                    )
                    disconnect_future.result(timeout=_SDK_DISCONNECT_TIMEOUT_S)
                else:
                    if not _run_disconnect_without_loop(disconnect_coro):
                        raise TimeoutError("late SDK disconnect exceeded its deadline")
            except Exception:
                if disconnect_future is not None:
                    disconnect_future.cancel()
                if child_process is not None:
                    _force_kill_sdk_child(pid, process=child_process)
        if (claimed_loop or claimed_thread) and loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if claimed_thread and loop_thread is not None:
            loop_thread.join(timeout=5.0)

    @property
    def instance_generation(self) -> str:
        return getattr(self, "_instance_generation", "")

    @property
    def generation(self) -> str:
        return getattr(self, "_instance_generation", "")

    # Upper bound on the pre-spawn wait for main's attestation file (#49 / b10 H3).
    _ATTEST_BARRIER_S = 1.5
    # Upper bound on the resumed-id provenance check before recording the launch.
    _RESUME_PROVENANCE_TIMEOUT_S = 1.0

    def _wait_for_attestation_barrier(self) -> None:
        """Wait up to 1.5 s for main's attestation before spawning the CLI (#49 / b10 H3).

        The attestation directory is agent-writable, so the probe (a) refuses a planned sid
        that is not a single safe path component, (b) opens ``session-attest`` and the sid dir
        with ``O_NOFOLLOW`` (a symlink never counts as present), and (c) runs on a daemon
        thread: a probe blocked on FUSE/NFS/a fifo is abandoned at the deadline and the CLI is
        spawned anyway. The barrier is a sync aid, never a gate.
        """
        try:
            from agent.claude_sdk_launch_table import has_recent_consumer

            if not has_recent_consumer(60.0):
                return
        except Exception:
            return

        try:
            from hermes_owner_grant.anchor import load_trusted_anchor

            anchor = load_trusted_anchor()
            grants_dir = getattr(anchor, "grants_dir", None)
            if not grants_dir:
                return
        except Exception:
            return

        try:
            from agent.claude_sdk_launch_table import attestation_files_present
            from hermes_owner_grant.attest import is_safe_session_component

            planned_sid = self.planned_cli_session_id()
            if not is_safe_session_component(planned_sid):
                logger.warning("attestation barrier skipped: planned Claude sid is not a safe id")
                return
            grants_root = str(grants_dir)
            found = threading.Event()
            stop = threading.Event()

            def _probe() -> None:
                while not stop.is_set():
                    try:
                        if attestation_files_present(grants_root, planned_sid):
                            found.set()
                            return
                    except Exception:
                        pass
                    stop.wait(0.05)

            threading.Thread(target=_probe, name="sdk-attest-barrier", daemon=True).start()
            deadline = time.monotonic() + self._ATTEST_BARRIER_S
            first = True
            try:
                while True:
                    if self._startup_is_retired():
                        break
                    # Real-time wait on the probe; the monotonic deadline below bounds the loop.
                    if found.wait(0.05 if first else 0.02):
                        break
                    first = False
                    now = time.monotonic()
                    if now >= deadline:
                        break
                    step = min(0.05, max(0.0, deadline - now))
                    time.sleep(step)
            finally:
                stop.set()
        except Exception:
            pass

    def _resume_is_authenticated(self, profile: str) -> bool:
        """Bounded provenance check for a resumed Claude sid (b10 review finding 1)."""
        try:
            from agent.claude_sdk_launch_table import authenticated_resume, run_bounded

            claude_sid = str(self._resume_session_id or "")
            hermes_sid = str(self._hermes_session_id or "")
            lineage = list(self._hermes_lineage or [])
            finished, value = run_bounded(
                lambda: authenticated_resume(claude_sid, hermes_sid, lineage, profile=profile),
                self._RESUME_PROVENANCE_TIMEOUT_S,
                name="sdk-resume-provenance",
            )
            return bool(finished and value is True)
        except Exception:
            logger.debug("resume provenance check failed", exc_info=True)
            return False

    def ensure_started(self) -> Optional[str]:
        """Start the loop thread, build the SDK client, connect. Idempotent —
        returns the session marker (SDK session ids arrive on first result)."""
        with self._turn_callback_lock:
            if self._retiring or self._closed:
                return None
            if self._client is not None:
                return self._session_id or "pending"
            if self._starting:
                startup_done = self._startup_done
                starter = False
            elif sdk_shutdown_begun():
                # Backend teardown has begun: a CLI spawned now would outlive
                # the process as an orphan (2026-09-25 incident).
                raise SdkShuttingDownError()
            else:
                self._starting = True
                self._startup_done.clear()
                startup_done = self._startup_done
                starter = True
        if not starter:
            startup_done.wait()
            with self._turn_callback_lock:
                if self._retiring or self._closed:
                    return None
                return self._session_id or "pending"
        startup_client: Any = None
        startup_loop: Any = None
        startup_loop_thread: Any = None
        startup_reader_task: Any = None
        try:
            # Hard default, enforced fail-closed: this provider targets the Claude
            # SUBSCRIPTION. If a metered ANTHROPIC_API_KEY is present the
            # underlying CLI would silently prefer it — refuse to start instead.
            # ANTHROPIC_TOKEN is in the set because it alone authenticates
            # Hermes' NATIVE Anthropic lane (x-api-key, metered) — but Hermes
            # also persists subscription setup tokens there, so only an
            # API-key-shaped value counts as a metered vector.
            for metered_var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
                value = os.environ.get(metered_var)
                if not value or self._allow_metered:
                    continue
                if metered_var == "ANTHROPIC_TOKEN" and _is_subscription_oauth_token(value):
                    continue
                raise RuntimeError(
                    f"claude-agent-sdk runtime refuses to start: {metered_var} "
                    "is set, which would silently switch billing from the "
                    "Claude subscription to metered API usage. Unset it, or "
                    "set agent.claude_agent_sdk.allow_metered_key: true in "
                    "config.yaml to explicitly allow it."
                )
            if self._client_factory is None:
                ok, msg = check_claude_sdk_available()
                if not ok:
                    raise RuntimeError(msg)
        except BaseException:
            with self._turn_callback_lock:
                self._starting = False
                self._startup_done.set()
            raise

        try:
            started_resources = self._start_loop_thread()
            if started_resources is not None:
                startup_loop, startup_loop_thread = started_resources
            else:
                # Keep compatibility with narrow test seams that implement
                # the starter without returning its snapshot.
                with self._turn_callback_lock:
                    startup_loop = self._loop
                    startup_loop_thread = self._loop_thread
            with self._turn_callback_lock:
                retired = self._retiring or self._closed
            if retired:
                self._cleanup_startup_resources(
                    loop=startup_loop, loop_thread=startup_loop_thread
                )
                return None
            self._claim_peer_name()
            profile = "default"
            try:
                from hermes_cli.profiles import get_active_profile_name

                profile = str(get_active_profile_name() or "default")
            except Exception:
                pass
            try:
                from agent.claude_sdk_launch_table import (
                    SID_ORIGIN_FRESH,
                    SID_ORIGIN_RESUMED,
                    record_launch,
                )

                # A resumed id was read from agent-writable session state: main may attest it
                # only when it already signed that id for this session (or a lineage ancestor).
                resumed = bool(self._resume_session_id)
                record_launch(
                    hermes_session_id=str(self._hermes_session_id or ""),
                    claude_session_id=self.planned_cli_session_id(),
                    profile=profile,
                    lineage=self._hermes_lineage,
                    sid_origin=SID_ORIGIN_RESUMED if resumed else SID_ORIGIN_FRESH,
                    resume_authenticated=(
                        self._resume_is_authenticated(profile) if resumed else False
                    ),
                )
            except Exception:
                logger.debug("recording launch table entry failed", exc_info=True)
            startup_client = self._build_client()
            # Assign BEFORE connect: a connect timeout/cancel leaves a
            # half-connected client whose CLI subprocess close() must still reap
            # — a None _client would skip disconnect and orphan it.
            # Registration and the shutdown fence share one lock: a session
            # is either visible to the shutdown reap or refused here, so no
            # CLI can be spawned (connect) behind the reaper's back.
            with self._turn_callback_lock:
                retired = self._retiring or self._closed
                refused = False
                if not retired:
                    if _admit_sdk_session(self):
                        self._client = startup_client
                    else:
                        refused = True
            if retired or refused:
                try:
                    from agent.claude_sdk_launch_table import retire

                    retire(self.planned_cli_session_id())
                except Exception:
                    pass
                self._cleanup_startup_resources(
                    client=startup_client,
                    loop=startup_loop,
                    loop_thread=startup_loop_thread,
                )
                if refused:
                    raise SdkShuttingDownError()
                return None
            self._wait_for_attestation_barrier()
            self._run_coro(startup_client.connect(), timeout=60.0)
            self._record_peer_cli(startup_client)
            with self._turn_callback_lock:
                retired = self._retiring or self._closed
            # A CLI spawned mid-connect can miss the reaper's pid snapshot;
            # the starter reaps its own child if shutdown began meanwhile.
            shutting_down = sdk_shutdown_begun()
            if retired or shutting_down:
                self._cleanup_startup_resources(
                    client=startup_client,
                    loop=startup_loop,
                    loop_thread=startup_loop_thread,
                )
                if shutting_down and not retired:
                    raise SdkShuttingDownError()
                return None
            # From here on exactly ONE consumer owns the SDK stream
            # (claude_agent_sdk_session_turn._reader_loop).
            # Started after connect so the client is live, before any turn so no
            # message can arrive unowned.
            self._start_reader()
            with self._turn_callback_lock:
                startup_reader_task = self._reader_task
                retired = self._retiring or self._closed
            if retired:
                self._cleanup_startup_resources(
                    client=startup_client,
                    loop=startup_loop,
                    loop_thread=startup_loop_thread,
                    reader_task=startup_reader_task,
                )
                return None
            logger.info(
                "claude-agent-sdk session started: model=%s mode=%s cwd=%s",
                self._model or "cli-default",
                self._permission_mode,
                self._cwd,
            )
            if self._permission_mode == "default":
                # Say the cost out loud once per session: on this lane the guardian is an SDK
                # one-shot (a full CLI spawn) per Bash call that reaches the callback. (cntrl carry)
                logger.warning(
                    "claude-agent-sdk permission_mode=default: every Bash call that reaches the "
                    "approval callback spawns a guardian one-shot (a Claude CLI process). Set "
                    "agent.claude_agent_sdk.permission_mode: auto to let the CLI's classifier "
                    "screen first; Hermes then screens only what it will not approve."
                )
            with self._turn_callback_lock:
                retired = self._retiring or self._closed
                marker = self._session_id or "pending"
            if retired:
                self._cleanup_startup_resources(
                    client=startup_client,
                    loop=startup_loop,
                    loop_thread=startup_loop_thread,
                    reader_task=startup_reader_task,
                )
                return None
            return marker
        except BaseException:
            if self._startup_is_retired():
                self._cleanup_startup_resources(
                    client=startup_client,
                    loop=startup_loop,
                    loop_thread=startup_loop_thread,
                    reader_task=startup_reader_task,
                )
            raise
        finally:
            with self._turn_callback_lock:
                self._starting = False
                self._startup_done.set()

    def rename(self, name: str) -> bool:
        """Rename the spawned CLI session so peers see the new name NOW.

        ``--name`` is fixed at process start, so a Hermes title change would otherwise show
        to ListAgents peers only after the next CLI rebuild. Claude Code's ``/rename`` is a
        local slash command (no model call) that updates the peer registry instantly —
        proven 2026-09-09: query("/rename X") answered "Session renamed to: X" in <0.1s and
        the peer list showed X. Returns True when the rename was applied or will apply on the
        next build (no client yet); False when a turn is in flight — the caller then defers
        (rotate at the next turn boundary). The ack is a CLI-initiated turn and would be
        delivered as a background result; it is swallowed by text match. (cntrl carry)"""
        cleaned = " ".join((name or "").split())
        if not cleaned:
            return False
        if cleaned == self._session_name:
            self._deferred_rename = None
            return True
        if self._client is not None and self._session_name and not self._peer_lease_move(cleaned):
            # Another live session holds the target name. Keep this CLI's own
            # (still leased, still unique) name; True because a rotation would
            # only rebuild the CLI unnamed.
            self._deferred_rename = None
            logger.warning(
                "claude-agent-sdk: rename to %r refused — another live session holds that peer name",
                cleaned,
            )
            return True
        with self._turn_callback_lock:
            if (
                self._turn_inbox is not None
                or getattr(self, "_turn_claim_requested", False)
                or getattr(self, "_rename_claim_requested", False)
                or getattr(self, "_unsolicited_burst_open", False)
                or self._stream_ended is not None
            ):
                # Turn admission and rename admission share this lock. A rename
                # that loses the race is applied by the owner at release.
                self._deferred_rename = cleaned
                return False
            client, loop = self._client, self._loop
            if client is None or loop is None:
                self._session_name = cleaned  # applied by build_option_fields on the next start
                self._deferred_rename = None
                return True
            self._rename_claim_requested = True
            self._deferred_rename = None
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._run_claimed_rename(cleaned), loop
                )
            except Exception:
                self._rename_claim_requested = False
                self._pending_rename_ack = None
                logger.debug("SDK /rename scheduling failed", exc_info=True)
                return False
        future.add_done_callback(_swallow_steer_result)
        logger.info("claude-agent-sdk: session renamed to %r", cleaned)
        return True

    async def _run_claimed_rename(self, name: str) -> None:
        """Own the shared reader stream until the CLI acknowledges ``/rename``."""
        inbox = asyncio.Queue()
        acknowledged = False
        claims = self._turn_claims
        if claims is None:
            raise RuntimeError("SDK message reader is not ready to claim rename")
        claim_ack = asyncio.get_running_loop().create_future()
        timed_out = False
        claim_granted = False
        rename_sent = False
        retry_after_burst = False
        try:
            claims.put_nowait(("rename", inbox, claim_ack))
            await claim_ack
            with self._turn_callback_lock:
                if self._stream_ended is not None:
                    return
                if self._turn_inbox is not inbox or not self._rename_claim_requested:
                    self._deferred_rename = name
                    return
                claim_granted = True
                self._pending_rename_ack = name
                self._session_name = name
            await self._client.query(f"/rename {name}")
            rename_sent = True
            while True:
                try:
                    message = await asyncio.wait_for(
                        inbox.get(), timeout=_RENAME_ACK_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    self._deferred_rename = name
                    logger.warning(
                        "claude-agent-sdk: /rename acknowledgement timed out; deferring %r",
                        name,
                    )
                    return
                if isinstance(message, _StreamEnd):
                    raise RuntimeError("SDK stream ended before rename acknowledgement")
                if type(message).__name__ == "AssistantMessage" and _is_rename_ack(
                    None,
                    ["".join(str(getattr(block, "text", "") or "")
                             for block in (getattr(message, "content", None) or []))],
                    name,
                ):
                    continue
                if type(message).__name__ == "ResultMessage" and _is_rename_ack(
                    getattr(message, "result", None), [], name
                ):
                    acknowledged = True
                    return
                # The reader should only deliver the claimed operation's ack
                # here. Keep unrelated messages from becoming a later turn's
                # answer if the CLI emits an unexpected event in the window.
                self._handle_unsolicited(message)
                if type(message).__name__ == "ResultMessage":
                    # A different CLI result is independent work. Deliver it,
                    # then release the stream so a host turn is not held behind
                    # a rename whose acknowledgement has not arrived yet.
                    return
        except Exception as exc:
            with self._turn_callback_lock:
                if self._stream_ended is None:
                    self._deferred_rename = name
                    retry_after_burst = isinstance(exc, _RenameClaimDeferred)
            raise
        finally:
            if self._turn_inbox is inbox and self._stream_ended is None:
                release_ack = asyncio.get_running_loop().create_future()
                self._turn_claims.put_nowait(("release_rename", inbox, release_ack))
                await release_ack
            elif self._turn_inbox is inbox:
                self._turn_inbox = None
            with self._turn_callback_lock:
                self._rename_claim_requested = False
            if acknowledged and getattr(self, "_pending_rename_ack", None) == name:
                self._pending_rename_ack = None
            if rename_sent and not timed_out and self._session_name == name:
                # /rename is a local CLI command applied on receipt (the ack
                # itself may be swallowed by the reader), so the CLI no longer
                # carries the old name: free it.
                self._release_peer_names(keep=name)
            if self._deferred_rename and claim_granted and not timed_out:
                self._apply_deferred_rename()
            elif retry_after_burst and not self._unsolicited_burst_open:
                self._apply_deferred_rename()

    # ---------- single-owner peer-name lease ----------

    def _peer_lease_owner(self) -> "_peer_lease.LeaseOwner":
        owner = getattr(self, "_peer_lease_owner_cache", None)
        if owner is None:
            owner = _peer_lease.LeaseOwner.current(
                getattr(self, "_hermes_session_id", None), self._instance_generation
            )
            self._peer_lease_owner_cache = owner
        return owner

    def _claim_peer_name(self) -> None:
        """Lease ``--name`` before the spawn; spawn unnamed when a live peer holds it."""
        name = self._session_name
        self._peer_name_refused = False
        if not name:
            return
        try:
            claimed = _peer_lease.claim(name, self._peer_lease_owner())
        except Exception:
            logger.debug("claude peer-name lease unavailable; spawning named", exc_info=True)
            return
        if claimed:
            self._peer_leases.add(name)
            self._release_peer_names(keep=name)
            return
        self._peer_name_refused = True
        logger.warning(
            "claude-agent-sdk: peer name %r is held by another live session; "
            "spawning this CLI without --name so peers never see two registrants",
            name,
        )

    def _record_peer_cli(self, client: Any) -> None:
        name = self._session_name
        if not name or name not in getattr(self, "_peer_leases", ()):
            return
        try:
            _peer_lease.record_cli(name, self._peer_lease_owner(), cli_pid=_sdk_child_pid(client))
        except Exception:
            logger.debug("claude peer-name lease: CLI pid not recorded", exc_info=True)

    def _peer_lease_move(self, new: str) -> bool:
        """Claim ``new`` while the live CLI still carries the old name (both held).

        A CLI that spawned unnamed (refused) may still rename onto a free name."""
        try:
            claimed = _peer_lease.move(
                self._session_name, new, self._peer_lease_owner(), keep_old=True
            )
        except Exception:
            logger.debug("claude peer-name lease move failed; renaming unleased", exc_info=True)
            return True
        if claimed:
            self._peer_leases.add(new)
            self._peer_name_refused = False
        return claimed

    def _release_peer_names(self, keep: Optional[str] = None) -> None:
        leases = getattr(self, "_peer_leases", None)
        if not leases:
            return
        for name in [n for n in leases if n != keep]:
            leases.discard(name)
            try:
                _peer_lease.release(name, self._peer_lease_owner())
            except Exception:
                logger.debug("claude peer-name lease release failed", exc_info=True)

    def defer_rename(self, name: str) -> None:
        """Park a rename for the next stream release (the host saw the session busy)."""
        cleaned = " ".join((name or "").split())
        if cleaned and cleaned != self._session_name:
            self._deferred_rename = cleaned

    def set_unsolicited_result_callback(self, callback: Optional[Callable[..., None]]) -> None:
        """Install a late background-delivery callback and replay buffered results."""
        with self._turn_callback_lock:
            self._on_unsolicited_result = callback
            pending = list(self._pending_unsolicited_deliveries) if callback else []
            if callback:
                self._pending_unsolicited_deliveries.clear()
        if callback is None:
            return
        for texts, items, delivery_id in pending:
            self._deliver_unsolicited_callback(callback, texts, items, delivery_id)

    @staticmethod
    def _deliver_unsolicited_callback(callback, texts, items, delivery_id=None) -> None:
        import inspect

        try:
            inspect.signature(callback).bind(texts, items, delivery_id)
        except (TypeError, ValueError):
            try:
                inspect.signature(callback).bind(texts, items)
            except (TypeError, ValueError):
                callback(texts)
            else:
                callback(texts, items)
        else:
            callback(texts, items, delivery_id)

    def _apply_deferred_rename(self) -> None:
        """Apply a parked rename once no turn owns the stream. Never raises."""
        name = getattr(self, "_deferred_rename", None)
        if not name:
            return
        try:
            self.rename(name)  # re-parks itself if a turn claimed the stream meanwhile
        except Exception:
            logger.debug("deferred SDK rename failed", exc_info=True)

    def close(self) -> None:
        # Tolerate a partially constructed object (tests build one via
        # __new__; __init__ may also have raised before the lock existed).
        admission_lock = getattr(self, "_turn_callback_lock", None)
        with admission_lock if admission_lock is not None else contextlib.nullcontext():
            timer = getattr(self, "_unsolicited_quiet_timer", None)
            if timer is not None:
                timer.cancel()
                self._unsolicited_quiet_timer = None
            if self._closed:
                startup_done = (
                    self._startup_done if getattr(self, "_starting", False) else None
                )
                admission_done = (
                    self._admission_done
                    if getattr(self, "_admission_inflight", 0)
                    else None
                )
                should_cleanup = False
            else:
                # Publish the fence before any blocking reader/disconnect work.
                # Callers can therefore reject stale turns without waiting for
                # the SDK loop thread to finish shutting down.
                self._closed = True
                startup_done = (
                    self._startup_done if getattr(self, "_starting", False) else None
                )
                admission_done = (
                    self._admission_done
                    if getattr(self, "_admission_inflight", 0)
                    else None
                )
                should_cleanup = True
        # These waits intentionally happen outside _turn_callback_lock. They are
        # bounded: a leaked startup/admission reservation must degrade to a
        # warning and teardown, never make close() hang before it can stop the
        # reader and disconnect the client that would unblock the turn.
        shutdown_deadline = time.monotonic() + _SDK_DISCONNECT_TIMEOUT_S
        for label, event in (("startup", startup_done), ("admission", admission_done)):
            if event is None:
                continue
            remaining = max(0.0, shutdown_deadline - time.monotonic())
            if not event.wait(timeout=remaining):
                logger.warning(
                    "claude-agent-sdk close: %s reservation did not release "
                    "within %.1fs; proceeding with teardown",
                    label,
                    _SDK_DISCONNECT_TIMEOUT_S,
                )
        pending_admission_future = None
        if admission_lock is not None:
            with admission_lock:
                pending_admission_future = getattr(
                    self, "_pending_turn_admission_future", None
                )
                self._pending_turn_admission_future = None
                self._close_owned_client = getattr(self, "_client", None)
                self._close_owned_loop = getattr(self, "_loop", None)
                self._close_owned_loop_thread = getattr(self, "_loop_thread", None)
                self._close_owned_reader_task = getattr(self, "_reader_task", None)
        if pending_admission_future is not None:
            pending_admission_future.cancel()
        if not should_cleanup:
            return
        interrupt_commit_lock = getattr(self, "_interrupt_commit_lock", None)
        if interrupt_commit_lock is not None:
            with interrupt_commit_lock:
                self._terminal_result_committed = False
                self._post_terminal_interrupt_pending = False
                self._interrupt_event.clear()
        else:
            interrupt_event = getattr(self, "_interrupt_event", None)
            if interrupt_event is not None:
                interrupt_event.clear()
        self._finalize_sdk_tasks()
        # Cancel the reader BEFORE disconnect so it unwinds on a live stream
        # instead of raising against a torn-down one.
        reader_task = getattr(self, "_reader_task", None)
        reader_loop = self._loop
        self._stop_reader()
        if reader_task is not None and reader_loop is not None:
            async def _wait_for_reader_stop() -> None:
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass

            reader_wait = _wait_for_reader_stop()
            try:
                self._run_coro(
                    reader_wait,
                    timeout=min(1.0, _SDK_DISCONNECT_TIMEOUT_S),
                )
            except Exception:
                if reader_loop.is_closed() or not reader_loop.is_running():
                    reader_wait.close()
        if self._client is not None and self._loop is not None:
            # Budget must exceed the SDK transport's own escalation ladder
            # (stdin lock + graceful wait + SIGTERM + SIGKILL); on timeout the
            # loop thread dies below and its shielded reap never finishes, so
            # fall back to killing the child directly.
            pid = _sdk_child_pid(self._client)
            # Capture the psutil identity before waiting on disconnect. If the
            # original child exits and its PID is reused during the timeout,
            # this object will fail psutil's identity check instead of
            # resolving the reused PID as a fresh kill target.
            child_process = _own_sdk_child_process(pid) if pid else None
            try:
                self._run_coro(
                    self._client.disconnect(), timeout=_SDK_DISCONNECT_TIMEOUT_S
                )
            except Exception:  # pragma: no cover - best-effort cleanup
                if child_process is not None:
                    _force_kill_sdk_child(pid, process=child_process)
            self._client = None
        self._stop_loop_thread()
        _forget_sdk_session(self)
        # After the CLI is gone: the name is free only once nobody carries it.
        self._release_peer_names()
        try:
            from agent.claude_sdk_launch_table import retire

            planned_id = getattr(self, "_planned_session_id", None)
            if planned_id:
                retire(str(planned_id))
        except Exception:
            logger.debug("retiring launch table entry failed", exc_info=True)

    def __enter__(self) -> "ClaudeAgentSdkSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def consume_interrupt(self) -> None:
        """Clear a pending interrupt signal — the caller honored it through
        another path (e.g. the runtime's cold-agent short-circuit)."""
        with self._interrupt_commit_lock:
            self._interrupt_event.clear()
            self._post_terminal_interrupt_pending = False

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to interrupt and unwind."""
        declined = False
        with self._interrupt_commit_lock:
            if self._terminal_result_committed:
                # Do not disturb a persistent client after terminal commit.
                # Runtime consumes this during its terminal handoff; a direct
                # session caller that leaves it unconsumed gets the signal as
                # the next turn's ordinary pre-set interrupt.
                self._post_terminal_interrupt_pending = True
                declined = True
        if declined:
            # Logged OUTSIDE the lock: this is the same lock the stream
            # consumer takes once per projected message.
            #
            # Deliberately not promised as "queued for the next turn": run_turn
            # consumes this flag before handing the result back, so under the
            # runtime the stop is DROPPED here, and the answer it raced is
            # delivered. Only a direct session caller that never consumes it
            # sees it re-armed on the next turn.
            logger.info(
                "claude-agent-sdk: /stop arrived after this turn committed its "
                "result — not sent to the CLI; the completed answer is "
                "delivered and the stop does not carry over"
            )
            return
        with self._interrupt_commit_lock:
            self._interrupt_event.set()
        if self._client is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._client.interrupt(), self._loop
                )
                future.add_done_callback(_swallow_interrupt_result)
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "SDK interrupt scheduling failed: %s",
                    _safe_sdk_error_text(exc),
                )

    def _native_peer_pending_locked(self) -> bool:
        """An injected peer input still owns the CLI. Caller holds _turn_callback_lock.

        Once its burst opens, the burst flag carries the claim; an input the CLI never
        turned into a turn expires, so it can't hold owner input back forever."""
        if not self._native_peer_in_flight:
            return False
        if self._unsolicited_burst_open or not getattr(self, "_native_peer_written", True):
            return True  # a write still in progress can't be presumed lost
        if time.monotonic() - self._native_peer_since < _NATIVE_PEER_ADMIT_GRACE_SECONDS:
            return True
        self._clear_native_peer_locked()
        return False

    def _clear_native_peer_locked(self) -> None:
        self._native_peer_in_flight = False
        self._native_peer_msg_id = None
        self._native_peer_seen = False
        self._native_peer_written = False

    def _expire_native_peer(self, gen: int) -> None:
        """Grace timer: release a confirmed write that never became a turn, then wake queued work."""
        with self._turn_callback_lock:
            if gen != self._native_peer_gen or not self._native_peer_in_flight:
                return
            if self._native_peer_pending_locked():
                return
        self._notify_idle_boundary()

    def _note_unsolicited_activity(self) -> None:
        """Only top-level CLI output extends an unclaimed turn's quiet window."""
        with self._turn_callback_lock:
            self._unsolicited_last_top_level_at = self._unsolicited_clock()
            if self._unsolicited_quiet_timer is None:
                timer = threading.Timer(UNSOLICITED_QUIET_S, self._expire_unsolicited_burst)
                timer.daemon = True
                self._unsolicited_quiet_timer = timer
                timer.start()

    def _note_unsolicited_tools(self, message: Any) -> None:
        """Track top-level tool calls until their matching top-level results arrive."""
        if getattr(message, "parent_tool_use_id", None) is not None:
            return
        name = type(message).__name__
        if name == "AssistantMessage":
            ids = {str(block.id) for block in (getattr(message, "content", None) or [])
                   if type(block).__name__ == "ToolUseBlock" and getattr(block, "id", None)}
            with self._turn_callback_lock:
                self._unsolicited_outstanding_tools.update(ids)
        elif name == "UserMessage":
            ids = {str(block.tool_use_id) for block in (getattr(message, "content", None) or [])
                   if type(block).__name__ == "ToolResultBlock" and getattr(block, "tool_use_id", None)}
            with self._turn_callback_lock:
                self._unsolicited_outstanding_tools.difference_update(ids)

    def _close_unsolicited_burst(self, reason: str, *, quiet_expired: bool = False) -> None:
        with self._turn_callback_lock:
            was_open = self._unsolicited_burst_open
            self._unsolicited_burst_open = False
            self._unsolicited_quiet_expired = quiet_expired
            self._unsolicited_last_top_level_at = None
            self._unsolicited_outstanding_tools.clear()
            timer = self._unsolicited_quiet_timer
            self._unsolicited_quiet_timer = None
            if timer is not None:
                timer.cancel()
        if was_open:
            logger.info("claude-agent-sdk: unsolicited burst closed session=%s reason=%s",
                        self._hermes_session_id or self._session_id or "pending", reason)

    def _expire_unsolicited_burst(self) -> None:
        """A missing ResultMessage cannot hold the next owner turn indefinitely."""
        with self._turn_callback_lock:
            since = self._unsolicited_last_top_level_at
            if not self._unsolicited_burst_open or since is None or self._closed:
                return
            remaining = UNSOLICITED_QUIET_S - (self._unsolicited_clock() - since)
            if remaining > 0 or self._unsolicited_outstanding_tools:
                # Tool execution may be quiet for far longer than the interval.
                timer = threading.Timer(remaining if remaining > 0 else UNSOLICITED_QUIET_S,
                                        self._expire_unsolicited_burst)
                timer.daemon = True
                self._unsolicited_quiet_timer = timer
                timer.start()
                return
            self._close_unsolicited_burst("quiet-interval", quiet_expired=True)
        self._apply_deferred_rename()
        self._notify_idle_boundary()

    def woken_turn_active(self) -> bool:
        with self._turn_callback_lock:
            return bool(self._native_peer_pending_locked() or self._unsolicited_burst_open)

    def interrupt_woken_turn(self) -> bool:
        """Stop an unclaimed CLI turn without leaving a host-turn interrupt pending."""
        with self._turn_callback_lock:
            if not (self._native_peer_pending_locked() or self._unsolicited_burst_open):
                return False
            client, loop = self._client, self._loop
        if client is None or loop is None:
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(client.interrupt(), loop)
            future.add_done_callback(_swallow_interrupt_result)
            return True
        except Exception:
            logger.debug("SDK woken-turn interrupt scheduling failed", exc_info=True)
            return False

    def set_idle_boundary_callback(self, callback: Optional[Callable[[], None]]) -> None:
        with self._turn_callback_lock:
            self._idle_boundary_callback = callback

    def _notify_idle_boundary(self) -> None:
        callback = self._idle_boundary_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.debug("SDK idle-boundary callback failed", exc_info=True)

    def native_peer_idle(self) -> bool:
        with self._turn_callback_lock:
            return not (self._turn_inbox is not None or self._turn_claim_requested
                        or self._rename_claim_requested or getattr(self, "_resume_backlog", None) is not None
                        or self._unsolicited_burst_open or self._native_peer_pending_locked()
                        or self._pending_steer_results or self._closed or self._retiring
                        or self._stream_ended is not None)

    def stop_task(self, task_id: str) -> bool:
        """Schedule Claude SDK's task stop on the session's loop thread."""
        client, loop = self._client, self._loop
        if client is None or loop is None or not hasattr(client, "stop_task"):
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(client.stop_task(task_id), loop)
            future.add_done_callback(_swallow_interrupt_result)
        except Exception as exc:  # pragma: no cover - loop teardown race
            logger.debug("SDK task stop scheduling failed: %s", _safe_sdk_error_text(exc))
            return False
        return True

    def steer(self, text: str) -> bool:
        """Inject a mid-turn user message using the SDK's own streaming input.

        Why this exists: on this lane the SDK owns tool execution, so Hermes'
        tool-batch drain points (agent/tool_executor.py) are never reached and
        a /steer would strand in ``_pending_steer`` until the turn finalizer
        handed it back — where the gateway only redelivers it if no other
        message is queued. Steers were therefore silently lost.

        ``ClaudeSDKClient.query()`` is documented as "send a new request in
        streaming mode", and we already connect in streaming mode (``connect()``
        with no prompt), so a second query() on the LIVE session is the SDK's
        own steering contract — not an interrupt and not a session restart,
        which is what the claude-cli-live transport has to do.

        Verified 2026-08-15 against SDK 0.2.120: a query() issued while a tool
        call was in flight did not raise and was honored at the next turn
        boundary ~3s later. It also produces a SECOND ResultMessage once the
        superseded work unwinds. The steer is submitted with a human origin, so
        the live turn keeps ownership until every accepted steer has settled and
        attributes that result itself; only a steer result arriving after the
        turn released is treated as late (see _handle_unsolicited).

        Returns True only when the steer was actually scheduled onto a live
        turn. False means "not applicable" — the caller must fall back to the
        ordinary pending-steer stash. Crucially this is never both: returning
        True suppresses the stash, so a steer cannot be delivered twice.
        """
        if not text or not text.strip():
            return False
        # No claimed turn means there is nothing to steer INTO. Sending anyway
        # would open a fresh unclaimed turn on this session whose output the
        # reader routes to the unsolicited path — a reply appearing from
        # nowhere. Decline instead and let the caller queue it normally. The
        # terminal fence is checked under the same lock as ResultMessage
        # acceptance, closing the result/steer race at the ownership edge.
        cleaned = text.strip()
        commit_lock = getattr(self, "_interrupt_commit_lock", None)
        with commit_lock if commit_lock is not None else contextlib.nullcontext():
            if getattr(self, "_terminal_result_committed", False):
                return False
            if self._turn_inbox is None:
                return False
            self._pending_steer_results = getattr(self, "_pending_steer_results", 0) + 1
            self._pending_steer_inputs = getattr(self, "_pending_steer_inputs", [])
            self._pending_steer_inputs.append(cleaned)
            self._pending_steer_marks = getattr(self, "_pending_steer_marks", [])
            self._pending_steer_marks.append({"text": cleaned, "ts": time.time(), "pos": None})
        client = self._client
        loop = self._loop
        if client is None or loop is None:
            with commit_lock if commit_lock is not None else contextlib.nullcontext():
                self._pending_steer_results = max(
                    0, getattr(self, "_pending_steer_results", 0) - 1
                )
                if self._pending_steer_inputs and self._pending_steer_inputs[-1] == cleaned:
                    self._pending_steer_inputs.pop()
                _forget_steer_mark(self, cleaned)
            return False
        try:
            query = client.query(_sdk_user_message_stream(cleaned, origin={"kind": "human"}))
            future = asyncio.run_coroutine_threadsafe(query, loop)

            def _finish_steer(done: Any) -> None:
                try:
                    done.result()
                except Exception:
                    # A successful query future only means the input was
                    # accepted by the SDK; its ResultMessage is still pending.
                    # A failed future must release the ownership reservation so
                    # the next ordinary ResultMessage is not mistaken for a
                    # steer result.
                    with commit_lock if commit_lock is not None else contextlib.nullcontext():
                        self._pending_steer_results = max(
                            0, getattr(self, "_pending_steer_results", 0) - 1
                        )
                        inputs = getattr(self, "_pending_steer_inputs", [])
                        if cleaned in inputs:
                            inputs.remove(cleaned)
                        _forget_steer_mark(self, cleaned)
                    logger.debug("SDK steer query failed after scheduling", exc_info=True)

            future.add_done_callback(_finish_steer)
        except Exception:
            if "query" in locals() and hasattr(query, "close"):
                query.close()
            with commit_lock if commit_lock is not None else contextlib.nullcontext():
                self._pending_steer_results = max(
                    0, getattr(self, "_pending_steer_results", 0) - 1
                )
                if self._pending_steer_inputs and self._pending_steer_inputs[-1] == cleaned:
                    self._pending_steer_inputs.pop()
                _forget_steer_mark(self, cleaned)
            logger.debug("SDK steer scheduling failed", exc_info=True)
            return False
        logger.info(
            "claude-agent-sdk: steered live turn via streaming input (%d chars)",
            len(cleaned),
        )
        return True

    def refresh_session_spawn_capability(self, *, min_interval: float = 60.0) -> bool:
        """Rotate the published hermes-tools capability at a turn boundary (best-effort, b3-30)."""
        published = getattr(self, "_session_spawn_capability", None)
        if not published:
            return False
        last = getattr(self, "_session_spawn_capability_refreshed_at", 0.0)
        if last and time.monotonic() - last < min_interval:
            return False
        try:
            from agent.transports.claude_agent_sdk_session_config import refresh_session_spawn_capability
            if not refresh_session_spawn_capability(*published):
                return False
        except Exception:
            logger.warning("session-spawn capability rotation failed", exc_info=True)
            return False
        self._session_spawn_capability_refreshed_at = time.monotonic()
        return True

    def send_peer_message(self, text: str, origin: dict[str, Any]) -> bool:
        """Inject one peer-origin input only while the CLI is idle."""
        if not text or not text.strip():
            return False
        if not self.is_live():
            return False
        client, loop = self._client, self._loop
        if client is None or loop is None:
            return False
        with self._turn_callback_lock:
            if not self.native_peer_idle():
                return False
            self._native_peer_gen += 1
            gen = self._native_peer_gen
            self._native_peer_in_flight = True
            self._native_peer_msg_id = str(origin.get("msg_id") or "")
            self._native_peer_seen = False
            self._native_peer_written = False
            self._native_peer_since = time.monotonic()
        # A peer message starts a CLI turn without run_turn; its tool calls need a live capability too.
        if callable(refresh_capability := getattr(self, "refresh_session_spawn_capability", None)):
            refresh_capability()
        query = None
        try:
            query = client.query(_sdk_user_message_stream(text.strip(), origin=origin))
            future = asyncio.run_coroutine_threadsafe(query, loop)

            def _finish_peer(done: Any) -> None:
                failed = done.cancelled() or done.exception() is not None
                with self._turn_callback_lock:
                    if gen != self._native_peer_gen or not self._native_peer_in_flight:
                        return
                    if failed:
                        self._clear_native_peer_locked()
                    else:
                        self._native_peer_written = True
                        self._native_peer_since = time.monotonic()
                if failed:
                    logger.warning("SDK peer query failed after scheduling: %r",
                                   None if done.cancelled() else done.exception())
                    self._notify_idle_boundary()
                    return
                timer = threading.Timer(_NATIVE_PEER_ADMIT_GRACE_SECONDS + 0.5,
                                        self._expire_native_peer, args=(gen,))
                timer.daemon = True
                timer.start()

            future.add_done_callback(_finish_peer)
            try:
                future.result(timeout=_NATIVE_PEER_QUERY_TIMEOUT_SECONDS)
            except TimeoutError:
                # Delivered means confirmed: keep waiting for the write rather than reporting
                # success early (a later failure would lose the mailbox row) or failure early
                # (a later success would deliver it twice).
                logger.info("claude-agent-sdk: native peer query still writing after %ss; waiting",
                            _NATIVE_PEER_QUERY_TIMEOUT_SECONDS)
                try:
                    future.result(timeout=_NATIVE_PEER_WRITE_MAX_SECONDS - _NATIVE_PEER_QUERY_TIMEOUT_SECONDS)
                except TimeoutError:
                    # Still stuck: cancel. A cancelled write never reached the CLI, so the
                    # row can safely go back to the mailbox.
                    future.cancel()
                    raise
        except Exception:
            with self._turn_callback_lock:
                if gen == self._native_peer_gen:
                    self._clear_native_peer_locked()
            if query is not None and hasattr(query, "close"):
                query.close()
            logger.debug("SDK peer query scheduling failed", exc_info=True)
            return False
        logger.info("claude-agent-sdk: injected native peer message (%d chars)", len(text.strip()))
        return True

    def is_live(self) -> bool:
        """Whether this cached adapter still owns a running CLI stream."""
        with self._turn_callback_lock:
            client, loop = self._client, self._loop
            if self._closed or self._retiring or self._stream_ended is not None:
                return False
        if client is None or loop is None or loop.is_closed() or not loop.is_running():
            return False
        pid = _sdk_child_pid(client)
        return bool(pid and _own_sdk_child_process(pid) is not None)

    def build_option_fields(self) -> dict[str, Any]:
        """The ClaudeAgentOptions field dict — plain data so tests can assert
        on it without importing the SDK."""
        mcp_servers: dict[str, Any] = {}
        # Hybrid in-process MCP bridge (ported from PR #56413) — exposes the
        # full Hermes tool registry, including proxified third-party MCP
        # servers, which the stdio `hermes-tools` wrapper cannot reach
        # because credentials live in the gateway process env and don't
        # propagate to the subprocess. Activation requires ALL of:
        #   1. the operator opted in via
        #      ``agent.claude_agent_sdk.hybrid_mcp_bridge: true``
        #      (checked here as well as at the runtime call site);
        #   2. the caller supplied both ``agent`` and ``tools``.
        # The bridge splits into two in-process servers:
        #   - ``hermes-tools`` — stdio-legacy names (see
        #     ``HERMES_TOOLS_LEGACY_NAMES``). Preserves operator grants
        #     stored in ``~/.claude/settings.json`` that key on
        #     ``mcp__hermes-tools__<tool>``: a box on
        #     ``permission_mode: default`` would otherwise face an
        #     approval storm for tools it already granted.
        #   - ``hermes-hybrid`` — everything else (proxified MCPs +
        #     agent-level tools).
        # The exclude list from config is applied to BOTH buckets so an
        # operator can keep the wide bridge for proxified MCPs without
        # inheriting a specific tool (delegate_task, cron_*, terminal, ...).
        hybrid_active = False
        hybrid_opted_in = _provider_flag("hybrid_mcp_bridge", default=False)
        if hybrid_opted_in and self._agent is not None and self._tools:
            try:
                from agent.transports.hermes_hybrid_mcp import (
                    HERMES_TOOLS_SERVER,
                    HYBRID_SERVER,
                    build_hybrid_mcp_server,
                )
                from agent.transports.hermes_tool_exposure import (
                    HERMES_TOOLS_LEGACY_NAMES,
                )

                exclude_names = _configured_hybrid_exclude()
                mcp_servers[HERMES_TOOLS_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    self._tools,
                    server_name=HERMES_TOOLS_SERVER,
                    only_names=HERMES_TOOLS_LEGACY_NAMES,
                    exclude_names=exclude_names,
                )
                # Registry diff rides the "extras" bucket only: the legacy
                # bucket stays byte-identical so ``mcp__hermes-tools__*``
                # grants keep matching. When the diff is empty the snapshot
                # object is forwarded untouched, so a deployment with nothing
                # late-registered builds exactly the bridge it built before.
                registry_extra = _mcp_registry_tool_specs(
                    self._agent, self._tools
                )
                hybrid_tools = (
                    list(self._tools) + registry_extra
                    if registry_extra
                    else self._tools
                )
                mcp_servers[HYBRID_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    hybrid_tools,
                    server_name=HYBRID_SERVER,
                    exclude_names=(
                        list(exclude_names) + list(HERMES_TOOLS_LEGACY_NAMES)
                    ),
                )
                hybrid_active = True
            except Exception as exc:  # noqa: BLE001
                # Never break session start on hybrid bridge failure — the
                # stdio wrapper still provides the curated tool set.
                logger.warning(
                    "hybrid MCP bridge failed to build (%s) — falling back "
                    "to stdio hermes-tools only",
                    exc,
                )
        # The stdio ``hermes-tools`` wrapper exposes ~25 curated tools that
        # the hybrid bridge already re-exposes under the same server name
        # (``hermes-tools``) when active — registering both concurrently
        # would send Claude two copies of every curated tool under the same
        # server. Skip stdio when hybrid is active. (Hybrid failure above
        # falls back to stdio via ``hybrid_active`` staying False.)
        if self._include_hermes_tools and not hybrid_active:
            mcp_servers["hermes-tools"] = _build_hermes_tools_mcp_config(
                hermes_session_id=self._hermes_session_id
            )
            # Remember where the scoped capability was published so each turn start can rotate
            # it in place (b3-30: a once-issued capability expired an hour into the session).
            tools_env = mcp_servers["hermes-tools"].get("env") or {}
            if tools_env.get("HERMES_SESSION_SPAWN_CAPABILITY_FILE"):
                self._session_spawn_capability = (
                    tools_env.get("HERMES_SESSION_ID", ""),
                    tools_env["HERMES_SESSION_SPAWN_CAPABILITY_FILE"],
                )
                self._session_spawn_capability_refreshed_at = time.monotonic()

        # Headerless third-party HTTP MCPs configured in Hermes (config.yaml
        # mcp_servers.<name>.url) can be exposed directly to the SDK because
        # the registry snapshot may not contain their late/proxified tools.
        # This is part of the SAME wide-surface security choice as the hybrid
        # bridge, never an independent back door: discovery runs only after
        # the explicit opt-in passed and both in-process bridge buckets built
        # successfully. Header-bearing entries are refused by the loader
        # because the SDK puts its MCP config in the Claude CLI argv.
        if hybrid_active:
            for entry_name, entry_cfg in _http_mcp_entries_from_config().items():
                if entry_name in mcp_servers:
                    continue
                mcp_servers[entry_name] = entry_cfg

        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if self._system_prompt_append:
            system_prompt = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt_append,
            }

        # This wrapper is Hermes' mandatory permission-policy owner, including
        # when runtime callback selection legitimately produced no downstream
        # callback.  Callbackless non-bypass sessions intentionally fail closed
        # here instead of falling through to SDK-native prompting/permission
        # behavior; bounded readers and trusted bypass still resolve only after
        # canonical validation, correlation validation, and immutable floors.
        can_use_tool = self._make_can_use_tool()

        # Metered-vector scrub for the spawned CLI (see _METERED_ENV_DENYLIST).
        # agent.claude_agent_sdk.allow_metered_key: true is the operator's
        # explicit "bill me metered" opt-in (the same flag the startup guard
        # honors), so it disables the scrub too — otherwise the documented
        # escape hatch would hand the CLI an environment with the key blanked.
        env_overrides = _sdk_env_overrides(
            metered_allowed=self._allow_metered,
            task_list_id=self._task_list_id,
            task_env=getattr(self, "_task_env", None),
            hermes_session_id=self._hermes_session_id,
            sdk_cwd=self._cwd,
        )
        if getattr(self, "_resume_interrupted_turn", False):
            # D62 L2, verified in CLI 2.1.284: on --resume, headless startup
            # re-runs the transcript's interrupted turn by itself.
            env_overrides["CLAUDE_CODE_RESUME_INTERRUPTED_TURN"] = "1"

        fields = {
            "model": _cli_model_id(self._model),
            "cwd": self._cwd,
            "permission_mode": self._permission_mode,
            "system_prompt": system_prompt,
            "mcp_servers": mcp_servers,
            "max_budget_usd": self._max_budget_usd,
            "can_use_tool": can_use_tool,
            "env": env_overrides,
            # Explicit SDK isolation. None (the SDK default, verified against
            # claude-agent-sdk 0.2.120) means the CLI loads ALL filesystem
            # settings — ~/.claude/settings.json, .claude/settings.json,
            # .claude/settings.local.json — so ambient operator/project
            # settings could re-permission tools or install hooks UNDERNEATH
            # the permission_mode/can_use_tool posture configured above. The
            # empty list is the SDK's documented isolation mode: settings
            # come from Hermes config only. Accepted side effect: "project"
            # is also what loads CLAUDE.md, which this runtime doesn't want —
            # it composes its own system-prompt append. Operators whose
            # deployment stores tool grants in ~/.claude/settings.json
            # (unattended cron turns with nobody to answer a prompt) opt
            # back in via agent.claude_agent_sdk.setting_sources — see
            # _configured_setting_sources.
            "setting_sources": _configured_setting_sources(),
            # Explicit, because the SDK's 1 MiB default is below what this
            # lane's own tool results routinely produce and overflowing it
            # kills the turn outright — see _configured_max_buffer_size.
            "max_buffer_size": getattr(self, "_max_buffer_size", None) or _configured_max_buffer_size(),
            # Native Read stays behind the protected-path-aware, bounded Hermes MCP
            # surface in every supported SDK permission mode.
            "disallowed_tools": ["Read"],
        }
        # CLI-side hooks: compaction watchdog / status, plus tool and subagent
        # lifecycle passthrough to Hermes plugin hooks.
        hooks: dict[str, list[Any]] = {}
        _compaction_hooks = self._build_compaction_hooks()
        if _compaction_hooks:
            for k, v in _compaction_hooks.items():
                hooks.setdefault(k, []).extend(v)
        _plugin_hooks = self._build_plugin_hooks()
        if _plugin_hooks:
            for k, v in _plugin_hooks.items():
                hooks.setdefault(k, []).extend(v)
        if hooks:
            fields["hooks"] = hooks
        if self._resume_session_id:
            fields["resume"] = self._resume_session_id
        else:
            fields["session_id"] = self._planned_session_id
        # Auto-mode classifier rules via the CLI's flag-settings layer
        # (--settings), which the CLI keeps enabled even with
        # setting_sources=[] — so ONLY {"autoMode": ...} crosses and the
        # isolation above is unchanged. Absent when off (the default). See
        # _configured_auto_mode_settings.
        auto_mode_settings = getattr(self, "_auto_mode_settings", None)
        if auto_mode_settings:
            fields["settings"] = auto_mode_settings
        # Operator-pinned Claude Code binary (see _configured_cli_path).
        cli_path = _configured_cli_path()
        if cli_path:
            fields["cli_path"] = cli_path
        # Explicitly loaded plugin roots (see _configured_plugins).
        plugins = _configured_plugins()
        if plugins:
            fields["plugins"] = plugins
        # Name the spawned session so peers can find and message it
        # (ListAgents/SendMessage) — the host-router seam.
        extra = dict(fields.get("extra_args") or {})
        next_name = self._deferred_rename or self._session_name
        if next_name and not getattr(self, "_peer_name_refused", False):
            extra.setdefault("name", next_name)
        # Injected turns (peer SendMessage deliveries, task notifications,
        # scheduled prompts) only appear on the stream as origin-tagged
        # UserMessages when the CLI replays user messages. Without this the
        # unsolicited path never sees the inbound peer message or what woke the
        # session, so neither renders in the transcript. Replayed echoes of
        # Hermes' own prompts are harmless: the projector keeps only tool
        # results from UserMessages. Constant per session (cache-safe).
        extra.setdefault("replay-user-messages", None)
        fields["extra_args"] = extra
        # Partial messages follow agent.claude_agent_sdk.streaming (the cntrl
        # fork defaults it on; upstream ships it off).
        # Reads the __init__ snapshot so option and quiet-watchdog semantics
        # can never diverge across a mid-session config edit.
        if self._streaming:
            fields["include_partial_messages"] = True
        if self._max_turns is not None:
            fields["max_turns"] = self._max_turns
        return fields

    def _build_client(self) -> Any:
        fields = self.build_option_fields()
        self._replay_user_messages = "replay-user-messages" in fields.get("extra_args", {})
        if self._client_factory is not None:
            return self._client_factory(options=fields)
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        return ClaudeSDKClient(options=ClaudeAgentOptions(**fields))
