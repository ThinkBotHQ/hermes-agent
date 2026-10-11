"""Per-turn driver and the session-lifetime reader of ``ClaudeAgentSdkSession``.

``run_turn`` (watchdog-supervised turn from the caller thread), ``_consume_turn``
(claims the shared stream, projects messages, settles the terminal result),
``_reader_loop`` (the single stream reader arbitrating turn claims and publishing
stream death) and unsolicited-result handling. The reader and the turn stay
together on purpose: they jointly drive one claim/release protocol over
``_turn_claims`` / ``_turn_inbox`` / ``_stream_ended``. Extracted from
``claude_agent_sdk_session.py``; every method resolves through
``ClaudeAgentSdkSession``'s MRO unchanged. ``ClaudeSdkEventProjector`` is bound
here — tests replacing the projector patch this module's name.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional

# TurnResult is the shared contract with the runtime glue — reused verbatim
# from the codex session module (same fields, same semantics) so
# ``run_claude_agent_sdk_turn`` mirrors ``run_codex_app_server_turn`` 1:1.
from agent.transports.codex_app_server_session import TurnResult
from agent.transports.claude_sdk_event_projector import ClaudeSdkEventProjector
from agent.transports.claude_agent_sdk_session_availability import (
    _safe_sdk_error_text,
    _safe_sdk_traceback,
    classify_auth_failure,
)
from agent.transports.claude_agent_sdk_session_input import (
    _coerce_turn_input,
    _sdk_user_message_stream,
    fit_sdk_turn_blocks,
)
from agent.transports.claude_agent_sdk_session_config import (
    _configured_post_tool_quiet_timeout,
    _configured_turn_idle_timeout,
    _configured_turn_max_seconds,
    _configured_turn_timeout,
    _configured_turn_tool_max_suspend,
)
from agent.transports.claude_sdk_peer_envelope import effective_origin
from agent.transports.claude_sdk_peer_envelope import parse as parse_peer_envelope
from agent.transports.claude_agent_sdk_session_child import SdkShuttingDownError
from agent.transports.claude_agent_sdk_session_watchdog import (
    _is_rename_ack,
    _DEFAULT_POST_TOOL_QUIET_STREAMING,
    _DEFAULT_TURN_IDLE_TIMEOUT,
    _POLL_STALL_FACTOR,
    _StreamEnd,
    _TOOL_MAX_SUSPEND,
    _TURN_ABORT_GRACE,
    _TurnWatch,
)

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


# Unsolicited deliveries retained while no result callback is wired (G2.4); newest kept.
_MAX_RETAINED_UNSOLICITED = 16
_MAX_RETAINED_UNSOLICITED_IDS = 512


def _remember_recent_id(owner: Any, seen_name: str, order_name: str, value: str) -> bool:
    """Remember an id while keeping each session-local dedupe index bounded."""
    seen = getattr(owner, seen_name, None)
    if seen is None:
        seen = set()
        setattr(owner, seen_name, seen)
    if value in seen:
        return False
    order = getattr(owner, order_name, None)
    if order is None:
        order = []
        setattr(owner, order_name, order)
    seen.add(value)
    order.append(value)
    if len(order) > _MAX_RETAINED_UNSOLICITED_IDS:
        seen.discard(order.pop(0))
    return True


def _is_human_origin(message: Any) -> bool:
    """Return whether the SDK attributed a result to an application query."""
    origin = getattr(message, "origin", None)
    return isinstance(origin, dict) and origin.get("kind") == "human"


def _is_injected_origin(origin: Any) -> bool:
    """True for CLI-injected provenance (peer, task-notification, channel, ...)."""
    return isinstance(origin, dict) and origin.get("kind") not in (None, "human")


def _starts_injected_turn(message: Any) -> bool:
    """An injected user message, or assistant output with no turn in flight."""
    if getattr(message, "parent_tool_use_id", None) is not None:
        return False
    name = type(message).__name__
    if name == "UserMessage":
        return _is_injected_origin(effective_origin(message))
    return name == "AssistantMessage"


def _is_own_prompt_echo(message: Any) -> bool:
    """A replayed Hermes prompt: the CLI has moved on to the host's own turn."""
    if type(message).__name__ != "UserMessage":
        return False
    if getattr(message, "parent_tool_use_id", None) is not None:
        return False
    if _is_injected_origin(effective_origin(message)):
        return False  # a peer envelope is never the host's own prompt
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return bool(content)
    if not isinstance(content, list):
        return False
    if not content:
        # The installed parser drops image blocks, so an image-only prompt replays as an empty list.
        # Only a top-level, unattributed message can be that echo (tool results and subagent turns
        # carry blocks or a parent tool id).
        return getattr(message, "parent_tool_use_id", None) is None
    return not any(type(block).__name__ == "ToolResultBlock" for block in content)


def _own_envelope_sender(prompt: Any) -> Optional[str]:
    """``hermes-session:<sender>`` when this turn's prompt is exactly one peer envelope, else None.

    A peer row delivered live runs as the host's own query; the CLI parses the envelope and stamps
    that query's echo AND its ResultMessage ``{'kind': 'peer', 'from': 'hermes-session:<sender>',
    'hostInjected': True}`` (live 2026-09-30, session c68a80)."""
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list):
        text = "\n".join(str(block.get("text") or "") for block in prompt
                         if isinstance(block, dict) and block.get("type") == "text")
    else:
        return None
    envelope = parse_peer_envelope(text)
    if not envelope:
        return None
    return f"hermes-session:{envelope.get('fromSession') or 'unknown'}"


def _is_own_envelope_result(origin: Any, own_sender: Optional[str]) -> bool:
    """The CLI attributed this injected-origin result to the host's own envelope query."""
    return bool(own_sender and isinstance(origin, dict) and origin.get("kind") == "peer"
                and origin.get("hostInjected") is True and str(origin.get("from") or "") == own_sender)


def _prefold_peer_items(session: Any) -> list:
    """The peer messages buffered in an injected burst before the host prompt folded into it."""
    return [dict(item) for item in getattr(session, "_unsolicited_items", None) or []
            if isinstance(item, dict) and item.get("kind") == "peer_in"]


class _ResumeInterruptedTurn:
    """``run_turn`` input that sends NOTHING (D62 L2).

    The CLI was spawned with ``CLAUDE_CODE_RESUME_INTERRUPTED_TURN=1`` on
    ``--resume``: at startup it re-runs the transcript's interrupted turn by
    itself (a meta "Continue from where you left off." after an interrupted
    tool round, or the unanswered prompt under its own uuid). The turn only
    claims the stream and consumes that re-run; when the CLI declines (nothing
    interrupted), ``resume_declined`` comes back with no API call.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "RESUME_INTERRUPTED_TURN"


RESUME_INTERRUPTED_TURN = _ResumeInterruptedTurn()
# How long a resume-only turn waits for the CLI's own re-run to start before it
# concludes the CLI declined (the re-run is queued at CLI startup, before stdin).
_RESUME_START_WAIT_SECONDS = 20.0


def _clear_unsolicited_projection(session: Any) -> None:
    """Discard the current unsolicited burst's text and structured projections."""
    session._unsolicited_text.clear()
    session._unsolicited_items.clear()
    session._unsolicited_tool_items.clear()
    session._unsolicited_api_signals = {
        "api_error_kind": None,
        "api_error_status": None,
        "api_retries": None,
        "rate_limit_rejected": None,
        "result_text": "",
    }
    session._unsolicited_start_notified = False
    session._unsolicited_seen.clear()
    session._unsolicited_delivery_id = None
    session._unsolicited_header_emitted = False
    # Keep this future-proof if a pending wake marker is added as a separate
    # field; today it is represented by the lifecycle item above.
    if hasattr(session, "_unsolicited_woken"):
        session._unsolicited_woken = None


def _claim_child_exit_emission(session: Any) -> bool:
    """Claim the one child-exit lifecycle event shared by both emission paths."""
    lock = getattr(session, "_child_exit_emission_lock", None)
    if lock is None:
        lock = threading.Lock()
        existing = getattr(session, "_child_exit_emission_lock", None)
        if existing is None:
            session._child_exit_emission_lock = lock
        else:
            lock = existing
    with lock:
        if getattr(session, "_child_exited_emitted", False):
            return False
        session._child_exited_emitted = True
        return True


class ClaudeSdkTurnMixin:
    """Turn driver, stream consumer and reader loop (see module docstring)."""

    _pending_rename_ack: Optional[str] = None
    _deferred_rename: Optional[str] = None
    _host_prompt_folded: bool = False

    def _deliver_prefold_peer_items(self) -> None:
        """A fold hands the injected turn's answer to the host turn, but the peer message that
        started it still needs its card: deliver the buffered peer_in items on the background
        lane (display + persistence) instead of discarding them with the partial text."""
        items = _prefold_peer_items(self)
        callback = getattr(self, "_on_unsolicited_result", None)
        if not items or callback is None:
            return
        try:
            import inspect

            inspect.signature(callback).bind([], items)
        except (TypeError, ValueError):
            return  # a texts-only callback has no peer-card channel
        try:
            callback([], items)
        except Exception:
            logger.warning("claude-agent-sdk: pre-fold peer card delivery raised", exc_info=True)

    # ---------- per-turn ----------

    def _retired_before_query_result(self) -> TurnResult:
        result = TurnResult(
            error="SDK session is rotating; rebuilding CLI",
            should_retire=True,
            api_call_made=False,
            thread_id=getattr(self, "_session_id", None),
        )
        result.retired_before_query = True
        with self._interrupt_commit_lock:
            result.interrupted = self._interrupt_event.is_set()
        return result

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: Optional[float] = None,
        post_tool_quiet_timeout: Optional[float] = None,
        watch_poll_interval: float = 1.0,
        abort_grace: float = _TURN_ABORT_GRACE,
    ) -> TurnResult:
        """Send a user message and block until the SDK's ResultMessage,
        projecting the typed stream into Hermes' messages shape.

        Turn lifetime is IDLE-based; wall clock alone never retires a turn.
        `turn_timeout` (explicit arg > agent.claude_agent_sdk.turn_idle_timeout
        > the deprecated turn_timeout key > 900s) is the idle limit: it fires
        only after that long with no SDK message of any kind (stream deltas,
        system/status included) AND nothing outstanding — no tool running, no
        live background Task, no approval awaiting a human, no compaction.
        turn_max_seconds (config, 0 = off) is the only absolute cap. A
        post-tool quiet watchdog (`post_tool_quiet_timeout`; default 300s with
        streaming on, disabled with streaming off) catches wedges early: armed
        when a tool result arrives, cleared by any later activity. On a trip
        the CLI is interrupted and given `abort_grace` to unwind — a clean unwind keeps
        the partial transcript and the resumable session id (no retire);
        only a grace expiry hard-cancels and retires."""
        result = TurnResult()
        resume_only = user_input is RESUME_INTERRUPTED_TURN
        prompt = user_input if resume_only else _coerce_turn_input(user_input)
        max_buffer_size = getattr(self, "_max_buffer_size", None)
        if max_buffer_size and not resume_only:
            prompt = fit_sdk_turn_blocks(prompt, max_buffer_size=max_buffer_size)
        if isinstance(prompt, str) and not prompt.strip():
            self.consume_interrupt()
            result.final_text = (
                "This Claude Agent SDK route can't process an empty message. "
                "Please send text or a supported image."
            )
            result.error = result.final_text
            result.api_call_made = False
            return result
        # Fence stale references before ensure_started: close() clears the
        # client, so an old caller must not resurrect a loop on this object.
        if self._admission_retired():
            return self._retired_before_query_result()
        try:
            started = self.ensure_started()
            if started is None:
                return self._retired_before_query_result()
        except SdkShuttingDownError as exc:
            # Not a config/auth fault and not a rotation: nothing was spawned,
            # and the runtime must neither rebuild nor mark the run fatal.
            logger.info("claude-agent-sdk turn refused: %s", exc)
            result.error = str(exc)
            result.api_call_made = False
            return result
        except Exception as exc:
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk startup failed: {safe_exc}"
            result.should_retire = True
            # Keep the traceback: `str(exc)` alone collapses a KeyError to a
            # bare quoted key (e.g. "'anyio'"), which is undiagnosable from the
            # user-facing error string. WARNING so it reaches errors.log.
            logger.warning(
                "claude-agent-sdk startup failed (%s: %s)\n"
                "Traceback (most recent call last):\n%s",
                type(exc).__name__,
                safe_exc,
                _safe_sdk_traceback(exc),
            )
            # A refusal to start is fatal to the run, not turn-scoped: the
            # metered-key guard and an uninstallable SDK are config errors
            # ("startup" — NOT "billing": kanban maps failure_reason
            # "billing" to the transient EX_TEMPFAIL requeue sentinel, and
            # retrying cannot fix a present metered key); an auth-classified
            # failure is "auth".
            result.fatal_reason = "auth" if hint else "startup"
            return result

        # b3-30: the hermes-tools capability was issued once at CLI spawn and expired an hour later;
        # rotate it at every turn start so session_send/session_create keep authenticating.
        refresh_capability = getattr(self, "refresh_session_spawn_capability", None)
        if callable(refresh_capability):
            refresh_capability()

        with self._interrupt_commit_lock:
            # Re-open interrupt admission for this turn.  A post-terminal
            # request that no outer runtime consumed belongs to this next
            # direct turn and becomes its normal pre-set interrupt.
            self._terminal_result_committed = False
            if self._post_terminal_interrupt_pending:
                self._post_terminal_interrupt_pending = False
                self._interrupt_event.set()

        # Text buffered before this turn whose terminal ResultMessage never
        # arrived is partial mid-burst content — a later unrelated result
        # must never pick it up (misattribution is the proven worse failure;
        # texts parked DURING the turn re-buffer via the residue drain and
        # are unaffected). Never a silent drop: WARN what is discarded.
        stale = list(self._unsolicited_text)
        stale_projection = not getattr(self, "_unsolicited_burst_open", False) and bool(
            stale
            or self._unsolicited_items
            or self._unsolicited_tool_items
            or self._unsolicited_seen
        )
        if stale_projection:
            _clear_unsolicited_projection(self)
            logger.warning(
                "claude-agent-sdk: discarding %d stale unsolicited text(s) "
                "(%d chars) buffered before this turn — their terminal "
                "ResultMessage never arrived; attaching them to a later "
                "unrelated result is the proven worse failure",
                len(stale), sum(len(t) for t in stale),
            )

        # An interrupt that arrived between turns or during connect (up to
        # 60s) targets THIS turn — honor it instead of erasing it. (The old
        # unconditional clear() silently swallowed that window.)
        if self._interrupt_event.is_set():
            self.consume_interrupt()
            result.interrupted = True
            return result

        import concurrent.futures

        # The idle limit. turn_timeout (arg and legacy config key) keeps
        # working under the idle semantics; it never bounds wall clock.
        budget = (
            float(turn_timeout)
            if turn_timeout is not None
            else (
                _configured_turn_idle_timeout()
                or _configured_turn_timeout()
                or _DEFAULT_TURN_IDLE_TIMEOUT
            )
        )
        max_seconds = _configured_turn_max_seconds() or 0.0
        tool_max_suspend = _configured_turn_tool_max_suspend() or _TOOL_MAX_SUSPEND
        if post_tool_quiet_timeout is not None:
            quiet = float(post_tool_quiet_timeout)
        else:
            configured_quiet = _configured_post_tool_quiet_timeout()
            if configured_quiet is not None:
                quiet = configured_quiet  # 0 = explicitly disabled
            else:
                quiet = (
                    _DEFAULT_POST_TOOL_QUIET_STREAMING if self._streaming else 0.0
                )
        poll = max(0.01, float(watch_poll_interval))

        # Reserve only the check-to-handoff gap. The reservation is released
        # once the reader acknowledges ownership; after that close()/rotation
        # use _turn_inbox/_turn_claim_requested and can tear down a live turn.
        reservation: Any = None
        bootstrap_coroutine: Any = None
        future: Any = None
        bootstrap_holder: dict[str, Any] = {}
        scheduled = False
        coroutine_entered = threading.Event()
        schedule_lock = threading.Lock()
        schedule_state = {
            "bootstrap_started": False,
            "bootstrap_closed": False,
        }

        def finalize_bootstrap_on_loop() -> None:
            with schedule_lock:
                if (
                    schedule_state["bootstrap_started"]
                    or schedule_state["bootstrap_closed"]
                ):
                    return
                schedule_state["bootstrap_closed"] = True
                coroutine = bootstrap_holder.pop("coroutine", None)
            if coroutine is not None:
                coroutine.close()

        def release_scheduled(_done: Any = None) -> None:
            self._release_turn_admission(reservation)
            with self._turn_callback_lock:
                if getattr(self, "_pending_turn_admission_future", None) is future:
                    self._pending_turn_admission_future = None
            with schedule_lock:
                bootstrap_started = schedule_state["bootstrap_started"]
            if not bootstrap_started and reservation is not None:
                # A cancelled concurrent future may settle on the caller
                # thread before the loop callback that owns bootstrap_turn.
                # Finalize there only by queueing the close back onto the SDK
                # loop; never close a submitted coroutine from this thread.
                try:
                    reservation.loop.call_soon_threadsafe(
                        finalize_bootstrap_on_loop
                    )
                except RuntimeError:
                    # The loop's own teardown path will finalize its pending
                    # callback; there is no safe caller-thread fallback here.
                    logger.debug(
                        "claude-agent-sdk bootstrap finalizer could not be queued",
                        exc_info=True,
                    )

        async def bootstrap_turn() -> Any:
            coroutine_entered.set()
            with self._turn_callback_lock:
                if getattr(self, "_pending_turn_admission_future", None) is future:
                    self._pending_turn_admission_future = None
            with schedule_lock:
                if schedule_state["bootstrap_closed"]:
                    return None
                schedule_state["bootstrap_started"] = True
            # The consumer coroutine is created on the loop thread. After
            # submission the caller owns only the concurrent future, so every
            # coroutine finalization stays on this loop's thread.
            return await self._consume_turn_with_admission(prompt, reservation)

        try:
            # Acquire inside the try so an asynchronous KeyboardInterrupt
            # between acquisition and cleanup cannot leak the reservation.
            # Create the token before registration. The caller retains this
            # reference if registration is interrupted after insertion.
            with self._turn_callback_lock:
                captured_loop = self._loop
                reservation = (
                    self._new_turn_admission_reservation(captured_loop)
                    if not self._retiring and not self._closed and captured_loop is not None
                    else None
                )
            reservation = self._reserve_turn_admission(reservation)
            if reservation is None:
                if self._admission_retired():
                    return self._retired_before_query_result()
                raise AssertionError("loop thread not started")
            loop = reservation.loop
            if loop.is_closed() or not loop.is_running():
                logger.warning(
                    "claude-agent-sdk turn admission captured a stopped loop; "
                    "retiring before query"
                )
                return self._retired_before_query_result()
            watch = _TurnWatch()
            watch.idle_limit = budget
            watch.tool_max_suspend = tool_max_suspend
            # The runtime's agent-side hook (activity generation bump): SDK
            # activity voids a pending turn_liveness abort claim.
            watch.on_activity = getattr(self, "_turn_activity_hook", None)
            self._turn_watch = watch
            trip: Optional[str] = None
            trip_elapsed = trip_idle = 0.0
            hard_trip = False
            turn_data: Optional[dict[str, Any]] = None
            bootstrap_coroutine = bootstrap_turn()
            bootstrap_holder["coroutine"] = bootstrap_coroutine
            try:
                future = asyncio.run_coroutine_threadsafe(bootstrap_coroutine, loop)
            except BaseException:
                bootstrap_holder.pop("coroutine", bootstrap_coroutine).close()
                bootstrap_coroutine = None
                raise
            finally:
                if future is not None:
                    future.add_done_callback(release_scheduled)
            # From this point, only the event loop owns the coroutine object.
            bootstrap_coroutine = None
            # Once submitted, the loop owns bootstrap_coroutine: the caller thread
            # must only cancel the future. release_scheduled queues any needed
            # finalization back to the loop; closing a submitted coroutine here
            # would tear down one the loop may already be executing.
            scheduled = True
            with self._turn_callback_lock:
                retired_after_schedule = self._retiring or self._closed
                # bootstrap_turn clears this slot on entry (under the same
                # lock) so close() only cancels a turn that has not started.
                # If the loop already entered it, publishing now would leave a
                # stale handle that a later close() cancels mid-turn.
                if (
                    not retired_after_schedule
                    and not future.done()
                    and not coroutine_entered.is_set()
                ):
                    self._pending_turn_admission_future = future
            if retired_after_schedule:
                # cancel() is the atomic handshake: True means the loop never
                # started the bootstrap, so nothing was submitted and retiring is
                # safe. False means it is running or already settled -- the loop
                # owns that result and the caller must harvest it below rather
                # than return a retire verdict that would replay the prompt.
                if not coroutine_entered.is_set() and future.cancel():
                    self._turn_watch = None
                    return self._retired_before_query_result()
                # The loop has entered bootstrap_turn. Its consumer owns the
                # decision about whether retirement preceded query submission;
                # caller-side retirement must never discard a settled result.
            if not coroutine_entered.wait(timeout=0.1):
                if (
                    loop.is_closed()
                    or not loop.is_running()
                    or self._admission_retired()
                ):
                    # A timed-out entry observation cannot authorize replay:
                    # only a successful cancel() proves no query was submitted.
                    if future.cancel():
                        self._turn_watch = None
                        return self._retired_before_query_result()
        except RuntimeError:
            if reservation is not None and (
                reservation.loop.is_closed() or not reservation.loop.is_running()
            ):
                self._turn_watch = None
                return self._retired_before_query_result()
            raise
        finally:
            if not scheduled:
                self._turn_watch = None
                # Release first so close() cannot wait on a reservation while
                # coroutine finalization is running in this caller thread.
                self._release_turn_admission(reservation)
                if bootstrap_coroutine is not None:
                    bootstrap_holder.pop("coroutine", bootstrap_coroutine).close()
        try:
            pending_verdict: Optional[str] = None
            prev_poll = time.monotonic()
            while turn_data is None and trip is None:
                try:
                    turn_data = future.result(timeout=poll)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    # py3.11 unifies TimeoutError/asyncio.TimeoutError/
                    # concurrent.futures.TimeoutError — a DONE future here
                    # means the COROUTINE settled in the race window: with an
                    # exception (typically its own TimeoutError — a socket/
                    # pipe timeout under the CLI), re-raise into the
                    # classification path instead of misreading it as a poll
                    # expiry (which would busy-spin until the budget); with a
                    # RESULT, harvest it — the turn completed.
                    if future.done():
                        if future.exception() is not None:
                            raise
                        turn_data = future.result()
                        continue
                    now = time.monotonic()
                    if now - prev_poll > _POLL_STALL_FACTOR * poll:
                        # The PROCESS was descheduled (swap/OOM), not the
                        # turn — that gap proves nothing about the stream.
                        watch.rebaseline()
                        prev_poll = now
                        pending_verdict = None
                        continue
                    prev_poll = now
                    verdict = watch.check(
                        budget=budget, quiet=quiet, max_seconds=max_seconds
                    )
                    if verdict is None:
                        pending_verdict = None
                        continue
                    if pending_verdict != verdict:
                        # Debounce: two consecutive polls must agree before
                        # a trip (closes the tick-vs-check race window).
                        pending_verdict = verdict
                        continue
                    now = time.monotonic()
                    trip = verdict
                    trip_elapsed = now - watch.started
                    trip_idle = now - watch.last_activity
            if trip is not None and turn_data is None:
                if future.done():
                    # Completed between check() and the trip decision —
                    # completion wins, the trip is void.
                    turn_data = future.result()
                    trip = None
                else:
                    # Interrupt through request_interrupt(): setting
                    # _interrupt_event is REQUIRED — _consume_turn's W22 EDE
                    # mask keys on the turn-local flag, so the CLI's
                    # interrupt-ack is masked instead of delivered as a
                    # fresh error.
                    self.request_interrupt()
                    try:
                        turn_data = future.result(timeout=abort_grace)
                    except (TimeoutError, concurrent.futures.TimeoutError):
                        if future.done():
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        elif not future.cancel():
                            # Settled in the cancel race window — a real
                            # result beats a fabricated hard trip.
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        else:
                            hard_trip = True
            if trip is not None and turn_data is not None and not hard_trip:
                if (
                    not turn_data["error"]
                    and turn_data.get("result_uuid")
                    and turn_data["final_text"]
                ):
                    # The turn FINISHED inside the grace window with a real
                    # answer — deliver it in full; the trip never happened.
                    # final_text is REQUIRED: the W22-masked interrupt ack
                    # also has error=None + a result uuid, but its final_text
                    # is empty (projection stopped at the interrupt) — that
                    # shape must stay a trip, or it degrades into a silent
                    # dropped turn. Consume our own interrupt signal so the
                    # mapping below cannot misread it. (A real user /stop
                    # racing this exact window loses its signal too — the
                    # completed answer it targeted is delivered, same as any
                    # near-boundary stop today.)
                    self.consume_interrupt()
                    # The watchdog's internal interrupt was withdrawn because
                    # the turn completed in full during grace. Clear the
                    # stream-consumer snapshot too; otherwise final mapping
                    # resurrects the voided trip as a user interruption.
                    turn_data["interrupt_observed"] = False
                    logger.info(
                        "claude-agent-sdk: turn completed during watchdog "
                        "grace (%.0fs elapsed) — delivered in full",
                        time.monotonic() - watch.started,
                    )
                    trip = None
        except Exception as exc:
            self.consume_interrupt()
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk turn failed: {safe_exc}"
            result.should_retire = True
            if hint is not None:
                # Auth failures are fatal; other mid-turn exceptions stay
                # transient (retire + retry semantics unchanged).
                result.fatal_reason = "auth"
            return result
        finally:
            # Detach only this turn's watchdog gate. A background Agent Task
            # can outlive the foreground turn and report later, so its
            # record is NOT finalized here: the stream teardown and close()
            # finalize whatever is still live on a real retire/close.
            watch.clear_tasks()
            self._turn_watch = None

        if turn_data is None:
            # Hard trip: the CLI ignored the interrupt for the whole grace —
            # nothing was harvested. Retire (today's shape, now the rare
            # fallback for a genuinely unresponsive CLI).
            self.consume_interrupt()
            result.interrupted = True
            # D62: a watchdog kill is not a user stop; the runtime continues it.
            result.watchdog_trip = True
            result.thread_id = getattr(self, "_session_id", None)
            result.error = self._format_trip_error(
                trip, budget, quiet, trip_elapsed, trip_idle, max_seconds
            )
            result.should_retire = True
            return result

        result.final_text = turn_data["final_text"]
        result.projected_messages = turn_data["messages"]
        result.terminal_answer = bool(turn_data.get("terminal_answer", False))
        result.tool_iterations = turn_data["tool_iterations"]
        result.token_usage_last = turn_data["usage"]
        result.token_usage_total = turn_data["usage"]
        result.num_turns = turn_data.get("num_turns", 1)
        result.model_last = turn_data.get("model")
        result.billing_mode = turn_data.get("billing_mode", "unknown")
        result.billing_evidence = turn_data.get("billing_evidence", {})
        result.total_cost_usd = turn_data.get("total_cost_usd")
        result.api_call_made = turn_data.get("api_call_made", True)
        # Preserve D0 API signals for the runtime's replay-safe transient retry
        # decision without expanding the shared Codex TurnResult contract.
        for signal in (
            "api_error_kind", "api_error_status", "api_retries",
            "rate_limit_rejected", "result_text",
        ):
            setattr(result, signal, turn_data.get(signal))
        result.thread_id = self._session_id
        result.turn_id = turn_data.get("result_uuid")
        # The stream consumer records interruption at the last point where it
        # can still precede terminal acceptance.  Never re-read the live event
        # here: a /stop can arrive after ResultMessage while ownership release
        # is finishing, and must not retroactively downgrade the committed
        # result.  The SDK-specific marker is deliberately dynamic so the
        # shared Codex TurnResult contract stays unchanged.
        result.terminal_result_accepted = bool(
            turn_data.get("terminal_result_accepted", False)
        )
        result.retired_before_query = bool(
            turn_data.get("retired_before_query", False)
        )
        # The runtime tells a CLI death before the query (api_call_made False)
        # from one after it by this marker; without it neither stream-death
        # recovery path in _run_sdk_attempts can see that the stream died.
        result.stream_ended = bool(turn_data.get("stream_ended", False))
        result.resume_declined = bool(turn_data.get("resume_declined", False))
        result.interrupted = bool(turn_data.get("interrupt_observed", False))
        # A non-terminal turn can spend time releasing foreground ownership
        # after the stream consumer's last snapshot. Restore the live read for
        # that path so a stop admitted during a stream-death/release handshake
        # remains authoritative. Terminal results stay fenced: once accepted,
        # no later event may retroactively downgrade the completed answer.
        if not result.terminal_result_accepted:
            with self._interrupt_commit_lock:
                result.interrupted = (
                    result.interrupted or self._interrupt_event.is_set()
                )
        # Consume either the honored pre-terminal interrupt or a late stop
        # aimed at the turn that has already committed. Both the event and
        # post-terminal pending state must clear together so neither can bleed
        # into the next turn on this session object.
        self.consume_interrupt()
        # The fence covered commit -> mapping. It ends HERE, with the result in
        # the caller's hands: between turns the CLI can still be running an
        # unsolicited background task, and a /stop then must reach the child
        # (and pre-set the next turn) exactly as it did before the fence.
        with self._interrupt_commit_lock:
            self._terminal_result_committed = False
        if turn_data["error"]:
            # A prior MCP tool use is not evidence that this terminal SDK
            # error belongs to MCP; preserve fail-closed Claude auth handling
            # unless the error text itself identifies an MCP origin.
            hint = classify_auth_failure(turn_data["error"])
            result.error = hint or turn_data["error"]
            if hint is not None:
                result.should_retire = True
                result.fatal_reason = "auth"
        if turn_data.get("billing_guard_violation"):
            # This is durable account/config state, not a transient provider
            # billing error: the generic "billing" fatal_reason is a retry
            # sentinel, which would repeatedly spend calls against the same
            # unsafe lane. "startup" stops the run until the operator fixes
            # Extra Usage/credentials or explicitly opts into metering.
            result.should_retire = True
            result.fatal_reason = "startup"
        if turn_data.get("stream_ended"):
            # A dead stream is permanent on this session object: the reader
            # exited, _stream_ended stays set, and ensure_started() keeps
            # returning early while _client is non-None — so WITHOUT retire,
            # every later turn short-circuits to this same error with zero
            # model calls, forever (the CLI died once and poisoned the
            # session; observed as the unrecoverable half of the 2026-08-09
            # desync incident). Retire lets the runtime rebuild a fresh CLI
            # on the next attempt/turn. Model-level result subtypes
            # (error_max_turns, error_max_budget_usd) stay non-retiring.
            result.should_retire = True
        if trip is not None:
            # Clean-ack trip: the CLI honored our interrupt inside the grace,
            # the partial transcript above is preserved, and the session id
            # is resumable — mirror the user-interrupt lane (close-but-keep-
            # resume-id) instead of retiring. The trip text WINS over
            # whatever error the drain surfaced (typically the W22-masked
            # interrupt ack) — EXCEPT auth and stream-death, which keep
            # their retire verdicts from the mapping above.
            if result.fatal_reason != "auth":
                result.error = self._format_trip_error(
                    trip, budget, quiet, trip_elapsed, trip_idle, max_seconds
                )
            result.interrupted = True
            result.watchdog_trip = True
        return result

    def _format_trip_error(
        self,
        kind: Optional[str],
        budget: float,
        quiet: float,
        elapsed: float,
        idle: float,
        max_seconds: float = 0.0,
    ) -> str:
        """≤200 chars (the gateway truncates at str(err)[:200]); keeps the
        literal "turn timed out" needle; names the cause."""
        if kind == "post_tool_quiet":
            return (
                f"turn timed out: no SDK activity for {idle:.0f}s after a "
                f"tool result (turn ran {elapsed:.0f}s, quiet limit "
                f"{quiet:.0f}s)"
            )
        if kind == "max_seconds":
            return (
                f"turn timed out after {elapsed:.0f}s: turn_max_seconds cap "
                f"{max_seconds:.0f}s reached (idle {idle:.0f}s)"
            )
        return (
            f"turn timed out after {idle:.0f}s idle: no SDK activity and no "
            f"tool, task or approval outstanding (idle limit {budget:.0f}s, "
            f"turn ran {elapsed:.0f}s)"
        )

    # ---------- internals ----------

    async def _consume_turn_with_admission(
        self, prompt: Any, reservation: Any
    ) -> dict[str, Any]:
        self._turn_admission_reservation = reservation
        try:
            return await self._consume_turn(prompt)
        finally:
            self._release_turn_admission(reservation)
            with self._turn_callback_lock:
                if self._turn_admission_reservation is reservation:
                    self._turn_admission_reservation = None

    async def _consume_turn(self, prompt: Any) -> dict[str, Any]:
        """The async side of one turn: query, then read THIS turn's messages
        off the reader loop's inbox until the ResultMessage.

        Deliberately NOT `receive_response()` — that helper serves the oldest
        buffered ResultMessage, which is not necessarily ours. See
        `_reader_loop` for why that is a permanent-corruption bug."""
        projector = ClaudeSdkEventProjector()
        out: dict[str, Any] = {
            "final_text": "",
            "terminal_answer": False,
            "messages": [],
            "tool_iterations": 0,
            "usage": None,
            "error": None,
            "result_uuid": None,
            "model": None,
            "stream_ended": False,
            "mcp_tool_seen": False,
            "billing_guard_violation": False,
            "billing_mode": self._reported_billing_mode(),
            "billing_evidence": dict(self._billing_evidence),
            "total_cost_usd": None,
            "num_turns": 1,
            "api_call_made": True,
            "query_submitted": False,
            "interrupt_observed": False,
            "terminal_result_accepted": False,
            "retired_before_query": False,
            "api_error_kind": None,
            "api_error_status": None,
            "api_retries": None,
            "rate_limit_rejected": None,
            "result_text": "",
        }

        boundary_interrupt = False
        own_envelope_sender = _own_envelope_sender(prompt)

        def _snapshot_interrupt() -> bool:
            with self._interrupt_commit_lock:
                observed = self._interrupt_event.is_set()
            out["interrupt_observed"] = observed
            return observed

        # Rotation may already have cancelled the reader by the time this
        # coroutine is scheduled. Prefer the retirement outcome over the
        # reader's terminal marker so the runtime can rebuild and retry.
        with self._turn_callback_lock:
            retired_before_query = bool(
                getattr(self, "_retiring", False) or getattr(self, "_closed", False)
            )
        if retired_before_query:
            _snapshot_interrupt()
            out["error"] = "SDK session is rotating; rebuilding CLI"
            out["stream_ended"] = True
            out["retired_before_query"] = True
            out["api_call_made"] = False
            return out

        ended = self._stream_ended
        if ended is not None:
            _snapshot_interrupt()
            out["error"] = "SDK message stream ended before this turn" + (
                f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
            )
            out["stream_ended"] = True
            return out
        if self._billing_guard_error is not None:
            _snapshot_interrupt()
            out["error"] = self._billing_guard_error
            out["billing_guard_violation"] = True
            out["billing_mode"] = self._reported_billing_mode()
            out["billing_evidence"] = dict(self._billing_evidence)
            out["api_call_made"] = False
            return out
        inbox: Any = asyncio.Queue()
        # Ask the sole reader to drain everything already ahead of this turn
        # in the resumed client's FIFO, then atomically install our inbox.  A
        # direct assignment here races an already-buffered background result:
        # the reader can observe the new inbox before it observes that older
        # result and mis-serve it as this query's answer.
        claims = self._turn_claims
        if claims is None:
            _snapshot_interrupt()
            out["error"] = "SDK message reader is not ready to claim this turn"
            out["stream_ended"] = True
            return out
        claim_ack = asyncio.get_running_loop().create_future()
        # Rotation shares this lock so it cannot pass the idle check between the claim
        # request and its publication to the reader.
        retired_before_query = False
        stream_ended_before_query = False
        with self._turn_callback_lock:
            if getattr(self, "_retiring", False) or getattr(self, "_closed", False):
                retired_before_query = True
            elif self._stream_ended is not None:
                stream_ended_before_query = True
            else:
                self._turn_claim_requested = True
                self._turn_claim_ack = claim_ack
                claims.put_nowait(("claim", inbox, claim_ack))
        if retired_before_query:
            _snapshot_interrupt()
            out["error"] = "SDK session is rotating; rebuilding CLI"
            out["stream_ended"] = True
            out["retired_before_query"] = True
            out["api_call_made"] = False
            return out
        if stream_ended_before_query:
            _snapshot_interrupt()
            out["error"] = "SDK message stream ended before this turn"
            out["stream_ended"] = True
            out["api_call_made"] = False
            return out
        interrupted = False
        billing_guarded = False
        held_interim_assistant: Any = None
        persisted_projection_count = 0

        def _flush_interim_assistant() -> None:
            nonlocal held_interim_assistant
            if held_interim_assistant is not None:
                self._notify_interim_assistant(held_interim_assistant)
                held_interim_assistant = None

        try:
            try:
                await claim_ack
            finally:
                with self._turn_callback_lock:
                    self._turn_claim_requested = False
                    if self._turn_claim_ack is claim_ack:
                        self._turn_claim_ack = None
            # Admission protects only the check-to-handoff window. Once the
            # reader has acknowledged this claim, ownership is expressed by
            # _turn_inbox and close() must be free to stop the reader.
            self._release_turn_admission(
                getattr(self, "_turn_admission_reservation", None)
            )
            ended = self._stream_ended
            if ended is not None:
                with self._interrupt_commit_lock:
                    interrupted = interrupted or self._interrupt_event.is_set()
                out["error"] = "SDK message stream ended before this turn" + (
                    f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
                )
                out["stream_ended"] = True
                out["interrupt_observed"] = interrupted
                return out
            resume_only = prompt is RESUME_INTERRUPTED_TURN
            query_input = (
                None if resume_only
                else _sdk_user_message_stream(prompt)
                if isinstance(prompt, list)
                else prompt
            )
            # This is the loop-owned admission point. Once set, a concurrent
            # retirement can only be reported after this query's result is
            # settled; it must never authorize replay of the side effects.
            with self._turn_callback_lock:
                if getattr(self, "_retiring", False) or getattr(self, "_closed", False):
                    out["error"] = "SDK session is rotating; rebuilding CLI"
                    out["stream_ended"] = True
                    out["retired_before_query"] = True
                    out["api_call_made"] = False
                    return out
                out["query_submitted"] = True
            if resume_only:
                # D62 L2: send nothing. The CLI's own re-run (queued at its
                # startup) is already on, or about to reach, this inbox; the
                # reader held any pre-claim frames for it (_resume_backlog).
                deadline = time.monotonic() + float(
                    getattr(self, "_resume_start_wait", _RESUME_START_WAIT_SECONDS)
                )
                while inbox.empty() and self._stream_ended is None:
                    if time.monotonic() >= deadline:
                        out["resume_declined"] = True
                        out["api_call_made"] = False
                        return out
                    await asyncio.sleep(0.1)
            else:
                await self._client.query(query_input)
            while True:
                message = await inbox.get()
                watch = self._turn_watch
                if watch is not None:
                    # Any stream message is liveness — stamp before anything
                    # else so the caller-thread watchdog sees it.
                    watch.tick()
                # A steer typed since the last message sits AFTER everything
                # projected so far and BEFORE whatever this message adds:
                # stamp that position now so its row is written where it was
                # sent (see _persist_steer_boundary), not before/after the turn.
                for _mark in getattr(self, "_pending_steer_marks", ()):
                    if _mark.get("pos") is None:
                        _mark["pos"] = len(out["messages"])
                if isinstance(message, _StreamEnd):
                    with self._interrupt_commit_lock:
                        interrupted = interrupted or self._interrupt_event.is_set()
                    out["error"] = (
                        "SDK message stream ended before this turn's result"
                        + (f": {_safe_sdk_error_text(message.error)}" if message.error else "")
                    )
                    out["stream_ended"] = True
                    break
                # Capture the SDK session id from ANY message that carries it —
                # the init SystemMessage announces it first, so even a turn
                # interrupted before its ResultMessage keeps a resumable id.
                early_sid = getattr(message, "session_id", None)
                if early_sid:
                    self._session_id = early_sid
                pending_rename = getattr(self, "_pending_rename_ack", None)
                if pending_rename and type(message).__name__ == "ResultMessage" and _is_rename_ack(
                    getattr(message, "result", None), [], pending_rename
                ):
                    # A /rename issued at the previous release answered after this turn
                    # claimed the stream: the CLI's ack, never this turn's result.
                    _flush_interim_assistant()
                    self._pending_rename_ack = None
                    continue
                if pending_rename and type(message).__name__ == "AssistantMessage" and _is_rename_ack(
                    None,
                    ["".join(str(getattr(block, "text", "") or "")
                             for block in (getattr(message, "content", None) or []))],
                    pending_rename,
                ):
                    # The same ack's assistant text: never relayed or projected as this
                    # turn's words. The pending name stays until its ResultMessage lands.
                    _flush_interim_assistant()
                    continue
                if type(message).__name__ == "ResultMessage" and _is_injected_origin(
                    getattr(message, "origin", None)
                ):
                    # ResultMessage.origin names the message that triggered its
                    # turn: a peer/task-notification result belongs to a separate
                    # injected turn. Hand it to the background path instead —
                    # unless the reader saw this turn's prompt folded INTO that
                    # injected turn; then its single result is this turn's answer.
                    # One fold unlocks exactly one result.
                    if getattr(self, "_host_prompt_folded", False):
                        self._host_prompt_folded = False
                    elif _is_own_envelope_result(getattr(message, "origin", None), own_envelope_sender):
                        # The CLI re-attributed this turn's own peer-envelope query as a
                        # host-injected peer turn: its result IS this turn's answer. Routing it
                        # away left the host turn running until the idle limit and parked every
                        # owner message queued behind it (live 2026-09-30, session c68a80).
                        own_envelope_sender = None
                    else:
                        self._handle_unsolicited(message)
                        continue
                self._handle_compact_boundary(message)
                origin = effective_origin(message)
                if (
                    type(message).__name__ == "UserMessage"
                    and isinstance(origin, dict)
                    and origin.get("kind") == "peer"
                ):
                    self._deliver_host_peer_item(message, origin)
                # Task lifecycle and provisional Bash records are stream
                # concerns, not foreground-turn visibility. Observe before
                # interrupt/billing gates so draining turns cannot orphan them.
                self._observe_sdk_lifecycle(message)
                if type(message).__name__ == "AssistantMessage":
                    message_error = getattr(message, "error", None)
                    if message_error:
                        out["api_error_kind"] = str(message_error)
                elif (
                    type(message).__name__ == "SystemMessage"
                    and getattr(message, "subtype", "") == "api_retry"
                ):
                    data = getattr(message, "data", None)
                    out["api_retries"] = dict(data) if isinstance(data, dict) else {}
                elif type(message).__name__ == "RateLimitEvent":
                    info = getattr(message, "rate_limit_info", None)
                    if str(getattr(info, "status", "")).lower() == "rejected":
                        out["rate_limit_rejected"] = {
                            "rate_limit_type": getattr(info, "rate_limit_type", None),
                            "resets_at": getattr(info, "resets_at", None),
                        }
                if self._billing_guard_error is not None and not billing_guarded:
                    billing_guarded = True
                    out["error"] = self._billing_guard_error
                    out["billing_guard_violation"] = True
                    try:
                        await self._client.interrupt()
                    except Exception:
                        logger.debug(
                            "claude-agent-sdk billing-guard interrupt failed",
                            exc_info=True,
                        )
                if self._interrupt_event.is_set():
                    # Previously `break`. Bailing out before this turn's
                    # ResultMessage orphaned it in the shared stream, and the
                    # NEXT turn was then served that stale result — the same
                    # off-by-one corruption `_reader_loop` exists to prevent,
                    # reachable through the interrupt path with no Agent tool
                    # involved. Keep draining to the result; stop projecting.
                    interrupted = True
                if type(message).__name__ == "StreamEvent":
                    if watch is not None:
                        # A partial delta proves the post-tool model call is
                        # alive — the quiet watchdog stands down.
                        watch.disarm_post_tool()
                    if not interrupted and not billing_guarded:
                        self._forward_stream_delta(message)
                    continue
                if type(message).__name__ == "ResultMessage":
                    out["api_error_status"] = getattr(
                        message, "api_error_status", None
                    )
                    out["result_text"] = _safe_sdk_error_text(
                        getattr(message, "result", None)
                    )
                    # A steer is a second query on the same SDK stream. Its
                    # human-origin terminal result owns the live Hermes turn;
                    # the prior query's un-attributed terminal result must not
                    # close that turn first.
                    with self._interrupt_commit_lock:
                        pending_steer = getattr(self, "_pending_steer_results", 0)
                        is_steer_result = pending_steer > 0 and _is_human_origin(message)
                        if is_steer_result:
                            self._pending_steer_results = pending_steer - 1
                            pending_inputs = getattr(self, "_pending_steer_inputs", [])
                            steer_text = pending_inputs.pop(0) if pending_inputs else None
                            marks = getattr(self, "_pending_steer_marks", [])
                            steer_mark = marks.pop(0) if marks else None
                        else:
                            steer_text = None
                            steer_mark = None
                        pending_steer_after = getattr(
                            self, "_pending_steer_results", 0
                        )
                    # The runtime owns the canonical Hermes transcript. Publish
                    # every projection settled before this SDK boundary there
                    # while the live turn still owns the stream. The same dicts
                    # stay in ``out['messages']`` and carry the normal persistence
                    # markers into the eventual turn-end flush.
                    if pending_steer > 0:
                        callback = getattr(self, "_on_steer_settled", None)
                        if callable(callback):
                            settled = out["messages"][persisted_projection_count:]
                            steer_pos = (steer_mark or {}).get("pos")
                            callback(
                                steer_text, settled,
                                None if steer_pos is None
                                else max(0, min(steer_pos - persisted_projection_count, len(settled))),
                            )
                        persisted_projection_count = len(out["messages"])
                    if pending_steer > 0 and (
                        not is_steer_result or pending_steer_after > 0
                    ):
                        _flush_interim_assistant()
                        continue
                self._note_mcp_tool_use(message, out)
                if not billing_guarded:
                    if not interrupted:
                        self._notify_tool_started(message)
                        self._notify_tool_use(message)
                    # Results must still close cards while the interrupted turn
                    # drains to its terminal ResultMessage.
                    self._notify_tool_results(message)
                    if not interrupted:
                        if type(message).__name__ == "AssistantMessage":
                            # A plain assistant paragraph may be the final
                            # answer (especially on injected/folded turns).
                            # Hold it until another assistant step proves the
                            # turn continues. Text paired with a tool call is
                            # unambiguously interim and can be relayed now.
                            _flush_interim_assistant()
                            blocks = getattr(message, "content", None) or []
                            has_text = any(
                                type(block).__name__ == "TextBlock"
                                and getattr(block, "text", "")
                                for block in blocks
                            )
                            has_tool_use = any(
                                type(block).__name__ == "ToolUseBlock"
                                for block in blocks
                            )
                            is_synthetic_error = bool(
                                getattr(message, "error", None)
                                and getattr(message, "model", None) == "<synthetic>"
                            )
                            if has_text and not has_tool_use and not is_synthetic_error:
                                held_interim_assistant = message
                            elif not is_synthetic_error:
                                self._notify_interim_assistant(message)
                projection, _result_is_error, _result_is_contradictory_success = (
                    self._project_message_step(projector, watch, message, out)
                )
                if projection.is_result:
                    # Terminal acceptance and interrupt observation are one
                    # atomic boundary.  If interrupt admission won the lock,
                    # REPORT it; if commit wins, later requests are queued and
                    # cannot call client.interrupt() during release.
                    #
                    # Reported, never folded into `interrupted`: that local
                    # gates delivery below, and the EDE mask further down whose
                    # contract is "never a fresh event read". A stop admitted
                    # during projection must not discard the answer this
                    # ResultMessage just completed, nor reclassify a genuine
                    # terminal error as an honored interrupt.
                    with self._interrupt_commit_lock:
                        # A steer may have been admitted while projection was
                        # running. Keep the result in this foreground turn and
                        # wait for that steer result before committing.
                        pending_steer_after_projection = getattr(
                            self, "_pending_steer_results", 0
                        )
                        if not pending_steer_after_projection:
                            boundary_interrupt = (
                                interrupted or self._interrupt_event.is_set()
                            )
                            self._terminal_result_committed = True
                    if pending_steer_after_projection:
                        continue
                    out["terminal_result_accepted"] = True
                if not interrupted and not billing_guarded:
                    if projection.messages:
                        out["messages"].extend(projection.messages)
                    projected_tools = any(
                        message.get("role") == "tool"
                        or bool(message.get("tool_calls"))
                        for message in projection.messages
                        if isinstance(message, dict)
                    )
                    assistant_outputs = [
                        message for message in projection.messages
                        if isinstance(message, dict) and message.get("role") == "assistant"
                    ]
                    if projection.is_tool_iteration or projected_tools:
                        out["terminal_answer"] = False
                    elif assistant_outputs:
                        last_assistant = assistant_outputs[-1]
                        out["terminal_answer"] = bool(
                            isinstance(last_assistant.get("content"), str)
                            and last_assistant["content"].strip()
                            and not last_assistant.get("tool_calls")
                        )
                    if projection.is_tool_iteration:
                        out["tool_iterations"] += 1
                        self._notify_tool_iteration()
                    if projection.final_text is not None and (
                        not _result_is_error or _result_is_contradictory_success
                    ):
                        out["final_text"] = projection.final_text
                if projection.is_result:
                    usage = getattr(message, "usage", None)
                    if isinstance(usage, dict):
                        out["usage"] = dict(usage)
                    sid = getattr(message, "session_id", None)
                    if sid:
                        self._session_id = sid
                    out["result_uuid"] = getattr(message, "uuid", None)
                    out["total_cost_usd"] = getattr(
                        message, "total_cost_usd", None
                    )
                    raw_num_turns = getattr(message, "num_turns", 1)
                    try:
                        out["num_turns"] = max(0, int(raw_num_turns))
                    except (TypeError, ValueError, OverflowError):
                        out["num_turns"] = 1
                    subtype = getattr(message, "subtype", "") or ""
                    if getattr(message, "is_error", False):
                        errors = getattr(message, "errors", None) or []
                        api_error_status = getattr(message, "api_error_status", None)
                        if subtype == "success" and not errors:
                            result_text = out["result_text"]
                            api_error_shape = bool(
                                api_error_status
                                or out["api_error_kind"]
                                or out["api_retries"] is not None
                                or result_text.startswith("API Error:")
                            )
                            if api_error_shape:
                                from agent.claude_sdk_transient import (
                                    classify_sdk_api_failure,
                                )

                                _verdict, error_class, _wait_hint = (
                                    classify_sdk_api_failure(out)
                                )
                                out["error"] = (
                                    f"Claude API error ({error_class}): "
                                    f"{result_text or subtype}"
                                    + (
                                        f" (HTTP {api_error_status})"
                                        if api_error_status else ""
                                    )
                                )
                                break
                            # Contradictory envelope: is_error=True yet
                            # subtype="success" with nothing in errors. The
                            # CLI emits this shape rarely (2026-08-11: it
                            # killed a cron run as "RuntimeError: SDK result
                            # error (subtype=success): success" — a dead job
                            # over a turn that had actually produced its
                            # answer). There is no error to report, so the
                            # error flag loses to the subtype; kept loud for
                            # diagnosis. A genuine failure carries a non-empty
                            # errors list or a non-success subtype and still
                            # takes the honest path below.
                            logger.warning(
                                "claude-agent-sdk: contradictory result "
                                "envelope (is_error=True, subtype=success, "
                                "no errors) — treated as success (diagnostic omitted)",
                            )
                            break
                        detail = _safe_sdk_error_text(
                            "; ".join(str(e) for e in errors)
                            or getattr(message, "result", None)
                            or subtype
                        )
                        err_text = f"SDK result error (subtype={subtype}): {detail}"
                        if api_error_status:
                            err_text += f" (HTTP {api_error_status})"
                        # A turn WE interrupted before any assistant content
                        # ends as is_error/error_during_execution in the CLI
                        # ("[ede_diagnostic] result_type=user…") — that is the
                        # interrupt being honored, not a failure; surfacing it
                        # paged the operator with a false "Processing stopped"
                        # (2026-08-09 barge-in incident). Mask ONLY that exact
                        # shape, only when THIS turn's interrupt flag is set
                        # (never a fresh event read — a late /stop must not
                        # reclassify a genuine error), and only when no auth
                        # needle is present (an auth failure must keep its
                        # error + retire path regardless of interrupts).
                        # A real transient failure colliding with an interrupt
                        # re-fires next turn with no interrupt pending and
                        # takes the honest path; the masked text is INFO-kept.
                        if (
                            interrupted
                            and subtype == "error_during_execution"
                            and classify_auth_failure(err_text) is None
                        ):
                            logger.info(
                                "claude-agent-sdk: masked %s on requested "
                                "interrupt (interrupt honored, not a "
                                "failure): %s",
                                subtype,
                                _safe_sdk_error_text(err_text),
                            )
                        else:
                            if not billing_guarded:
                                out["error"] = err_text
                    elif subtype not in ("", "success") and not billing_guarded:
                        # e.g. error_max_turns / error_max_budget_usd — surface
                        # honestly; the partial transcript is still projected.
                        out["error"] = f"SDK turn ended: {subtype}"
                    break
        finally:
            if self._turn_inbox is inbox:
                # Relinquish ownership through the same reader arbiter.  It
                # drains immediately buffered post-result residue into this
                # inbox before acknowledging release, preserving the existing
                # overlap/own-answer dedup handling below.
                if self._stream_ended is None and self._turn_claims is not None:
                    release_ack = asyncio.get_running_loop().create_future()
                    self._turn_claims.put_nowait(("release", inbox, release_ack))
                    await release_ack
                else:
                    self._turn_inbox = None
                    self._host_prompt_folded = False
                # Stream death can win the release handshake after the
                # terminal ResultMessage was already delivered.  The reader's
                # death path acknowledges queued operations so waiters wake,
                # but it deliberately does not install/apply their ownership.
                # Re-check after the acknowledgement: preserve the valid turn
                # result while retiring the now-dead session, and never leave
                # a between-turn inbox falsely claimed.
                if self._stream_ended is not None:
                    if self._turn_inbox is inbox:
                        self._turn_inbox = None
                    self._host_prompt_folded = False
                    out["stream_ended"] = True
                elif self._turn_inbox is None:
                    # The stream is free: apply a rename refused while this turn owned it.
                    self._apply_deferred_rename()
            with self._interrupt_commit_lock:
                # Any steer whose query was abandoned by interruption, stream
                # death, or release must not be mistaken for a later turn's
                # result. Completed steers already decremented this counter.
                # A steer the turn never settled (its answer, if any, arrives
                # late as unsolicited output) still needs its row: the owner's
                # words go into the transcript where they were typed, ahead of
                # the reply that answers them.
                leftovers = list(getattr(self, "_pending_steer_marks", ()))
                inserted = 0
                for _mark in leftovers:
                    _text = str(_mark.get("text") or "")
                    if not _text:
                        continue
                    _pos = _mark.get("pos")
                    _pos = len(out["messages"]) if _pos is None else _pos + inserted
                    _pos = max(persisted_projection_count, min(_pos, len(out["messages"])))
                    from agent.claude_sdk_runtime_continuity import steer_insert_index

                    _pos = persisted_projection_count + steer_insert_index(
                        out["messages"][persisted_projection_count:], _pos - persisted_projection_count
                    )
                    out["messages"].insert(_pos, {"role": "user", "content": _text, "timestamp": _mark.get("ts")})
                    inserted += 1
                self._pending_steer_results = 0
                self._pending_steer_inputs = []
                self._pending_steer_marks = []
            # Anything the reader parked after our ResultMessage belongs to a
            # CLI-initiated turn that overlapped ours. Route it now — left in
            # a discarded queue it would be lost, and left in the stream it
            # would become the next turn's answer. EXCEPT a residue result
            # that repeats THIS turn's own answer: delivering it through the
            # background lane re-presents the agent's own text as a fake
            # completion (the 2026-08-06 echo class) — suppress it, dedup-mark
            # it, and consume its burst buffer. Genuinely different residue is
            # a real background completion and still delivers.
            final_norm = " ".join(str(out["final_text"]).split())
            while not inbox.empty():
                residue = inbox.get_nowait()
                if final_norm and type(residue).__name__ == "ResultMessage":
                    res_text = getattr(residue, "result", None)
                    if (
                        isinstance(res_text, str)
                        and " ".join(res_text.split()) == final_norm
                    ):
                        uuid = getattr(residue, "uuid", None)
                        if uuid:
                            _remember_recent_id(
                                self, "_unsolicited_delivered", "_unsolicited_delivered_order", uuid
                            )
                        self._unsolicited_results += 1
                        buffered = len(self._unsolicited_text)
                        _clear_unsolicited_projection(self)
                        logger.warning(
                            "claude-agent-sdk: residue ResultMessage %s "
                            "matches this turn's own answer — suppressed, "
                            "never delivered as a background result "
                            "(%d buffered text(s) consumed)",
                            uuid, buffered,
                        )
                        continue
                self._handle_unsolicited(residue)
            if held_interim_assistant is not None and not out.get("terminal_result_accepted"):
                _flush_interim_assistant()
        out["interrupt_observed"] = interrupted or boundary_interrupt
        out["billing_mode"] = self._reported_billing_mode()
        out["billing_evidence"] = dict(self._billing_evidence)
        return out

    # ---------- ungated per-message steps of _consume_turn ----------
    # Everything that depends on turn STATE (the interrupted / billing-guarded
    # gates on delivery, and every exit) stays in the loop scaffold above; the
    # two steps below run for every non-delta message regardless of that
    # state and only feed it.

    @staticmethod
    def _note_mcp_tool_use(message: Any, out: dict[str, Any]) -> None:
        """Record that this turn issued an ``mcp__`` tool (ungated)."""
        for block in getattr(message, "content", None) or []:
            if (
                type(block).__name__ == "ToolUseBlock"
                and str(getattr(block, "name", "")).startswith("mcp__")
            ):
                out["mcp_tool_seen"] = True

    def _deliver_host_peer_item(self, message: Any, origin: dict) -> None:
        """Render a peer prompt folded into this host turn at its stream position."""
        uuid = str(getattr(message, "uuid", None) or "")
        msg_id = str(origin.get("msg_id") or uuid)
        seen = getattr(self, "_host_peer_seen", None)
        if seen is None:
            seen = self._host_peer_seen = set()
        if uuid and not _remember_recent_id(
            self, "_host_peer_seen", "_host_peer_seen_order", uuid
        ):
            return
        content = getattr(message, "content", None)
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(
                str(getattr(block, "text", "") or "")
                for block in content or []
                if type(block).__name__ == "TextBlock"
            )
        item = {
            "kind": "peer_in",
            "text": str(origin.get("body") or text),
            "from": str(origin.get("from") or ""),
            "name": str(origin.get("name") or ""),
            "from_session": str(origin.get("fromSession") or ""),
            "msg_id": str(origin.get("msg_id") or uuid),
            "uuid": uuid,
        }
        if not item["text"]:
            return
        self._deliver_or_buffer_unsolicited([], [item], f"deliv-peer-{msg_id}" if msg_id else None)

    def _deliver_or_buffer_unsolicited(
        self, texts: list[str], items: list[dict], delivery_id: Optional[str]
    ) -> None:
        callback = getattr(self, "_on_unsolicited_result", None)
        if callback is None:
            pending = getattr(self, "_pending_unsolicited_deliveries", None)
            if pending is None:
                pending = self._pending_unsolicited_deliveries = []
            pending.append((list(texts), [dict(item) for item in items], delivery_id))
            # With background delivery off no callback is ever wired: keep only the newest few so a
            # long-lived session cannot grow this without bound.
            if len(pending) > _MAX_RETAINED_UNSOLICITED:
                del pending[: len(pending) - _MAX_RETAINED_UNSOLICITED]
            logger.warning(
                "claude-agent-sdk: retained unsolicited delivery until its callback is wired"
            )
            return
        try:
            self._deliver_unsolicited_callback(callback, texts, items, delivery_id)
        except Exception:
            logger.warning(
                "claude-agent-sdk: unsolicited-result delivery callback raised — answer may be lost",
                exc_info=True,
            )

    @staticmethod
    def _project_message_step(
        projector: ClaudeSdkEventProjector,
        watch: Optional[_TurnWatch],
        message: Any,
        out: dict[str, Any],
    ) -> tuple[Any, bool, bool]:
        """Project one stream message and do the ungated bookkeeping around it.

        Runs whatever the turn state: the projector recognises the terminal
        result either way, the watchdog needs the outstanding-tool evidence, and
        the model id is captured even on interrupted turns. Returns the
        projection plus the two terminal-error flags that the gated delivery
        reads to withhold a diagnostic ``result`` string from the transcript.
        """
        projection = projector.project(message)
        if watch is not None:
            # Outstanding-tool evidence: ToolUseBlocks issue, tool
            # results resolve (server tools never enter — the
            # projector resolves them inside their own assistant
            # message). A turn with a tool in flight is suspended
            # from BOTH watchdog rules.
            issued_ids = [
                call.get("id")
                for m in projection.messages
                if m.get("role") == "assistant"
                for call in (m.get("tool_calls") or [])
                if isinstance(call, dict)
            ]
            issued = sum(
                len(m.get("tool_calls") or [])
                for m in projection.messages
                if m.get("role") == "assistant"
            )
            if issued:
                watch.note_tools_issued(issued, ids=issued_ids)
            if projection.is_tool_iteration:
                watch.note_tools_resolved(
                    sum(
                        1
                        for m in projection.messages
                        if m.get("role") == "tool"
                    ),
                    ids=[
                        m.get("tool_call_id")
                        for m in projection.messages
                        if m.get("role") == "tool"
                    ],
                )
                # Codex-parity arm point: a tool result just landed;
                # silence from here on is the wedge signature.
                watch.arm_post_tool()
            elif projection.messages or projection.final_text is not None:
                watch.disarm_post_tool()
        if projection.model:
            # Last reported id wins; captured even on interrupted
            # turns — the tokens were still spent on that model.
            out["model"] = projection.model
        # A genuine terminal SDK error carries a result string, but it
        # is transport diagnostics, not an assistant answer. The outer
        # runtime needs `out["error"]` to activate provider fallback;
        # emitting this text first persists it as an assistant message
        # and poisons the fallback's history with a false “I am
        # blocked” claim. Contradictory success envelopes remain a
        # success exactly as handled in the terminal branch.
        _result_subtype = getattr(message, "subtype", "") or ""
        _result_is_error = bool(
            projection.is_result
            and (
                getattr(message, "is_error", False)
                or _result_subtype not in ("", "success")
            )
        )
        _result_is_contradictory_success = bool(
            _result_is_error
            and _result_subtype == "success"
            and not (getattr(message, "errors", None) or [])
            and not getattr(message, "api_error_status", None)
            and not out.get("api_error_kind")
            and out.get("api_retries") is None
            and not str(out.get("result_text") or "").startswith("API Error:")
        )
        return projection, _result_is_error, _result_is_contradictory_success

    # ---------- stream ownership ----------

    async def _reader_loop(self) -> None:
        """Sole owner of the SDK message stream for the client's lifetime.

        Hermes used to call `receive_response()` once per turn. That helper
        returns the OLDEST buffered ResultMessage — not necessarily this
        turn's — and the SDK offers no per-turn correlator (`query()` takes
        only a `session_id`, which is constant across turns; `ResultMessage`
        has a `uuid` with no request-side counterpart). Ordering is therefore
        the ONLY thing keeping answers matched to questions.

        The Claude Code CLI breaks that ordering by itself: when a background
        Agent task completes it injects a `<task-notification>` and runs a
        FULL assistant turn nobody asked for, leaving an unconsumed
        ResultMessage in the shared FIFO. Every later turn then pops a stale
        result. Because a desynced turn still looks like success — no error,
        no timeout, no interrupt — no retire path fires, so the skew is
        permanent and silent. Observed live 2026-07-25: four such turns left a
        constant off-by-4 for hours (dasbrow-hermes-coder#2).

        One reader for the whole client lifetime makes the rule enforceable:
        a message arriving while no turn is in flight is unsolicited BY
        DEFINITION, and gets routed away instead of poisoning the next turn."""
        from agent.transports.claude_agent_sdk_session import _RenameClaimDeferred

        end: _StreamEnd
        iterator = self._client.receive_messages().__aiter__()
        message_task = asyncio.ensure_future(iterator.__anext__())
        claim_task = asyncio.ensure_future(self._turn_claims.get())
        pending_turn_claims = []
        try:
            while True:
                done, _pending = await asyncio.wait(
                    (message_task, claim_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if message_task in done:
                    try:
                        message = message_task.result()
                    except StopAsyncIteration:
                        end = _StreamEnd(error=None)
                        break
                    self._observe_billing_evidence(message)
                    watch = getattr(self, "_turn_watch", None)
                    if watch is not None:
                        # Any CLI output is liveness for the in-flight turn —
                        # including an injected burst routed away below, which
                        # _consume_turn's own tick never sees.
                        watch.tick()
                    inbox = self._turn_inbox
                    if inbox is not None and self._unsolicited_burst_open:
                        # An injected turn (peer message, task notification)
                        # started before this claim and has not ended: the CLI
                        # runs turns sequentially, so everything up to its
                        # ResultMessage still belongs to it, not to the host's
                        # turn. Only an echo of the host's own prompt proves
                        # the CLI moved on.
                        if _is_own_prompt_echo(message):
                            # The CLI replays a queued prompt when it consumes it.
                            # Mid-burst, that means it folded the host prompt INTO
                            # the injected turn at a tool boundary (live probe
                            # 2026-09-24): the injected turn's one ResultMessage,
                            # peer origin and all, now answers the host turn.
                            # (cntrl carry) Its pre-fold partial text must never
                            # attach to a later unrelated result.
                            self._host_prompt_folded = True
                            self._close_unsolicited_burst()
                            self._native_peer_in_flight = False
                            self._native_peer_msg_id = None
                            self._native_peer_seen = False
                            self._deliver_prefold_peer_items()
                            if self._unsolicited_text or self._unsolicited_items:
                                logger.warning(
                                    "claude-agent-sdk: host prompt replayed mid-burst; discarding "
                                    "%d stale unsolicited text(s) buffered before it (a folded "
                                    "injected turn, or a burst whose result never arrived)",
                                    len(self._unsolicited_text),
                                )
                            _clear_unsolicited_projection(self)
                            inbox.put_nowait(message)
                        else:
                            if (getattr(message, "parent_tool_use_id", None) is None
                                    and type(message).__name__ != "ResultMessage"):
                                self._note_unsolicited_activity()
                            self._handle_unsolicited(message)
                            if type(message).__name__ == "ResultMessage":
                                self._close_unsolicited_burst()
                                self._apply_deferred_rename()
                                self._notify_idle_boundary()
                    elif inbox is not None and self._unsolicited_quiet_expired:
                        # A terminal result can arrive after the quiet boundary. It still
                        # belongs to the old burst until the CLI echoes this host prompt.
                        if _is_own_prompt_echo(message):
                            self._unsolicited_quiet_expired = False
                            inbox.put_nowait(message)
                        else:
                            self._handle_unsolicited(message)
                            if type(message).__name__ == "ResultMessage":
                                self._notify_idle_boundary()
                    elif inbox is not None:
                        inbox.put_nowait(message)
                    elif getattr(self, "_resume_backlog", None) is not None:
                        # D62 L2: the CLI's own re-run of the interrupted turn
                        # can start before run_turn claims the stream; it is
                        # that turn's output, not an unsolicited burst.
                        self._resume_backlog.append(message)
                    else:
                        if _starts_injected_turn(message):
                            self._unsolicited_burst_open = True
                            self._unsolicited_quiet_expired = False
                        if (self._unsolicited_burst_open and getattr(message, "parent_tool_use_id", None) is None
                                and type(message).__name__ != "ResultMessage"):
                            self._note_unsolicited_activity()
                        self._handle_unsolicited(message)
                        if type(message).__name__ == "ResultMessage":
                            self._close_unsolicited_burst()
                            self._apply_deferred_rename()
                            self._notify_idle_boundary()
                    # Message wins ties with a claim. Re-arm first, then loop:
                    # every immediately available pre-claim FIFO entry is
                    # classified unsolicited before the claim is acknowledged.
                    message_task = asyncio.ensure_future(iterator.__anext__())
                    continue

                operation, inbox, claim_ack = claim_task.result()
                claim_task = asyncio.ensure_future(self._turn_claims.get())
                if claim_ack.cancelled():
                    continue
                if operation == "claim":
                    if self._turn_inbox is not None:
                        # A short claimed operation (currently /rename) may
                        # have won admission just before this turn. Preserve
                        # FIFO claim order instead of failing the turn.
                        pending_turn_claims.append((inbox, claim_ack))
                        continue
                    self._host_prompt_folded = False
                    self._turn_inbox = inbox
                    backlog = getattr(self, "_resume_backlog", None)
                    if backlog is not None:
                        self._resume_backlog = None
                        for held in backlog:
                            inbox.put_nowait(held)
                elif operation == "rename":
                    if self._unsolicited_burst_open:
                        claim_ack.set_exception(
                            _RenameClaimDeferred("an injected SDK burst owns the stream")
                        )
                        continue
                    if self._turn_inbox is not None:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream already has a turn owner")
                        )
                        continue
                    self._host_prompt_folded = False
                    self._turn_inbox = inbox
                elif operation in ("release", "release_rename"):
                    if self._turn_inbox is not inbox:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream release owner mismatch")
                        )
                        continue
                    self._host_prompt_folded = False
                    self._turn_inbox = None
                    claim_ack.set_result(None)
                    if pending_turn_claims:
                        next_inbox, next_ack = pending_turn_claims.pop(0)
                        if operation == "release_rename":
                            while not inbox.empty():
                                next_inbox.put_nowait(inbox.get_nowait())
                        self._host_prompt_folded = False
                        self._turn_inbox = next_inbox
                        if not next_ack.done():
                            next_ack.set_result(None)
                    elif operation == "release_rename":
                        while not inbox.empty():
                            self._handle_unsolicited(inbox.get_nowait())
                    continue
                else:  # pragma: no cover - internal invariant
                    claim_ack.set_exception(
                        RuntimeError(f"unknown SDK stream ownership operation: {operation}")
                    )
                    continue
                claim_ack.set_result(None)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            message_task.cancel()
            claim_task.cancel()
            # Cancellation is the reader's terminal edge too. A queued claim
            # must be released below or the foreground coroutine waits until
            # its watchdog fires against a session that is already gone.
            end = _StreamEnd(error=None)
        except Exception as exc:  # pragma: no cover - stream torn down
            logger.debug(
                "claude-agent-sdk reader loop ended: %s",
                _safe_sdk_error_text(exc),
            )
            end = _StreamEnd(error=_safe_sdk_error_text(exc))
        finally:
            if not message_task.done():
                message_task.cancel()
            if not claim_task.done():
                claim_task.cancel()
        # The stream is gone (CLI exited or transport died). Mark it before
        # acknowledging every queued ownership operation.  A foreground turn
        # can pass its initial stream check, enqueue a claim behind buffered
        # messages, and then lose the reader to EOF; resolving the claim makes
        # it re-check this terminal state instead of waiting out turn_timeout.
        self._stream_ended = end
        self._finalize_sdk_tasks()
        pending_ack = self._turn_claim_ack
        if pending_ack is not None and not pending_ack.done():
            pending_ack.set_result(None)
        pending_claims = []
        if claim_task.done() and not claim_task.cancelled():
            try:
                pending_claims.append(claim_task.result())
            except Exception:  # pragma: no cover - queue task failed
                pass
        claims = self._turn_claims
        if claims is not None:
            while True:
                try:
                    pending_claims.append(claims.get_nowait())
                except asyncio.QueueEmpty:
                    break
        for _operation, _owner, claim_ack in pending_claims:
            if not claim_ack.done():
                claim_ack.set_result(None)
        for _owner, claim_ack in pending_turn_claims:
            if not claim_ack.done():
                claim_ack.set_result(None)
        inbox = self._turn_inbox
        if inbox is not None:
            inbox.put_nowait(end)

    def _ensure_unsolicited_header_emitted(self) -> None:
        if getattr(self, "_unsolicited_header_emitted", False):
            return
        self._unsolicited_header_emitted = True
        callback = getattr(self, "_on_unsolicited_header", None)
        if callback is None:
            return
        delivery_id = getattr(self, "_unsolicited_delivery_id", None)
        if not delivery_id:
            import uuid
            delivery_id = f"deliv-{uuid.uuid4().hex[:12]}"
            self._unsolicited_delivery_id = delivery_id
        items = getattr(self, "_unsolicited_items", None) or []
        header_items = [
            dict(item) for item in items
            if isinstance(item, dict) and item.get("kind") in ("lifecycle", "peer_in")
        ]
        try:
            callback(delivery_id, header_items)
        except Exception:
            logger.debug("claude-agent-sdk unsolicited-header callback failed", exc_info=True)

    def _handle_unsolicited(self, message: Any) -> None:
        """Route a message that arrived with no turn in flight.

        These are real CLI output (typically a finished background Agent task
        reporting in). They answer nothing Hermes asked, so they must never
        enter a turn's result — but their CONTENT is completed work the user
        is waiting on. Capture the complete injected turn, including its peer
        envelope and tool round, and hand the ordered projection over on the
        terminal ResultMessage. Without a callback, the historical WARN-drop
        stands."""
        if isinstance(message, _StreamEnd):
            # The runtime retirement path is the sole child-exit emitter. The
            # reader can observe EOF just before that path retires the turn;
            # emitting here as well creates two lifecycle rows with distinct
            # timestamps, so the shared guard cannot deduplicate them.
            self._unsolicited_stream_end_emitted = True
            return
        sid = getattr(message, "session_id", None)
        if sid:
            self._session_id = sid
        name = type(message).__name__
        signals = getattr(self, "_unsolicited_api_signals", None)
        if not isinstance(signals, dict):
            signals = self._unsolicited_api_signals = {
                "api_error_kind": None,
                "api_error_status": None,
                "api_retries": None,
                "rate_limit_rejected": None,
                "result_text": "",
            }
        if name == "AssistantMessage" and getattr(message, "error", None):
            signals["api_error_kind"] = str(message.error)
        elif name == "SystemMessage" and getattr(message, "subtype", "") == "api_retry":
            data = getattr(message, "data", None)
            signals["api_retries"] = dict(data) if isinstance(data, dict) else {}
        elif name == "RateLimitEvent":
            info = getattr(message, "rate_limit_info", None)
            if str(getattr(info, "status", "")).lower() == "rejected":
                signals["rate_limit_rejected"] = {
                    "rate_limit_type": getattr(info, "rate_limit_type", None),
                    "resets_at": getattr(info, "resets_at", None),
                }
        is_synthetic_error = bool(
            name == "AssistantMessage"
            and getattr(message, "error", None)
            and getattr(message, "model", None) == "<synthetic>"
        )
        if name == "ResultMessage" and _is_human_origin(message):
            # Human-origin results answer a query submitted by this host (most
            # commonly a steer). If ownership was released before the result
            # arrived, it is late, not a background task result. Clear any
            # preceding projection so it cannot be joined to the next burst.
            _clear_unsolicited_projection(self)
            logger.debug(
                "claude-agent-sdk: ignored late human-origin ResultMessage %s",
                getattr(message, "uuid", None),
            )
            return
        self._observe_sdk_lifecycle(message)
        if name == "StreamEvent":
            self._ensure_unsolicited_header_emitted()
            self._forward_stream_delta(message)
        if name == "AssistantMessage" and not is_synthetic_error:
            # CLI-initiated Agent work has no foreground turn, but its child
            # tool/text activity still belongs on the existing subagent feed.
            self._notify_tool_started(message)
            self._notify_tool_use(message)
        if name == "UserMessage":
            origin = effective_origin(message)
            if (isinstance(origin, dict) and origin.get("kind") == "peer"
                    and str(origin.get("msg_id") or "") == getattr(self, "_native_peer_msg_id", None)):
                self._native_peer_seen = True
            # Close what the line above opened. A woken turn (peer message, task
            # notification) opened a card per tool call but never closed one — the
            # foreground loop is the only other caller — so every tool it ran was
            # sealed at settle as "Result unavailable" although it had succeeded.
            self._notify_tool_results(message)
        if name == "ResultMessage":
            self._unsolicited_results += 1
            # Any unsolicited result ends the CLI turn; don't gate the release on
            # recognising the echo, or a missed echo holds owner input forever.
            self._native_peer_in_flight = False
            self._native_peer_msg_id = None
            self._native_peer_seen = False
        if getattr(message, "parent_tool_use_id", None):
            # Subagent streams belong to the parent tool card, not to the
            # session-level unsolicited turn.
            return
        if not hasattr(self, "_unsolicited_items"):
            return  # no projection state (partially built session): nothing to buffer into
        items = self._unsolicited_items
        tool_items = self._unsolicited_tool_items
        seen = self._unsolicited_seen

        def _message_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            parts = [
                str(getattr(block, "text", "") or "")
                for block in content or []
                if type(block).__name__ == "TextBlock" and getattr(block, "text", "")
            ]
            return "\n".join(parts)

        def _tool_args(block: Any) -> dict:
            args = getattr(block, "input", None) or {}
            return dict(args) if isinstance(args, dict) else {"input": args}

        def _woken_source(origin: dict) -> Optional[str]:
            kind = origin.get("kind")
            subkind = origin.get("subkind")
            if kind == "peer":
                return "peer"
            if kind == "channel":
                return "channel"
            if kind == "task-notification":
                return f"{kind}/{subkind}" if subkind else kind
            return None

        def _woken_by(origin: dict, content: Any) -> str:
            for key in ("name", "from", "task", "task_description", "taskDescription", "description", "body"):
                value = origin.get(key)
                if value:
                    return str(value)
            return _message_text(content) or "unknown"

        def _append_woken(origin: dict, content: Any, uuid: str) -> None:
            source = _woken_source(origin)
            if source is None or any(
                item.get("kind") == "lifecycle" and item.get("event") == "woken"
                for item in items
            ):
                return
            items.insert(0, {
                "kind": "lifecycle",
                "event": "woken",
                "source": source,
                "by": _woken_by(origin, content),
                "uuid": uuid,
            })

        if name == "UserMessage":
            origin = effective_origin(message)
            if isinstance(origin, dict):
                _append_woken(origin, getattr(message, "content", None), str(getattr(message, "uuid", None) or ""))
            is_woken = isinstance(origin, dict) and (
                origin.get("kind") == "peer" or origin.get("kind") == "task-notification"
            )
            is_peer = isinstance(origin, dict) and (
                origin.get("kind") == "peer"
                or (origin.get("kind") == "task-notification"
                    and origin.get("subkind") == "peer-send-message")
            )
            if is_woken and not self._unsolicited_start_notified:
                self._unsolicited_start_notified = True
                callback = getattr(self, "_on_unsolicited_start", None)
                prompt = str(origin.get("body") or _message_text(getattr(message, "content", None)))
                if callback is not None and prompt.strip():
                    try:
                        callback(prompt)
                    except Exception:
                        logger.debug("claude-agent-sdk unsolicited-start callback failed", exc_info=True)
            uuid = str(getattr(message, "uuid", None) or "")
            if (is_woken or is_peer) and not getattr(self, "_unsolicited_delivery_id", None):
                msg_id = str(origin.get("msg_id") or uuid)
                self._unsolicited_delivery_id = f"deliv-peer-{msg_id}" if msg_id else None
            if is_peer:
                if uuid and uuid in seen:
                    return
                if uuid:
                    seen.add(uuid)
                text = str(origin.get("body") or _message_text(getattr(message, "content", None)))
                if text:
                    items.append({
                        "kind": "peer_in",
                        "text": text,
                        "from": str(origin.get("from") or ""),
                        "name": str(origin.get("name") or ""),
                        "from_session": str(origin.get("fromSession") or ""),
                        "msg_id": str(origin.get("msg_id") or uuid),
                        "uuid": uuid,
                    })
                return

            content = getattr(message, "content", None)
            if not isinstance(content, list):
                return
            from agent.transports.claude_sdk_event_projector import (
                _flatten_tool_result_content,
            )
            if uuid and uuid in seen:
                return
            if uuid:
                seen.add(uuid)
            for block in content:
                if type(block).__name__ != "ToolResultBlock":
                    continue
                tool_use_id = str(getattr(block, "tool_use_id", "") or "")
                if not tool_use_id:
                    continue
                item = tool_items.get(tool_use_id)
                if item is None:
                    item = {
                        "kind": "tool",
                        "tool_use_id": tool_use_id,
                        "name": "tool",
                        "args": {},
                        "result": "",
                        "is_error": False,
                    }
                    items.append(item)
                    tool_items[tool_use_id] = item
                from agent.secret_hygiene import mask_tool_result_text
                item["result"] = mask_tool_result_text(_flatten_tool_result_content(
                    getattr(block, "content", None)
                ))
                item["is_error"] = bool(getattr(block, "is_error", False))
            return

        if name == "ResultMessage":
            uuid = getattr(message, "uuid", None)
            if uuid and uuid in self._unsolicited_delivered:
                _clear_unsolicited_projection(self)
                logger.debug(
                    "claude-agent-sdk: duplicate unsolicited ResultMessage "
                    "%s ignored", uuid,
                )
                return
            result_text = getattr(message, "result", None)
            safe_result_text = _safe_sdk_error_text(result_text)
            signals["api_error_status"] = getattr(message, "api_error_status", None)
            signals["result_text"] = safe_result_text
            signals = dict(signals)
            errors = getattr(message, "errors", None) or []
            api_error_shape = bool(
                getattr(message, "is_error", False)
                and (getattr(message, "subtype", "") or "") == "success"
                and not errors
                and (
                    signals["api_error_status"]
                    or signals["api_error_kind"]
                    or signals["api_retries"] is not None
                    or safe_result_text.startswith("API Error:")
                )
            )
            texts = list(self._unsolicited_text)
            unsolicited_items = list(self._unsolicited_items)
            delivery_id = getattr(self, "_unsolicited_delivery_id", None)
            _clear_unsolicited_projection(self)
            if api_error_shape:
                from agent.claude_sdk_transient import classify_sdk_api_failure

                _verdict, error_class, _wait_hint = classify_sdk_api_failure(signals)
                error_record = (
                    f"Claude API error ({error_class}): "
                    f"{safe_result_text or getattr(message, 'subtype', 'success')}"
                    + (
                        f" (HTTP {signals['api_error_status']})"
                        if signals["api_error_status"] else ""
                    )
                )
                delivered_items = [
                    item for item in unsolicited_items
                    if not (
                        item.get("kind") == "text"
                        and (
                            item.get("text") == safe_result_text
                            or str(item.get("text", "")).startswith("API Error:")
                        )
                    )
                ]
                delivered_items.append({
                    "kind": "lifecycle",
                    "event": "api_error",
                    "error": error_record,
                    "source": "claude_agent_sdk",
                    "uuid": str(getattr(message, "uuid", None) or ""),
                })
                delivered_texts = [
                    t for t in texts
                    if t != safe_result_text and not str(t).startswith("API Error:")
                ]
                if uuid:
                    _remember_recent_id(
                        self, "_unsolicited_delivered", "_unsolicited_delivered_order", uuid
                    )
                self._deliver_or_buffer_unsolicited(delivered_texts, delivered_items, delivery_id)
                return
            pending_rename = getattr(self, "_pending_rename_ack", None)
            if pending_rename and _is_rename_ack(result_text, texts, pending_rename):
                # The CLI's own "/rename" acknowledgement — never a background result.
                self._pending_rename_ack = None
                if uuid:
                    _remember_recent_id(
                        self, "_unsolicited_delivered", "_unsolicited_delivered_order", uuid
                    )
                logger.debug("claude-agent-sdk: swallowed /rename ack %r", (result_text or "")[:60])
                return
            if isinstance(result_text, str) and result_text.strip():
                # The CLI's result text repeats the turn's final assistant
                # message — never hand the same text over twice.
                buffered_text = "\n".join(texts)
                normalized_result = " ".join(result_text.split())
                already_buffered = any(
                    normalized_result == " ".join(text.split())
                    for text in [*texts, buffered_text]
                )
                if not already_buffered:
                    texts.append(result_text)
                if not already_buffered and not any(
                    item.get("kind") == "text"
                    and " ".join(str(item.get("text") or "").split()) == normalized_result
                    for item in unsolicited_items
                ):
                    unsolicited_items.append({"kind": "text", "text": result_text})
            if uuid:
                _remember_recent_id(
                    self, "_unsolicited_delivered", "_unsolicited_delivered_order", uuid
                )
            if not texts and not unsolicited_items:
                logger.warning(
                    "claude-agent-sdk: unsolicited ResultMessage carried no "
                    "content (total=%d) — nothing to deliver",
                    self._unsolicited_results,
                )
                return
            logger.info(
                "claude-agent-sdk: delivering unsolicited result burst "
                "(background task finished, total=%d, %d message(s), "
                "%d chars, %d item(s))",
                self._unsolicited_results, len(texts),
                sum(len(t) for t in texts), len(unsolicited_items),
            )
            self._deliver_or_buffer_unsolicited(texts, unsolicited_items, delivery_id)
        elif name == "AssistantMessage" and not is_synthetic_error:
            # Buffer top-level text, one entry per message so the burst
            # delivers in message granularity; subagent streams
            # (parent_tool_use_id set) are noise — the same gate
            # _forward_stream_delta uses.
            message_uuid = str(getattr(message, "uuid", None) or "")
            if message_uuid and message_uuid in seen:
                return
            if message_uuid:
                seen.add(message_uuid)
            parts = []
            for block in getattr(message, "content", None) or []:
                if type(block).__name__ == "TextBlock":
                    text = getattr(block, "text", "") or ""
                    if text and not is_synthetic_error:
                        parts.append(text)
                        items.append({"kind": "text", "text": text})
                    continue
                if type(block).__name__ != "ToolUseBlock":
                    continue
                tool_use_id = str(getattr(block, "id", "") or "")
                if not tool_use_id:
                    continue
                name = str(getattr(block, "name", "") or "unknown")
                args = _tool_args(block)
                item = {
                    "kind": "tool",
                    "tool_use_id": tool_use_id,
                    "name": name,
                    "args": args,
                    "result": "",
                    "is_error": False,
                }
                items.append(item)
                tool_items[tool_use_id] = item
                if name == "SendMessage":
                    peer_text = args.get("message") or args.get("content") or ""
                    if not isinstance(peer_text, str):
                        peer_text = str(peer_text)
                    peer = args.get("to") or args.get("recipient") or ""
                    items.append({
                        "kind": "peer_out",
                        "text": peer_text,
                        "to": str(peer),
                        "tool_use_id": tool_use_id,
                    })
            message_text = "\n".join(parts)
            if message_text:
                self._unsolicited_text.append(message_text)
            logger.info(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )
        else:
            logger.debug(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )

    def _start_reader(self) -> None:
        """Spawn the reader on the session loop. Idempotent."""
        if self._reader_task is not None:
            return

        async def _spawn() -> Any:
            self._turn_claims = asyncio.Queue()
            # Armed only for a D62 L2 spawn; the first turn claim disarms it.
            self._resume_backlog = [] if getattr(self, "_resume_interrupted_turn", False) else None
            return asyncio.ensure_future(self._reader_loop())

        self._reader_task = self._run_coro(_spawn(), timeout=10.0)

    def _stop_reader(self) -> None:
        task = self._reader_task
        self._reader_task = None
        if task is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(task.cancel)
        except Exception:  # pragma: no cover - loop already gone
            pass
