"""Auto-continue: resume a turn killed by a process/machine death, plus queued-prompt drain and
busy-submit handling. Bodies are rebound onto server.py's globals at install time
(method_ctx.bind_module), so they reference server.py globals bare."""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module

# A concluded turn (success, handled error, interrupt) clears its durable marker (turn_marker.py) in _run_prompt_submit's
# finally; only a process death leaves it behind, so a marker at session.resume proves the turn never finished AND the
# client never saw a terminal frame. Fresh: re-submit automatically (as the messaging gateway does). Stale: clear it
# and let the partial transcript speak.
# A fresh interruption continues the existing session without resending the prompt. Prompt replay is available only
# through the explicit ``desktop.auto_continue.replay_interrupted_prompt`` opt-in; stale markers are cleared.
_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT = 15


def _auto_continue_config() -> tuple[bool, float, int]:
    """(enabled, freshness window in seconds, max attempts) from ``desktop.auto_continue`` in config.yaml."""
    desktop = _load_cfg().get("desktop")
    cfg = desktop.get("auto_continue") if isinstance(desktop, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    try:
        minutes = float(cfg.get("freshness_minutes", _AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT))
    except (TypeError, ValueError):
        minutes = float(_AUTO_CONTINUE_FRESHNESS_MINUTES_DEFAULT)
    return (is_truthy_value(cfg.get("enabled"), default=True), max(0.0, minutes) * 60.0,
            _coerce_int_config_value(cfg.get("max_attempts"), 2, min_value=0))


def _session_home(session: dict) -> Path:
    """The HERMES_HOME the session's durable state lives in (profile-aware)."""
    return Path(session.get("profile_home") or _hermes_home)


def _retire_turn_marker(session: dict, *keys: str) -> None:
    """Drop the crash marker right before the terminal frame (not at turn-thread end: post-turn work outlives the
    client's answer, and quitting in that window would leave a marker that re-runs a finished turn). Extra ``keys``
    cover a session_key that compression rotated mid-turn."""
    home = _session_home(session)
    for key in dict.fromkeys((*keys, str(session.get("session_key") or ""))):
        if key:
            clear_turn_marker(home, key)


def _auto_continue_note(prompt: str) -> str:
    # Explicit opt-in for installations that want cold resume to replay the interrupted prompt.
    return (f"{_AUTO_CONTINUE_NOTE_PREFIX} — the app or its backend process stopped before the turn could finish. "
            "Some of the work may already be complete; check the current state before redoing anything, then "
            f"finish the task. The interrupted request was:]\n\n{prompt}")


def _auto_continue_replay_enabled() -> bool:
    desktop = _load_cfg().get("desktop")
    cfg = desktop.get("auto_continue") if isinstance(desktop, dict) else None
    cfg = cfg if isinstance(cfg, dict) else {}
    return is_truthy_value(cfg.get("replay_interrupted_prompt"), default=False)


def _continue_note(reason: str) -> str:
    # D62: a continuation, never a replay: the interrupted prompt is NOT embedded (the Claude session still holds
    # it). Same opening as the recovery notes, so history renders it as an auto_continue row.
    return (f"{_AUTO_CONTINUE_NOTE_PREFIX} ({reason}). Continue where you left off. Some steps may already be "
            "complete: check their results before redoing anything, and do not repeat actions that already took "
            "effect.]")


def _continue_session_turn(sid: str, session: dict, *, reason: str, on_admitted=None) -> dict | None:
    """D62: continue a session's interrupted work in its SAME Claude session (error-card Retry, usage-limit resume).

    Never re-sends the old prompt: the SDK runtime first lets the CLI re-run its own interrupted turn
    (CLAUDE_CODE_RESUME_INTERRUPTED_TURN), and only when the CLI declines is the continuation note the turn's input.
    Returns a descriptor when a continuation turn was started, else None (busy, finalized, hosted room)."""
    if session.get("source") == "bot_room":
        return None
    with session["history_lock"]:
        if (session.get("running") or session.get("_finalized") or _ac_owner_stop_hold(session)
                or _ac_sdk_woken(session)
                or _ac_owner_pending(session)):
            return None
        session["running"] = True
        session["last_active"] = time.time()

    def release() -> None:
        with session["history_lock"]:
            session["running"] = False

    def kickoff() -> None:
        with _session_profile_runtime_scope(session):
            rid = f"__continue__{int(time.time() * 1000)}"
            try:
                _start_agent_build(sid, session)
                err = _wait_agent(session, rid, timeout=120.0)
            except Exception:
                logger.warning("continue: agent build failed for %s", sid, exc_info=True)
                err = {"error": {"message": "agent build failed"}}
            if err or _ensure_active_session_slot(sid, session) is not None:
                release()
                return
            if on_admitted is not None:
                try:
                    on_admitted()
                except Exception:
                    logger.warning("continue admission callback failed for %s", sid, exc_info=True)
            agent = session.get("agent")
            if agent is not None:
                with contextlib.suppress(Exception):
                    agent._claude_sdk_continue_requested = True
            try:
                _emit("status.update", sid, {"kind": "process", "text": "Interrupted, continuing…"})
                _emit("message.start", sid)
                _run_prompt_submit(rid, sid, session, _continue_note(reason), display_kind="auto_continue")
            except Exception as exc:
                _notif_log_failure("continue dispatch failed", exc)
                _notif_release_turn(session)

    if _start_session_work(kickoff, name=f"continue-{sid}") is None:
        release()
        return None
    return {"status": "continuing", "reason": reason}


def _usage_park_dispatch(record: dict) -> bool:
    """Scheduler callback: continue the parked session if it is live here (else it stays parked)."""
    key = str(record.get("session_key") or "")
    home = Path(record.get("profile_home") or _hermes_home).resolve()
    for sid, session in list(_sessions.items()):
        if (isinstance(session, dict) and str(session.get("session_key") or "") == key
                and _session_home(session).resolve() == home):
            from agent import claude_sdk_usage_park as usage_park

            def admitted() -> None:
                usage_park.cancel(key, home=home)

            # Admission completes inside the worker, after agent build and the ownership fence.
            _continue_session_turn(sid, session, reason="usage limit reset", on_admitted=admitted)
            return False
    return False


def _ensure_usage_park_scheduler() -> None:
    from agent import claude_sdk_usage_park as usage_park
    from tui_gateway import usage_park_scheduler

    with contextlib.suppress(Exception):
        if usage_park.configured_policy()[0]:
            usage_park_scheduler.ensure_started(_usage_park_dispatch)


def _maybe_schedule_auto_continue(sid: str, session: dict, session_key: str) -> dict | None:
    """Kick off a continuation turn for a crash-interrupted session (session.resume cold paths). Returns a descriptor
    for the resume payload when scheduled, else None. The turn runs on a background thread after the deferred agent
    build via _run_prompt_submit, so the client that just resumed streams it."""
    # A parked usage-limit session resumes from the scheduler (state.db survives restarts); resume paths start it.
    _ensure_usage_park_scheduler()
    # Hosted room turns are recovered by their durable task/lease state machine; generic auto-continue would bypass
    # its execution generation and duplicate work.
    if session.get("source") == "bot_room":
        return None
    home = _session_home(session)
    if (marker := read_turn_marker(home, session_key)) is None:
        return None
    if not marker.get("auto_continue", True):
        return None  # The mailbox owns recovery and receipt identity for imported turns.
    enabled, freshness_secs, max_attempts = _auto_continue_config()
    age = time.time() - marker["started_at"]
    if not enabled or age > freshness_secs or marker["attempts"] >= max_attempts:
        clear_turn_marker(home, session_key)  # stale/disabled/crash-looping: a manual message continues
        return None
    if (session.get("_auto_continue_scheduled") or _ac_owner_stop_hold(session)
            or _ac_owner_pending(session) or _ac_sdk_woken(session)):
        return None
    session["_auto_continue_scheduled"] = True
    replay_prompt = _auto_continue_replay_enabled()
    attempt = marker["attempts"] + 1
    text = (_auto_continue_note(marker["prompt"]) if replay_prompt
            else _continue_note("previous turn was interrupted"))

    def kickoff() -> None:
        with _session_profile_runtime_scope(session):
            run_kickoff()

    def run_kickoff() -> None:
        rid = f"__auto_continue__{int(time.time() * 1000)}"
        try:
            _start_agent_build(sid, session)
            err = _wait_agent(session, rid, timeout=120.0)
        except Exception:
            logger.warning("auto-continue agent build failed for %s", sid, exc_info=True)
            err = {"error": {"message": "agent build failed"}}
        if err:  # leave the marker: the next resume retries (bounded by attempts)
            session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            if (session.get("running") or session.get("_turn_cancel_requested") or session.get("_finalized")
                    or _ac_owner_stop_hold(session) or _ac_owner_pending(session) or _ac_sdk_woken(session)):
                session["_auto_continue_scheduled"] = False  # a real user prompt beat us; it clears the marker
                return
            session["running"] = True
            session["last_active"] = time.time()
        # Ownership admission BEFORE message.start: a sibling backend sharing this HERMES_HOME may have written the
        # marker and still be mid-turn. Leave the marker so a later resume retries.
        # Running the continuation anyway would be the double-writer this fence exists to prevent. See
        # #94778.
        if _ensure_active_session_slot(sid, session) is not None:
            logger.info("auto-continue for %s refused: session has another live owner", session_key)
            with session["history_lock"]:
                session["running"] = False
                session["_auto_continue_scheduled"] = False
            return
        with session["history_lock"]:
            # Marker inputs read back by _run_prompt_submit: attempt count (crash breaker) and the ORIGINAL prompt (no
            # nested notes). Set here, not at schedule time, so a bail above leaves nothing for a racing user turn.
            session["_auto_continue_attempt"], session["_auto_continue_prompt"] = attempt, marker["prompt"]
        agent = session.get("agent")
        if agent is not None and not replay_prompt:
            with contextlib.suppress(Exception):
                agent._claude_sdk_continue_requested = True
        try:
            from gateway.warning_notifications import render_notification
            diagnostic = marker.get("notification_category") == "diagnostic"

            def announce():
                _emit("status.update", sid, {"kind": "process", "text": "Resuming interrupted turn…"})
                _emit("message.start", sid)
            render_notification(announce, platform="tui", diagnostic=diagnostic)
            _run_prompt_submit(rid, sid, session, text, display_kind="auto_continue",
                **({"display_metadata": {"notification_category": "diagnostic"}} if diagnostic else {}))
        except Exception as exc:
            _notif_log_failure("auto-continue dispatch failed", exc)
            _notif_release_turn(session)  # rebound from session_notifications
    if _start_session_work(kickoff, name=f"auto-continue-{sid}") is None:
        session["_auto_continue_scheduled"] = False
        return None
    logger.info("auto-continue scheduled for session %s (attempt %d, interrupted %.0fs ago)", session_key, attempt, age)
    return {"attempt": attempt, "interrupted_at": marker["started_at"]}


def _ac_inflight_original(session: dict) -> str:
    turn = session.get("inflight_turn")
    return str(turn.get("user") or "").strip() if isinstance(turn, dict) else ""


def _ac_queued_is_distinct(entry: dict) -> bool:
    """An envelope that must keep its own slot: attachments, a relayed author, or a display kind (a peer card, an
    owner-forward chip, a hidden widget send). Merging it would move its images, sender or kind onto other text."""
    return bool(entry.get("image_paths") or entry.get("turn_author") or entry.get("display_kind"))


def _ac_is_owner_entry(entry: dict) -> bool:
    return entry.get("display_kind") in (None, "owner_forward") and not str(entry.get("rid") or "").startswith(
        "peer-mailbox:")


def _ac_owner_pending(session: dict) -> bool:
    """Fresh owner input holds auto-started work back; bounded and self-releasing (tui_gateway/owner_hold.py)."""
    from tui_gateway.owner_hold import owner_pending

    return owner_pending(session)


def _ac_owner_stop_hold(session: dict) -> bool:
    """The owner's Stop hold on the auto-started chain; released after the bound (tui_gateway/owner_hold.py)."""
    from tui_gateway.owner_hold import stop_hold_active

    return stop_hold_active(session)


def _ac_sdk_woken(session: dict) -> bool:
    from tui_gateway.session_mailbox import _live_claude_sdk

    sdk_session = _live_claude_sdk(session)
    check = getattr(sdk_session, "woken_turn_active", None)
    return bool(check()) if callable(check) else False


def _enqueue_prompt(session: dict, text: Any, transport: Any, image_paths: list[str] | None = None,
                    turn_author: dict | None = None, display_kind: str | None = None,
                    display_metadata: dict | None = None) -> None:
    """Queue a message for the next turn. Text-only arrivals share a slot and merge losslessly (like the
    consecutive-user merge in ``repair_message_sequence``); image-bearing, authored and typed ones stay separate
    envelopes so attachment chronology, the sender and the display kind survive. ``transport`` is pinned so the
    drained turn streams to its sender. ``display_metadata`` rides only with a ``display_kind``: without one it is a
    plain submit's ``title_preview``, which feeds the first-turn titler only (a busy session is already past it)."""
    image_paths = list(image_paths or [])
    # Scrub live-turn self-duplicates first so the text merge below can't glue "{original}\n\n{later}" and re-fire the
    # original after a correction settles.
    # See #84417.
    _drop_queued_duplicates_of_inflight_user(session)
    typed = {"display_kind": display_kind} if display_kind else {}
    if typed and isinstance(display_metadata, dict) and display_metadata:
        typed["display_metadata"] = dict(display_metadata)
    queued = {"text": text, "transport": transport, **({"image_paths": image_paths} if image_paths else {}),
              **({"turn_author": turn_author} if turn_author else {}), **typed}
    plain_text = isinstance(text, str) and not _ac_queued_is_distinct(queued)
    # A plain self-copy of the live prompt would restart it on drain; an authored or typed copy is another act.
    if plain_text and text.strip() == _ac_inflight_original(session) != "":
        return
    existing = session.get("queued_prompt")
    if (existing and plain_text and isinstance(existing.get("text"), str)
            and not _ac_queued_is_distinct(existing) and not session.get("queued_prompts")):
        prev = existing["text"]
        existing["text"] = f"{prev}\n\n{text}" if prev and text else (prev or text)
    elif existing:
        session.setdefault("queued_prompts", []).append(queued)
    else:
        session["queued_prompt"] = queued
    from tui_gateway.owner_hold import note_owner_queued

    # A queued owner message keeps peer/background work behind it until its next turn boundary.
    if any(_ac_is_owner_entry(entry) for entry in _owner_hold_entries(session)):
        note_owner_queued(session)


def _owner_hold_entries(session: dict) -> list:
    head = session.get("queued_prompt")
    return ([head] if head else []) + list(session.get("queued_prompts") or [])


def _sanitize_queued_entry_vs_inflight_user(entry: Any, original: str) -> dict | None:
    """Drop (``None``) a text-only self-duplicate of the live user text, or rewrite a merged slot
    ``"{original}\\n\\n{later}"`` to ``later`` so the correction survives without re-firing the original. Image-bearing,
    authored and typed envelopes are left alone: chronology, the sender's own words and the display kind are kept.

    Returns ``None`` to drop the envelope, or a (possibly rewritten) dict to keep. A merged slot
    ``"{original}\\n\\n{later}"`` (from ``_enqueue_prompt``'s consecutive text merge) is rewritten to just
    ``later`` so a later correction is not lost and the original is not re-fired (#84417).
    """
    if not isinstance(entry, dict):
        return None
    text = entry.get("text")
    if not original or _ac_queued_is_distinct(entry) or not isinstance(text, str):
        return entry
    # A lossless text-merge may have glued the live original onto a later follow-up: keep the remainder.
    rest = next((text[len(original + sep):] for sep in ("\n\n", "\n") if text.startswith(original + sep)), text).strip()
    return None if not rest or rest == original else (entry if rest == text.strip() else {**entry, "text": rest})


def _drop_queued_duplicates_of_inflight_user(session: dict) -> None:
    """Remove server-queue copies of the live turn's original user text: a mid-turn ``prompt.submit`` of the same text
    queued while redirect was unavailable must not drain and restart the original.

    A mid-turn ``prompt.submit`` of the same text can land in ``queued_prompt`` when redirect is not yet
    available (model not active, build window, tool boundary). If the user then corrects the turn with a
    different prompt via redirect, that stale self-duplicate must not ``_drain_queued_prompt`` after the
    redirected turn completes — otherwise the original prompt restarts as a fresh agent turn (#84417).
    """
    if not (original := _ac_inflight_original(session)):
        return
    head = session.get("queued_prompt")
    cleaned = (_sanitize_queued_entry_vs_inflight_user(e, original)
               for e in ([head] if head else []) + list(session.get("queued_prompts") or []))
    _ac_set_queue(session, [c for c in cleaned if c is not None])


def _ac_set_queue(session: dict, entries: list) -> None:
    """Write ``entries`` back as queued_prompt (head) + queued_prompts (rest)."""
    session["queued_prompt"] = entries[0] if entries else None
    if len(entries) > 1:
        session["queued_prompts"] = entries[1:]
    else:
        session.pop("queued_prompts", None)


def _interrupt_busy_session(sid: str, session: dict, agent: Any) -> None:
    """Interrupt a busy turn on a worker thread, never under ``history_lock`` (some providers can't apply ``interrupt()``
    until a blocking call returns; inline it stalled ``session.resume``). At most one interrupt worker per session so
    repeated steering can't leak threads."""
    use_agent = agent is not None and hasattr(agent, "interrupt")
    if not use_agent and not _session_uses_compute_host(session):
        return
    with session["history_lock"]:
        if session.get("_busy_interrupt_pending"):
            return
        session["_busy_interrupt_pending"] = True

    def interrupt() -> None:
        try:
            with contextlib.suppress(Exception):
                agent.interrupt() if use_agent else _get_compute_host_supervisor().interrupt(sid)
        finally:
            with session["history_lock"]:
                session["_busy_interrupt_pending"] = False
    threading.Thread(target=interrupt, daemon=True, name=f"busy-interrupt-{sid}").start()


def _ac_try_correction(rid, session: dict, agent: Any, method: str, plain_text: str, status: str) -> dict | None:
    """Apply ``agent.<method>(plain_text)`` (steer/redirect); on acceptance record the correction, scrub stale
    self-duplicates so the live turn's original text is not re-fired after settle, and return the ``status`` reply.
    None → caller falls through to the queue path."""
    try:
        if not getattr(agent, method)(plain_text):
            return None
    except Exception:
        return None
    with session["history_lock"]:
        _record_inflight_correction(session, plain_text)
        _drop_queued_duplicates_of_inflight_user(session)
        session["last_active"] = time.time()
    return _ok(rid, {"status": status})


def _handle_busy_submit(rid, sid: str, session: dict, text: Any, transport: Any, queued: bool = False,
                        turn_author: dict | None = None, display_kind: str | None = None,
                        display_metadata: dict | None = None) -> dict | None:
    """Apply ``display.busy_input_mode`` to a mid-turn prompt instead of rejecting it (rejection made clients busy-retry
    and drop sends): ``interrupt`` (default) → redirect, falling back to hard interrupt + queue; ``queue`` → queue only;
    ``steer`` → inject after the current atomic action. ``queued=True`` (client queue drain) forces queue mode: a "run
    after" message must NEVER become a live correction."""
    mode = "queue" if queued else _load_busy_input_mode()
    agent = session.get("agent")
    with session["history_lock"]:
        if not session.get("running"):
            return None  # turn ended since prompt.submit's busy check; caller retries on the idle session
        image_paths = [] if display_kind == "owner_forward" else list(session.get("attached_images", []))
        if image_paths:
            session["attached_images"] = []  # claim now so a later paste isn't consumed when the turn yields
    plain_text = _coerce_message_text(text).strip() if not image_paths and _is_text_only_busy_payload(text) else ""
    # Text-only corrections steer/redirect in place when supported; media payloads and older agents fall through to
    # the proven interrupt + queue path.
    if plain_text and agent is not None and display_kind != "owner_forward":
        supported = {
            "steer": hasattr(agent, "steer"),
            "interrupt": getattr(agent, "_supports_active_turn_redirect", False) is True and hasattr(agent, "redirect")}
        method, status = {"steer": ("steer", "steered"), "interrupt": ("redirect", "redirected")}.get(mode, (None, None))
        if (method and supported[mode]
                and (resp := _ac_try_correction(rid, session, agent, method, plain_text, status)) is not None):
            return resp
    # Queue before asking the live turn to stop. Never call a provider/compute-host method under history_lock: an
    # interrupt can wait behind the op it cancels.
    with session["history_lock"]:
        if not session.get("running"):
            if image_paths:
                session["attached_images"] = image_paths + list(session.get("attached_images", []))
            return None
        _enqueue_prompt(session, text, transport, image_paths=image_paths, turn_author=turn_author,
                        display_kind=display_kind, display_metadata=display_metadata)
        session["last_active"] = time.time()
    # Attachments need their own model invocation: queue without cancelling so the user gets both results in order.
    # ``steer`` must NEVER escalate to a hard interrupt: it would kill the live turn AND drop ``AIAgent._pending_steer``
    # (earlier accepted steers); steer fall-throughs stay FIFO-queued.
    # A burst of user messages while the agent is busy can land as a mix of accepted steers (stashed in
    # ``AIAgent._pending_steer``) and fall-through queue envelopes (payload not steerable, ``steer()``
    # rejected/raised). A hard interrupt here kills the live turn AND ``AIAgent.interrupt()`` drops the
    # pending steer buffer — silently destroying the earlier messages of the burst. See #86134.
    if mode == "interrupt" and not image_paths:
        _interrupt_busy_session(sid, session, agent)
    return _ok(rid, {"status": "queued"})


def _drain_queued_prompt(rid, sid: str, session: dict) -> bool:
    """Fire a queued next-turn prompt if one is waiting and the session is idle. True when dispatched: the caller
    skips lower-priority follow-ups this cycle (the user's message wins)."""
    with _session_turn_admission(session) as admitted:
        if not admitted or session.get("_closing") or not (queued := session.get("queued_prompt")) or session.get("running"):
            return False
        if _ac_sdk_woken(session):
            # Elapsed time is never a boundary: a prompt sent now would be folded into the unclaimed CLI
            # turn mid-loop. The owner-hold backstop nudges the CLI while preserving this entry.
            return False
        entries = [queued, *(session.get("queued_prompts") or [])]
        owner_index = next((i for i, entry in enumerate(entries) if _ac_is_owner_entry(entry)), None)
        if _ac_owner_stop_hold(session) and owner_index is None:
            return False
        index = owner_index if owner_index is not None else 0
        queued = entries.pop(index)
        queue_generation = int(session.get("_queued_prompt_generation", 0))
        _ac_set_queue(session, entries)
        session["running"] = True
        if _ac_is_owner_entry(queued):
            session["_owner_stop_hold"] = False
            from tui_gateway.owner_hold import owner_entry_delivered

            owner_entry_delivered(session)
        queued_transport = queued.get("transport")
        # The queuer's transport is pinned so the drained turn reaches the client that sent it — but
        # ATTACHED, not rebound: a mid-turn prompt from a second client used to silence the first for the
        # whole drained turn. A peer that disconnected while its prompt sat in the queue is skipped: the
        # prompt still runs, only the dead pin is dropped.
        if queued_transport is not None and not _transport_is_dead(queued_transport):
            _attach_session_transport(session, queued_transport)
    use_compute_host = _session_uses_compute_host(session)
    with session["history_lock"]:
        if int(session.get("_queued_prompt_generation", 0)) != queue_generation:
            # Generation bump cancelled the claim (Stop, compress re-anchor, …): don't dispatch, but restore the
            # envelope (claimed head first, then whatever advanced into the slot) so a legitimate follow-up isn't dropped.
            # See #84417.
            advanced = session.get("queued_prompt")
            _ac_set_queue(session, [queued, *([advanced] if advanced else []), *(session.get("queued_prompts") or [])])
            session["running"] = False
            return True
    kwargs: dict = {"queued_prompt_generation": queue_generation}
    if queued.get("image_paths"):
        kwargs["image_paths"] = queued["image_paths"]
    # A typed turn (peer card, owner-forward chip, hidden send) drains as the same kind it was submitted as (F5).
    if queued.get("display_kind"):
        kwargs["display_kind"] = queued["display_kind"]
        if isinstance(queued.get("display_metadata"), dict):
            kwargs["display_metadata"] = dict(queued["display_metadata"])
    # The compute-host frame has no author field, so only the inline runner receives it.
    author_kwargs = {"turn_author": queued["turn_author"]} if queued.get("turn_author") else {}
    dispatch_failed = False
    dispatch_error = None
    try:
        if not use_compute_host:
            if _run_prompt_submit(rid, sid, session, queued["text"], **kwargs, **author_kwargs) is False:
                _notif_release_turn(session)
                dispatch_failed = True
                dispatch_error = "Queued prompt could not start; it remains queued."
        elif (resp := _submit_prompt_to_compute_host(rid, sid, session, queued["text"], **kwargs)).get("error"):
            with session["history_lock"]:
                session["running"] = False
                _clear_inflight_turn(session)
            dispatch_failed = True
            dispatch_error = str((resp.get("error") or {}).get("message") or "queued prompt failed")
    except Exception as exc:
        _notif_log_failure("queued prompt dispatch failed", exc)
        _notif_release_turn(session)
        dispatch_failed = True
        dispatch_error = "Queued prompt could not start; it remains queued."
    if dispatch_failed:
        with session["history_lock"]:
            # Keep the failed owner at the front; another boundary may retry it.
            _ac_set_queue(session, [queued, *([session["queued_prompt"]] if session.get("queued_prompt") else []),
                                    *(session.get("queued_prompts") or [])])
            if _ac_is_owner_entry(queued):
                from tui_gateway.owner_hold import note_owner_queued
                note_owner_queued(session)
        _emit("error", sid, {"message": dispatch_error})
    return True


def _inflight_snapshot(session: dict) -> dict | None:
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        return None
    user, assistant = str(turn.get("user") or "").strip(), str(turn.get("assistant") or "")
    streaming, error = bool(turn.get("streaming")), str(turn.get("error") or "").strip()
    if not (user or assistant or streaming or error):
        return None
    snapshot = {"assistant": assistant, "streaming": streaming, "user": user}
    if isinstance(display_kind := turn.get("display_kind"), str) and display_kind:
        snapshot["display_kind"] = display_kind
    if isinstance(display_metadata := turn.get("display_metadata"), dict):
        snapshot["display_metadata"] = dict(display_metadata)
    raw_offsets = turn.get("correction_offsets") or []
    correction_pairs = [(str(c), raw_offsets[i] if i < len(raw_offsets) else None)
                        for i, c in enumerate(turn.get("corrections") or []) if str(c).strip()]
    if correction_pairs:
        # Mid-turn redirects alongside (not over) the original prompt so resume can rebuild every user bubble; offsets
        # only when every correction has one so clients can trust the pairing.
        snapshot["corrections"] = [c for c, _ in correction_pairs]
        if all(isinstance(offset, int) and offset >= 0 for _, offset in correction_pairs):
            snapshot["correction_offsets"] = [int(offset) for _, offset in correction_pairs]  # type: ignore[arg-type]
    if error:
        # Retained failed turn (_fail_inflight_turn): a resuming client must rebuild the failed bubble, not render the
        # partial text as a healthy reply.
        snapshot.update(error=error, status=str(turn.get("status") or "error"), recoverable=bool(turn.get("recoverable")))
        if isinstance(surface := turn.get("error_surface"), dict) and surface:
            snapshot["error_surface"] = surface
    return snapshot


def _emit_terminal_turn_error(
    sid: str, session: dict, error: Any, error_surface: Optional[dict] = None, *, retire_marker: bool = True) -> None:
    """Close a failed turn with the same ``status: "error"`` ``message.complete`` frame as the returned-error path,
    retaining the turn so a client that missed the frame recovers it from ``session.resume``'s ``inflight``.
    ``error_surface`` ({layer, code, retryable}) is classified from an exception if absent."""
    agent = session.get("agent")
    if error_surface is None and isinstance(error, BaseException):
        with contextlib.suppress(Exception):
            from agent.error_surface import build_error_surface_from_exception
            error_surface = build_error_surface_from_exception(
                error, provider=str(getattr(agent, "provider", "") or ""), model=str(getattr(agent, "model", "") or ""),
                api_key=getattr(agent, "api_key", None))
    with session["history_lock"]:
        _fail_inflight_turn(session, error, error_surface=error_surface)
        turn = session.get("inflight_turn") or {}
        message, partial = str(turn.get("error") or "turn failed"), str(turn.get("assistant") or "")
        cols = int(session.get("cols", 80))
    text = partial or turn_error_text(message, error_surface)
    rendered = ""
    with contextlib.suppress(Exception):
        rendered = render_message(text, cols)
    payload = {"text": text, "usage": _get_usage(agent) if agent is not None else {}, "status": "error",
               "error": message, "recoverable": True, **({"error_surface": error_surface} if error_surface else {}),
               **({"partial": True} if partial else {}), **({"rendered": rendered} if rendered else {})}
    if retire_marker:
        _retire_turn_marker(session)
    _emit("message.complete", sid, payload)


def _restore_agent_history_after_turn_error(session: dict, agent) -> bool:
    """Keep a failed turn's working transcript: ``AIAgent`` persists its messages independently, so after a raise the
    next prompt must see them, not the pre-turn snapshot."""
    agent_messages = getattr(agent, "_session_messages", None)
    if not isinstance(agent_messages, list):
        return False
    with session["history_lock"]:
        session["history"] = list(agent_messages)
        session["history_version"] = int(session.get("history_version", 0)) + 1
    return True


def _queued_prompt_snapshot(session: dict) -> dict | None:
    """The accepted next-turn prompt without its transport handle, for the live-session projection (Desktop may
    reconnect while it is still queued)."""
    queued = session.get("queued_prompt")
    user = _inflight_text(queued.get("text")) if isinstance(queued, dict) else ""
    if not user:
        return None
    snapshot = {"user": user}
    if queued.get("display_kind"):
        snapshot["display_kind"] = queued["display_kind"]
        if isinstance(queued.get("display_metadata"), dict):
            snapshot["display_metadata"] = dict(queued["display_metadata"])
    return snapshot


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
