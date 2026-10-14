"""cntrl_sync — Hermes plugin entry point.

Registers the three kanban lifecycle hooks that flow status up to cntrl and
starts the pull loop that brings labelled cntrl tasks down. All logic lives in
``core.py`` so it can run without a plugin context (standalone runner, tests).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from . import core

log = logging.getLogger("plugins.cntrl_sync")

_cfg: dict = dict(core.DEFAULTS)
_state: Any = None
_stop = threading.Event()


def _client() -> Optional[core.CntrlClient]:
    key = core.api_key_from_env(_cfg)
    if not key:
        return None
    return core.CntrlClient(str(_cfg["base_url"]), key)


def _conn(board: Optional[str]):
    from hermes_cli.kanban_db_connect import connect

    return connect(board=board or (_cfg.get("board") or None))


def _push(task_id: Optional[str], board: Optional[str], reason: Optional[str] = None) -> None:
    if not task_id:
        return
    client = _client()
    if client is None:
        return
    try:
        conn = _conn(board)
        try:
            out = core.flow_up(client, conn, task_id, reason=reason)
        finally:
            conn.close()
        if out:
            log.info("cntrl_sync: %s -> cntrl %s (%s)", task_id, out.get("status"), out.get("cntrl_id"))
    except Exception as exc:  # observers must never break a board transition
        log.warning("cntrl_sync: flow-up for %s failed: %s", task_id, exc)


def _on_completed(task_id: str = None, board: str = None, **_: Any) -> None:  # type: ignore[assignment]
    _push(task_id, board)


def _on_blocked(task_id: str = None, board: str = None, reason: str = None, **_: Any) -> None:  # type: ignore[assignment]
    _push(task_id, board, reason=reason)


def _on_updated(task_id: str = None, board: str = None, changed_fields: Any = None, **_: Any) -> None:  # type: ignore[assignment]
    # Assignment/priority/title edits carry no status change; only a status
    # field in changed_fields is worth a round trip.
    fields = {str(f) for f in (changed_fields or [])}
    if "status" in fields:
        _push(task_id, board)


def _poll_once(failures: int) -> tuple[int, float]:
    """Run one pull; return the new failure count and the wait before the next.

    An unreachable cntrl logs once when the outage starts and once when it
    ends; the pulls in between back off instead of warning every interval."""
    base = max(5, int(_cfg.get("poll_seconds") or 0))
    try:
        report = core.run_once(_state, cfg=_cfg)
    except Exception as exc:
        if failures == 0:
            log.warning("cntrl_sync: pull failed: %s (backing off until cntrl answers)", exc)
        else:
            log.debug("cntrl_sync: pull failed again (%d): %s", failures + 1, exc)
        return failures + 1, core.poll_backoff_seconds(base, failures + 1)
    if failures:
        log.info("cntrl_sync: pull recovered after %d failed attempts", failures)
    if report["created"] or report["updated"] or report["closed"] or report["errors"]:
        log.info("cntrl_sync: pull %s", {k: v for k, v in report.items() if k != "errors"} | {"errors": len(report["errors"])})
    return 0, base


def _poll_loop() -> None:
    failures, delay = 0, max(5, int(_cfg.get("poll_seconds") or 0))
    while not _stop.wait(delay):
        failures, delay = _poll_once(failures)


def register(ctx) -> None:
    global _cfg, _state
    _cfg = dict(core.DEFAULTS)
    for key in core.DEFAULTS:
        try:
            value = ctx.get_config(key, None)
        except Exception:
            value = None
        if value is not None:
            _cfg[key] = value
    _state = ctx.state

    ctx.register_hook("kanban_task_completed", _on_completed)
    ctx.register_hook("kanban_task_blocked", _on_blocked)
    ctx.register_hook("on_kanban_task_updated", _on_updated)

    if not core.api_key_from_env(_cfg):
        log.warning("cntrl_sync: $%s is not set — hooks registered, pull loop off", _cfg.get("api_key_env"))
        return
    if int(_cfg.get("poll_seconds") or 0) <= 0:
        log.info("cntrl_sync: poll_seconds=0 — pull loop off, hooks only")
        return
    t = threading.Thread(target=_poll_loop, name="cntrl-sync-pull", daemon=True)
    t.start()
    log.info("cntrl_sync: registered (pull every %ss from %s)", _cfg.get("poll_seconds"), _cfg.get("base_url"))
