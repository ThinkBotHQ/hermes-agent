"""cntrl ↔ Hermes kanban one-way sync — core logic.

No dependency on the Hermes plugin context, so the same code runs inside the
gateway (``__init__.register``), from the standalone runner, and under tests.
Design: ``docs/cntrl/kanban-one-way-sync.md``.

Down (cntrl → Hermes): a cntrl task that carries the opt-in label and is not
bizops is created on the Hermes board with ``idempotency_key = "cntrl:<uuid>"``
(the only external-key slot Hermes has), refreshed when title/description/
priority change, and closed when cntrl closes it.

Up (Hermes → cntrl): status only, plus one comment on ``done``/``blocked``.
Closing from the down direction never fires Hermes lifecycle hooks, and the
up direction skips a move when cntrl already holds the mapped status, so the
two directions cannot echo each other.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

log = logging.getLogger("plugins.cntrl_sync")

KEY_PREFIX = "cntrl:"

# cntrl status -> Hermes status (down). Hermes owns run state, so cntrl's
# in_progress lands as todo: Hermes is never told it is running.
STATUS_DOWN: dict[str, str] = {
    "todo": "todo",
    "backlog": "todo",
    "ready": "todo",
    "pending_approval": "todo",
    "in_progress": "todo",
    "active": "todo",
    "in_review": "review",
    "review": "review",
    "done": "done",
    "cancelled": "archived",
    "failed": "archived",
}

# Hermes status -> cntrl status (up). blocked has no cntrl status: it maps to
# todo and the comment carries the reason.
STATUS_UP: dict[str, str] = {
    "triage": "todo",
    "todo": "todo",
    "scheduled": "todo",
    "ready": "todo",
    "running": "in_progress",
    "review": "in_review",
    "blocked": "todo",
    "done": "done",
    "archived": "cancelled",
}

PRIORITY_DOWN: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

DEFAULTS: dict[str, Any] = {
    "base_url": "http://127.0.0.1:3101",
    "api_key_env": "CNTRL_API_KEY",
    "poll_seconds": 30,
    "label": "hermes",
    "landing": "triage",
    "board": "",
    "profile_map": {},
}


# ---------- pure helpers ----------


def hermes_key(cntrl_id: str) -> str:
    return f"{KEY_PREFIX}{cntrl_id}"


def cntrl_id_from_key(key: Optional[str]) -> Optional[str]:
    if isinstance(key, str) and key.startswith(KEY_PREFIX) and len(key) > len(KEY_PREFIX):
        return key[len(KEY_PREFIX):]
    return None


def _labels(task: dict) -> list[str]:
    raw = task.get("labels") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(x).strip().lower() for x in raw if str(x).strip()]


def qualifies(task: dict, label: str) -> bool:
    """The down-sync predicate: opted in by label, and not a bizops task."""
    if not isinstance(task, dict) or not task.get("id"):
        return False
    if label.lower() not in _labels(task):
        return False
    if str(task.get("category") or "").strip().lower() == "bizops":
        return False
    return True


def priority_for(task: dict) -> int:
    raw = task.get("priority")
    if isinstance(raw, (int, float)):
        return int(raw)
    return PRIORITY_DOWN.get(str(raw or "medium").strip().lower(), 1)


def fingerprint(task: dict) -> str:
    parts = [str(task.get("title") or ""), str(task.get("description") or ""), str(priority_for(task))]
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


def body_for(task: dict, base_url: str) -> str:
    """Description first, then a fenced block with what Hermes has no column for."""
    meta = {
        "cntrl_id": task.get("id"),
        "workspace_id": task.get("workspaceId"),
        "project_id": task.get("projectId"),
        "labels": _labels(task),
        "due_date": task.get("dueDate"),
        "priority": task.get("priority"),
        "url": f"{base_url.rstrip('/')}/tasks/{task.get('id')}",
    }
    desc = str(task.get("description") or "").rstrip()
    block = "```cntrl\n" + json.dumps(meta, ensure_ascii=False, indent=2) + "\n```"
    return f"{desc}\n\n{block}" if desc else block


def assignee_for(task: dict, profile_map: dict) -> Optional[str]:
    if not profile_map:
        return None
    for key in (task.get("assignedTo"), task.get("assigneeEmail"), task.get("assignee")):
        if key and str(key) in profile_map:
            return str(profile_map[str(key)])
        if key and str(key).lower() in profile_map:
            return str(profile_map[str(key).lower()])
    return None


# ---------- egress masking ----------


def _mask_outbound(value: Any) -> Any:
    """Mask secrets in text bound for cntrl (``[REDACTED:<kind>:<tag>]``); raises
    ``agent.secret_egress.EgressMaskError`` instead of sending unmasked content."""
    from agent.secret_egress import mask_egress_value

    return mask_egress_value(value, surface="cntrl_sync")


# ---------- cntrl HTTP client ----------


class CntrlError(RuntimeError):
    pass


class CntrlClient:
    """Minimal urllib client for the cntrl task API (Bearer tb_ key)."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        # Every payload leaves masked (secret hygiene, fail closed): no request is sent when
        # masking fails. flow_up already masks its comments; this covers any other caller.
        data = json.dumps(_mask_outbound(body)).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise CntrlError(f"{method} {path} -> HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise CntrlError(f"{method} {path} -> {exc.reason}") from None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return raw.decode("utf-8", "replace")

    @staticmethod
    def _unwrap_list(payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [t for t in payload if isinstance(t, dict)]
        if isinstance(payload, dict):
            for key in ("tasks", "items", "data", "results"):
                if isinstance(payload.get(key), list):
                    return [t for t in payload[key] if isinstance(t, dict)]
        return []

    @staticmethod
    def _unwrap_task(payload: Any) -> dict:
        if isinstance(payload, dict):
            if isinstance(payload.get("task"), dict):
                return payload["task"]
            if isinstance(payload.get("data"), dict):
                return payload["data"]
            return payload
        return {}

    def my_tasks(self, limit: int = 200) -> list[dict]:
        return self._unwrap_list(self._req("GET", f"/api/tasks/my-tasks?limit={int(limit)}"))

    def get_task(self, task_id: str) -> dict:
        return self._unwrap_task(self._req("GET", f"/api/tasks/{urllib.parse.quote(task_id)}"))

    def move(self, task_id: str, status: str, position: int) -> Any:
        return self._req("PUT", f"/api/tasks/{urllib.parse.quote(task_id)}/move", {"status": status, "position": int(position)})

    def comment(self, task_id: str, text: str, field: str = "content") -> Any:
        return self._req("POST", f"/api/tasks/{urllib.parse.quote(task_id)}/comments", {field: text})


# ---------- state ----------


class FileState:
    """PluginState-compatible ``get``/``set`` on a JSON file (standalone runs, tests)."""

    def __init__(self, path: str):
        self.path = path

    def _read(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        data = self._read()
        data[key] = value
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


# ---------- sync ----------


def _task_id(created: Any) -> str:
    return str(getattr(created, "id", created))


def sync_down(client: CntrlClient, conn: sqlite3.Connection, *, cfg: dict, state: Any) -> dict:
    """Pull qualifying cntrl tasks onto the Hermes board. Returns a report."""
    from hermes_cli import kanban_db as kdb

    label = str(cfg.get("label") or DEFAULTS["label"])
    landing = str(cfg.get("landing") or DEFAULTS["landing"]).lower()
    base_url = str(cfg.get("base_url") or DEFAULTS["base_url"])
    profile_map = cfg.get("profile_map") or {}
    seen: dict = dict(state.get("seen", {}) or {})
    report = {"seen": 0, "qualified": 0, "created": 0, "updated": 0, "closed": 0, "errors": []}

    tasks = client.my_tasks()
    report["seen"] = len(tasks)
    for task in tasks:
        if not qualifies(task, label):
            continue
        report["qualified"] += 1
        cid = str(task["id"])
        key = hermes_key(cid)
        cstatus = str(task.get("status") or "").strip().lower()
        target = STATUS_DOWN.get(cstatus)
        fp = fingerprint(task)
        try:
            row = conn.execute(
                "SELECT id, status FROM tasks WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is None:
                if target in ("done", "archived"):
                    continue  # never import a task that is already closed
                created = kdb.create_task(
                    conn,
                    title=str(task.get("title") or "(untitled)")[:500],
                    body=body_for(task, base_url),
                    assignee=assignee_for(task, profile_map),
                    created_by="cntrl_sync",
                    priority=priority_for(task),
                    idempotency_key=key,
                    triage=(landing != "ready"),
                    board=(cfg.get("board") or None),
                )
                tid = _task_id(created)
                report["created"] += 1
            else:
                tid, hstatus = str(row["id"]), str(row["status"])
                prev = seen.get(cid) or {}
                if prev.get("fp") != fp:
                    conn.execute(
                        "UPDATE tasks SET title = ?, body = ?, priority = ? WHERE id = ?",
                        (
                            str(task.get("title") or "(untitled)")[:500],
                            body_for(task, base_url),
                            priority_for(task),
                            tid,
                        ),
                    )
                    conn.commit()
                    report["updated"] += 1
                if target == "done" and hstatus not in ("done", "archived"):
                    # complete_task only accepts running/ready/blocked/review;
                    # a task Hermes never started is archived instead.
                    if not kdb.complete_task(conn, tid, result="Closed in cntrl", fire_lifecycle_hook=False):
                        kdb.archive_task(conn, tid)
                    report["closed"] += 1
                elif target == "archived" and hstatus != "archived":
                    kdb.archive_task(conn, tid)
                    report["closed"] += 1
            seen[cid] = {"hermes_id": tid, "fp": fp, "cntrl_status": cstatus, "ts": int(time.time())}
        except Exception as exc:  # one bad task must not stop the rest
            log.warning("cntrl_sync: task %s failed: %s", cid, exc)
            report["errors"].append(f"{cid}: {exc}")
    state.set("seen", seen)
    state.set("last_pull", int(time.time()))
    return report


def flow_up(client: CntrlClient, conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None) -> Optional[dict]:
    """Push one Hermes task's status to cntrl. None when the task is not a cntrl task."""
    from hermes_cli import kanban_db as kdb

    task = kdb.get_task(conn, task_id)
    if task is None:
        return None
    cid = cntrl_id_from_key(getattr(task, "idempotency_key", None))
    if cid is None:
        return None
    target = STATUS_UP.get(str(task.status))
    if target is None:
        return None
    remote = client.get_task(cid)
    if str(remote.get("status") or "").lower() == target and task.status not in ("blocked",):
        return {"cntrl_id": cid, "status": target, "skipped": "already there"}
    note: Optional[str] = None
    if task.status == "done":
        result = (getattr(task, "result", None) or "").strip()
        note = "Hermes completed this task." + (f"\n\n{result}" if result else "")
    elif task.status == "blocked":
        note = "Hermes blocked this task." + (f" Reason: {reason}" if reason else "")
    # Masked before anything is sent: a masking failure blocks the move and the comment.
    note = _mask_outbound(note) if note else note
    client.move(cid, target, int(remote.get("position") or 0))
    if note:
        client.comment(cid, note)
    return {"cntrl_id": cid, "status": target}


# ---------- config + standalone ----------


def load_settings(config: Optional[dict] = None) -> dict:
    """``plugins.entries.cntrl_sync.settings`` from config.yaml, over DEFAULTS."""
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    entry = (((config.get("plugins") or {}).get("entries") or {}).get("cntrl_sync") or {})
    settings = entry.get("settings") if isinstance(entry, dict) else None
    out = dict(DEFAULTS)
    if isinstance(settings, dict):
        out.update({k: v for k, v in settings.items() if v is not None})
    return out


def api_key_from_env(cfg: dict) -> str:
    return os.environ.get(str(cfg.get("api_key_env") or DEFAULTS["api_key_env"]), "").strip()


POLL_BACKOFF_MAX_SECONDS = 900


def poll_backoff_seconds(base: float, failures: int) -> float:
    """Wait after ``failures`` consecutive failed pulls: doubles from ``base``, capped."""
    if failures <= 0:
        return base
    return min(max(base, POLL_BACKOFF_MAX_SECONDS), base * 2 ** min(failures, 16))


def run_once(state: Any, *, cfg: Optional[dict] = None) -> dict:
    """One pull against the configured board. Used by the poll loop and the CLI runner."""
    from hermes_cli.kanban_db_connect import connect

    cfg = cfg or load_settings()
    key = api_key_from_env(cfg)
    if not key:
        raise CntrlError(f"no cntrl API key in ${cfg.get('api_key_env')}")
    client = CntrlClient(str(cfg["base_url"]), key)
    conn = connect(board=(cfg.get("board") or None))
    try:
        return sync_down(client, conn, cfg=cfg, state=state)
    finally:
        conn.close()
