"""Unit tests for cntrl_sync.core — real kanban_db on a temp file, fake cntrl."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cntrl-plugins"))

from cntrl_sync import core  # noqa: E402


class FakeCntrl:
    def __init__(self, tasks):
        self.tasks = {t["id"]: dict(t) for t in tasks}
        self.moves: list[tuple] = []
        self.comments: list[tuple] = []

    def my_tasks(self, limit=200):
        return list(self.tasks.values())

    def get_task(self, task_id):
        return dict(self.tasks[task_id])

    def move(self, task_id, status, position):
        self.moves.append((task_id, status, position))
        self.tasks[task_id]["status"] = status

    def comment(self, task_id, text, field="content"):
        self.comments.append((task_id, text))


def _task(i, **over):
    base = {
        "id": f"00000000-0000-0000-0000-00000000000{i}",
        "title": f"Task {i}",
        "description": f"Do thing {i}",
        "status": "todo",
        "priority": "high",
        "labels": ["hermes"],
        "category": "engineering",
        "assignedTo": "user-1",
        "position": 3,
    }
    base.update(over)
    return base


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    from hermes_cli.kanban_db_connect import connect

    c = connect(db_path=tmp_path / "kanban.db")
    yield c
    c.close()


@pytest.fixture
def state(tmp_path):
    return core.FileState(str(tmp_path / "state.json"))


CFG = dict(core.DEFAULTS, profile_map={"user-1": "default"})


def test_qualifies_needs_label_and_not_bizops():
    assert core.qualifies(_task(1), "hermes")
    assert not core.qualifies(_task(1, labels=["ops"]), "hermes")
    assert core.qualifies(_task(1, labels=["Hermes"]), "hermes")
    assert not core.qualifies(_task(1, category="bizops"), "hermes")
    assert not core.qualifies({"title": "no id", "labels": ["hermes"]}, "hermes")


def test_key_round_trip():
    assert core.cntrl_id_from_key(core.hermes_key("abc")) == "abc"
    assert core.cntrl_id_from_key("other:abc") is None
    assert core.cntrl_id_from_key(None) is None


def test_body_carries_the_fields_hermes_has_no_column_for():
    body = core.body_for(_task(1, dueDate="2026-09-30"), "http://c")
    assert body.startswith("Do thing 1")
    assert '"due_date": "2026-09-30"' in body
    assert "http://c/tasks/00000000-0000-0000-0000-000000000001" in body


def test_sync_down_creates_once_and_lands_in_triage(conn, state):
    client = FakeCntrl([_task(1), _task(2, labels=["ops"])])
    r1 = core.sync_down(client, conn, cfg=CFG, state=state)
    assert (r1["qualified"], r1["created"], r1["updated"], r1["closed"]) == (1, 1, 0, 0)
    row = conn.execute("SELECT * FROM tasks WHERE idempotency_key = ?", (core.hermes_key(_task(1)["id"]),)).fetchone()
    assert row is not None
    assert row["status"] == "triage"
    assert row["priority"] == 2
    assert row["assignee"] == "default"
    assert row["created_by"] == "cntrl_sync"
    # second pull: idempotent
    r2 = core.sync_down(client, conn, cfg=CFG, state=state)
    assert (r2["created"], r2["updated"], r2["closed"]) == (0, 0, 0)
    assert conn.execute("SELECT count(*) AS n FROM tasks").fetchone()["n"] == 1


def test_sync_down_lands_ready_when_configured(conn, state):
    client = FakeCntrl([_task(1)])
    core.sync_down(client, conn, cfg=dict(CFG, landing="ready"), state=state)
    assert conn.execute("SELECT status FROM tasks").fetchone()["status"] == "ready"


def test_sync_down_refreshes_changed_fields(conn, state):
    client = FakeCntrl([_task(1)])
    core.sync_down(client, conn, cfg=CFG, state=state)
    client.tasks[_task(1)["id"]].update(title="Renamed", priority="low")
    r = core.sync_down(client, conn, cfg=CFG, state=state)
    assert r["updated"] == 1
    row = conn.execute("SELECT title, priority FROM tasks").fetchone()
    assert (row["title"], row["priority"]) == ("Renamed", 0)


def test_sync_down_closes_when_cntrl_closes(conn, state):
    client = FakeCntrl([_task(1), _task(2)])
    core.sync_down(client, conn, cfg=CFG, state=state)
    client.tasks[_task(1)["id"]]["status"] = "cancelled"
    client.tasks[_task(2)["id"]]["status"] = "done"
    r = core.sync_down(client, conn, cfg=CFG, state=state)
    assert r["closed"] == 2
    statuses = {row["idempotency_key"]: row["status"] for row in conn.execute("SELECT idempotency_key, status FROM tasks")}
    assert statuses[core.hermes_key(_task(1)["id"])] == "archived"
    # never started in Hermes -> complete_task refuses -> archived, not done
    assert statuses[core.hermes_key(_task(2)["id"])] == "archived"
    assert client.moves == []  # closing from cntrl never echoes back up


def test_sync_down_never_imports_an_already_closed_task(conn, state):
    client = FakeCntrl([_task(1, status="done"), _task(2, status="cancelled")])
    r = core.sync_down(client, conn, cfg=CFG, state=state)
    assert r["qualified"] == 2 and r["created"] == 0
    assert conn.execute("SELECT count(*) AS n FROM tasks").fetchone()["n"] == 0


def test_flow_up_moves_and_comments(conn, state):
    from hermes_cli import kanban_db as kdb

    client = FakeCntrl([_task(1)])
    core.sync_down(client, conn, cfg=dict(CFG, landing="ready"), state=state)
    tid = conn.execute("SELECT id FROM tasks").fetchone()["id"]
    assert kdb.complete_task(conn, tid, result="all green", fire_lifecycle_hook=False)
    out = core.flow_up(client, conn, tid)
    assert out == {"cntrl_id": _task(1)["id"], "status": "done"}
    assert client.moves == [(_task(1)["id"], "done", 3)]
    assert client.comments and "all green" in client.comments[0][1]
    # same status again: no second move
    again = core.flow_up(client, conn, tid)
    assert again["skipped"] == "already there"
    assert len(client.moves) == 1


def test_flow_up_ignores_non_cntrl_tasks(conn, state):
    from hermes_cli import kanban_db as kdb

    client = FakeCntrl([])
    created = kdb.create_task(conn, title="local only", triage=True)
    tid = str(getattr(created, "id", created))
    assert core.flow_up(client, conn, tid) is None
    assert client.moves == []


def test_load_settings_reads_canonical_plugin_entry():
    cfg = core.load_settings({"plugins": {"entries": {"cntrl_sync": {"settings": {"label": "agent", "poll_seconds": 5}}}}})
    assert cfg["label"] == "agent" and cfg["poll_seconds"] == 5
    assert cfg["base_url"] == core.DEFAULTS["base_url"]


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("CNTRL_API_KEY", " tb_abc ")
    assert core.api_key_from_env(core.DEFAULTS) == "tb_abc"
    monkeypatch.delenv("CNTRL_API_KEY")
    assert core.api_key_from_env(core.DEFAULTS) == ""
