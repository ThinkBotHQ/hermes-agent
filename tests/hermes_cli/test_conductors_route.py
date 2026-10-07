"""Tests for GET /api/profiles/conductors route (#49 Unit B6)."""

from __future__ import annotations

import json
import os
import pwd
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
from hermes_cli import web_server
import hermes_cli.web_server_sessions as _web_server_sessions
from hermes_cli.web_routers import conductors, profiles as profiles_mod
from hermes_state import SessionDB
from tui_gateway.conductor_roster import _clear_cache as _clear_roster_cache


@pytest.fixture
def client(monkeypatch):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous_auth_required is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous_auth_required


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    home.mkdir()
    temp_dir = (tmp_path / "temp").resolve()
    temp_dir.mkdir()

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: type("Pw", (), {"pw_dir": str(home)})())
    monkeypatch.setenv("HOME", str(home))

    state_dir = home / ".claude" / "state"
    state_dir.mkdir(parents=True)
    index_file = state_dir / "tb-marker-index.jsonl"
    monkeypatch.setenv("TB_MARKER_INDEX_PATH", str(index_file))

    db_path = home / "state.db"
    seed = SessionDB(db_path)
    seed.create_session("sess-worker-1", "desktop")
    seed.close()

    monkeypatch.setattr(_web_server_sessions, "_open_session_db_for_profile",
                        lambda profile, *, read_only: SessionDB(db_path))
    monkeypatch.setattr(profiles_mod, "_profile_targets",
                        lambda label: [("default", home)])

    import tui_gateway.server as server
    sessions: dict = {}
    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")

    _clear_roster_cache()
    conductors._clear_cache()

    yield SimpleNamespace(
        home=home,
        temp_dir=temp_dir,
        index_file=index_file,
        db_path=db_path,
        server=server,
        sessions=sessions,
    )

    _clear_roster_cache()
    conductors._clear_cache()


def _make_marker(home: Path, proj_name: str, run_id: str = "run-1", session_id: str = "sid-1", **kwargs) -> Path:
    proj_dir = home / proj_name
    state_dir = proj_dir / ".claude" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    marker_path = state_dir / "tb-build-active.json"
    # Default to "armed an hour ago": a fixed calendar date ages past the
    # abandoned threshold and turns every default marker abandoned.
    armed_at_str = kwargs.get("armed_at") or (
        datetime.now(timezone.utc) - timedelta(hours=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = {
        "schema": "tb-build/v1",
        "run_id": run_id,
        "session_id": session_id,
        "plan": f"{proj_name}/plan.md",
        "waves_total": 4,
        "waves_done": 1,
        "armed_at": armed_at_str,
        "context_path": str(proj_dir),
        **kwargs,
    }
    marker_path.write_text(json.dumps(data), encoding="utf-8")
    mtime_val = kwargs.get("mtime")
    if mtime_val is None:
        try:
            mtime_val = datetime.fromisoformat(armed_at_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            mtime_val = None
    if mtime_val is not None:
        os.utime(marker_path, (mtime_val, mtime_val))
    return marker_path


def _add_index_entry(index_file: Path, marker_path: Path, run_id: str = "run-1", session_id: str = "sid-1", context_path: str | None = None):
    entry = {
        "session_id": session_id,
        "marker_path": str(marker_path),
        "context_path": context_path or str(marker_path.parent.parent.parent),
        "state_root": str(marker_path.parent),
        "run_id": run_id,
    }
    with open(index_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def test_conductors_endpoint_requires_auth():
    """Unauthenticated GET /api/profiles/conductors returns 401."""
    unauthed_client = TestClient(web_server.app)
    resp = unauthed_client.get("/api/profiles/conductors")
    assert resp.status_code == 401


def test_returns_schema_hermes_conductors_v1(client, env):
    """Returns hermes-conductors/v1 schema with rows from temp marker index."""
    m_path = _make_marker(env.home, "my-repo", run_id="run-101", session_id="claude-sid-1")
    _add_index_entry(env.index_file, m_path, run_id="run-101", session_id="claude-sid-1")

    resp = client.get("/api/profiles/conductors")
    assert resp.status_code == 200
    data = resp.json()

    assert data["schema"] == "hermes-conductors/v1"
    assert isinstance(data["generated_at"], (int, float))
    assert data["sources"]["marker_index"] == "ok"
    assert data["sources"]["state_dirs"] >= 1
    assert data["sources"]["skipped"] == 0
    assert data["abandoned"] == 0

    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["key"]
    assert row["project"]["name"] == "my-repo"
    assert row["project"]["root_display"] == "~/my-repo"
    assert row["build"]["run_id"] == "run-101"
    assert row["orchestrator"]["claude_sid_short"] == "claude-s"


def test_stale_only_when_idle_and_over_2_hours(client, env, monkeypatch):
    """Row is stale only when owner session is idle AND newest activity > 2 h."""
    now_ts = 1790670000.0
    monkeypatch.setattr(time, "time", lambda: now_ts)
    monkeypatch.setattr(time, "monotonic", lambda: now_ts)

    # Build A: Owner is busy -> active even though marker is 10 h old
    m_a = _make_marker(
        env.home, "repo-active", run_id="run-act", session_id="claude-act",
        armed_at="2026-09-28T10:00:00Z",
        lease_expires_at=now_ts - 36000,
    )
    _add_index_entry(env.index_file, m_a, run_id="run-act", session_id="claude-act")

    cli_a = ClaudeAgentSdkSession(cwd="/tmp", hermes_session_id="sess-worker-1")
    cli_a._client = object()
    cli_a._session_id = "claude-act"
    agent_a = SimpleNamespace(session_id="sess-worker-1", api_mode="claude_agent_sdk", _claude_sdk_session=cli_a)
    env.sessions["rt-1"] = {"session_key": "sess-worker-1", "agent": agent_a, "running": True}

    # Build B: Owner gone, lease expired 30 min ago, activity 45 min ago (< 2h) -> idle
    m_b = _make_marker(
        env.home, "repo-idle", run_id="run-idle", session_id="claude-idle",
        armed_at=datetime.fromtimestamp(now_ts - 2700, timezone.utc).isoformat().replace("+00:00", "Z"),
        lease_expires_at=now_ts - 1800,
    )
    _add_index_entry(env.index_file, m_b, run_id="run-idle", session_id="claude-idle")

    # Build C: Owner gone, lease expired 3 h ago, activity 3 h ago (> 2h) -> stale
    m_c = _make_marker(
        env.home, "repo-stale", run_id="run-stale", session_id="claude-stale",
        armed_at=datetime.fromtimestamp(now_ts - 10800, timezone.utc).isoformat().replace("+00:00", "Z"),
        lease_expires_at=now_ts - 10800,
    )
    _add_index_entry(env.index_file, m_c, run_id="run-stale", session_id="claude-stale")

    resp = client.get("/api/profiles/conductors")
    assert resp.status_code == 200
    rows = {r["build"]["run_id"]: r["build"]["liveness"] for r in resp.json()["rows"]}

    assert rows["run-act"] == "active"
    assert rows["run-idle"] == "idle"
    assert rows["run-stale"] == "stale"


def test_abandoned_over_7_days(client, env, monkeypatch):
    """Rows whose owner is not live and lease ran out > 7 days go behind abandoned."""
    now_ts = 1790670000.0
    monkeypatch.setattr(time, "time", lambda: now_ts)
    monkeypatch.setattr(time, "monotonic", lambda: now_ts)

    # Build A: Owner not live, lease expired 8 days ago (> 7 days) -> abandoned
    eight_days_ago = now_ts - (8 * 86400)
    m_aban = _make_marker(
        env.home, "repo-aban", run_id="run-aban", session_id="claude-aban",
        armed_at=datetime.fromtimestamp(eight_days_ago, timezone.utc).isoformat().replace("+00:00", "Z"),
        lease_expires_at=eight_days_ago,
    )
    _add_index_entry(env.index_file, m_aban, run_id="run-aban", session_id="claude-aban")

    # Build B: Owner not live, lease expired 2 days ago (< 7 days) -> normal stale row
    two_days_ago = now_ts - (2 * 86400)
    m_stale = _make_marker(
        env.home, "repo-recent", run_id="run-recent", session_id="claude-recent",
        armed_at=datetime.fromtimestamp(two_days_ago, timezone.utc).isoformat().replace("+00:00", "Z"),
        lease_expires_at=two_days_ago,
    )
    _add_index_entry(env.index_file, m_stale, run_id="run-recent", session_id="claude-recent")

    resp = client.get("/api/profiles/conductors")
    assert resp.status_code == 200
    data = resp.json()

    assert data["abandoned"] == 1
    # §4: abandoned rows stay in the payload, flagged, behind the renderer's "Show abandoned" toggle.
    by_run = {r["build"]["run_id"]: r for r in data["rows"]}
    assert by_run["run-recent"]["abandoned"] is False
    assert by_run["run-aban"]["abandoned"] is True


def test_fresh_reuse_within_5s(client, env, monkeypatch):
    """Second call within 5 s reuses result without re-reading."""
    current_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: current_time[0])
    monkeypatch.setattr(time, "monotonic", lambda: current_time[0])

    m_path = _make_marker(env.home, "repo-reuse", run_id="run-reuse", session_id="claude-reuse")
    _add_index_entry(env.index_file, m_path, run_id="run-reuse", session_id="claude-reuse")

    read_count = [0]
    orig_read = conductors._build_conductors_payload

    def counted_read():
        read_count[0] += 1
        return orig_read()

    monkeypatch.setattr(conductors, "_build_conductors_payload", counted_read)

    # Call 1: reads
    resp1 = client.get("/api/profiles/conductors")
    assert resp1.status_code == 200
    assert read_count[0] == 1

    # Call 2 at +2 s: reuses result within 5 s
    current_time[0] += 2.0
    resp2 = client.get("/api/profiles/conductors")
    assert resp2.status_code == 200
    assert read_count[0] == 1


def test_fresh_1_honoured_only_after_5s(client, env, monkeypatch):
    """fresh=1 is ignored within 5 s; honoured only when cache is older than 5 s."""
    current_time = [2000.0]
    monkeypatch.setattr(time, "time", lambda: current_time[0])
    monkeypatch.setattr(time, "monotonic", lambda: current_time[0])

    m_path = _make_marker(env.home, "repo-fresh", run_id="run-fresh", session_id="claude-fresh")
    _add_index_entry(env.index_file, m_path, run_id="run-fresh", session_id="claude-fresh")

    read_count = [0]
    orig_read = conductors._build_conductors_payload

    def counted_read():
        read_count[0] += 1
        return orig_read()

    monkeypatch.setattr(conductors, "_build_conductors_payload", counted_read)

    # Call 1: initial read
    resp1 = client.get("/api/profiles/conductors")
    assert resp1.status_code == 200
    assert read_count[0] == 1

    # Call 2 at +2 s with fresh=1: NOT honoured (reused within 5 s)
    current_time[0] += 2.0
    resp2 = client.get("/api/profiles/conductors", params={"fresh": 1})
    assert resp2.status_code == 200
    assert read_count[0] == 1

    # Call 3 at +6 s (> 5 s) with fresh=1: HONOURED (re-reads)
    current_time[0] += 4.1  # elapsed = 6.1 s
    resp3 = client.get("/api/profiles/conductors", params={"fresh": 1})
    assert resp3.status_code == 200
    assert read_count[0] == 2


def test_no_subprocess_in_steady_state(client, env, monkeypatch):
    """No git/gh subprocess is spawned in steady state."""
    m_path = _make_marker(env.home, "repo-subp", run_id="run-subp", session_id="claude-subp", branch="feat/no-subp")
    _add_index_entry(env.index_file, m_path, run_id="run-subp", session_id="claude-subp")

    def forbidden(*args, **kwargs):
        raise RuntimeError("Subprocess execution is strictly forbidden in steady state")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    resp = client.get("/api/profiles/conductors")
    assert resp.status_code == 200
    assert resp.json()["rows"][0]["build"]["run_id"] == "run-subp"


def test_masking_applied_to_free_text(client, env):
    """Masking is applied to free text (secrets redacted, ANSI stripped)."""
    secret = "sk-ant-api03-abcdef1234567890abcdef1234567890"
    m_path = _make_marker(
        env.home, "repo-mask", run_id="run-mask", session_id="claude-mask",
        plan_title=f"\x1b[31mPlan with {secret}\x1b[0m",
        waiting_on=f"Waiting for key {secret}",
    )
    _add_index_entry(env.index_file, m_path, run_id="run-mask", session_id="claude-mask")

    resp = client.get("/api/profiles/conductors")
    assert resp.status_code == 200
    text = resp.text

    assert secret not in text
    assert "\x1b[31m" not in text
    row = resp.json()["rows"][0]
    assert "[REDACTED]" in row["build"]["plan_title"] or "sk-ant-***" in row["build"]["plan_title"] or secret not in row["build"]["plan_title"]
