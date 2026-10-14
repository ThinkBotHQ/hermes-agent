"""The pull loop backs off while cntrl is unreachable and logs once per outage."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cntrl-plugins"))

import cntrl_sync  # noqa: E402
from cntrl_sync import core  # noqa: E402

EMPTY = {"created": 0, "updated": 0, "closed": 0, "errors": []}


@pytest.fixture
def pulls(monkeypatch):
    outcomes: list = []

    def fake_run_once(state, *, cfg=None):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(core, "run_once", fake_run_once)
    monkeypatch.setattr(cntrl_sync, "_cfg", dict(core.DEFAULTS, poll_seconds=30))
    return outcomes


def _drive(outcomes, n):
    failures, delays = 0, []
    for _ in range(n):
        failures, delay = cntrl_sync._poll_once(failures)
        delays.append(delay)
    return failures, delays


def test_an_outage_warns_once_and_recovery_is_logged(pulls, caplog):
    pulls.extend([core.CntrlError("refused")] * 5 + [EMPTY])
    with caplog.at_level(logging.DEBUG, logger="plugins.cntrl_sync"):
        failures, _ = _drive(pulls, 6)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert failures == 0
    assert any("recovered after 5" in r.getMessage() for r in caplog.records)


def test_waits_grow_while_down_stay_capped_and_reset_on_success(pulls):
    pulls.extend([core.CntrlError("refused")] * 12 + [EMPTY])
    _, delays = _drive(pulls, 13)
    down, after = delays[:-1], delays[-1]
    assert all(b >= a for a, b in zip(down, down[1:]))
    assert down[0] > 30
    assert max(down) == core.POLL_BACKOFF_MAX_SECONDS
    assert after == 30


def test_backoff_never_waits_less_than_the_configured_interval():
    for base in (5, 30, 3600):
        for n in range(0, 40):
            assert core.poll_backoff_seconds(base, n) >= base


def test_pull_opens_the_board_without_the_deprecated_import(tmp_path, monkeypatch):
    from hermes_cli import plugin_compat

    def refuse(*args, **kwargs):
        raise AssertionError(f"deprecated kanban_db import used: {args}")

    monkeypatch.setattr(plugin_compat, "warn_once", refuse)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("CNTRL_TEST_KEY", "k")
    seen = {}

    def fake_sync_down(client, conn, *, cfg, state):
        seen["conn"] = conn
        return EMPTY

    monkeypatch.setattr(core, "sync_down", fake_sync_down)
    report = core.run_once(None, cfg=dict(core.DEFAULTS, api_key_env="CNTRL_TEST_KEY"))
    assert report == EMPTY and seen["conn"] is not None

    conn = cntrl_sync._conn(None)
    conn.close()
