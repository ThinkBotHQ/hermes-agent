"""`agent.claude_agent_sdk.env` — operator knobs for the spawned CLI.

The Claude Code CLI reads operational knobs from its environment that the SDK
exposes no typed option for. Measured against claude-agent-sdk 0.2.120 via
get_context_usage(): CLAUDE_CODE_AUTO_COMPACT_WINDOW=300000 moves maxTokens to
300000 and the autocompact threshold to 267000, while
CLAUDE_CODE_MAX_CONTEXT_TOKENS and CLAUDE_AUTOCOMPACT_PCT_OVERRIDE are inert.
Three of four plausible knobs doing nothing is why this is a generic config
surface rather than a named option per knob.

The security edge: this env is applied AFTER the metered-billing scrub, so
without a guard `env: {ANTHROPIC_API_KEY: ...}` would overwrite the scrub's ""
and silently re-arm metered billing behind `allow_metered_key: false`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.transports import claude_agent_sdk_session_config as M



@pytest.fixture
def env_config(monkeypatch):
    """Drive _configured_sdk_env / the metered flag / the scrub from the test."""

    def _apply(env=None, metered_allowed=False, scrubbed=None, task_tools=False):
        monkeypatch.setattr(
            M, "_provider_config", lambda: {"env": env} if env is not None else {}
        )
        monkeypatch.setattr(M, "_provider_flag", lambda name, default=False: (
            task_tools if name == "task_tools" else metered_allowed
        ))
        monkeypatch.setattr(M, "_scrubbed_sdk_env", lambda: dict(scrubbed or {}))

    return _apply


def test_values_are_stringified(env_config):
    """YAML gives ints for numeric knobs; the CLI env must be all strings."""
    env_config(env={"CLAUDE_CODE_AUTO_COMPACT_WINDOW": 300000})

    assert M._configured_sdk_env() == {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "300000"}


def test_absent_or_non_mapping_env_yields_empty(env_config):
    for bad in (None, "not-a-map", ["a"], 7):
        env_config(env=bad)
        assert M._configured_sdk_env() == {}


def test_none_values_are_skipped(env_config):
    """`KEY:` with no value parses as None; str(None) would set the literal
    string "None" in the CLI's environment, so it is dropped instead."""
    env_config(env={"A": None, "B": "keep"})

    assert M._configured_sdk_env() == {"B": "keep"}


def test_configured_env_reaches_the_overrides(env_config):
    env_config(env={"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "300000"})

    assert M._sdk_env_overrides()["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "300000"


def test_sdk_state_root_is_profile_scoped_and_private(env_config, monkeypatch, tmp_path):
    from hermes_cli.control_plane_env import scrub_desktop_control_plane_env

    profile_home = tmp_path / "profile"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    env_config()

    env = M._sdk_env_overrides(sdk_cwd=str(cwd), hermes_session_id="session-a")

    state_dir = Path(env["TB_STATE_ROOT"])
    assert state_dir.is_relative_to(profile_home / "sdk-state")
    assert not state_dir.is_relative_to(cwd)
    assert state_dir.is_dir()
    assert state_dir.stat().st_mode & 0o777 == 0o700
    assert scrub_desktop_control_plane_env({"TB_STATE_ROOT": str(state_dir)}) == {
        "TB_STATE_ROOT": str(state_dir)
    }


def test_sdk_state_root_is_distinct_for_different_cwds(env_config, monkeypatch, tmp_path):
    profile_home = tmp_path / "profile"
    cwd_a = tmp_path / "repo-a"
    cwd_b = tmp_path / "repo-b"
    cwd_a.mkdir()
    cwd_b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    env_config()

    state_a = M._sdk_env_overrides(sdk_cwd=str(cwd_a), hermes_session_id="session-a")["TB_STATE_ROOT"]
    state_b = M._sdk_env_overrides(sdk_cwd=str(cwd_b), hermes_session_id="session-b")["TB_STATE_ROOT"]

    assert state_a != state_b


def test_sdk_session_exports_its_launch_cwd_as_claude_project_dir(env_config, monkeypatch, tmp_path):
    """Conductor resolves its state root and marker ownership from CLAUDE_PROJECT_DIR, which Claude
    Code sets to its launch dir; an SDK session must set the same, beating a parent session's value."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/some/parent/session/project")
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    env_config()

    env = M._sdk_env_overrides(sdk_cwd=str(repo / "sub" / ".."), hermes_session_id="s")
    assert env["CLAUDE_PROJECT_DIR"] == str(repo.resolve())
    # No session cwd (aux one-shots, bare process): not a project, nothing exported.
    assert "CLAUDE_PROJECT_DIR" not in M._sdk_env_overrides(hermes_session_id="s")


def test_operator_configured_claude_project_dir_wins(env_config, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    env_config(env={"CLAUDE_PROJECT_DIR": "/pinned/project"})
    env = M._sdk_env_overrides(sdk_cwd=str(tmp_path), hermes_session_id="s")
    assert env["CLAUDE_PROJECT_DIR"] == "/pinned/project"


def test_native_task_tools_inject_defaults_and_keep_operator_overrides(env_config):
    env_config(task_tools=True)

    overrides = M._sdk_env_overrides(task_list_id="hermes-session-1")

    assert overrides["CLAUDE_CODE_ENABLE_TODO_TOOLS"] == "1"
    assert overrides["CLAUDE_CODE_TASK_LIST_ID"] == "hermes-session-1"

    env_config(
        env={
            "CLAUDE_CODE_ENABLE_TODO_TOOLS": "operator",
            "CLAUDE_CODE_TASK_LIST_ID": "operator-list",
        },
        task_tools=True,
    )
    overrides = M._sdk_env_overrides(task_list_id="hermes-session-1")
    assert overrides["CLAUDE_CODE_ENABLE_TODO_TOOLS"] == "operator"
    assert overrides["CLAUDE_CODE_TASK_LIST_ID"] == "operator-list"


def test_native_task_tools_disabled_injects_nothing(env_config):
    env_config(task_tools=False)

    overrides = M._sdk_env_overrides(task_list_id="hermes-session-1")

    assert "CLAUDE_CODE_ENABLE_TODO_TOOLS" not in overrides
    assert "CLAUDE_CODE_TASK_LIST_ID" not in overrides


def test_native_task_tools_inherited_env_beats_defaults_but_yaml_wins(env_config, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TODO_TOOLS", "0")
    monkeypatch.setenv("CLAUDE_CODE_TASK_LIST_ID", "operator-list")
    env_config(task_tools=True)

    inherited = M._sdk_env_overrides(task_list_id="derived-list")
    assert inherited["CLAUDE_CODE_ENABLE_TODO_TOOLS"] == "0"
    assert inherited["CLAUDE_CODE_TASK_LIST_ID"] == "operator-list"

    env_config(
        env={
            "CLAUDE_CODE_ENABLE_TODO_TOOLS": "1",
            "CLAUDE_CODE_TASK_LIST_ID": "yaml-list",
        },
        task_tools=True,
    )
    configured = M._sdk_env_overrides(task_list_id="derived-list")
    assert configured["CLAUDE_CODE_ENABLE_TODO_TOOLS"] == "1"
    assert configured["CLAUDE_CODE_TASK_LIST_ID"] == "yaml-list"


def test_scrub_is_preserved_alongside_configured_env(env_config):
    """A knob must not displace the metered scrub that ships with it."""
    env_config(
        env={"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "300000"},
        scrubbed={"ANTHROPIC_API_KEY": ""},
    )

    overrides = M._sdk_env_overrides()

    assert overrides["ANTHROPIC_API_KEY"] == ""
    assert overrides["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "300000"


def test_config_env_cannot_resurrect_a_scrubbed_credential(env_config, caplog):
    """The whole point of the guard: config must not defeat allow_metered_key: false."""
    env_config(
        env={"ANTHROPIC_API_KEY": "sk-ant-metered"},
        metered_allowed=False,
        scrubbed={"ANTHROPIC_API_KEY": ""},
    )

    with caplog.at_level("WARNING", logger=M.logger.name):
        overrides = M._sdk_env_overrides()

    assert overrides["ANTHROPIC_API_KEY"] == ""
    assert "metered billing vector" in caplog.text


@pytest.mark.parametrize("key", M._METERED_ENV_DENYLIST)
def test_every_denylisted_key_is_guarded(env_config, key):
    env_config(env={key: "x"}, metered_allowed=False, scrubbed={key: ""})

    assert M._sdk_env_overrides()[key] == ""


def test_metered_opt_in_permits_the_key(env_config):
    """With the explicit opt-in the scrub is off and the operator's value stands."""
    env_config(env={"ANTHROPIC_API_KEY": "sk-ant-metered"}, metered_allowed=True)

    assert M._sdk_env_overrides()["ANTHROPIC_API_KEY"] == "sk-ant-metered"


def test_subscription_shaped_anthropic_token_is_preserved(env_config, monkeypatch):
    """ANTHROPIC_TOKEN is shared by metered and setup-token flows in Hermes."""
    monkeypatch.setattr(M, "_is_subscription_oauth_token", lambda value: True)
    env_config(env={"ANTHROPIC_TOKEN": "oauth-token"}, metered_allowed=False)

    assert M._sdk_env_overrides()["ANTHROPIC_TOKEN"] == "oauth-token"


def test_metered_anthropic_token_is_scrubbed_from_parent(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_TOKEN", "metered-token")
    monkeypatch.setattr(M, "_is_subscription_oauth_token", lambda value: False)

    assert M._scrubbed_sdk_env()["ANTHROPIC_TOKEN"] == ""


# ── Interpreter-path scrub ────────────────────────────────────────────────────
# The desktop backend inherits PYTHONPATH=<repo>:<venv>/lib/python3.11/site-packages from
# Electron; the SDK merges os.environ into the CLI child, which hands it to every plugin MCP it
# spawns, and a plugin's `uv run --python >=3.12` server then imports 3.11-built pydantic_core
# and dies (conductor tb-workers "Connection closed", 2026-09-11).


def test_pythonpath_and_pythonhome_are_blanked_when_present(env_config, monkeypatch):
    env_config(env=None)
    monkeypatch.setenv("PYTHONPATH", "/repo:/repo/.venv/lib/python3.11/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/repo/.venv")

    overrides = M._sdk_env_overrides()

    assert overrides["PYTHONPATH"] == ""
    assert overrides["PYTHONHOME"] == ""


def test_absent_interpreter_vars_are_not_introduced(env_config, monkeypatch):
    """Only PRESENT keys are overridden — an empty var the child never had is a new fact."""
    env_config(env=None)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)

    overrides = M._sdk_env_overrides()

    assert "PYTHONPATH" not in overrides
    assert "PYTHONHOME" not in overrides


def test_operator_env_may_still_set_pythonpath_deliberately(env_config, monkeypatch):
    """The scrub is a default, not a security boundary: `env: {PYTHONPATH: ...}` is a knob and wins."""
    env_config(env={"PYTHONPATH": "/deliberate"})
    monkeypatch.setenv("PYTHONPATH", "/repo/.venv/lib/python3.11/site-packages")

    assert M._sdk_env_overrides()["PYTHONPATH"] == "/deliberate"


def test_interpreter_scrub_is_independent_of_the_metered_opt_in(env_config, monkeypatch):
    """allow_metered_key disables the BILLING scrub only; a metered opt-in still must not
    hand plugin MCPs a wrong-ABI import path."""
    env_config(env=None, metered_allowed=True)
    monkeypatch.setenv("PYTHONPATH", "/repo/.venv/lib/python3.11/site-packages")

    assert M._sdk_env_overrides()["PYTHONPATH"] == ""


# ── npm environment hygiene ───────────────────────────────────────────────────
# The desktop app is launched by `npm run`, so npm_config_* and npm_package_*
# describe the live checkout and are stale for anything an agent runs (same list
# as apps/desktop/electron/terminal-ipc.ts). This is hygiene, not a fix for a
# known failure.


def test_npm_vars_are_blanked_when_present_in_parent_env(env_config, monkeypatch):
    """(a) with npm_config_local_prefix, npm_config_prefix and npm_package_name set
    in the parent env, the env overrides handed to the SDK map each of them to ""."""
    env_config(env=None)
    monkeypatch.setenv("npm_config_local_prefix", "/workspace/repo")
    monkeypatch.setenv("npm_config_prefix", "/usr/local")
    monkeypatch.setenv("npm_package_name", "hermes-desktop")

    overrides = M._sdk_env_overrides()

    assert overrides["npm_config_local_prefix"] == ""
    assert overrides["npm_config_prefix"] == ""
    assert overrides["npm_package_name"] == ""


def test_absent_npm_vars_yield_no_npm_keys_in_overrides(env_config, monkeypatch):
    """(b) a parent env with none of them yields no npm keys in the overrides."""
    env_config(env=None)
    for key in list(os.environ):
        if key == "npm_config_prefix" or key.startswith(("npm_config_", "npm_package_")):
            monkeypatch.delenv(key, raising=False)

    overrides = M._sdk_env_overrides()

    npm_keys = [
        k for k in overrides
        if k == "npm_config_prefix" or k.startswith(("npm_config_", "npm_package_"))
    ]
    assert npm_keys == []


def test_unrelated_keys_are_not_overridden_by_npm_scrub(env_config, monkeypatch):
    """(c) unrelated keys (e.g. PATH, NPM_TOKEN in upper case, INIT_CWD) are NOT overridden."""
    env_config(env=None)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("NPM_TOKEN", "secret-npm-token")
    monkeypatch.setenv("INIT_CWD", "/workspace/repo")

    overrides = M._sdk_env_overrides()

    assert "PATH" not in overrides
    assert "NPM_TOKEN" not in overrides
    assert "INIT_CWD" not in overrides


def test_existing_interpreter_scrub_still_behaves_as_before(env_config, monkeypatch):
    """(d) the existing PYTHONPATH/PYTHONHOME scrub still behaves as before."""
    env_config(env=None)
    monkeypatch.setenv("PYTHONPATH", "/repo:/repo/.venv/lib/python3.11/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/repo/.venv")
    monkeypatch.setenv("npm_config_prefix", "/usr/local")

    overrides = M._sdk_env_overrides()

    assert overrides["PYTHONPATH"] == ""
    assert overrides["PYTHONHOME"] == ""
    assert overrides["npm_config_prefix"] == ""


def test_nested_in_claude_child_masks_the_parent_task_list(env_config, monkeypatch):
    """Inside a Claude Code child the inherited list id is the PARENT's: never reuse it."""
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_TASK_LIST_ID", "parent-session-list")
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TODO_TOOLS", "1")
    env_config(task_tools=True)
    enabled = M._sdk_env_overrides(task_list_id="derived-list")
    assert enabled["CLAUDE_CODE_TASK_LIST_ID"] == "derived-list"

    env_config(task_tools=False)
    disabled = M._sdk_env_overrides(task_list_id="derived-list")
    assert disabled["CLAUDE_CODE_TASK_LIST_ID"] == ""
    assert disabled["CLAUDE_CODE_ENABLE_TODO_TOOLS"] == ""


def test_attest_dir_hint_present_when_anchor_loads_and_absent_otherwise(env_config, monkeypatch, tmp_path):
    env_config()
    # Absent when anchor fails to load
    monkeypatch.setattr(
        "hermes_owner_grant.anchor.load_trusted_anchor",
        lambda **kwargs: (_ for _ in ()).throw(Exception("no anchor")),
    )
    overrides = M._sdk_env_overrides()
    assert "HERMES_SESSION_ATTEST_DIR" not in overrides

    # Present when anchor loads
    from hermes_owner_grant import anchor as anchor_mod
    mock_anchor = anchor_mod.Anchor(
        owner_uid=501,
        grants_dir=str(tmp_path / "grants"),
        keys=(),
        verifier_sha256=None,
        sha256="0" * 64,
    )
    monkeypatch.setattr(
        "hermes_owner_grant.anchor.load_trusted_anchor",
        lambda **kwargs: mock_anchor,
    )
    overrides = M._sdk_env_overrides()
    assert overrides["HERMES_SESSION_ATTEST_DIR"] == str(tmp_path / "grants" / "session-attest")


def test_tb_state_root_set_from_verified_bound_binding(env_config, monkeypatch, tmp_path):
    import json
    from hermes_owner_grant import anchor as anchor_mod, envelope as env_mod

    fixture_path = Path(__file__).resolve().parent.parent / "hermes_owner_grant" / "fixtures" / "attest_v1_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    grants_dir = tmp_path / "grants"
    key = fixture["anchor_key"]
    anchor = anchor_mod.Anchor(
        owner_uid=fixture["owner_uid"],
        grants_dir=str(grants_dir),
        keys=(anchor_mod.AnchorKey(
            kid=key["kid"], alg=key["alg"], pub=env_mod.b64url_decode(key["pub"]),
            status=key["status"], not_before=key["not_before"], retired_at=key["retired_at"],
        ),),
        verifier_sha256=None,
        sha256="0" * 64,
    )
    monkeypatch.setattr("hermes_owner_grant.anchor.load_trusted_anchor", lambda **kwargs: anchor)

    binding_dir = grants_dir / "session-bindings" / fixture["profile"]
    binding_dir.mkdir(parents=True, exist_ok=True)
    binding_file = binding_dir / f"{fixture['session']}.json"
    binding_file.write_text(json.dumps(fixture["binding_envelope"]), encoding="utf-8")

    env_config()
    env = M._sdk_env_overrides(
        sdk_cwd=str(tmp_path / "repo"),
        hermes_session_id=fixture["session"],
        profile=fixture["profile"],
    )
    # b10 review (grok #2): the verified binding picks WHICH path is hashed; the state root
    # itself stays under HERMES_HOME, never the (agent-writable, committable) project.
    import hashlib
    import os
    from hermes_constants import get_hermes_home

    expected = get_hermes_home() / "sdk-state" / hashlib.sha256(
        os.fsencode(Path(fixture["project_root"]).resolve())
    ).hexdigest()[:16]
    assert env["TB_STATE_ROOT"] == str(expected)
    assert env["TB_STATE_ROOT"] != fixture["project_root"]


def test_tb_state_root_not_set_on_tamper_unbound_missing(env_config, monkeypatch, tmp_path):
    import json
    from hermes_owner_grant import anchor as anchor_mod, attest as attest_mod, envelope as env_mod
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    from cryptography.hazmat.primitives import serialization

    fixture_path = Path(__file__).resolve().parent.parent / "hermes_owner_grant" / "fixtures" / "attest_v1_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    grants_dir = tmp_path / "grants"
    key = fixture["anchor_key"]

    priv = ed.Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    test_kid = env_mod.kid_for_pub(pub_bytes)

    anchor = anchor_mod.Anchor(
        owner_uid=fixture["owner_uid"],
        grants_dir=str(grants_dir),
        keys=(
            anchor_mod.AnchorKey(
                kid=key["kid"], alg=key["alg"], pub=env_mod.b64url_decode(key["pub"]),
                status=key["status"], not_before=key["not_before"], retired_at=key["retired_at"],
            ),
            anchor_mod.AnchorKey(
                kid=test_kid, alg="Ed25519", pub=pub_bytes,
                status="active", not_before=0, retired_at=None,
            ),
        ),
        verifier_sha256=None,
        sha256="0" * 64,
    )
    monkeypatch.setattr("hermes_owner_grant.anchor.load_trusted_anchor", lambda **kwargs: anchor)

    binding_dir = grants_dir / "session-bindings" / fixture["profile"]
    binding_dir.mkdir(parents=True, exist_ok=True)

    # 1. Missing binding file
    env_config()
    env_missing = M._sdk_env_overrides(
        sdk_cwd=str(tmp_path / "repo"),
        hermes_session_id="missing_session",
        profile=fixture["profile"],
    )
    assert env_missing["TB_STATE_ROOT"] != fixture["project_root"]
    assert "sdk-state" in env_missing["TB_STATE_ROOT"]

    # 2. Tampered binding file
    tampered_file = binding_dir / "tampered_session.json"
    tampered_data = dict(fixture["binding_envelope"])
    tampered_data["sig"] = "A" * 86
    tampered_file.write_text(json.dumps(tampered_data), encoding="utf-8")
    env_tampered = M._sdk_env_overrides(
        sdk_cwd=str(tmp_path / "repo"),
        hermes_session_id="tampered_session",
        profile=fixture["profile"],
    )
    assert env_tampered["TB_STATE_ROOT"] != fixture["project_root"]
    assert "sdk-state" in env_tampered["TB_STATE_ROOT"]

    # 3. State unbound binding file
    unbound_payload = {
        "v": 1,
        "aud": ["hermes-main"],
        "owner_uid": fixture["owner_uid"],
        "profile": fixture["profile"],
        "hermes_session_id": "unbound_session",
        "state": "unbound",
        "seq": 1,
        "binding_nonce": "nonce123",
        "bound_at": 1789999940000,
        "project_root": None,
    }
    unbound_env = attest_mod.seal_binding(
        json.dumps(unbound_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        test_kid,
        priv.sign,
    )
    unbound_file = binding_dir / "unbound_session.json"
    unbound_file.write_text(unbound_env.to_json(), encoding="utf-8")
    env_unbound = M._sdk_env_overrides(
        sdk_cwd=str(tmp_path / "repo"),
        hermes_session_id="unbound_session",
        profile=fixture["profile"],
    )
    assert env_unbound["TB_STATE_ROOT"] != fixture["project_root"]
    assert "sdk-state" in env_unbound["TB_STATE_ROOT"]


def test_operator_configured_tb_state_root_wins(env_config, monkeypatch, tmp_path):
    import json
    from hermes_owner_grant import anchor as anchor_mod, envelope as env_mod

    fixture_path = Path(__file__).resolve().parent.parent / "hermes_owner_grant" / "fixtures" / "attest_v1_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    grants_dir = tmp_path / "grants"
    key = fixture["anchor_key"]
    anchor = anchor_mod.Anchor(
        owner_uid=fixture["owner_uid"],
        grants_dir=str(grants_dir),
        keys=(anchor_mod.AnchorKey(
            kid=key["kid"], alg=key["alg"], pub=env_mod.b64url_decode(key["pub"]),
            status=key["status"], not_before=key["not_before"], retired_at=key["retired_at"],
        ),),
        verifier_sha256=None,
        sha256="0" * 64,
    )
    monkeypatch.setattr("hermes_owner_grant.anchor.load_trusted_anchor", lambda **kwargs: anchor)

    binding_dir = grants_dir / "session-bindings" / fixture["profile"]
    binding_dir.mkdir(parents=True, exist_ok=True)
    binding_file = binding_dir / f"{fixture['session']}.json"
    binding_file.write_text(json.dumps(fixture["binding_envelope"]), encoding="utf-8")

    # Operator-configured TB_STATE_ROOT in config.yaml env
    env_config(env={"TB_STATE_ROOT": "/custom/operator/tb_state_root"})
    env = M._sdk_env_overrides(
        sdk_cwd=str(tmp_path / "repo"),
        hermes_session_id=fixture["session"],
        profile=fixture["profile"],
    )
    assert env["TB_STATE_ROOT"] == "/custom/operator/tb_state_root"


def test_bound_tb_state_root_is_hashed_under_hermes_home_never_the_project(
    env_config, monkeypatch, tmp_path
):
    """b10 review (grok #2): conductor state must not land inside the bound git worktree."""
    import hashlib
    import json
    import os
    from hermes_owner_grant import anchor as anchor_mod, envelope as env_mod

    fixture_path = Path(__file__).resolve().parent.parent / "hermes_owner_grant" / "fixtures" / "attest_v1_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    profile_home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    grants_dir = tmp_path / "grants"
    key = fixture["anchor_key"]
    anchor = anchor_mod.Anchor(
        owner_uid=fixture["owner_uid"],
        grants_dir=str(grants_dir),
        keys=(anchor_mod.AnchorKey(
            kid=key["kid"], alg=key["alg"], pub=env_mod.b64url_decode(key["pub"]),
            status=key["status"], not_before=key["not_before"], retired_at=key["retired_at"],
        ),),
        verifier_sha256=None,
        sha256="0" * 64,
    )
    monkeypatch.setattr("hermes_owner_grant.anchor.load_trusted_anchor", lambda **kwargs: anchor)
    binding_dir = grants_dir / "session-bindings" / fixture["profile"]
    binding_dir.mkdir(parents=True, exist_ok=True)
    (binding_dir / f"{fixture['session']}.json").write_text(
        json.dumps(fixture["binding_envelope"]), encoding="utf-8"
    )
    env_config()

    roots = []
    for sub in ("repo-a", "repo-b/nested"):
        cwd = tmp_path / sub
        cwd.mkdir(parents=True)
        env = M._sdk_env_overrides(
            sdk_cwd=str(cwd), hermes_session_id=fixture["session"], profile=fixture["profile"]
        )
        roots.append(env["TB_STATE_ROOT"])

    project_root = fixture["project_root"]
    state_dir = Path(roots[0])
    assert roots[0] == roots[1], "a bound session's cwd moves share one state root"
    assert state_dir.is_relative_to(profile_home / "sdk-state")
    assert state_dir.name == hashlib.sha256(
        os.fsencode(Path(project_root).resolve())
    ).hexdigest()[:16]
    assert not str(state_dir).startswith(project_root)
    assert state_dir.is_dir()
    assert state_dir.stat().st_mode & 0o777 == 0o700

    # Unbound: today's cwd hash, also under HERMES_HOME.
    unbound = M._sdk_env_overrides(
        sdk_cwd=str(tmp_path / "repo-a"), hermes_session_id="unbound_other", profile=fixture["profile"]
    )["TB_STATE_ROOT"]
    assert unbound != roots[0]
    assert Path(unbound).name == hashlib.sha256(
        os.fsencode((tmp_path / "repo-a").resolve())
    ).hexdigest()[:16]
