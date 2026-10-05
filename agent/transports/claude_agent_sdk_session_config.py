"""Provider configuration readers for the claude-agent-sdk session.

``agent.claude_agent_sdk`` config/flag access, permission-mode and setting-source
readers, the child environment (scrubbed of metered billing vectors), stdout
framing and timeout knobs, the HTTP MCP entries and the stdio hermes-tools MCP
config. Extracted from ``claude_agent_sdk_session.py``; ``_provider_config``,
``_provider_flag`` and ``_sdk_env_overrides`` are re-exported by the facade for
the runtime and the auxiliary client.
"""

from __future__ import annotations

import logging
import math
import os
import secrets
import stat
import sys
import sysconfig
import contextlib
import hashlib
from pathlib import Path
from typing import Any, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


# HERMES_TERMINAL_SECURITY_MODE → SDK permission_mode, read at session
# construction (default "auto"). Precedence: explicit constructor arg, then
# the agent.claude_agent_sdk.permission_mode config key (an SDK mode literal
# — see _configured_permission_mode), then this env mapping.
#
# SDK default posture is intentionally stricter than generic terminal `auto`:
# `default` preserves Hermes' per-tool approval bridge. Mapping `auto` to
# `acceptEdits` skips that bridge entirely, so even the fixed bounded MCP read
# surface is denied/unguarded depending on CLI state. Hermes YOLO requests are
# represented internally as ``bypassPermissions`` but normalized back to
# ``default`` before SDK option construction; the audited callback performs the
# bypass only after immutable floors.
_HERMES_TO_SDK_PERMISSION_MODE = {
    "auto": "default",
    "approval-required": "default",
    "unrestricted": "bypassPermissions",
    "yolo": "bypassPermissions",
}


# The SDK's own permission_mode literals (verified against the installed
# claude-agent-sdk 0.2.120 ClaudeAgentOptions type).
_SDK_PERMISSION_MODES = (
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
    "dontAsk",
    "auto",
)


def _configured_sdk_env() -> dict:
    """agent.claude_agent_sdk.env — extra environment for the CLI subprocess.

    The Claude Code CLI reads operational knobs from its environment that the
    SDK exposes no typed option for. Measured on 0.2.120: only
    ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` moves the context ceiling and the
    autocompact threshold (300000 -> maxTokens 300000, threshold 267000);
    ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` and ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``
    are inert. That ratio is exactly why this is a generic config surface and
    not a named option per knob — the knobs are undocumented and shift.

    Values are stringified; a non-mapping or unreadable config yields {} so a
    bad edit cannot strip the scrubbed env that ships alongside it.
    """
    raw = _provider_config().get("env")
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, value in raw.items():
        if value is None:
            continue
        try:
            out[str(key)] = str(value)
        except Exception:
            logger.warning(
                "agent.claude_agent_sdk.env[%r] is not stringifiable — ignoring", key
            )
    return out


def _sdk_env_overrides(
    *,
    metered_allowed: Optional[bool] = None,
    task_list_id: Optional[str] = None,
    task_env: Optional[dict[str, str]] = None,
    hermes_session_id: Optional[str] = None,
    sdk_cwd: Optional[str] = None,
    profile: Optional[str] = None,
) -> dict[str, str]:
    """The full env override set handed to the spawned CLI.

    Metered-vector scrub first (see _METERED_ENV_DENYLIST), then the
    interpreter-path scrub (see _CHILD_INTERPRETER_ENV_DENYLIST) and npm
    environment hygiene (see _scrubbed_npm_env).
    agent.claude_agent_sdk.allow_metered_key: true is the operator's explicit
    "bill me metered" opt-in (the same flag the startup guard honors), so it
    disables the scrub too — otherwise the documented escape hatch would hand
    the CLI an environment with the key blanked.

    Operator-configured env is applied last so deliberate knobs win over
    defaults, but it must NOT win over the scrub: a plain update() would let
    ``env: {ANTHROPIC_API_KEY: ...}`` overwrite the scrub's "" and silently
    re-arm metered billing behind allow_metered_key: false. Denylisted keys
    are therefore dropped (loudly) unless the metered opt-in is set.
    """
    if metered_allowed is None:
        metered_allowed = _provider_flag("allow_metered_key")
    overrides: dict[str, str] = {} if metered_allowed else _scrubbed_sdk_env()
    # Interpreter-path scrub (see _CHILD_INTERPRETER_ENV_DENYLIST). Applied
    # before the operator env so a deliberate ``env: {PYTHONPATH: ...}`` in
    # config.yaml still wins — that is a knob, not a billing vector.
    overrides.update(_scrubbed_interpreter_env())
    # npm environment hygiene (see apps/desktop/electron/terminal-ipc.ts).
    overrides.update(_scrubbed_npm_env())
    # A caller that already resolved the task env for this conversation passes
    # it in, so a rebuilt child gets byte-identical task variables.
    overrides.update(
        dict(task_env) if task_env is not None
        else _effective_sdk_task_env(task_list_id=task_list_id)
    )
    if sdk_cwd and _provider_flag("retry_watchdog", True):
        # D62 L1 (agent.claude_agent_sdk.retry_watchdog, default on): the CLI
        # keeps retrying 429/529/no-response INSIDE the same API call (bool
        # env, verified in CLI 2.1.284). Real sessions only (they pass their
        # cwd); aux one-shots keep the bounded default. Operator env still wins.
        overrides["CLAUDE_CODE_RETRY_WATCHDOG"] = "1"
    from tools.environments.local_env_policy import (
        desktop_control_plane_env_blanks, is_desktop_control_plane_env)
    for key, value in _configured_sdk_env().items():
        if is_desktop_control_plane_env(key):
            logger.warning(
                "agent.claude_agent_sdk.env[%s] is a desktop control-plane variable — ignoring "
                "(no agent may hold the dashboard credential or the renderer debug port)", key)
            continue
        if not metered_allowed and _is_metered_sdk_env_value(key, value):
            logger.warning(
                "agent.claude_agent_sdk.env[%s] is a metered billing vector — "
                "ignoring (set allow_metered_key: true to permit it)",
                key,
            )
            continue
        overrides[key] = value
    # Desktop control plane (tools.environments.local_env_policy.DESKTOP_CONTROL_PLANE_ENV_KEYS):
    # the SDK spawns {**os.environ, **options.env}, so every present key is overridden to "" —
    # the CLI, its Bash tool and every plugin MCP/hook it launches inherit no dashboard token
    # and no CDP port. Unconditional: allow_metered_key does not re-arm it. (The desktop
    # backend also pops the sealed subset from os.environ at startup, so there it is absent.)
    overrides.update(desktop_control_plane_env_blanks())

    # Explicit session identity for the spawned CLI. The SDK spawns the CLI with
    # the inherited parent process env ({**os.environ, ..., **options.env}), so an
    # un-overridden key inherits the ambient process value. In a multiplexed
    # backend (tui_gateway, gateway), os.environ["HERMES_SESSION_ID"] holds
    # whichever session last initialized; without an explicit override, a sibling
    # session's id leaks into the Claude CLI (and its Bash tool subprocesses).
    if hermes_session_id is not None:
        overrides["HERMES_SESSION_ID"] = str(hermes_session_id)
    else:
        try:
            from gateway.session_context import _SESSION_ID, _UNSET, session_context_engaged
            val = _SESSION_ID.get()
            if val is not _UNSET and val:
                overrides["HERMES_SESSION_ID"] = str(val)
            elif session_context_engaged() and os.environ.get("HERMES_SESSION_ID"):
                overrides["HERMES_SESSION_ID"] = ""
        except Exception:
            pass

    # HERMES_SESSION_ATTEST_DIR hint = <grants_dir>/session-attest (spec §0.4, §4, §8 row H4).
    # It is a hint only; nothing trusts it.
    anchor = None
    try:
        from hermes_owner_grant.anchor import load_trusted_anchor
        anchor = load_trusted_anchor()
    except Exception:
        anchor = None

    if "HERMES_SESSION_ATTEST_DIR" not in _configured_sdk_env():
        if anchor is not None:
            grants_dir = getattr(anchor, "grants_dir", None)
            if grants_dir:
                overrides["HERMES_SESSION_ATTEST_DIR"] = os.path.join(str(grants_dir), "session-attest")

    # Conductor's worker-spawn hook honors TB_STATE_ROOT. Keep its artifacts
    # out of the checked-out project and scope them to the active Hermes
    # profile; a cwd hash gives each project a stable, filesystem-safe root.
    # Only a real SDK session (which passes its cwd) gets a root; the process cwd
    # is not a project.
    if not sdk_cwd:
        return overrides
    from hermes_constants import get_hermes_home

    project_cwd = Path(sdk_cwd).expanduser().resolve()
    # Claude Code exports CLAUDE_PROJECT_DIR (the directory it was launched in) to everything it runs;
    # SDK-spawned CLIs don't, so conductor's state root and marker ownership fell back to whatever
    # cwd a Bash call had (a subfolder, a lane) and split the build state. Set it to this session's
    # launch cwd, replacing any value inherited from a parent Claude process; an explicit
    # agent.claude_agent_sdk.env entry still wins.
    if "CLAUDE_PROJECT_DIR" not in _configured_sdk_env():
        overrides["CLAUDE_PROJECT_DIR"] = str(project_cwd)

    # TB_STATE_ROOT: when the session has a BOUND binding, set it to the binding's project_root,
    # but ONLY after verifying main's signed binding record (spec §0.4, §4, §6 C1, §8 row H4).
    # Any failure -> fall back to cwd behavior. Operator-configured TB_STATE_ROOT wins.
    if "TB_STATE_ROOT" not in _configured_sdk_env():
        bound_project_root = None
        effective_session_id = overrides.get("HERMES_SESSION_ID") or hermes_session_id
        effective_profile = profile
        if not effective_profile:
            try:
                from gateway.session_context import _SESSION_PROFILE, _UNSET
                p_val = _SESSION_PROFILE.get()
                if p_val is not _UNSET and p_val:
                    effective_profile = str(p_val)
            except Exception:
                pass
        if not effective_profile:
            try:
                from hermes_cli.profiles import current_profile_name
                effective_profile = current_profile_name(default="default") or "default"
            except Exception:
                effective_profile = os.environ.get("HERMES_PROFILE") or "default"

        if anchor is not None and effective_session_id and effective_profile:
            try:
                grants_dir = getattr(anchor, "grants_dir", None)
                if (
                    grants_dir
                    and "/" not in str(effective_profile)
                    and "\\" not in str(effective_profile)
                    and str(effective_profile) not in (".", "..")
                    and "/" not in str(effective_session_id)
                    and "\\" not in str(effective_session_id)
                    and str(effective_session_id) not in (".", "..")
                ):
                    binding_path = os.path.join(
                        str(grants_dir), "session-bindings", str(effective_profile), f"{effective_session_id}.json"
                    )
                    from hermes_owner_grant.attest import read_binding_file, verify_binding_envelope
                    binding_env = read_binding_file(binding_path)
                    res = verify_binding_envelope(
                        binding_env,
                        expected_profile=str(effective_profile),
                        expected_session=str(effective_session_id),
                        anchor=anchor,
                    )
                    if res.ok and res.project_root:
                        bound_project_root = res.project_root
            except Exception:
                bound_project_root = None

        # The state root is ALWAYS a hashed dir under HERMES_HOME, never the project itself:
        # conductor state inside the checkout is agent-writable and committable. A verified
        # binding only changes which path is hashed (the bound project instead of the launch
        # cwd), so cwd moves, subfolders and lanes of a bound session share one state root.
        state_key = project_cwd
        if bound_project_root:
            try:
                state_key = Path(str(bound_project_root)).resolve()
            except (OSError, RuntimeError, ValueError):
                state_key = project_cwd
        state_hash = hashlib.sha256(os.fsencode(state_key)).hexdigest()[:16]
        state_dir = get_hermes_home() / "sdk-state" / state_hash
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        state_dir.chmod(0o700)
        overrides["TB_STATE_ROOT"] = str(state_dir)
    return overrides


_TASK_ENV_KEYS = ("CLAUDE_CODE_ENABLE_TODO_TOOLS", "CLAUDE_CODE_TASK_LIST_ID")
# Set by the Claude Code CLI in its own child processes. When Hermes itself runs
# inside a Claude session (a Hermes SDK child running tests, a nested backend,
# `hermes` typed in a Claude-driven terminal), the inherited task vars belong to
# THAT parent session, not to an operator: honouring them would make every
# session this process starts read and write the parent's task list.
_CLAUDE_CHILD_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID")


def _inside_claude_code_child() -> bool:
    return any(os.environ.get(marker) for marker in _CLAUDE_CHILD_MARKERS)


def _effective_sdk_task_env(*, task_list_id: Optional[str] = None) -> dict[str, str]:
    """Resolve task-tool env with YAML > inherited process env > injected defaults.

    Inherited values count as operator intent only outside a Claude Code child;
    inside one they are the parent session's and are masked instead.
    """
    values: dict[str, str] = {}
    nested = _inside_claude_code_child()
    if _provider_flag("task_tools"):
        values["CLAUDE_CODE_ENABLE_TODO_TOOLS"] = "1"
        if task_list_id:
            values["CLAUDE_CODE_TASK_LIST_ID"] = str(task_list_id)
    for key in _TASK_ENV_KEYS:
        if key not in os.environ:
            continue
        if nested:
            # Never let the parent's list leak into this session's child; the
            # SDK merges options.env after the inherited env, so an explicit
            # blank is what masks it when no per-session value was set above.
            values.setdefault(key, "")
        else:
            values[key] = str(os.environ[key])
    configured = _configured_sdk_env()
    for key in ("CLAUDE_CODE_ENABLE_TODO_TOOLS", "CLAUDE_CODE_TASK_LIST_ID"):
        if key in configured:
            values[key] = configured[key]
    return values


def _configured_permission_mode() -> Optional[str]:
    """agent.claude_agent_sdk.permission_mode from config.yaml, validated.

    Takes a validated SDK permission-mode intent (one of
    _SDK_PERMISSION_MODES — note "auto" here is the SDK's own mode, NOT the
    HERMES_TERMINAL_SECURITY_MODE value of the same name). The literal
    ``bypassPermissions`` intent is emitted as callback-capable ``default`` and
    emulated inside Hermes after immutable approval floors. Empty/absent —
    the default — keeps current behavior: the HERMES_TERMINAL_SECURITY_MODE
    mapping stands, so existing deployments harden without env archaeology
    only when they opt in. Unknown values are ignored with a warning:
    permissions must never silently loosen (or tighten into an unusable
    mode) on a typo."""
    raw = str(_provider_config().get("permission_mode") or "").strip()
    if not raw:
        return None
    if raw not in _SDK_PERMISSION_MODES:
        logger.warning(
            "agent.claude_agent_sdk.permission_mode=%r is not a valid SDK "
            "permission mode (one of %s) — ignoring it; the "
            "HERMES_TERMINAL_SECURITY_MODE mapping stands.",
            raw,
            ", ".join(_SDK_PERMISSION_MODES),
        )
        return None
    return raw


# The SDK's own setting-source literals (verified against the installed
# claude-agent-sdk 0.2.120 SettingSource type).
_SDK_SETTING_SOURCES = ("user", "project", "local")


def _configured_setting_sources() -> list:
    """agent.claude_agent_sdk.setting_sources from config.yaml, validated.

    Default (absent/empty) is FULL ISOLATION — the SDK loads no filesystem
    settings, so ambient ``~/.claude`` or project files cannot re-permission
    tools or install hooks underneath the configured posture. Deployments
    whose operating model deliberately stores tool grants in the operator's
    own ``~/.claude/settings.json`` — an unattended box whose cron turns
    must pre-approve WebSearch/MCP tools with no human to answer a prompt —
    opt back in explicitly (``setting_sources: ["user"]``). Unknown entries
    are dropped with a warning: a typo must neither silently widen isolation
    nor quietly load an unintended source."""
    raw = _provider_config().get("setting_sources")
    if not isinstance(raw, (list, tuple)):
        return []
    sources: list = []
    for entry in raw:
        name = str(entry or "").strip()
        if name in _SDK_SETTING_SOURCES:
            if name not in sources:
                sources.append(name)
        elif name:
            logger.warning(
                "agent.claude_agent_sdk.setting_sources entry %r is not a "
                "valid SDK setting source (one of %s) — dropping it.",
                name,
                ", ".join(_SDK_SETTING_SOURCES),
            )
    return sources


# ---------- auto-mode classifier rules ----------
# Claude Code's auto-mode classifier reads its rule config (``autoMode``:
# allow / soft_deny / hard_deny / environment ...) from the user, FLAG and
# policy settings layers only — never project/local (repo-controllable). The
# flag layer is the CLI's ``--settings`` argument (the SDK's ``settings``
# option), and the CLI always keeps flagSettings/policySettings enabled even
# when ``--setting-sources=`` is empty. So Hermes can hand the classifier its
# rules while ``setting_sources`` stays [] — verified against Claude Code
# 2.1.283 and the SDK-bundled 2.1.259 via the get_settings control request.
#
# The value is serialized as INLINE JSON (the CLI treats a --settings value
# that starts with "{" and ends with "}" as JSON, not a path), so no settings
# file is written and nothing needs cleaning up at session close. It does
# travel in the CLI argv, like the rest of the SDK's option flags.
_AUTO_MODE_OFF = "off"
_AUTO_MODE_INHERIT_USER = "inherit_user"


def _user_claude_settings_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".claude", "settings.json")


# ~/.claude/settings.json is read at session start from a path Hermes does not
# control. Bound the read: a FIFO would block open()/read() forever, a symlink
# to /dev/zero (or a huge file) would exhaust memory, and deeply nested JSON
# raises RecursionError from the scanner. Real settings files are a few KiB.
_USER_SETTINGS_MAX_BYTES = 1024 * 1024


class _UnusableSettingsFile(Exception):
    """~/.claude/settings.json exists but is not a readable regular file
    within the size cap; the message is the reason, for the warning."""


def _read_user_settings_text(path: str) -> str:
    """The file's text, read without ever blocking and never past the cap.

    O_NONBLOCK makes open() return at once on a FIFO; fstat on the opened fd
    then refuses anything that is not a regular file (FIFO, character or
    block device, directory, socket). Following the owner's own symlink to a
    regular file is fine — dotfile managers do exactly that — because the
    type check is on what the fd actually refers to. The read loop stops at
    cap + 1 bytes, so a file that grows after the fstat still cannot be
    slurped whole."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _UnusableSettingsFile("not a regular file")
        if st.st_size > _USER_SETTINGS_MAX_BYTES:
            raise _UnusableSettingsFile(
                f"larger than {_USER_SETTINGS_MAX_BYTES} bytes"
            )
        chunks = []
        remaining = _USER_SETTINGS_MAX_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining <= 0:
            raise _UnusableSettingsFile(
                f"larger than {_USER_SETTINGS_MAX_BYTES} bytes"
            )
    finally:
        os.close(fd)
    return b"".join(chunks).decode("utf-8")


def _read_user_auto_mode() -> Optional[dict]:
    """The ``autoMode`` object from ~/.claude/settings.json, or None (warned).

    Reads ONLY that key; the rest of the file (permissions, hooks, plugins,
    env, model, statusLine ...) is parsed to reach it and then discarded.
    Never blocks and never raises: a missing, non-regular, oversized,
    undecodable, malformed or pathologically nested file is warned and off."""
    import json

    path = _user_claude_settings_path()
    try:
        data = json.loads(_read_user_settings_text(path))
    except FileNotFoundError:
        logger.warning(
            "agent.claude_agent_sdk.auto_mode=inherit_user but %s does not "
            "exist — auto-mode rules are off for this session.", path,
        )
        return None
    except _UnusableSettingsFile as exc:
        logger.warning(
            "agent.claude_agent_sdk.auto_mode=inherit_user but %s is %s — "
            "auto-mode rules are off for this session.", path, exc,
        )
        return None
    except (OSError, UnicodeDecodeError, ValueError, RecursionError,
            MemoryError) as exc:
        logger.warning(
            "agent.claude_agent_sdk.auto_mode=inherit_user but %s could not "
            "be read as JSON (%s) — auto-mode rules are off for this session.",
            path, type(exc).__name__,
        )
        return None
    if not isinstance(data, dict):
        logger.warning(
            "agent.claude_agent_sdk.auto_mode=inherit_user but %s is not a "
            "JSON object — auto-mode rules are off for this session.", path,
        )
        return None
    auto_mode = data.get("autoMode")
    if not isinstance(auto_mode, dict):
        logger.warning(
            "agent.claude_agent_sdk.auto_mode=inherit_user but %s has no "
            "autoMode object (found %s) — auto-mode rules are off for this "
            "session.",
            path, "nothing" if auto_mode is None else type(auto_mode).__name__,
        )
        return None
    return auto_mode


def _configured_auto_mode_settings() -> Optional[str]:
    """agent.claude_agent_sdk.auto_mode from config.yaml, resolved to the SDK
    ``settings`` value: inline JSON holding ONLY ``{"autoMode": ...}``, or None.

    Accepted values:

    * ``off`` / absent / empty (default) — nothing is passed. An unquoted
      YAML ``off`` parses as boolean False and means the same.
    * ``inherit_user`` — ONLY the ``autoMode`` object from the operator's
      ``~/.claude/settings.json``.
    * a mapping — the autoMode object itself, passed through as-is.

    Every failure — unknown value, missing/unreadable/malformed settings file,
    a non-object ``autoMode``, an unserializable mapping — logs a warning and
    resolves to off: session start never fails on this key, and a typo never
    widens what reaches the CLI. The caller snapshots the result once per
    session (``__init__``) so the options stay byte-identical for the whole
    conversation; edits take effect on the next session."""
    import json

    raw = _provider_config().get("auto_mode")
    if raw is None or raw is False:
        return None
    auto_mode: Optional[dict]
    if isinstance(raw, dict):
        auto_mode = raw
    elif isinstance(raw, str):
        name = raw.strip().lower()
        if name in ("", _AUTO_MODE_OFF):
            return None
        if name != _AUTO_MODE_INHERIT_USER:
            logger.warning(
                "agent.claude_agent_sdk.auto_mode=%r is not valid (use %r, "
                "%r or an autoMode mapping) — auto-mode rules are off.",
                raw, _AUTO_MODE_OFF, _AUTO_MODE_INHERIT_USER,
            )
            return None
        auto_mode = _read_user_auto_mode()
    else:
        logger.warning(
            "agent.claude_agent_sdk.auto_mode must be %r, %r or an autoMode "
            "mapping, not %s — auto-mode rules are off.",
            _AUTO_MODE_OFF, _AUTO_MODE_INHERIT_USER, type(raw).__name__,
        )
        return None
    if not auto_mode:
        return None
    try:
        # Serialized now, so a later mutation of the source object (or an
        # edit of the file) can never reach a live session's options.
        return json.dumps({"autoMode": auto_mode}, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "agent.claude_agent_sdk.auto_mode mapping is not JSON-serializable "
            "(%s) — auto-mode rules are off.", type(exc).__name__,
        )
        return None


# ---------- stdout framing ----------
# The SDK reads the CLI's NDJSON stdout through a line framer and kills the
# whole message reader when one message exceeds max_buffer_size — a FATAL
# CLIJSONDecodeError, not a skipped message, so the turn dies mid-flight and
# the session is retired (see the retire matrix in claude_sdk_runtime). The
# SDK's own default is 1 MiB (subprocess_cli._DEFAULT_MAX_BUFFER_SIZE), which
# a single large tool result clears easily: production forensics on
# 2026-08-17 21:03 EDT caught a turn killed this way right after an Edit,
# with the CLI transcript's largest persisted line at 347 KB — the oversized
# message never reached disk.
#
# Upstream treats the option, not the default, as the fix: PR #416 proposed
# raising the default to 10 MiB and was withdrawn ("the existing
# max_buffer_size parameter already provides the needed functionality"), and
# issue #98 was closed by adding the option. So Hermes sets it explicitly
# rather than carrying an SDK patch. 10 MiB matches the figure that
# discussion converged on.
#
# NOTE: the SDK measures this with len() on a str, so the unit is Unicode
# CODE POINTS despite the "bytes" wording in its error text (upstream issue
# #1165, open). Worst-case real memory for a multibyte-heavy message is
# therefore ~4x this number — still bounded, and the point of the limit is to
# stop an unterminated line growing without end, not to be exact.
_DEFAULT_MAX_BUFFER_SIZE = 10 * 1024 * 1024


def _configured_max_buffer_size() -> int:
    """agent.claude_agent_sdk.max_buffer_size, validated.

    Same warn-and-fall-back contract as the timeout validators: bools are
    rejected before int() (YAML `true` must not become 1), and non-numeric,
    zero or negative values warn and yield the built-in default. There is
    deliberately no `0 = unlimited`: the limit is the only backstop against a
    CLI that never terminates a line, and removing it trades a killed turn
    for an OOM on a memory-constrained host."""
    raw = _provider_config().get("max_buffer_size")
    if raw is None:
        return _DEFAULT_MAX_BUFFER_SIZE
    if isinstance(raw, bool):
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is a boolean, not a "
            "size — ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is not a number — "
            "ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    if isinstance(raw, float) and not raw.is_integer():
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is not a whole number "
            "— ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    if value <= 0:
        logger.warning(
            "agent.claude_agent_sdk.max_buffer_size=%r is out of range — "
            "ignoring it (using the built-in default).", raw,
        )
        return _DEFAULT_MAX_BUFFER_SIZE
    return value


def _configured_max_turns() -> Optional[int]:
    """Resolve the canonical ``agent.max_turns`` cap for the SDK option.

    The SDK client is session-scoped, so this is called only while constructing
    a session. Unlimited Hermes values omit the option and preserve the SDK's
    own unlimited behavior; finite values become ``ClaudeAgentOptions.max_turns``.
    """
    try:
        from hermes_cli.config import TURN_LIMIT_UNLIMITED, load_config_readonly, resolve_turn_limit

        config = load_config_readonly() or {}
        raw = (config.get("agent") or {}).get("max_turns")
        value = resolve_turn_limit(raw)
        return None if value == TURN_LIMIT_UNLIMITED else value
    except Exception:
        logger.debug("claude-agent-sdk max_turns resolution failed", exc_info=True)
        return None


def _configured_cli_path() -> str:
    """agent.claude_agent_sdk.cli_path from config.yaml, validated.

    The SDK prefers its own bundled Claude Code binary over the ``claude`` on
    PATH, and that bundle lags the CLI releases by weeks — long enough that a
    freshly shipped model id is rejected with "Claude Code X does not support
    this model" while ``claude update`` on the same machine already has it.
    Pointing at the operator's launcher (``~/.local/bin/claude``) keeps the
    runtime current with that update instead of with the SDK release cadence.
    Empty/absent keeps the SDK default. A path that is not an executable file
    is dropped with a warning: a typo must never silently run a different
    binary than the one the operator asked for. (cntrl carry)"""
    raw = _provider_config().get("cli_path")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    path = os.path.expanduser(raw.strip())
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        logger.warning(
            "agent.claude_agent_sdk.cli_path %r is not an executable file — "
            "ignoring it (the SDK's bundled Claude Code CLI will be used).",
            raw,
        )
        return ""
    return path


def _configured_plugins() -> list:
    """agent.claude_agent_sdk.plugins from config.yaml, validated.

    Explicit Claude Code plugin directories to load into the spawned CLI
    (``--plugin-dir``), each ``{"type": "local", "path": ...}`` for the SDK.
    This is the isolation-preserving way to bring ONE plugin (its skills,
    agents, hooks and MCP servers) into Hermes turns: ``setting_sources``
    stays ``[]`` so the operator's whole ``~/.claude`` — every enabled plugin,
    every session-tracker hook, every MCP server, the permission allowlist —
    does not ride along underneath the configured posture. Entries that are
    not a plugin root (no ``.claude-plugin/plugin.json``) are dropped with a
    warning: a typo must never silently load nothing while looking configured."""
    raw = _provider_config().get("plugins")
    if not isinstance(raw, (list, tuple)):
        return []
    plugins: list = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            logger.warning(
                "agent.claude_agent_sdk.plugins entry %r is not a path — dropping it.",
                entry,
            )
            continue
        path = os.path.expanduser(entry.strip())
        if not os.path.isfile(os.path.join(path, ".claude-plugin", "plugin.json")):
            logger.warning(
                "agent.claude_agent_sdk.plugins entry %r is not a Claude Code plugin "
                "root (no .claude-plugin/plugin.json) — dropping it.",
                entry,
            )
            continue
        if all(existing["path"] != path for existing in plugins):
            plugins.append({"type": "local", "path": path})
    return plugins


_DEFAULT_SESSION_NAME_TEMPLATE = "hermes:{title}"
# Keep well under what Claude Code shows so ListAgents rows stay readable.
_SESSION_NAME_MAX = 60


def _configured_session_name_template() -> str:
    """agent.claude_agent_sdk.session_name — the ``--name`` template for the
    spawned Claude Code session.

    Without it the CLI derives a name from its cwd, so every Hermes session on
    a machine looks like ``justin-7`` and none can be addressed by a peer.
    Naming them makes a Hermes session a first-class target for the
    ListAgents/SendMessage pair the CLI already ships — what the cntrl
    host-router is built on. Placeholders: ``{title}`` (Hermes session title),
    ``{session}`` (short session id), ``{profile}``, ``{model}``. Set to ""
    to restore the CLI's own cwd-derived naming. (cntrl carry)"""
    raw = _provider_config().get("session_name")
    if raw is None:
        return _DEFAULT_SESSION_NAME_TEMPLATE
    if not isinstance(raw, str):
        logger.warning(
            "agent.claude_agent_sdk.session_name %r is not a string — using the default template.",
            raw,
        )
        return _DEFAULT_SESSION_NAME_TEMPLATE
    return raw


def render_sdk_session_name(
    template: str, *, title: str = "", session: str = "", profile: str = "", model: str = ""
) -> str:
    """Fill a session-name template, falling back title -> session -> profile.

    An empty template means "let the CLI name it", and so does a template whose
    placeholders all resolve empty: a bare ``hermes:`` row carries no identity
    and would be worse than the CLI's own name."""
    if not template.strip():
        return ""
    values = {
        "title": (title or "").strip(),
        "session": (session or "").strip()[-6:],
        "profile": (profile or "").strip(),
        "model": (model or "").strip(),
    }
    if not values["title"]:
        values["title"] = values["session"] or values["profile"]
    if not any(values.values()):
        return ""
    try:
        name = template.format(**values)
    except (KeyError, IndexError, ValueError):
        logger.warning(
            "agent.claude_agent_sdk.session_name %r has an unknown placeholder — using the default.",
            template,
        )
        name = _DEFAULT_SESSION_NAME_TEMPLATE.format(**values)
    return " ".join(name.split())[:_SESSION_NAME_MAX]


def _configured_timeout_seconds(key: str, *, allow_zero: bool) -> Optional[float]:
    """Numeric seconds from `agent.claude_agent_sdk.<key>`, validated.

    Same warn-and-fall-back contract as _configured_max_budget_usd: bools are
    rejected before float() (YAML `true` must not become 1.0), non-numeric and
    negative values warn and yield None (= use the built-in default). `0` is
    key-specific: allowed only where "disabled" is a documented meaning."""
    raw = _provider_config().get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is a boolean, not seconds — "
            "ignoring it (using the built-in default).", key, raw,
        )
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        # NaN would pass every range check below and silently disable the
        # rule it feeds (`idle >= nan` is never true); Inf would too.
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is not a number of seconds — "
            "ignoring it (using the built-in default).", key, raw,
        )
        return None
    if value < 0 or (value == 0 and not allow_zero):
        logger.warning(
            "agent.claude_agent_sdk.%s=%r is out of range — ignoring it "
            "(using the built-in default).", key, raw,
        )
        return None
    return value


def _configured_turn_timeout() -> Optional[float]:
    """agent.claude_agent_sdk.turn_timeout (seconds) — DEPRECATED alias of
    turn_idle_timeout, read with the same IDLE semantics when that key is
    unset (it no longer bounds wall clock). Positive only."""
    return _configured_timeout_seconds("turn_timeout", allow_zero=False)


def _configured_turn_idle_timeout() -> Optional[float]:
    """agent.claude_agent_sdk.turn_idle_timeout (seconds): retire a turn only
    after this long with no activity and nothing outstanding. Positive only —
    there is deliberately no `0 = unlimited`: an unbounded idle rule would let
    a wedged CLI hold the session until an outer ceiling notices."""
    return _configured_timeout_seconds("turn_idle_timeout", allow_zero=False)


def _configured_turn_tool_max_suspend() -> Optional[float]:
    """agent.claude_agent_sdk.turn_tool_max_suspend (seconds): how long one
    outstanding ordinary tool may suspend the idle rule. Positive only."""
    return _configured_timeout_seconds("turn_tool_max_suspend", allow_zero=False)


def _configured_turn_max_seconds() -> Optional[float]:
    """agent.claude_agent_sdk.turn_max_seconds (seconds): optional absolute
    wall-clock cap on one turn. `0`/absent = off (the default)."""
    return _configured_timeout_seconds("turn_max_seconds", allow_zero=True)


def _configured_post_tool_quiet_timeout() -> Optional[float]:
    """agent.claude_agent_sdk.post_tool_quiet_timeout (seconds).
    `0` = explicitly disabled. Absent/None = streaming-dependent default
    (90s with streaming on, disabled with streaming off)."""
    return _configured_timeout_seconds("post_tool_quiet_timeout", allow_zero=True)


def _mcp_registry_tool_specs(agent, snapshot: list) -> list:
    """MCP schemas the agent's gating allows that its snapshot is missing.

    The snapshot is missing them whenever Hermes' own tool-search deferral is
    active: ``get_tool_definitions`` collapses MCP tools behind
    ``tool_search``/``describe``/``call`` before the bridge is built, so they
    never reach the subprocess at all — no bucket carries them, and direct
    registration in ``mcp_servers`` covers neither stdio servers (no URL) nor
    servers whose URL is refused.

    That deferral is redundant on this lane. The spawned CLI runs its own tool
    search over whatever it is handed (``ToolSearch`` returning
    ``tool_reference`` entries), and unlike ours it keeps every tool
    reachable — it only decides what to put in context, which is where the
    token cost is actually paid. Ours removes the tool outright, so the
    subprocess cannot surface what it never received. Reading the
    pre-assembly schemas hands it the full surface and lets its own deferral
    do the job.

    This is a diff, not a concatenation: only ``mcp__``-prefixed names the
    snapshot does not already carry are returned, so a server the snapshot
    already covers is never registered twice.

    Gating is the agent's own — the same ``enabled_toolsets`` /
    ``disabled_toolsets`` it was built with, which is what
    ``refresh_agent_mcp_tools`` uses for the same rebuild. Reading the
    registry unfiltered would resurrect a server the operator disabled for
    this platform, and registry dispatch is not allowlist-gated.

    Never raises: any failure yields ``[]`` and leaves the bridge exactly as
    it would have been.
    """
    # Honour the same opt-out the snapshot refresh honours: background review
    # sets `_skip_mcp_refresh` deliberately, and reading the registry here
    # would walk straight past it.
    if getattr(agent, "_skip_mcp_refresh", False):
        return []

    try:
        from tools.mcp_tool_discovery import has_registered_mcp_tools

        if not has_registered_mcp_tools():
            return []
    except Exception:
        logger.debug("MCP registry probe failed", exc_info=True)
        return []

    try:
        from model_tools import get_tool_definitions

        from agent.transports.hermes_tool_exposure import normalize_tool_spec

        specs = get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True,
            # Pre-assembly schemas: tool_search would otherwise hand back its
            # tier-2 placeholder and the bridge would register that instead
            # of the real tool.
            skip_tool_search_assembly=True,
        )
    except Exception:
        logger.debug("MCP registry tool-definition read failed", exc_info=True)
        return []

    def _name(spec: Any) -> Optional[str]:
        try:
            normalized = normalize_tool_spec(spec, strip_mcp_prefix=False)
        except Exception:
            return None
        return normalized[0] if normalized else None

    seen = {n for n in (_name(s) for s in (snapshot or [])) if n}
    out: list = []
    for spec in specs or []:
        name = _name(spec)
        if not name or not name.startswith("mcp__") or name in seen:
            continue
        seen.add(name)
        out.append(spec)
    return out


def _http_mcp_entries_from_config() -> dict[str, dict]:
    """Return explicitly opted-in, credential-free HTTP MCPs for the SDK.

    Direct registration shares the ``hybrid_mcp_bridge`` opt-in and exclusion
    list. Header-bearing, templated, userinfo-bearing, and credential-like
    query or fragment URLs are refused because the SDK serializes this config
    into the Claude CLI's ``--mcp-config`` process argument. Returns ``{}`` on
    any read/parse failure and never logs a URL or credential value.
    """
    import os as _os
    import re as _re
    from urllib.parse import unquote as _unquote, urlsplit as _urlsplit

    try:
        import hermes_yaml as _yaml
    except Exception:
        return {}

    home = _os.environ.get("HERMES_HOME") or _os.path.expanduser("~/.hermes")
    profile = _os.environ.get("HERMES_PROFILE") or "default"
    paths = [
        _os.path.join(home, "config.yaml"),
        _os.path.join(home, "profiles", profile, "config.yaml"),
    ]

    merged: dict[str, Any] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = _yaml.safe_load(fh) or {}
            block = (data.get("mcp_servers") or {}) if isinstance(data, dict) else {}
            if isinstance(block, dict):
                merged.update(block)
        except FileNotFoundError:
            continue
        except Exception:
            logger.debug("failed to read %s for HTTP MCP discovery", p, exc_info=True)
            continue

    if not _provider_flag("hybrid_mcp_bridge", default=False):
        return {}

    _env_re = _re.compile(r"\$\{[^}]*\}")
    _secret_field_re = _re.compile(
        r"(?:api[_-]?key|token|secret|password|credential|authorization|auth)",
        _re.I,
    )
    excluded_names = set(_configured_hybrid_exclude())

    out: dict[str, dict] = {}
    for raw_name, cfg in merged.items():
        if not isinstance(cfg, dict):
            continue
        url = cfg.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        name = str(raw_name)
        url = url.strip()
        if name in excluded_names:
            logger.info("claude-agent-sdk: HTTP MCP %r excluded by config", name)
            continue
        if cfg.get("headers"):
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because it has headers",
                name,
            )
            continue
        if _env_re.search(url):
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is templated",
                name,
            )
            continue
        try:
            parsed = _urlsplit(url)
        except ValueError:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is malformed",
                name,
            )
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL is not HTTP(S)",
                name,
            )
            continue
        # Direct HTTP MCP configuration is passed to the SDK/CLI process.
        # Treat every query, fragment, or userinfo component as secret-bearing:
        # a denylist cannot safely distinguish signed/public query parameters
        # from encoded credentials (including nested percent encoding).
        if parsed.username is not None or parsed.query or parsed.fragment:
            logger.warning(
                "claude-agent-sdk: refusing HTTP MCP %r because its URL has userinfo, query, or fragment",
                name,
            )
            continue
        out[name] = {"type": "http", "url": url}
    return out


def _hermes_repo_root() -> str:
    """Repo root for the hermes-tools MCP subprocess (PYTHONPATH)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


# The SDK spawns the Claude Code CLI with the FULL inherited parent env and
# merges ``ClaudeAgentOptions.env`` ON TOP of it (subprocess_cli.py builds
# ``{**os.environ, ..., **options.env, ...}``), so a key can never be REMOVED
# from the options side — the only lever is an explicit override. Every
# metered billing vector below is overridden to "" (the CLI and the AWS/GCP
# SDKs treat an empty value as unset), so a credentialed gateway environment
# cannot silently re-route this provider's billing off the Claude
# subscription. Why each class of key is stripped:
#
#   ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN — the CLI prefers these over
#     subscription OAuth: the exact silent-rebilling the fail-closed startup
#     guard exists to stop. The guard covers Hermes' own process at startup;
#     this scrub covers the spawned CLI, which otherwise inherits them.
#   CLAUDE_CODE_USE_BEDROCK + AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
#     AWS_SESSION_TOKEN — flips the CLI onto AWS Bedrock, billing the AWS
#     account (metered) instead of the subscription; the AWS static
#     credentials are the vector that makes the flip actually authenticate.
#   CLAUDE_CODE_USE_VERTEX + GOOGLE_APPLICATION_CREDENTIALS — the same
#     takeover via Google Vertex, billing the GCP project.
#
# Deliberately NOT stripped: HOME/PATH (the CLI needs them to run and to find
# its credential store) and the subscription token flow itself —
# CLAUDE_CODE_OAUTH_TOKEN / ~/.claude — which is what this provider runs on.
_METERED_ENV_DENYLIST = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _is_subscription_oauth_token(value: str) -> bool:
    """True when an ANTHROPIC_TOKEN value is OAuth/setup-token shaped — the
    subscription lane itself, not a metered vector. Unknown shapes count as
    metered (fail closed), including when the classifier cannot be imported."""
    try:
        from agent.anthropic_adapter import _is_oauth_token
    except Exception:
        return False
    try:
        return bool(_is_oauth_token(value))
    except Exception:
        return False


def _is_metered_sdk_env_value(key: str, value: str) -> bool:
    """Whether an SDK child env value can switch billing off subscription.

    ``ANTHROPIC_TOKEN`` is ambiguous in Hermes: a setup/OAuth token is the
    desired subscription credential, while every unrecognised shape is
    treated as metered.  The other denylisted variables are always metered
    routing vectors when non-empty.
    """
    if key not in _METERED_ENV_DENYLIST or not value:
        return False
    if key == "ANTHROPIC_TOKEN":
        return not _is_subscription_oauth_token(value)
    return True


def _scrubbed_sdk_env() -> dict[str, str]:
    """Empty-string overrides for every metered billing vector currently set
    in the parent environment. Only PRESENT keys are overridden — writing
    ``""`` for absent ones would introduce empty vars the child never had
    (an empty AWS_ACCESS_KEY_ID can itself confuse AWS credential chains)."""
    return {
        key: ""
        for key in _METERED_ENV_DENYLIST
        if _is_metered_sdk_env_value(key, os.environ.get(key, ""))
    }


# Interpreter-path vectors that must not reach the spawned CLI. The desktop
# backend runs with PYTHONPATH=<repo>:<venv>/lib/python3.11/site-packages
# (apps/desktop/electron/main.ts adds the venv path so a system python can
# import hermes_cli). The SDK transport merges os.environ into the CLI child,
# and the CLI hands its env to every plugin MCP server and hook it spawns — so
# a plugin's own `uv run --python >=3.12` interpreter imported 3.11-built C
# extensions (pydantic_core) from Hermes' site-packages and died on the ABI
# mismatch: conductor's tb-workers "Connection closed", cached by Claude Code
# for 15 min per process (~/.claude/mcp-needs-auth-cache.json). Nothing under
# the CLI needs Hermes' import path: the hermes-tools MCP runs on
# sys.executable, where Hermes is editable-installed. Proven 2026-09-11: the
# same command with PYTHONPATH unset connected in 1.4s. Same class of fix as
# tools/browser_use_cli.py::_base_subprocess_env.
_CHILD_INTERPRETER_ENV_DENYLIST = ("PYTHONPATH", "PYTHONHOME")


def _scrubbed_interpreter_env() -> dict[str, str]:
    """Empty-string overrides for PYTHONPATH/PYTHONHOME when the parent has them
    set. Only PRESENT keys are overridden; "" is how the SDK's ``{**os.environ,
    **options.env}`` merge can unset a key, and CPython treats an empty
    PYTHONPATH exactly like an absent one (verified: no path entries added)."""
    return {
        key: ""
        for key in _CHILD_INTERPRETER_ENV_DENYLIST
        if os.environ.get(key)
    }


# The desktop app is launched by `npm run`, so its npm_config_* / npm_package_*
# describe the live checkout and are stale for anything an agent runs (same list
# as apps/desktop/electron/terminal-ipc.ts). This is hygiene, not a fix for a
# known failure.
_CHILD_NPM_ENV_PREFIXES = ("npm_config_", "npm_package_")


def _scrubbed_npm_env() -> dict[str, str]:
    """Empty-string overrides for npm variables inherited from `npm run`. Only
    PRESENT keys are overridden; "" is how the SDK's ``{**os.environ,
    **options.env}`` merge neutralises an inherited key."""
    return {
        key: ""
        for key in list(os.environ)
        if key == "npm_config_prefix" or key.startswith(_CHILD_NPM_ENV_PREFIXES)
    }


# The SDK serializes the stdio MCP config — env INCLUDED — into the claude
# CLI's --mcp-config argument, i.e. onto the subprocess argv, which any local
# user can read via ps. Nothing secret may ever ride this dict: the env is a
# minimal ALLOWLIST, never a copy of the credentialed environment. Keyed
# Hermes tools inside the server degrade via their own check_fns — the
# subscription lane's fail-closed posture.
_MCP_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "PYTHONUTF8",
    "HERMES_HOME",
    "HERMES_KANBAN_TASK",
    "HERMES_MCP_STATE_DB",  # the shims' documented state-DB override — a path, not a secret
    "HERMES_QUIET",
    "HERMES_REDACT_SECRETS",
)


def _provider_config() -> dict:
    """The `agent.claude_agent_sdk` config block ({} when absent/unreadable)."""
    try:
        from hermes_cli.config import load_config_readonly

        block = ((load_config_readonly() or {}).get("agent", {}) or {}).get(
            "claude_agent_sdk", {}
        )
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def _provider_flag(config_key: str, default: bool = False) -> bool:
    """Behavioural flag read from `agent.claude_agent_sdk.<key>` in config.yaml.

    config.yaml is the ONLY interface. AGENTS.md keeps non-secret behavioural
    settings out of `HERMES_*` environment variables, so there is deliberately
    no env override here — a deployment sets the key in config.yaml.
    Canonical defaults live in `hermes_cli/config.py::DEFAULT_CONFIG`.
    """
    value = _provider_config().get(config_key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


def _configured_hybrid_exclude() -> list:
    """agent.claude_agent_sdk.hybrid_mcp_bridge_exclude from config.yaml.

    Names to drop from the hybrid bridge (both buckets). Match on the raw
    Hermes registry name, no ``mcp__`` prefix. Non-string entries are
    dropped silently — a typo is a config error the operator will notice
    when the tool doesn't disappear, not a reason to widen exposure.
    """
    raw = _provider_config().get("hybrid_mcp_bridge_exclude")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        name = entry.strip()
        if name and name not in out:
            out.append(name)
    return out


def _build_hermes_tools_mcp_config(
    hermes_session_id: Optional[str] = None,
) -> dict[str, Any]:
    """The stdio MCP server exposing Hermes tools into the SDK agent loop —
    the exact server the codex runtime uses (backend-agnostic), launched with
    this venv's interpreter. McpStdioServerConfig has no cwd field, so the
    repo root rides PYTHONPATH."""
    env = {
        key: os.environ[key]
        for key in _MCP_ENV_ALLOWLIST
        if os.environ.get(key)
    }
    # Multiplexed gateway turns bind HERMES_HOME through a context override,
    # not process-wide os.environ; the child must discover the same profile's
    # owner lease and state registry.
    # Claude CLI may resolve the venv's python symlink before respawning this
    # server. Preserve the active venv's packages for that base interpreter,
    # without carrying a desktop or project PYTHONPATH into the MCP child.
    python_paths = [_hermes_repo_root()]
    for key in ("purelib", "platlib"):
        site_packages = sysconfig.get_paths().get(key)
        if site_packages and os.path.isdir(site_packages) and site_packages not in python_paths:
            python_paths.append(site_packages)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if hermes_session_id is None:
        try:
            from gateway.session_context import _SESSION_ID, _UNSET
            val = _SESSION_ID.get()
            if val is not _UNSET and val:
                hermes_session_id = str(val)
        except Exception:
            pass
    if hermes_session_id:
        # Lets the stateless session_search shim exclude the calling
        # session's own lineage from recall results (#26567). The shim reads
        # the canonical HERMES_SESSION_ID — set explicitly with THIS
        # session's id rather than allowlisting the ambient variable, so a
        # multi-session host can never leak a sibling session's id into the
        # subprocess.
        env["HERMES_SESSION_ID"] = str(hermes_session_id)
        spawn_policy = _provider_config().get("session_spawn")
        spawn_enabled = bool(spawn_policy.get("enabled", True)) if isinstance(spawn_policy, dict) else True
        # cntrl carry: session_send (durable peer messaging) rides the same scoped capability.
        send_policy = _provider_config().get("session_send")
        send_enabled = bool(send_policy.get("enabled", True)) if isinstance(send_policy, dict) else True
        if spawn_enabled or send_enabled:
            from agent.transports.hermes_gateway_session_bridge import capability_file_path, issue_scoped_capability
            if capability := issue_scoped_capability(str(hermes_session_id)):
                from hermes_constants import get_hermes_home
                path = capability_file_path(str(hermes_session_id), get_hermes_home())
                _publish_session_spawn_capability(path, capability)
                env["HERMES_HOME"] = str(get_hermes_home())
                env["HERMES_SESSION_SPAWN_CAPABILITY_FILE"] = str(path)
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [
            # -P: never put the CLI's cwd on sys.path. A session working in a
            # Hermes checkout (a lane worktree) would otherwise import that
            # tree's bootstrap, which re-roots dependencies onto the lane's
            # own venv and drops these site-packages (#66).
            "-P",
            "-m",
            "agent.transports.hermes_tools_mcp_server",
            "--profile",
            "claude-agent-sdk",
        ],
        "env": env,
    }


def _publish_session_spawn_capability(path: Path, value: str) -> None:
    """Publish one capability with private directory/file permissions and no symlink follow."""
    path.parent.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.mkdir(exist_ok=True, mode=0o700)
    getuid = getattr(os, "getuid", None)  # absent on Windows, where st_uid carries no ownership
    owner = getuid() if getuid is not None else None
    for directory in (path.parent.parent, path.parent):
        parent_st = os.lstat(directory)
        if not stat.S_ISDIR(parent_st.st_mode) or (owner is not None and parent_st.st_uid != owner):
            raise PermissionError("session-spawn capability directory is not owner-controlled")
        os.chmod(directory, 0o700, follow_symlinks=False)
    with contextlib.suppress(FileNotFoundError):
        path_st = os.lstat(path)
        if not stat.S_ISREG(path_st.st_mode) or (owner is not None and path_st.st_uid != owner):
            raise PermissionError("session-spawn capability path is not owner-controlled")
    # Write a private sibling and rename it over the path: the MCP child re-reads this file on every
    # call, and turn-start rotation must never let it observe a missing or half-written capability.
    staging = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(staging, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    except Exception:
        with contextlib.suppress(OSError):
            staging.unlink()
        raise


def refresh_session_spawn_capability(hermes_session_id: Optional[str], path: Any) -> Optional[str]:
    """Rotate the capability published at ``path`` for a live SDK session (b3-30).

    The capability used to be issued once at CLI spawn and expired an hour later with nothing
    re-issuing it, so every SDK session older than an hour got HTTP 401 on session_send.  Called at
    each SDK turn start: publish a fresh token in place (the MCP child re-reads the file per call),
    then retire the previous one after a short grace so an in-flight call is not cut."""
    if not hermes_session_id or not path:
        return None
    from pathlib import Path as _Path

    from agent.transports.hermes_gateway_session_bridge import issue_scoped_capability, retire_scoped_capability
    target = _Path(path)
    previous = ""
    with contextlib.suppress(OSError):
        previous = target.read_text(encoding="utf-8").strip()
    token = issue_scoped_capability(str(hermes_session_id))
    if not token:
        return None
    try:
        _publish_session_spawn_capability(target, token)
    except Exception as exc:  # noqa: BLE001 - keep the previous capability serving
        # Never revoke_scoped_capability() here: it also unlinks the file that still holds `previous`.
        retire_scoped_capability(token, grace_seconds=0)
        logger.warning("session-spawn capability refresh failed for %s: %s", hermes_session_id, exc)
        return None
    if previous and previous != token:
        retire_scoped_capability(previous)
    return token
