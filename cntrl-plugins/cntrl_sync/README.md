# cntrl_sync — Hermes plugin

One-way sync between the cntrl bizops board and one user's Hermes board. Design and
mapping tables: `docs/cntrl/kanban-one-way-sync.md`.

- **Down** (pull loop): cntrl tasks with the `hermes` label and `category != bizops` are created
  on the Hermes board (`idempotency_key = cntrl:<uuid>`), refreshed when title/description/
  priority change, and closed when cntrl closes them.
- **Up** (hooks): `kanban_task_completed`, `kanban_task_blocked`, and a status change in
  `on_kanban_task_updated` move the cntrl task and leave one comment on done/blocked.

## Install (out-of-tree)

```bash
ln -s "$PWD/cntrl-plugins/cntrl_sync" "$HERMES_HOME/plugins/cntrl_sync"
```

`config.yaml`:

```yaml
plugins:
  enabled: [cntrl_sync]
  entries:
    cntrl_sync:
      settings:
        base_url: http://127.0.0.1:3101
        api_key_env: CNTRL_API_KEY      # tb_… key, minted by POST /api/api-keys
        poll_seconds: 30                # 0 = hooks only
        label: hermes
        landing: triage                 # or ready (dispatcher may run it)
        profile_map:
          "<cntrl user id or email>": default
```

Put the key in `$HERMES_HOME/.env` as `CNTRL_API_KEY=tb_…`.

While cntrl is unreachable the pull loop logs one warning, then waits twice as
long after each failed pull (up to 15 minutes), and logs once when a pull
succeeds again.

## Run one pull by hand

```bash
HERMES_HOME=$PWD/.hermes-test .venv/bin/python -c "
import sys; sys.path.insert(0,'cntrl-plugins')
from cntrl_sync import core
print(core.run_once(core.FileState('.hermes-test/cntrl_sync_state.json')))"
```

## Tests

```bash
.venv/bin/python -m pytest -q cntrl-plugins/cntrl_sync/tests
```
