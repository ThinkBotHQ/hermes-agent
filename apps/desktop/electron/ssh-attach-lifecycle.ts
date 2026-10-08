/**
 * ssh-attach-lifecycle.ts
 *
 * Attach-only lifecycle for SSH remote mode in Hermes Desktop.
 *
 * Unlike standard SSH remote mode (which manages, spawns with --isolated,
 * and terminates a dedicated remote backend), attach mode connects to a
 * persistent `hermes serve` already running under a host supervisor (e.g. systemd).
 *
 * It reads the host rendezvous records published by the server in:
 *   ${HERMES_GATEWAY_LOCK_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/hermes/gateway-locks}
 *
 * Specifically:
 *   - host-serve.json: port, pid, role ('serve'), tokenFingerprint
 *   - host-serve.token: 0600 live session token
 *
 * Lifecycle operations:
 *   - attach(): validates record shape & role ('serve', not 'desktop-serve'),
 *     verifies token fingerprint, forwards a local port to the remote port,
 *     verifies owner identity via GET /api/host/identity, and waits for readiness.
 *   - detach(): cancels the local port forward. NEVER terminates or kills
 *     the remote process.
 */

import crypto from 'node:crypto'

import { sshAttachDialIdentity } from './connection-registry'
import { pickLocalPort as defaultPickLocalPort } from './ssh-connection'

export class AttachNoBackendError extends Error {
  readonly kind = 'attach-no-backend'
  constructor(message = 'Host supervisor is not running hermes serve (host-serve.json / host-serve.token missing)') {
    super(message)
    this.name = 'AttachNoBackendError'
  }
}

export class AttachTokenMismatchError extends Error {
  readonly kind = 'attach-token-mismatch'
  constructor(message = 'Host serve token fingerprint does not match host-serve.json') {
    super(message)
    this.name = 'AttachTokenMismatchError'
  }
}

export class AttachIdentityError extends Error {
  readonly kind = 'attach-identity-error'
  constructor(message = 'Host identity probe failed or PID does not match') {
    super(message)
    this.name = 'AttachIdentityError'
  }
}

export function tokenFingerprint(token: string): string {
  if (!token) {
    return ''
  }

  return crypto.createHash('sha256').update(token, 'utf8').digest('hex').slice(0, 16)
}

const ATTACH_DELIM = '__HERMES_ATTACH_DELIM__'
const ATTACH_MISSING = '__HERMES_ATTACH_MISSING__'

const READ_HOST_SERVE_CMD =
  `lockdir="\${HERMES_GATEWAY_LOCK_DIR:-\${XDG_STATE_HOME:-$HOME/.local/state}/hermes/gateway-locks}"\n` +
  `if [ ! -f "$lockdir/host-serve.json" ] || [ ! -f "$lockdir/host-serve.token" ]; then\n` +
  `  echo "${ATTACH_MISSING}"\n` +
  `  exit 0\n` +
  `fi\n` +
  `cat "$lockdir/host-serve.json"\n` +
  `printf '\\n${ATTACH_DELIM}\\n'\n` +
  `cat "$lockdir/host-serve.token"`

export interface AttachSshConnection {
  exec: (cmd: string) => Promise<string>
  forward: (localPort: number, remotePort: number, remoteHost?: string) => Promise<void>
  cancelForward: (localPort: number, remotePort: number, remoteHost?: string) => Promise<void>
}

export interface AttachOptions {
  pickLocalPort?: () => unknown
  waitForHermes?: (baseUrl: string, token: string) => Promise<void>
  fetchFn?: typeof fetch
  signal?: AbortSignal
}

export interface AttachResult {
  baseUrl: string
  token: string
  localPort: number
  remotePort: number
  pid: number
  attachOnly: boolean
}

export async function attach(ssh: AttachSshConnection, opts: AttachOptions = {}): Promise<AttachResult> {
  let output = ''

  try {
    output = await ssh.exec(READ_HOST_SERVE_CMD)
  } catch (error: any) {
    throw new AttachNoBackendError(
      `Host supervisor is not running hermes serve (failed to read host-serve files: ${error?.message || error})`
    )
  }

  const trimmed = output.trim()

  if (!trimmed || trimmed.includes(ATTACH_MISSING) || !trimmed.includes(ATTACH_DELIM)) {
    throw new AttachNoBackendError(
      'Host supervisor is not running hermes serve (host-serve.json or host-serve.token missing)'
    )
  }

  const parts = trimmed.split(ATTACH_DELIM)
  const rawJson = (parts[0] || '').trim()
  const rawToken = (parts[1] || '').trim()

  if (!rawJson || !rawToken) {
    throw new AttachNoBackendError(
      'Host supervisor is not running hermes serve (empty host-serve.json or host-serve.token)'
    )
  }

  let record: any

  try {
    record = JSON.parse(rawJson)
  } catch {
    throw new AttachNoBackendError('Host supervisor is not running hermes serve (host-serve.json is invalid JSON)')
  }

  if (typeof record !== 'object' || record === null) {
    throw new AttachNoBackendError('Host supervisor is not running hermes serve (host-serve.json is invalid)')
  }

  if (record.role !== 'serve') {
    throw new AttachNoBackendError(
      `Host supervisor is not running hermes serve (role is "${record.role}", expected "serve")`
    )
  }

  const pid = Number(record.pid)
  const port = Number(record.port)

  if (!Number.isInteger(pid) || pid <= 0 || !Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new AttachNoBackendError(
      'Host supervisor is not running hermes serve (invalid pid or port in host-serve.json)'
    )
  }

  const expectedFingerprint = String(record.tokenFingerprint || '')
  const computedFingerprint = tokenFingerprint(rawToken)

  if (!expectedFingerprint || expectedFingerprint !== computedFingerprint) {
    throw new AttachTokenMismatchError(
      `Host serve token fingerprint does not match host-serve.json (expected "${expectedFingerprint}", got "${computedFingerprint}")`
    )
  }

  const localPort = Number(opts.pickLocalPort ? await opts.pickLocalPort() : await defaultPickLocalPort())
  await ssh.forward(localPort, port)
  const baseUrl = `http://127.0.0.1:${localPort}`

  try {
    const fetchFn = opts.fetchFn || fetch

    const response = await fetchFn(`${baseUrl}/api/host/identity`, {
      method: 'GET',
      headers: {
        'X-Hermes-Session-Token': rawToken,
        'X-Hermes-Token': rawToken,
        Authorization: `Bearer ${rawToken}`
      },
      signal: opts.signal
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const identity = (await response.json()) as any

    if (!identity || typeof identity !== 'object' || identity.pid !== pid) {
      throw new Error(`Identity PID mismatch: expected ${pid}, got ${identity?.pid}`)
    }
  } catch (err: any) {
    try {
      await ssh.cancelForward(localPort, port)
    } catch {
      // Best-effort forward cleanup on probe failure
    }

    throw new AttachIdentityError(
      `Host identity probe failed or PID does not match: ${err instanceof Error ? err.message : String(err)}`
    )
  }

  if (opts.waitForHermes) {
    await opts.waitForHermes(baseUrl, rawToken)
  }

  return {
    baseUrl,
    token: rawToken,
    localPort,
    remotePort: port,
    pid,
    attachOnly: true
  }
}

export interface AttachWhenReadyOptions extends AttachOptions {
  /** How long to keep trying while the host has no backend yet. */
  waitMs?: number
  intervalMs?: number
  sleep?: (ms: number) => Promise<void>
  now?: () => number
}

/**
 * attach(), for a host that was just started. A freshly booted machine answers
 * ssh a few seconds before its supervisor has `hermes serve` up, and attach()
 * reports that window as "no backend". Retry only that error, for a bounded
 * time; every other failure (token mismatch, identity, ssh) surfaces at once.
 * Still never starts anything on the host.
 */
export async function attachWhenReady(ssh: AttachSshConnection, opts: AttachWhenReadyOptions = {}): Promise<AttachResult> {
  const { intervalMs = 2000, now = Date.now, sleep, waitMs = 45_000, ...attachOpts } = opts
  const signal = attachOpts.signal
  const pause = sleep ?? ((ms: number) => abortableDelay(ms, signal))
  const deadline = now() + waitMs
  let last: unknown = null

  for (;;) {
    // A cancelled bootstrap must not start another read on the host.
    if (signal?.aborted) {
      throw last ?? new AttachNoBackendError('Attach was cancelled before the host backend came up')
    }

    try {
      return await attach(ssh, attachOpts)
    } catch (error) {
      last = error

      if (!(error instanceof AttachNoBackendError) || signal?.aborted || now() + intervalMs > deadline) {
        throw error
      }

      await pause(intervalMs)
    }
  }
}

// Resolves after `ms`, or at once when the signal aborts: a cancelled wait must
// not hold its bootstrap (and whatever is queued behind it) for the full pause.
function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    if (signal?.aborted) {
      resolve()

      return
    }

    const done = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', done)
      resolve()
    }

    const timer = setTimeout(done, ms)

    signal?.addEventListener('abort', done, { once: true })
  })
}

/**
 * An attach-only connection whose host does not answer has nothing it can do
 * about it unless it knows the Fly app to start. Say so, instead of leaving
 * the generic ssh timeout text as the only clue.
 */
export function attachTimeoutHint(
  message: string,
  context: { flyApp?: string; isAttach?: boolean; kind?: string }
): string {
  if (!context.isAttach || context.flyApp || context.kind !== 'timeout') {
    return message
  }

  return `${message} If this host is a Fly machine it may be stopped: set "Fly app" on this connection and Hermes will start it.`
}

export interface AttachStateCandidate {
  dialIdentity?: string
  kind?: string
  registryConnectionId?: string
}

/**
 * Which ssh state a re-attach may use: the connection's own scope, else the
 * ssh-attach state registered for the same registry connection id. Never
 * another connection's state. Borrowing the primary's would read a different
 * host's rendezvous files and hand this connection that host's base URL and
 * token.
 */
export function resolveAttachState<T extends AttachStateCandidate>(
  states: Map<string, T>,
  scope: string,
  connectionId?: null | string,
  expectedDialIdentity?: string
): [string, T] | null {
  const own = states.get(scope)

  if (own && own.kind === 'ssh-attach' &&
      (expectedDialIdentity === undefined || own.dialIdentity === expectedDialIdentity)) {
    return [scope, own]
  }

  if (connectionId) {
    for (const [key, candidate] of states) {
      if (candidate.kind === 'ssh-attach' && candidate.registryConnectionId === connectionId &&
          (expectedDialIdentity === undefined || candidate.dialIdentity === expectedDialIdentity)) {
        return [key, candidate]
      }
    }
  }

  return null
}

export function planReattach<T extends AttachStateCandidate>(input: {
  connectionId?: null | string
  registryEntry?: null | { flyApp?: string; host: string; id?: string; kind?: string; port?: number; user?: string }
  scope: string
  states: Map<string, T>
}): { ok: true; scope: string; state: T } | { ok: false; reason: 'state-unavailable' } {
  if (input.connectionId && !input.registryEntry) {
    return { ok: false, reason: 'state-unavailable' }
  }

  const identity = input.registryEntry ? sshAttachDialIdentity(input.registryEntry) : undefined
  const resolved = resolveAttachState(input.states, input.scope, input.connectionId, identity)

  if (!resolved) {
    return { ok: false, reason: 'state-unavailable' }
  }

  return { ok: true, scope: resolved[0], state: resolved[1] }
}

/**
 * Every ssh scope whose attach state belongs to a registry connection. The
 * primary's state lives at the bare primary scope rather than under
 * `conn:<id>::`, so a prefix match alone misses it when the connection is
 * removed or re-pointed.
 */
export function attachScopesOwnedBy<T extends AttachStateCandidate>(
  states: Map<string, T>,
  connectionId: string
): string[] {
  if (!connectionId) {
    return []
  }

  return [...states]
    .filter(([, state]) => state.kind === 'ssh-attach' && state.registryConnectionId === connectionId)
    .map(([scope]) => scope)
}

export async function detach(
  ssh: Pick<AttachSshConnection, 'cancelForward'>,
  handle: { localPort: number; remotePort: number }
): Promise<void> {
  if (handle?.localPort && handle?.remotePort) {
    await ssh.cancelForward(handle.localPort, handle.remotePort)
  }
}

export async function reattach(
  ssh: AttachSshConnection,
  handle: { localPort: number; remotePort: number },
  opts: AttachOptions = {}
): Promise<AttachResult> {
  await detach(ssh, handle)

  return attach(ssh, opts)
}
