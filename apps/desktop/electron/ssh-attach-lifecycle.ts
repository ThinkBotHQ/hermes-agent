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
  `lockdir="\${HERMES_GATEWAY_LOCK_DIR:-\${XDG_STATE_HOME:-\$HOME/.local/state}/hermes/gateway-locks}"\n` +
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

export async function detach(
  ssh: Pick<AttachSshConnection, 'cancelForward'>,
  handle: { localPort: number; remotePort: number }
): Promise<void> {
  if (handle?.localPort && handle?.remotePort) {
    await ssh.cancelForward(handle.localPort, handle.remotePort)
  }
}
