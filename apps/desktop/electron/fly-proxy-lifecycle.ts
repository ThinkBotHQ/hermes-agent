import { execFile, spawn } from 'node:child_process'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { pickLocalPort as defaultPickLocalPort } from './ssh-connection'

export class FlyCliMissingError extends Error {
  readonly kind = 'fly-cli-missing'

  constructor(message = 'The Fly CLI (fly) is not installed. Install it with: brew install flyctl') {
    super(message)
    this.name = 'FlyCliMissingError'
  }
}

export class FlyAuthError extends Error {
  readonly kind = 'fly-auth'

  constructor(message = 'The Fly CLI is not logged in. Run: fly auth login') {
    super(message)
    this.name = 'FlyAuthError'
  }
}

function firstLineOf(text: string): string {
  const line = text
    .split(/\r?\n/)
    .map(l => l.trim())
    .find(l => l.length > 0)

  return line || ''
}

export class FlyMachineError extends Error {
  readonly kind = 'fly-machine'

  constructor(message: string, stderr?: string) {
    const firstLine = stderr ? firstLineOf(stderr) : ''
    const fullMessage = firstLine && !message.includes(firstLine) ? `${message}: ${firstLine}` : message

    super(fullMessage)
    this.name = 'FlyMachineError'
  }
}

export class FlyProxyError extends Error {
  readonly kind = 'fly-proxy'

  constructor(message: string, stderr?: string) {
    const firstLine = stderr ? firstLineOf(stderr) : ''
    const fullMessage = firstLine && !message.includes(firstLine) ? `${message}: ${firstLine}` : message

    super(fullMessage)
    this.name = 'FlyProxyError'
  }
}

const FLY_APP_RE = /^[a-z0-9][a-z0-9-]{0,62}$/

export function validateFlyApp(name: string): string {
  const trimmed = typeof name === 'string' ? name.trim() : ''

  if (!FLY_APP_RE.test(trimmed)) {
    throw new Error(
      `Invalid Fly app name "${name}". Fly app names must be 1-63 characters, start with a lowercase letter or digit, and contain only lowercase letters, digits, and hyphens.`
    )
  }

  return trimmed
}

export function flyHostKeyAlias(app: string): string {
  const valid = validateFlyApp(app)

  return `fly-proxy.${valid}`
}

export interface ResolveFlyBinaryDeps {
  env?: NodeJS.ProcessEnv
  existsSync?: (filePath: string) => boolean
  homedir?: () => string
}

export function resolveFlyBinary(deps?: ResolveFlyBinaryDeps): string {
  const env = deps?.env ?? process.env
  const existsSync = deps?.existsSync ?? fs.existsSync
  const home = deps?.homedir ? deps.homedir() : os.homedir()

  const rawPath = env.PATH || ''
  const searchDirs: string[] = []

  for (const dir of rawPath.split(path.delimiter)) {
    const trimmed = dir.trim()

    if (trimmed && !searchDirs.includes(trimmed)) {
      searchDirs.push(trimmed)
    }
  }

  const commonDirs = ['/opt/homebrew/bin', '/usr/local/bin', path.join(home, '.fly', 'bin')]

  for (const dir of commonDirs) {
    if (!searchDirs.includes(dir)) {
      searchDirs.push(dir)
    }
  }

  const names =
    process.platform === 'win32' ? ['fly.exe', 'fly', 'flyctl.exe', 'flyctl'] : ['fly', 'flyctl']

  for (const dir of searchDirs) {
    for (const name of names) {
      const candidate = path.join(dir, name)

      try {
        if (existsSync(candidate)) {
          return candidate
        }
      } catch {
        // ignore errors
      }
    }
  }

  throw new FlyCliMissingError()
}

const AUTH_RE = /no access token|not logged in|auth login|unauthorized/i

function classifyFlyCliError(err: any, fallbackMessage: string): never {
  if (err?.code === 'ENOENT') {
    throw new FlyCliMissingError()
  }

  const stderr = String(err?.stderr || '')
  const message = String(err?.message || '')

  if (AUTH_RE.test(stderr) || AUTH_RE.test(message)) {
    throw new FlyAuthError()
  }

  throw new FlyMachineError(fallbackMessage, stderr || message)
}

export interface EnsureFlyMachineStartedOptions {
  binaryDeps?: ResolveFlyBinaryDeps
  execFile?: typeof execFile
  flyBinary?: string
  onStatus?: (status: 'starting-host') => void
  signal?: AbortSignal
}

export interface EnsureFlyMachineResult {
  machineId: string
  started: boolean
}

export async function ensureFlyMachineStarted(
  app: string,
  opts: EnsureFlyMachineStartedOptions = {}
): Promise<EnsureFlyMachineResult> {
  const validApp = validateFlyApp(app)
  const flyBinary = opts.flyBinary || resolveFlyBinary(opts.binaryDeps)
  const execFileFn = opts.execFile || execFile

  let listStdout = ''
  let listStderr = ''

  try {
    const res = await new Promise<{ stderr: string; stdout: string }>((resolve, reject) => {
      execFileFn(
        flyBinary,
        ['machines', 'list', '-a', validApp, '--json'],
        { maxBuffer: 10 * 1024 * 1024, signal: opts.signal },
        (err, stdout, stderr) => {
          if (err) {
            ;(err as any).stdout = stdout
            ;(err as any).stderr = stderr
            reject(err)
          } else {
            resolve({ stderr: String(stderr || ''), stdout: String(stdout || '') })
          }
        }
      )
    })

    listStdout = res.stdout
    listStderr = res.stderr
  } catch (err: any) {
    classifyFlyCliError(err, 'Failed to list machines')
  }

  let machines: any[] = []

  try {
    machines = JSON.parse(listStdout)
  } catch {
    throw new FlyMachineError(`Failed to parse machines list for app "${validApp}"`, listStderr)
  }

  if (!Array.isArray(machines) || machines.length === 0) {
    throw new FlyMachineError(`No Fly machines found for app "${validApp}".`)
  }

  const started = machines.find(m => m && m.state === 'started')

  if (started && started.id) {
    return { machineId: String(started.id), started: false }
  }

  const candidate = machines.find(m => m && (m.state === 'stopped' || m.state === 'suspended'))

  if (!candidate || !candidate.id) {
    throw new FlyMachineError(
      `No suitable stopped or suspended Fly machine found for app "${validApp}".`
    )
  }

  const machineId = String(candidate.id)

  opts.onStatus?.('starting-host')

  try {
    await new Promise<{ stderr: string; stdout: string }>((resolve, reject) => {
      execFileFn(
        flyBinary,
        ['machine', 'start', machineId, '-a', validApp],
        { maxBuffer: 10 * 1024 * 1024, signal: opts.signal },
        (err, stdout, stderr) => {
          if (err) {
            ;(err as any).stdout = stdout
            ;(err as any).stderr = stderr
            reject(err)
          } else {
            resolve({ stderr: String(stderr || ''), stdout: String(stdout || '') })
          }
        }
      )
    })
  } catch (err: any) {
    classifyFlyCliError(err, `Failed to start machine ${machineId}`)
  }

  return { machineId, started: true }
}

export interface FlyProxyHandle {
  alive(): boolean
  app: string
  localPort: number
  onExit(cb: (code: number | null, signal: NodeJS.Signals | null) => void): void
  pid: number
  remotePort: number
  stop(): Promise<void>
}

export interface StartFlyProxyOptions {
  binaryDeps?: ResolveFlyBinaryDeps
  flyBinary?: string
  killGraceMs?: number
  ownerPid?: number
  pickLocalPort?: () => number | Promise<number>
  readyTimeoutMs?: number
  signal?: AbortSignal
  spawn?: typeof spawn
}

interface ActiveProxyEntry {
  handle: FlyProxyHandle
  killSync: () => void
  proxyPid: number
  stop: () => Promise<void>
  supervisorPid: number
}

const activeProxies = new Set<ActiveProxyEntry>()

function isPidDead(pid: number): boolean {
  if (pid <= 0) {
    return true
  }

  try {
    process.kill(pid, 0)

    return false
  } catch (err: any) {
    return err.code === 'ESRCH'
  }
}

function checkTcpListening(port: number): Promise<boolean> {
  return new Promise(resolve => {
    const socket = new net.Socket()

    socket.setTimeout(250)

    socket.once('connect', () => {
      socket.destroy()
      resolve(true)
    })

    socket.once('timeout', () => {
      socket.destroy()
      resolve(false)
    })

    socket.once('error', () => {
      socket.destroy()
      resolve(false)
    })

    socket.connect(port, '127.0.0.1')
  })
}

// SIGKILL the supervisor's whole process group (POSIX). The supervisor is a
// group leader, so this reaches the proxy even before its pid was reported.
function killProxyGroup(supervisorPid: number): void {
  if (process.platform === 'win32' || supervisorPid <= 0) {
    return
  }

  try {
    process.kill(-supervisorPid, 'SIGKILL')
  } catch {
    // group already gone
  }
}

export async function startFlyProxy(
  app: string,
  remotePort: number,
  opts: StartFlyProxyOptions = {}
): Promise<FlyProxyHandle> {
  const validApp = validateFlyApp(app)

  if (!Number.isInteger(remotePort) || remotePort < 1 || remotePort > 65535) {
    throw new Error(`Invalid remote port: ${remotePort}`)
  }

  const flyBinary = opts.flyBinary || resolveFlyBinary(opts.binaryDeps)
  const spawnFn = opts.spawn || spawn

  const pickPort = opts.pickLocalPort || defaultPickLocalPort
  const localPort = Number(await pickPort())

  if (!Number.isInteger(localPort) || localPort <= 0 || localPort > 65535) {
    throw new Error(`Invalid local port: ${localPort}`)
  }

  let supervisorPid = 0
  let proxyPid = 0
  let isAlive = true
  let exited = false
  let stderr = ''

  const exitCallbacks: Array<(code: number | null, signal: NodeJS.Signals | null) => void> = []

  let stoppingPromise: Promise<void> | null = null

  const stop = (): Promise<void> => {
    if (stoppingPromise) {
      return stoppingPromise
    }

    stoppingPromise = (async () => {
      isAlive = false

      if (proxyPid > 0) {
        try {
          process.kill(proxyPid, 'SIGTERM')
        } catch {
          // ignore
        }
      }

      if (supervisorPid > 0 && supervisorPid !== proxyPid) {
        try {
          process.kill(supervisorPid, 'SIGTERM')
        } catch {
          // ignore
        }
      }

      const graceMs = opts.killGraceMs ?? 2000
      const startWait = Date.now()

      while (Date.now() - startWait < graceMs) {
        const proxyGone = proxyPid <= 0 || isPidDead(proxyPid)
        const supervisorGone = supervisorPid <= 0 || isPidDead(supervisorPid)

        if (proxyGone && supervisorGone) {
          break
        }

        await new Promise(r => setTimeout(r, 50))
      }

      if (proxyPid > 0 && !isPidDead(proxyPid)) {
        try {
          process.kill(proxyPid, 'SIGKILL')
        } catch {
          // ignore
        }
      }

      if (supervisorPid > 0 && !isPidDead(supervisorPid)) {
        killProxyGroup(supervisorPid)

        try {
          process.kill(supervisorPid, 'SIGKILL')
        } catch {
          // ignore
        }
      }

      const hardTimeout = Date.now() + 1000

      while (Date.now() < hardTimeout) {
        const proxyGone = proxyPid <= 0 || isPidDead(proxyPid)
        const supervisorGone = supervisorPid <= 0 || isPidDead(supervisorPid)

        if (proxyGone && supervisorGone) {
          break
        }

        await new Promise(r => setTimeout(r, 50))
      }

      activeProxies.delete(entry)
    })()

    return stoppingPromise
  }

  const killSync = () => {
    isAlive = false

    if (proxyPid > 0) {
      try {
        process.kill(proxyPid, 'SIGKILL')
      } catch {
        // ignore
      }
    }

    killProxyGroup(supervisorPid)

    if (supervisorPid > 0 && supervisorPid !== proxyPid) {
      try {
        process.kill(supervisorPid, 'SIGKILL')
      } catch {
        // ignore
      }
    }
  }

  const handle: FlyProxyHandle = {
    alive: () => isAlive && !exited && proxyPid > 0 && !isPidDead(proxyPid),
    app: validApp,
    localPort,
    onExit: cb => {
      exitCallbacks.push(cb)
    },
    get pid() {
      return proxyPid
    },
    remotePort,
    stop
  }

  const entry: ActiveProxyEntry = {
    handle,
    killSync,
    proxyPid,
    stop,
    supervisorPid
  }

  activeProxies.add(entry)

  // Spawn proxy or supervisor:
  // On win32 skip the supervisor and spawn the CLI directly (document this limitation in a comment).
  // Limitation on win32: Windows has no POSIX /bin/sh or kill -0 signal semantics; without a native
  // job-object wrapper, orphan cleanup on Windows relies on the process exit handler rather than a
  // supervisor script.
  if (process.platform === 'win32') {
    const child = spawnFn(
      flyBinary,
      ['proxy', `${localPort}:${remotePort}`, '-a', validApp, '-b', '127.0.0.1'],
      {
        stdio: ['ignore', 'pipe', 'pipe']
      }
    )

    supervisorPid = child.pid ?? 0
    proxyPid = child.pid ?? 0
    entry.supervisorPid = supervisorPid
    entry.proxyPid = proxyPid

    child.stderr?.on('data', chunk => {
      stderr = (stderr + chunk.toString()).slice(0, 4096)
    })

    child.on('exit', (code, sig) => {
      exited = true
      isAlive = false

      for (const cb of exitCallbacks) {
        try {
          cb(code, sig)
        } catch {
          // ignore
        }
      }

      activeProxies.delete(entry)
    })
  } else {
    const supervisorScript =
      `"$1" proxy "$3:$4" -a "$2" -b 127.0.0.1 &\n` +
      `PROXY_PID=$!\n` +
      `printf '__FLY_PROXY_PID__:%d\\n' "$PROXY_PID"\n` +
      `trap 'kill -TERM "$PROXY_PID" 2>/dev/null; wait "$PROXY_PID" 2>/dev/null; exit 0' TERM INT HUP\n` +
      `(\n` +
      `  while kill -0 "$5" 2>/dev/null; do\n` +
      `    if ! kill -0 "$PROXY_PID" 2>/dev/null; then\n` +
      `      exit 0\n` +
      `    fi\n` +
      `    sleep 1\n` +
      `  done\n` +
      `  kill -TERM "$PROXY_PID" 2>/dev/null\n` +
      `  sleep 2\n` +
      `  kill -KILL "$PROXY_PID" 2>/dev/null\n` +
      `) 2>/dev/null &\n` +
      `WATCHDOG_PID=$!\n` +
      `wait "$PROXY_PID" 2>/dev/null\n` +
      `EXIT_CODE=$?\n` +
      `kill -TERM "$WATCHDOG_PID" 2>/dev/null\n` +
      `wait "$WATCHDOG_PID" 2>/dev/null\n` +
      `exit $EXIT_CODE\n`

    const ownerPid = opts.ownerPid ?? process.pid

    const child = spawnFn(
      '/bin/sh',
      [
        '-c',
        supervisorScript,
        'sh',
        flyBinary,
        validApp,
        String(localPort),
        String(remotePort),
        String(ownerPid)
      ],
      {
        // Own process group: a hard kill of the group takes the proxy with the
        // supervisor even when the supervisor never gets to run its trap.
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe']
      }
    )

    supervisorPid = child.pid ?? 0
    proxyPid = child.pid ?? 0
    entry.supervisorPid = supervisorPid
    entry.proxyPid = proxyPid

    child.stdout?.on('data', chunk => {
      const text = chunk.toString()
      const match = text.match(/__FLY_PROXY_PID__:(\d+)/)

      if (match) {
        proxyPid = Number(match[1])
        entry.proxyPid = proxyPid
      }
    })

    child.stderr?.on('data', chunk => {
      stderr = (stderr + chunk.toString()).slice(0, 4096)
    })

    child.on('exit', (code, sig) => {
      exited = true
      isAlive = false

      for (const cb of exitCallbacks) {
        try {
          cb(code, sig)
        } catch {
          // ignore
        }
      }

      activeProxies.delete(entry)
    })
  }

  // Poll TCP readiness
  const timeoutMs = opts.readyTimeoutMs ?? 20_000
  const startTime = Date.now()

  while (Date.now() - startTime < timeoutMs) {
    if (exited) {
      await stop()
      throw new FlyProxyError('Fly proxy exited early', stderr)
    }

    if (opts.signal?.aborted) {
      await stop()
      throw new FlyProxyError('Fly proxy start aborted')
    }

    const ready = await checkTcpListening(localPort)

    if (ready) {
      return handle
    }

    await new Promise(r => setTimeout(r, 50))
  }

  await stop()

  throw new FlyProxyError(
    `Fly proxy timed out waiting for readiness on 127.0.0.1:${localPort} within ${timeoutMs}ms`,
    stderr
  )
}

export async function stopAllFlyProxies(): Promise<void> {
  const entries = Array.from(activeProxies)

  await Promise.allSettled(entries.map(e => e.stop()))
}

export function killAllFlyProxiesSync(): void {
  for (const entry of activeProxies) {
    entry.killSync()
  }

  activeProxies.clear()
}
