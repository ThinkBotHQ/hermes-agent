import { execFile } from 'node:child_process'
import net from 'node:net'

import {
  ensureFlyMachineStarted,
  flyHostKeyAlias,
  type FlyProxyHandle,
  startFlyProxy
} from './fly-proxy-lifecycle'

interface AttachTarget {
  flyApp?: string
  host: string
  port: number
  user?: string
}

export interface AttachTransportDeps {
  directRetryMs?: number
  ensureFlyMachineStarted?: (app: string, opts?: { signal?: AbortSignal }) => Promise<unknown>
  onStatus?: (status: 'starting-host') => void
  probeTcp?: typeof probeTcp
  signal?: AbortSignal
  startFlyProxy?: (app: string, port: number, opts?: { signal?: AbortSignal }) => Promise<AttachProxyHandle>
}

export type AttachProxyHandle = Pick<FlyProxyHandle, 'alive' | 'localPort' | 'onExit' | 'stop'>

export type AttachTransport =
  | { route: 'direct'; host: string; port: number; user?: string; /** this resolution started a stopped machine */ hostStarted?: boolean }
  | {
      route: 'fly-proxy'
      /** this resolution started a stopped machine */
      hostStarted?: boolean
      host: '127.0.0.1'
      port: number
      user?: string
      hostKeyAlias: string
      proxy: AttachProxyHandle
    }

export function probeTcp(host: string, port: number, timeoutMs = 1500): Promise<boolean> {
  return new Promise(resolve => {
    const socket = net.connect({ host, port })
    let settled = false

    const finish = (reachable: boolean) => {
      if (settled) {
        return
      }

      settled = true
      socket.destroy()
      resolve(reachable)
    }

    socket.setTimeout(timeoutMs)
    socket.once('connect', () => finish(true))
    socket.once('timeout', () => finish(false))
    socket.once('error', () => finish(false))
  })
}

function assertNotAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new Error('SSH attach transport aborted')
  }
}

export async function resolveAttachTransport(
  target: AttachTarget,
  deps: AttachTransportDeps = {}
): Promise<AttachTransport> {
  const direct = { route: 'direct' as const, host: target.host, port: target.port, user: target.user }

  if (!target.flyApp) {
    return direct
  }

  const probe = deps.probeTcp || probeTcp
  assertNotAborted(deps.signal)

  if (await probe(target.host, target.port, 1500)) {
    assertNotAborted(deps.signal)

    return direct
  }

  assertNotAborted(deps.signal)
  deps.onStatus?.('starting-host')
  const ensured = await (deps.ensureFlyMachineStarted || ensureFlyMachineStarted)(target.flyApp, { signal: deps.signal })
  // Only when the machine was actually stopped and this call started it (not for one already running).
  const started = (ensured as null | undefined | { started?: boolean })?.started === true ? { hostStarted: true as const } : {}

  const deadline = Date.now() + (deps.directRetryMs ?? 3000)

  while (Date.now() <= deadline) {
    assertNotAborted(deps.signal)

    if (await probe(target.host, target.port, 1500)) {
      assertNotAborted(deps.signal)

      return { ...direct, ...started }
    }

    if (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, Math.min(500, deadline - Date.now())))
    }
  }

  assertNotAborted(deps.signal)
  const proxy = await (deps.startFlyProxy || startFlyProxy)(target.flyApp, target.port, { signal: deps.signal })

  try {
    assertNotAborted(deps.signal)

    return {
      route: 'fly-proxy', host: '127.0.0.1', port: proxy.localPort, user: target.user,
      hostKeyAlias: flyHostKeyAlias(target.flyApp), proxy, ...started
    }
  } catch (error) {
    await proxy.stop()
    throw error
  }
}

type ExecFileFn = (
  file: string,
  args: string[],
  options: { maxBuffer: number },
  callback: (error: Error | null, stdout: string, stderr: string) => void
) => unknown

export async function hostKeyFingerprint(name: string, execFileFn: ExecFileFn = execFile as ExecFileFn): Promise<string | null> {
  try {
    const output = await new Promise<string>((resolve, reject) => {
      execFileFn('ssh-keygen', ['-F', name, '-l'], { maxBuffer: 64 * 1024 }, (error, stdout) => {
        if (error) {
          reject(error)
        } else {
          resolve(String(stdout))
        }
      })
    })

    return output.match(/\bSHA256:[A-Za-z0-9+/=]+/)?.[0] || null
  } catch {
    return null
  }
}

export async function hasKnownHost(name: string, execFileFn: ExecFileFn = execFile as ExecFileFn): Promise<boolean> {
  try {
    const output = await new Promise<string>((resolve, reject) => {
      execFileFn('ssh-keygen', ['-F', name], { maxBuffer: 64 * 1024 }, (error, stdout) => {
        if (error) {
          reject(error)
        } else {
          resolve(String(stdout))
        }
      })
    })

    return output.includes('ssh-') || output.includes('ecdsa-')
  } catch {
    return false
  }
}
