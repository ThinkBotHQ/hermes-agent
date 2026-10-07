import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import net from 'node:net'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, test } from 'vitest'

import { resolveAttachTransport } from './attach-transport'
import {
  ensureFlyMachineStarted,
  FlyAuthError,
  FlyCliMissingError,
  flyHostKeyAlias,
  FlyMachineError,
  FlyProxyError,
  killAllFlyProxiesSync,
  resolveFlyBinary,
  startFlyProxy,
  stopAllFlyProxies,
  validateFlyApp
} from './fly-proxy-lifecycle'

function checkPortListening(port: number): Promise<boolean> {
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

describe('fly-proxy-lifecycle', () => {
  let tempDir: string
  let fakeFlyPath: string
  let fakeFlyLog: string

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fly-proxy-test-'))
    fakeFlyPath = path.join(tempDir, 'fake-fly')
    fakeFlyLog = path.join(tempDir, 'fake-fly.log')

    const scriptContent = `#!/usr/bin/env node
const fs = require('node:fs')
const net = require('node:net')

const argv = process.argv.slice(2)
const logFile = process.env.FAKE_FLY_LOG

if (logFile) {
  try {
    fs.appendFileSync(logFile, JSON.stringify({ pid: process.pid, argv }) + '\\n')
  } catch {}
}

if (process.env.FAKE_FLY_AUTH_ERROR === '1') {
  process.stderr.write("Error: No access token available. Please login with 'flyctl auth login'\\n")
  process.exit(1)
}

if (argv[0] === 'machines' && argv[1] === 'list') {
  if (process.env.FAKE_FLY_MACHINES_ERROR) {
    process.stderr.write(process.env.FAKE_FLY_MACHINES_ERROR + '\\n')
    process.exit(1)
  }
  const machines = process.env.FAKE_FLY_MACHINES
    ? JSON.parse(process.env.FAKE_FLY_MACHINES)
    : [{ id: 'm-started', state: 'started' }]
  process.stdout.write(JSON.stringify(machines) + '\\n')
  process.exit(0)
}

if (argv[0] === 'machine' && argv[1] === 'start') {
  if (process.env.FAKE_FLY_START_ERROR) {
    process.stderr.write(process.env.FAKE_FLY_START_ERROR + '\\n')
    process.exit(1)
  }
  process.stdout.write('Machine ' + (argv[2] || '') + ' started\\n')
  process.exit(0)
}

if (argv[0] === 'proxy') {
  if (process.env.FAKE_FLY_PROXY_FAIL_STDERR) {
    process.stderr.write(process.env.FAKE_FLY_PROXY_FAIL_STDERR + '\\n')
    process.exit(1)
  }
  if (process.env.FAKE_FLY_PROXY_NEVER_LISTEN === '1') {
    setInterval(() => {}, 1000)
    return
  }
  const portArg = argv.find(a => /^(?:127\\.0\\.0\\.1:)?\\d+:\\d+$/.test(a))
  const match = portArg ? portArg.match(/^(?:127\\.0\\.0\\.1:)?(\\d+):\\d+$/) : null
  const localPort = match ? Number(match[1]) : 0
  if (!localPort) {
    process.stderr.write('Missing port in proxy command\\n')
    process.exit(1)
  }
  const server = net.createServer(socket => {
    socket.on('error', () => {})
    socket.end('ok')
  })
  server.on('error', () => {})
  server.listen(localPort, '127.0.0.1', () => {})
  setInterval(() => {}, 60000)
}
`

    fs.writeFileSync(fakeFlyPath, scriptContent, { mode: 0o755 })
  })

  afterEach(async () => {
    delete process.env.FAKE_FLY_LOG
    delete process.env.FAKE_FLY_AUTH_ERROR
    delete process.env.FAKE_FLY_MACHINES
    delete process.env.FAKE_FLY_MACHINES_ERROR
    delete process.env.FAKE_FLY_START_ERROR
    delete process.env.FAKE_FLY_PROXY_FAIL_STDERR
    delete process.env.FAKE_FLY_PROXY_NEVER_LISTEN

    await stopAllFlyProxies()
    killAllFlyProxiesSync()

    if (tempDir) {
      try {
        fs.rmSync(tempDir, { force: true, recursive: true })
      } catch {
        // ignore
      }
    }
  })

  function readFakeFlyLogs(): Array<{ argv: string[]; pid: number }> {
    if (!fs.existsSync(fakeFlyLog)) {
      return []
    }

    const lines = fs.readFileSync(fakeFlyLog, 'utf8').trim().split('\n').filter(Boolean)

    return lines.map(line => JSON.parse(line))
  }

  test('(a) validateFlyApp accepts valid and rejects invalid names', () => {
    assert.equal(validateFlyApp('tb-worker-host'), 'tb-worker-host')
    assert.equal(validateFlyApp('  tb-worker-host  '), 'tb-worker-host')
    assert.equal(validateFlyApp('a'), 'a')
    assert.equal(validateFlyApp('0abc'), '0abc')
    assert.equal(validateFlyApp('a'.repeat(63)), 'a'.repeat(63))

    assert.throws(() => validateFlyApp(''), /Invalid Fly app name/)
    assert.throws(() => validateFlyApp('-x'), /Invalid Fly app name/)
    assert.throws(() => validateFlyApp('a b'), /Invalid Fly app name/)
    assert.throws(() => validateFlyApp('a;b'), /Invalid Fly app name/)
    assert.throws(() => validateFlyApp('A'), /Invalid Fly app name/)
    assert.throws(() => validateFlyApp('a'.repeat(64)), /Invalid Fly app name/)
    assert.throws(() => validateFlyApp('app$name'), /Invalid Fly app name/)
  })

  test('flyHostKeyAlias returns fly-proxy.<app>', () => {
    assert.equal(flyHostKeyAlias('tb-worker-host'), 'fly-proxy.tb-worker-host')
    assert.throws(() => flyHostKeyAlias('INVALID_APP'), /Invalid Fly app name/)
  })

  test('(b) ensureFlyMachineStarted handles started, stopped, and empty machine lists', async () => {
    process.env.FAKE_FLY_LOG = fakeFlyLog

    // Started machine: no start call, no status notification
    process.env.FAKE_FLY_MACHINES = JSON.stringify([{ id: 'm-started-1', state: 'started' }])

    let statusCalled = false

    const resStarted = await ensureFlyMachineStarted('tb-worker-host', {
      flyBinary: fakeFlyPath,
      onStatus: () => {
        statusCalled = true
      }
    })

    assert.deepEqual(resStarted, { machineId: 'm-started-1', started: false })
    assert.equal(statusCalled, false)

    const logsAfterStarted = readFakeFlyLogs()

    const startCallsAfterStarted = logsAfterStarted.filter(
      l => l.argv[0] === 'machine' && l.argv[1] === 'start'
    )

    assert.equal(startCallsAfterStarted.length, 0)

    // Stopped machine: calls onStatus('starting-host'), runs machine start <id> -a <app>
    process.env.FAKE_FLY_MACHINES = JSON.stringify([
      { id: 'm-stopped-1', state: 'stopped' },
      { id: 'm-stopped-2', state: 'stopped' }
    ])

    const statusCalls: string[] = []

    const resStopped = await ensureFlyMachineStarted('tb-worker-host', {
      flyBinary: fakeFlyPath,
      onStatus: s => {
        statusCalls.push(s)
      }
    })

    assert.deepEqual(resStopped, { machineId: 'm-stopped-1', started: true })
    assert.deepEqual(statusCalls, ['starting-host'])

    const logsAfterStopped = readFakeFlyLogs()

    const startCallsAfterStopped = logsAfterStopped.filter(
      l => l.argv[0] === 'machine' && l.argv[1] === 'start'
    )

    assert.equal(startCallsAfterStopped.length, 1)
    assert.deepEqual(startCallsAfterStopped[0].argv, [
      'machine',
      'start',
      'm-stopped-1',
      '-a',
      'tb-worker-host'
    ])

    // Suspended machine: also starts
    process.env.FAKE_FLY_MACHINES = JSON.stringify([{ id: 'm-suspended-1', state: 'suspended' }])

    const resSuspended = await ensureFlyMachineStarted('tb-worker-host', {
      flyBinary: fakeFlyPath
    })

    assert.deepEqual(resSuspended, { machineId: 'm-suspended-1', started: true })

    // No machines: throws FlyMachineError
    process.env.FAKE_FLY_MACHINES = JSON.stringify([])

    await assert.rejects(
      () =>
        ensureFlyMachineStarted('tb-worker-host', {
          flyBinary: fakeFlyPath
        }),
      (err: any) => {
        assert.ok(err instanceof FlyMachineError)
        assert.equal(err.kind, 'fly-machine')
        assert.match(err.message, /no.*machine/i)

        return true
      }
    )
  })

  test('(c) missing binary throws FlyCliMissingError; auth failure throws FlyAuthError', async () => {
    // Missing binary
    assert.throws(
      () => resolveFlyBinary({ env: { PATH: '' }, existsSync: () => false }),
      (err: any) => {
        assert.ok(err instanceof FlyCliMissingError)
        assert.equal(err.kind, 'fly-cli-missing')
        assert.match(err.message, /brew install flyctl/)

        return true
      }
    )

    // Auth failure from fake fly
    process.env.FAKE_FLY_AUTH_ERROR = '1'

    await assert.rejects(
      () =>
        ensureFlyMachineStarted('tb-worker-host', {
          flyBinary: fakeFlyPath
        }),
      (err: any) => {
        assert.ok(err instanceof FlyAuthError)
        assert.equal(err.kind, 'fly-auth')
        assert.match(err.message, /fly auth login/)

        return true
      }
    )
  })

  test('(d) startFlyProxy resolves with listening localPort and passes correct argv to fake fly', async () => {
    process.env.FAKE_FLY_LOG = fakeFlyLog

    const handle = await startFlyProxy('tb-worker-host', 22, {
      flyBinary: fakeFlyPath,
      readyTimeoutMs: 5000
    })

    assert.equal(handle.app, 'tb-worker-host')
    assert.equal(handle.remotePort, 22)
    assert.ok(handle.localPort > 0)
    assert.ok(handle.pid > 0)
    assert.equal(handle.alive(), true)

    // Verify it is listening on loopback
    const listening = await checkPortListening(handle.localPort)
    assert.equal(listening, true)

    // Verify argv passed to fake fly
    const logs = readFakeFlyLogs()
    const proxyCalls = logs.filter(l => l.argv[0] === 'proxy')
    assert.ok(proxyCalls.length >= 1)

    const proxyArgv = proxyCalls[0].argv
    assert.ok(proxyArgv.includes('-a'))
    assert.equal(proxyArgv[proxyArgv.indexOf('-a') + 1], 'tb-worker-host')
    assert.ok(proxyArgv.some(a => a.endsWith(':22')))

    // Verify loopback bind address
    assert.ok(proxyArgv.includes('-b'))
    assert.equal(proxyArgv[proxyArgv.indexOf('-b') + 1], '127.0.0.1')

    await handle.stop()
  })

  test('(e) stop(): port no longer accepts connections, proxy pid is gone, calling stop() twice is fine', async () => {
    const handle = await startFlyProxy('tb-worker-host', 22, {
      flyBinary: fakeFlyPath,
      readyTimeoutMs: 5000
    })

    const localPort = handle.localPort
    const proxyPid = handle.pid

    assert.equal(await checkPortListening(localPort), true)
    assert.doesNotThrow(() => process.kill(proxyPid, 0))

    await handle.stop()

    assert.equal(handle.alive(), false)
    assert.equal(await checkPortListening(localPort), false)
    assert.throws(() => process.kill(proxyPid, 0), /ESRCH/)

    // Calling stop() a second time is fine and resolves without error
    await assert.doesNotReject(() => handle.stop())
  })

  test.skipIf(process.platform === 'win32')(
    '(f) CRASH: owner killed with -9 causes proxy to exit within 8s',
    async () => {
      process.env.FAKE_FLY_LOG = fakeFlyLog

      const ownerScript = `
        import { startFlyProxy } from './electron/fly-proxy-lifecycle'
        const handle = await startFlyProxy('tb-worker-host', 22, {
          flyBinary: ${JSON.stringify(fakeFlyPath)},
          readyTimeoutMs: 10000
        })
        process.stdout.write(JSON.stringify({ pid: handle.pid, localPort: handle.localPort }) + '\\n')
        setInterval(() => {}, 60000)
      `

      const owner = spawn(
        process.execPath,
        ['--import', 'tsx', '-e', ownerScript],
        {
          cwd: path.resolve(__dirname, '..'),
          env: { ...process.env, FAKE_FLY_LOG: fakeFlyLog },
          stdio: ['ignore', 'pipe', 'pipe']
        }
      )

      let stdout = ''
      owner.stdout.on('data', d => {
        stdout += d.toString()
      })

      const info = await new Promise<{ localPort: number; pid: number }>((resolve, reject) => {
        const timeout = setTimeout(() => {
          owner.kill('SIGKILL')
          reject(new Error('Owner script timed out waiting for proxy start. stdout: ' + stdout))
        }, 10000)

        owner.stdout.on('data', () => {
          const lines = stdout.split('\n').filter(Boolean)

          for (const line of lines) {
            try {
              const parsed = JSON.parse(line)

              if (parsed.pid && parsed.localPort) {
                clearTimeout(timeout)
                resolve(parsed)

                return
              }
            } catch {
              // ignore
            }
          }
        })

        owner.on('error', err => {
          clearTimeout(timeout)
          reject(err)
        })

        owner.on('exit', code => {
          clearTimeout(timeout)
          reject(new Error(`Owner exited prematurely with code ${code}. stdout: ${stdout}`))
        })
      })

      assert.ok(info.pid > 0)
      assert.ok(info.localPort > 0)
      assert.doesNotThrow(() => process.kill(info.pid, 0))
      assert.equal(await checkPortListening(info.localPort), true)

      // Kill owner with kill -9
      owner.kill('SIGKILL')

      // Assert within 8s that proxy pid is gone and port is closed
      const startTime = Date.now()
      let proxyDead = false
      let portClosed = false

      while (Date.now() - startTime < 8000) {
        try {
          process.kill(info.pid, 0)
        } catch (err: any) {
          if (err.code === 'ESRCH') {
            proxyDead = true
          }
        }

        const listening = await checkPortListening(info.localPort)

        if (!listening) {
          portClosed = true
        }

        if (proxyDead && portClosed) {
          break
        }

        await new Promise(r => setTimeout(r, 200))
      }

      assert.equal(proxyDead, true, 'Proxy process should be terminated after owner kill -9')
      assert.equal(portClosed, true, 'Proxy local port should be closed after owner kill -9')
    },
    15000
  )

  test('(g) proxy that exits immediately with stderr text throws FlyProxyError with stderr text and leaves no child', async () => {
    process.env.FAKE_FLY_LOG = fakeFlyLog
    process.env.FAKE_FLY_PROXY_FAIL_STDERR = 'WireGuard handshake failed: authentication error'

    await assert.rejects(
      () =>
        startFlyProxy('tb-worker-host', 22, {
          flyBinary: fakeFlyPath,
          readyTimeoutMs: 3000
        }),
      (err: any) => {
        assert.ok(err instanceof FlyProxyError)
        assert.equal(err.kind, 'fly-proxy')
        assert.match(err.message, /WireGuard handshake failed/)

        return true
      }
    )

    // Ensure no child left behind
    const logs = readFakeFlyLogs()
    const proxyCalls = logs.filter(l => l.argv[0] === 'proxy')

    for (const call of proxyCalls) {
      assert.throws(() => process.kill(call.pid, 0), /ESRCH/)
    }
  })

  test('rejects an invalid remote port before spawning', async () => {
    for (const port of [0, 65536, 22.5, NaN]) {
      await assert.rejects(() => startFlyProxy('tb-worker-host', port, { flyBinary: fakeFlyPath }), /remote port/i)
    }

    assert.deepEqual(readFakeFlyLogs(), [])
  })

  test('caps reported proxy stderr at 4 KB and keeps its head', async () => {
    process.env.FAKE_FLY_PROXY_FAIL_STDERR = `HEAD-${'x'.repeat(8000)}-TAIL`

    await assert.rejects(() => startFlyProxy('tb-worker-host', 22, {
      flyBinary: fakeFlyPath,
      readyTimeoutMs: 3000
    }), (error: any) => {
      assert.match(error.message, /HEAD-/)
      assert.doesNotMatch(error.message, /-TAIL/)
      assert.ok(error.message.length < 4500)

      return true
    })
  })

  test('two apps get isolated proxy ports and host key aliases', async () => {
    const deps = {
      directRetryMs: 0,
      ensureFlyMachineStarted: async () => {},
      probeTcp: async () => false,
      startFlyProxy: (app: string, port: number) => startFlyProxy(app, port, { flyBinary: fakeFlyPath })
    }

    const [first, second] = await Promise.all([
      resolveAttachTransport({ host: 'host-a', port: 2222, user: 'hermes', flyApp: 'app-a' }, deps),
      resolveAttachTransport({ host: 'host-b', port: 2222, user: 'hermes', flyApp: 'app-b' }, deps)
    ])

    assert.equal(first.route, 'fly-proxy')
    assert.equal(second.route, 'fly-proxy')

    if (first.route !== 'fly-proxy' || second.route !== 'fly-proxy') {
      throw new Error('Expected proxy routes')
    }

    assert.notEqual(first.port, second.port)
    assert.notEqual(first.hostKeyAlias, second.hostKeyAlias)
    await first.proxy.stop()
    assert.equal(second.proxy.alive(), true)
    assert.equal(await checkPortListening(second.port), true)
  })

  test('(h) proxy that never listens throws FlyProxyError after readyTimeoutMs and leaves no child', async () => {
    process.env.FAKE_FLY_LOG = fakeFlyLog
    process.env.FAKE_FLY_PROXY_NEVER_LISTEN = '1'

    await assert.rejects(
      () =>
        startFlyProxy('tb-worker-host', 22, {
          flyBinary: fakeFlyPath,
          readyTimeoutMs: 500
        }),
      (err: any) => {
        assert.ok(err instanceof FlyProxyError)
        assert.equal(err.kind, 'fly-proxy')
        assert.match(err.message, /timed out|ready/i)

        return true
      }
    )

    // Ensure child is dead
    const logs = readFakeFlyLogs()
    const proxyCalls = logs.filter(l => l.argv[0] === 'proxy')

    for (const call of proxyCalls) {
      assert.throws(() => process.kill(call.pid, 0), /ESRCH/)
    }
  })

  test('(i) stopAllFlyProxies stops two live handles', async () => {
    const handle1 = await startFlyProxy('tb-worker-host', 22, {
      flyBinary: fakeFlyPath,
      readyTimeoutMs: 5000
    })

    const handle2 = await startFlyProxy('tb-worker-host', 80, {
      flyBinary: fakeFlyPath,
      readyTimeoutMs: 5000
    })

    assert.equal(handle1.alive(), true)
    assert.equal(handle2.alive(), true)
    assert.equal(await checkPortListening(handle1.localPort), true)
    assert.equal(await checkPortListening(handle2.localPort), true)

    await stopAllFlyProxies()

    assert.equal(handle1.alive(), false)
    assert.equal(handle2.alive(), false)
    assert.equal(await checkPortListening(handle1.localPort), false)
    assert.equal(await checkPortListening(handle2.localPort), false)
    assert.throws(() => process.kill(handle1.pid, 0), /ESRCH/)
    assert.throws(() => process.kill(handle2.pid, 0), /ESRCH/)
  })
})
