import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { hostKeyFingerprint, resolveAttachTransport } from './attach-transport'

const input = { host: 'host.example', port: 2222, user: 'hermes' }

describe('resolveAttachTransport', () => {
  test('without Fly, returns direct without probing or calling Fly', async () => {
    const result = await resolveAttachTransport(input, {
      probeTcp: () => { throw new Error('probed') },
      ensureFlyMachineStarted: () => { throw new Error('started') },
      startFlyProxy: () => { throw new Error('proxied') }
    })

    assert.deepEqual(result, { route: 'direct', ...input })
  })

  test('reachable Fly host stays direct without a Fly command', async () => {
    const calls: string[] = []

    const result = await resolveAttachTransport({ ...input, flyApp: 'app-a' }, {
      probeTcp: async () => true,
      ensureFlyMachineStarted: async () => { calls.push('start') },
      startFlyProxy: async () => { calls.push('proxy'); throw new Error('unexpected') }
    })

    assert.deepEqual(result, { route: 'direct', ...input })
    assert.deepEqual(calls, [])
  })

  test('wakes host then uses direct route when retry succeeds', async () => {
    const calls: string[] = []

    const result = await resolveAttachTransport({ ...input, flyApp: 'app-a' }, {
      probeTcp: async () => {
        calls.push('probe')

        return calls.length > 2
      },
      ensureFlyMachineStarted: async () => { calls.push('wake') },
      startFlyProxy: async () => { throw new Error('unexpected') },
      directRetryMs: 3000,
      onStatus: status => calls.push(status)
    })

    assert.equal(result.route, 'direct')
    assert.deepEqual(calls.slice(0, 3), ['probe', 'starting-host', 'wake'])
  })

  test('falls back to app-specific proxy after direct retries', async () => {
    const proxy = { localPort: 40222, alive: () => true, stop: async () => {}, onExit: () => {} }

    const result = await resolveAttachTransport({ ...input, flyApp: 'app-a' }, {
      probeTcp: async () => false,
      ensureFlyMachineStarted: async () => {},
      startFlyProxy: async () => proxy,
      directRetryMs: 0
    })

    assert.deepEqual(result, {
      route: 'fly-proxy', host: '127.0.0.1', port: 40222, user: 'hermes',
      hostKeyAlias: 'fly-proxy.app-a', proxy
    })
  })

  test('abort after proxy creation stops it', async () => {
    const controller = new AbortController()
    let stopped = false

    await assert.rejects(() => resolveAttachTransport({ ...input, flyApp: 'app-a' }, {
      probeTcp: async () => false,
      ensureFlyMachineStarted: async () => {},
      startFlyProxy: async () => {
        controller.abort()

        return { localPort: 40222, alive: () => true, stop: async () => { stopped = true }, onExit: () => {} }
      },
      directRetryMs: 0,
      signal: controller.signal
    }), /abort/i)
    assert.equal(stopped, true)
  })

  test('Fly errors propagate unchanged', async () => {
    const failure = Object.assign(new Error('Run fly auth login'), { kind: 'fly-auth' })

    await assert.rejects(() => resolveAttachTransport({ ...input, flyApp: 'app-a' }, {
      probeTcp: async () => false,
      ensureFlyMachineStarted: async () => { throw failure },
      directRetryMs: 0
    }), error => error === failure)
  })
})

test('host key fingerprint lookup returns only SHA256 fingerprint', async () => {
  const fingerprint = await hostKeyFingerprint('fly-proxy.app-a', (_file, args, _opts, callback) => {
    assert.deepEqual(args, ['-F', 'fly-proxy.app-a', '-l'])
    callback(null, '# Host fly-proxy.app-a found\n256 SHA256:abc123 fly-proxy.app-a (ED25519)\n', '')
  })

  assert.equal(fingerprint, 'SHA256:abc123')
})

test('host key fingerprint lookup tolerates ssh-keygen failure', async () => {
  const fingerprint = await hostKeyFingerprint('missing-host', (_file, _args, _opts, callback) => {
    callback(new Error('no host key'), '', '')
  })

  assert.equal(fingerprint, null)
})
