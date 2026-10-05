import assert from 'node:assert/strict'
import crypto from 'node:crypto'

import { describe, test } from 'vitest'

import {
  attach,
  AttachIdentityError,
  AttachNoBackendError,
  AttachTokenMismatchError,
  detach,
  tokenFingerprint
} from './ssh-attach-lifecycle'

function fakeSsh(
  rules: Array<[RegExp | ((cmd: string) => boolean), string | Error | ((cmd: string) => string)]> = []
) {
  const calls: string[] = []
  const forwards: Array<{ localPort: number; remotePort: number; remoteHost?: string }> = []
  const cancelledForwards: Array<{ localPort: number; remotePort: number; remoteHost?: string }> = []

  return {
    calls,
    forwards,
    cancelledForwards,
    async exec(cmd: string) {
      calls.push(cmd)
      for (const [matcher, resp] of rules) {
        const hit = typeof matcher === 'function' ? matcher(cmd) : matcher.test(cmd)
        if (hit) {
          const out = typeof resp === 'function' ? resp(cmd) : resp
          if (out instanceof Error) throw out
          return out
        }
      }
      return ''
    },
    async forward(localPort: number, remotePort: number, remoteHost = '127.0.0.1') {
      forwards.push({ localPort, remotePort, remoteHost })
    },
    async cancelForward(localPort: number, remotePort: number, remoteHost = '127.0.0.1') {
      cancelledForwards.push({ localPort, remotePort, remoteHost })
    }
  }
}

function makeHostRecord(over: Record<string, any> = {}) {
  const token = typeof over.token === 'string' ? over.token : 'test-session-token-xyz'
  const record: Record<string, any> = {
    role: 'serve',
    pid: 12345,
    createTime: 1727500000.0,
    host: '127.0.0.1',
    port: 54321,
    protocolVersion: 1,
    tokenFingerprint: tokenFingerprint(token),
    profiles: ['default'],
    updatedAt: '2026-09-28T00:00:00Z',
    ...over
  }
  delete record.token
  return { record, token }
}

describe('ssh-attach-lifecycle', () => {
  test('(a) happy path returns baseUrl/token and calls forward with record.port', async () => {
    const { record, token } = makeHostRecord()
    const ssh = fakeSsh([
      [/host-serve/, `${JSON.stringify(record)}\n__HERMES_ATTACH_DELIM__\n${token}`]
    ])

    let waitForHermesCalledWith: { baseUrl: string; token: string } | null = null

    const result = await attach(ssh, {
      pickLocalPort: async () => 50001,
      waitForHermes: async (baseUrl, t) => {
        waitForHermesCalledWith = { baseUrl, token: t }
      },
      fetchFn: async (url: string | URL | Request) => {
        const urlStr = String(url)
        if (urlStr.endsWith('/api/host/identity')) {
          return new Response(JSON.stringify({ pid: record.pid, role: 'serve' }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
          })
        }
        return new Response('Not found', { status: 404 })
      }
    })

    assert.equal(result.baseUrl, 'http://127.0.0.1:50001')
    assert.equal(result.token, token)
    assert.equal(result.localPort, 50001)
    assert.equal(result.remotePort, record.port)
    assert.equal(result.pid, record.pid)
    assert.deepEqual(ssh.forwards, [
      { localPort: 50001, remotePort: record.port, remoteHost: '127.0.0.1' }
    ])
    assert.deepEqual(waitForHermesCalledWith, {
      baseUrl: 'http://127.0.0.1:50001',
      token
    })
  })

  test('(b) missing files throws AttachNoBackendError and opens no forward', async () => {
    // Missing files return the missing marker or empty output
    for (const missingOutput of ['__HERMES_ATTACH_MISSING__', '']) {
      const ssh = fakeSsh([[/host-serve/, missingOutput]])

      await assert.rejects(
        () =>
          attach(ssh, {
            pickLocalPort: async () => 50001
          }),
        (err: any) => {
          assert.ok(err instanceof AttachNoBackendError)
          assert.equal(err.kind, 'attach-no-backend')
          assert.match(err.message, /supervisor|missing|running/i)
          return true
        }
      )

      assert.equal(ssh.forwards.length, 0, 'no forward should be opened when files are missing')
    }
  })

  test('(b-2) role is desktop-serve or not serve throws AttachNoBackendError and opens no forward', async () => {
    const { record, token } = makeHostRecord({ role: 'desktop-serve' })
    const ssh = fakeSsh([
      [/host-serve/, `${JSON.stringify(record)}\n__HERMES_ATTACH_DELIM__\n${token}`]
    ])

    await assert.rejects(
      () =>
        attach(ssh, {
          pickLocalPort: async () => 50001
        }),
      (err: any) => {
        assert.ok(err instanceof AttachNoBackendError)
        assert.equal(err.kind, 'attach-no-backend')
        return true
      }
    )

    assert.equal(ssh.forwards.length, 0, 'no forward should be opened when role is not serve')
  })

  test('(c) fingerprint mismatch throws AttachTokenMismatchError and opens no forward', async () => {
    const { record } = makeHostRecord({ tokenFingerprint: '0123456789abcdef' })
    const differentToken = 'a-different-token-whose-hash-wont-match'
    const ssh = fakeSsh([
      [/host-serve/, `${JSON.stringify(record)}\n__HERMES_ATTACH_DELIM__\n${differentToken}`]
    ])

    await assert.rejects(
      () =>
        attach(ssh, {
          pickLocalPort: async () => 50001
        }),
      (err: any) => {
        assert.ok(err instanceof AttachTokenMismatchError)
        assert.equal(err.kind, 'attach-token-mismatch')
        return true
      }
    )

    assert.equal(ssh.forwards.length, 0, 'no forward should be opened when token fingerprint mismatches')
  })

  test('(d) identity pid mismatch throws AttachIdentityError and forward is cancelled', async () => {
    const { record, token } = makeHostRecord({ pid: 12345 })
    const ssh = fakeSsh([
      [/host-serve/, `${JSON.stringify(record)}\n__HERMES_ATTACH_DELIM__\n${token}`]
    ])

    await assert.rejects(
      () =>
        attach(ssh, {
          pickLocalPort: async () => 50001,
          fetchFn: async (url: string | URL | Request) => {
            const urlStr = String(url)
            if (urlStr.endsWith('/api/host/identity')) {
              // Returning different PID 99999
              return new Response(JSON.stringify({ pid: 99999, role: 'serve' }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
              })
            }
            return new Response('Not found', { status: 404 })
          }
        }),
      (err: any) => {
        assert.ok(err instanceof AttachIdentityError)
        assert.equal(err.kind, 'attach-identity-error')
        return true
      }
    )

    assert.equal(ssh.forwards.length, 1, 'forward must have been initially established')
    assert.deepEqual(ssh.cancelledForwards, [
      { localPort: 50001, remotePort: record.port, remoteHost: '127.0.0.1' }
    ], 'forward must be cancelled after identity check failure')
  })

  test('(e) detach cancels the forward and issues NO exec containing kill/pkill', async () => {
    const ssh = fakeSsh()
    await detach(ssh, { localPort: 50001, remotePort: 54321 })

    assert.deepEqual(ssh.cancelledForwards, [
      { localPort: 50001, remotePort: 54321, remoteHost: '127.0.0.1' }
    ])
    assert.ok(
      !ssh.calls.some(cmd => /kill|pkill/.test(cmd)),
      'detach must issue NO exec containing kill or pkill'
    )
  })
})
