import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { errorRecoveryPlan } from '@/lib/error-surface'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $sessionStates,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState,
  reconcileBusyStatesOnReconnect,
  SESSION_WATCHDOG_TIMEOUT_MS,
  setSessionTurnReconciler
} from './session-states'

// Read from the store rather than restated here: these assert what happens on
// either side of the threshold, not what the threshold is.
const WATCHDOG_MS = SESSION_WATCHDOG_TIMEOUT_MS

function state(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState(null), storedSessionId: 's1', ...over }
}

describe('session watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
    setSessionTurnReconciler(null)
  })

  it('marks a silent session stalled without pretending it finished', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))

    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toContain('s1')
    expect($stalledSessionIds.get()).toContain('s1')
  })

  it('clears stalled on new activity and rearms the watchdog', () => {
    const working = state({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt2', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s2')

    publishSessionState('rt2', { ...working, awaitingResponse: true })
    expect($stalledSessionIds.get()).not.toContain('s2')

    vi.advanceTimersByTime(WATCHDOG_MS - 1)
    expect($stalledSessionIds.get()).not.toContain('s2')
    expect($workingSessionIds.get()).toContain('s2')
  })

  it('clears both running and stalled on an authoritative terminal transition', () => {
    const working = state({ busy: true, storedSessionId: 's3' })
    publishSessionState('rt3', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s3')

    publishSessionState('rt3', { ...working, busy: false })

    expect($workingSessionIds.get()).not.toContain('s3')
    expect($stalledSessionIds.get()).not.toContain('s3')
  })

  it('never marks a session stalled when it settles before the window', () => {
    const working = state({ busy: true, storedSessionId: 's4' })
    publishSessionState('rt4', working)
    publishSessionState('rt4', { ...working, busy: false })
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).not.toContain('s4')
    expect($stalledSessionIds.get()).not.toContain('s4')
  })

  it('clears stalled state and disarms timers on a gateway wipe', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toEqual(['s1'])

    clearAllSessionStates()
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toEqual([])
    expect($stalledSessionIds.get()).toEqual([])
  })
})

describe('computed $workingSessionIds', () => {
  beforeEach(() => {
    clearAllSessionStates()
  })

  afterEach(() => {
    clearAllSessionStates()
  })

  it('reflects busy sessions under the id their surfaces key on', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    publishSessionState('rt2', state({ busy: false, storedSessionId: 's2' }))
    // Not yet persisted, so the runtime id is the only id it has — and the one
    // the row is keyed by until the backend hands a stored id back.
    publishSessionState('rt3', state({ busy: true, storedSessionId: null }))

    expect($workingSessionIds.get()).toEqual(['s1', 'rt3'])
  })

  it('updates when session state changes', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual(['s1'])

    publishSessionState('rt1', state({ busy: false, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual([])
  })
})

const SILENCE_MS = LIVE_TURN_EVENT_SILENCE_MS

function partial(text: string, over: Partial<ClientSessionState> = {}): ClientSessionState {
  return state({
    awaitingResponse: true,
    busy: true,
    messages: [
      {
        id: 'a1',
        parts: [{ type: 'text', text }],
        pending: true,
        role: 'assistant'
      }
    ],
    model: 'any-model',
    sawAssistantPayload: true,
    streamId: 'a1',
    turnLive: true,
    turnStartedAt: Date.now(),
    ...over
  })
}

describe('live turn event silence', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
    setSessionTurnReconciler(vi.fn().mockResolvedValue({ complete: false, messages: [] }))
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
    setSessionTurnReconciler(null)
  })

  it('force-settles a silent live turn even after a partial payload and offers retry', async () => {
    $activeSessionId.set('rt1')
    publishSessionState('rt1', partial('partial answer', { model: 'glm-5.3-flash', storedSessionId: 's1' }))
    noteSessionEvent('rt1')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const settled = $sessionStates.get().rt1
    expect($workingSessionIds.get()).not.toContain('s1')
    expect(settled?.busy).toBe(false)
    expect(settled?.awaitingResponse).toBe(false)
    expect(settled?.turnLive).toBe(false)
    expect(settled?.messages.every(message => !message.pending)).toBe(true)
    expect(
      settled?.messages.some(message =>
        message.parts.some(part => part.type === 'text' && part.text === 'partial answer')
      )
    ).toBe(true)

    const failed = settled?.messages.find(message => message.errorSurface)
    expect(failed?.errorSurface?.retryable).toBe(true)
    expect(errorRecoveryPlan(failed?.errorSurface).retry).toBe(true)
    expect(failed?.error).not.toMatch(/glm|deepseek|interrupted mid-run/i)
  })

  it('does not latch the owner-Stop interrupt when it settles, so a recovered reply still draws', async () => {
    $activeSessionId.set('rt-recover')
    publishSessionState('rt-recover', partial('before the crash', { storedSessionId: 's-recover' }))
    noteSessionEvent('rt-recover')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const settled = $sessionStates.get()['rt-recover']
    expect(settled?.busy).toBe(false)
    // `interrupted` is the owner's Stop latch: it makes message.start, deltas, tool events and
    // message.complete be dropped until the next prompt.submit. Crash recovery answers after the
    // silence window, and that reply must not need another owner message to appear.
    expect(settled?.interrupted).toBe(false)
  })

  it('force-settles a silent live turn that never produced a payload', async () => {
    $activeSessionId.set('rt-empty')
    publishSessionState(
      'rt-empty',
      state({ awaitingResponse: true, busy: true, model: 'deepseek-chat', storedSessionId: 's-empty', turnLive: true })
    )
    noteSessionEvent('rt-empty')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const settled = $sessionStates.get()['rt-empty']
    expect($workingSessionIds.get()).not.toContain('s-empty')
    expect(settled?.busy).toBe(false)
    expect(settled?.turnLive).toBe(false)
    const failed = settled?.messages.find(message => message.role === 'assistant' && message.errorSurface)
    expect(failed?.errorSurface?.retryable).toBe(true)
    expect(errorRecoveryPlan(failed?.errorSurface).retry).toBe(true)
  })

  it('does not settle a live turn that keeps producing events', async () => {
    publishSessionState('rt-live', partial('still working', { storedSessionId: 's-live' }))
    noteSessionEvent('rt-live')

    await vi.advanceTimersByTimeAsync(SILENCE_MS - 1)
    noteSessionEvent('rt-live')
    await vi.advanceTimersByTimeAsync(SILENCE_MS - 1)

    expect($workingSessionIds.get()).toContain('s-live')
    expect($sessionStates.get()['rt-live']?.messages.some(message => message.errorSurface)).toBe(false)
  })

  it('does not settle a turn the user is still answering', async () => {
    publishSessionState('rt-ask', partial('need a choice', { needsInput: true, storedSessionId: 's-ask' }))
    noteSessionEvent('rt-ask')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    expect($workingSessionIds.get()).toContain('s-ask')
    expect($sessionStates.get()['rt-ask']?.messages.some(message => message.errorSurface)).toBe(false)
  })

  it('settles only the session that stopped producing events', async () => {
    $activeSessionId.set('rt-a')
    publishSessionState('rt-a', partial('a', { storedSessionId: 's-a' }))
    publishSessionState('rt-b', partial('b', { storedSessionId: 's-b' }))
    noteSessionEvent('rt-a')
    noteSessionEvent('rt-b')

    await vi.advanceTimersByTimeAsync(SILENCE_MS - 1)
    noteSessionEvent('rt-b')
    await vi.advanceTimersByTimeAsync(1)

    expect($workingSessionIds.get()).not.toContain('s-a')
    expect($workingSessionIds.get()).toContain('s-b')
    expect($sessionStates.get()['rt-a']?.messages.some(message => message.errorSurface?.retryable)).toBe(true)
    expect($sessionStates.get()['rt-b']?.busy).toBe(true)
  })

  it('does not stamp a retry when the turn settles before the silence window', () => {
    const working = partial('done soon', { storedSessionId: 's-done' })
    publishSessionState('rt-done', working)
    noteSessionEvent('rt-done')
    publishSessionState('rt-done', {
      ...working,
      awaitingResponse: false,
      busy: false,
      messages: working.messages.map(message => ({ ...message, pending: false })),
      turnLive: false
    })

    vi.advanceTimersByTime(SILENCE_MS)

    expect($sessionStates.get()['rt-done']?.messages.some(message => message.errorSurface)).toBe(false)
    expect($workingSessionIds.get()).not.toContain('s-done')
  })

  it('adopts the completed persisted reply when the renderer missed completion', async () => {
    $activeSessionId.set('rt-complete')

    const persisted = [
      { id: 'u1', rowId: 1, parts: [{ type: 'text' as const, text: 'help me' }], role: 'user' as const },
      {
        id: 'a1-durable',
        rowId: 2,
        parts: [{ type: 'text' as const, text: 'Here is the complete solution.' }],
        pending: false,
        role: 'assistant' as const
      }
    ]

    const reconcile = vi.fn().mockResolvedValue({ complete: true, messages: persisted, turnKey: 'row:1' })
    setSessionTurnReconciler(reconcile)

    const live = partial('Here is the complete solution.', {
      messages: [
        { id: 'u1', rowId: 1, parts: [{ type: 'text', text: 'help me' }], role: 'user' },
        { id: 'a1', parts: [{ type: 'text', text: 'Here is the complete solution.' }], pending: true, role: 'assistant' }
      ],
      storedSessionId: 's-complete'
    })

    publishSessionState('rt-complete', live)
    reconcileBusyStatesOnReconnect()
    noteSessionEvent('rt-complete')

    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const settled = $sessionStates.get()['rt-complete']
    expect(reconcile).toHaveBeenCalledWith({
      baselineAssistantRowIds: [],
      sessionId: 's-complete',
      turnKey: 'user-row:1'
    })
    expect(settled?.messages).toEqual(persisted)
    expect(settled?.messages.some(message => message.errorSurface)).toBe(false)
    expect(settled?.messages.every(message => !message.pending)).toBe(true)
    expect(settled?.busy).toBe(false)
    expect(settled?.awaitingResponse).toBe(false)
    expect(settled?.turnLive).toBe(false)
  })

  it('sets a visible interrupted state when persistence says the turn is incomplete', async () => {
    $activeSessionId.set('rt-midstream')

    const midstreamTurn: ClientSessionState = state({
      awaitingResponse: true,
      busy: true,
      messages: [
        {
          id: 'u1',
          parts: [{ type: 'text', text: 'help me' }],
          role: 'user'
        },
        {
          id: 'a1',
          parts: [{ type: 'text', text: 'Halfway through...' }],
          pending: true,
          role: 'assistant'
        }
      ],
      model: 'glm-5.3-flash',
      sawAssistantPayload: true,
      storedSessionId: 's-midstream',
      streamId: 'a1',
      turnLive: true,
      turnStartedAt: Date.now()
    })

    publishSessionState('rt-midstream', midstreamTurn)
    noteSessionEvent('rt-midstream')

    setSessionTurnReconciler(vi.fn().mockResolvedValue({ complete: false, messages: [], turnKey: 'row:1' }))
    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    const settled = $sessionStates.get()['rt-midstream']
    const failed = settled?.messages.find(message => message.errorSurface)
    expect(failed).toBeDefined()
    expect(failed?.errorSurface?.layer).toBe('streaming')
    expect(failed?.errorSurface?.code).toBe('stream_drop')
    // Action is not a prompt re-send: errorRecoveryPlan provides continue/retry,
    // which maps to ContinueRetryAction / session.continue in the UI
    const plan = errorRecoveryPlan(failed?.errorSurface)
    expect(plan.retry).toBe(true)
    expect(failed?.error).toBe('The connection dropped before the reply finished.')
  })

  it('falls back to the interrupted state when persistence reconciliation rejects or times out', async () => {
    $activeSessionId.set('rt-reject')

    const pending = new Promise<never>(() => undefined)
    const reconcile = vi.fn().mockRejectedValueOnce(new Error('offline')).mockReturnValue(pending)
    setSessionTurnReconciler(reconcile)

    publishSessionState('rt-reject', partial('partial', { storedSessionId: 's-reject' }))
    noteSessionEvent('rt-reject')
    await vi.advanceTimersByTimeAsync(SILENCE_MS)
    expect($sessionStates.get()['rt-reject']?.messages.some(message => message.errorSurface)).toBe(true)
    expect($sessionStates.get()['rt-reject']?.turnLive).toBe(false)

    $activeSessionId.set('rt-timeout')

    publishSessionState('rt-timeout', partial('partial', { storedSessionId: 's-timeout' }))
    noteSessionEvent('rt-timeout')
    await vi.advanceTimersByTimeAsync(SILENCE_MS + 5000)
    expect($sessionStates.get()['rt-timeout']?.messages.some(message => message.errorSurface)).toBe(true)
    expect($sessionStates.get()['rt-timeout']?.turnLive).toBe(false)
  })

  it('keeps liveness for a new turn after a completed reply, including queued follow-ups', async () => {
    setSessionTurnReconciler(vi.fn().mockResolvedValue({ complete: false, messages: [], turnKey: 'message:user-queued-followup' }))

    const previous = {
      completedAt: Date.now() / 1000,
      durableComplete: true,
      id: 'a-previous',
      rowId: 2,
      parts: [{ type: 'text' as const, text: 'Previous complete answer.' }],
      pending: false,
      role: 'assistant' as const
    }

    const queued = {
      id: 'user-queued-followup',
      rowId: 3,
      parts: [{ type: 'text' as const, text: 'follow up' }],
      role: 'user' as const
    }

    for (const [runtimeId, followup] of [
      ['rt-backend-start', null],
      ['rt-queued-start', queued]
    ] as const) {
      $activeSessionId.set(runtimeId)

      publishSessionState(
        runtimeId,
        state({
          busy: true,
          messages: [
            { id: 'u1', rowId: 1, parts: [{ type: 'text', text: 'first prompt' }], role: 'user' },
            previous,
            ...(followup ? [followup] : [])
          ],
          storedSessionId: runtimeId,
          turnStartedAt: Date.now(),
          turnLive: true
        })
      )
      noteSessionEvent(runtimeId)
      await vi.advanceTimersByTimeAsync(SILENCE_MS)
      expect($sessionStates.get()[runtimeId]?.messages.some(message => message.errorSurface)).toBe(true)
      expect($sessionStates.get()[runtimeId]?.turnLive).toBe(false)
    }
  })

  it('ignores a reconcile result when another session event arrives during the request', async () => {
    let resolve!: (value: { complete: true; messages: [] ; turnKey: string }) => void
    setSessionTurnReconciler(() => new Promise(done => { resolve = done }))
    publishSessionState('rt-race', partial('partial', { storedSessionId: 's-race' }))
    noteSessionEvent('rt-race')
    await vi.advanceTimersByTimeAsync(SILENCE_MS)

    noteSessionEvent('rt-race')
    resolve({ complete: true, messages: [], turnKey: 'message:u1' })
    await vi.advanceTimersByTimeAsync(0)

    expect($sessionStates.get()['rt-race']?.messages.some(message => message.errorSurface)).toBe(false)
    expect($sessionStates.get()['rt-race']?.turnLive).toBe(true)
  })
})
