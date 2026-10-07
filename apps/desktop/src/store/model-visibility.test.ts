import type { ModelOptionProvider } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $seenModels,
  $visibleModels,
  clearSeenModels,
  collapseModelFamilies,
  defaultVisibleKeys,
  effectiveVisibleKeys,
  emptyProviderSentinelKey,
  isProviderSentinel,
  markModelSeen,
  markProvidersSeen,
  modelVisibilityKey,
  resolveVisibleKeys,
  seedSeenModels,
  setProviderVisibility,
  setSeenModels,
  setVisibleModels,
  toggleModelVisibility
} from './model-visibility'
import { $connection } from './session'

const provider = (slug: string, models: string[]): ModelOptionProvider => ({
  models,
  name: slug,
  slug
})

describe('model visibility', () => {
  it('keeps newly configured providers visible when stored choices are stale', () => {
    const stored = new Set([modelVisibilityKey('copilot', 'claude-sonnet-4.6')])

    const visible = effectiveVisibleKeys(stored, [
      provider('copilot', ['claude-sonnet-4.6']),
      provider('local-ollama', ['qwen3:latest', 'llama3.2:latest'])
    ])

    expect(visible.has(modelVisibilityKey('copilot', 'claude-sonnet-4.6'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'llama3.2:latest'))).toBe(true)
  })

  it('does not re-add models the user already judged for a curated provider', () => {
    const stored = new Set([modelVisibilityKey('local-ollama', 'qwen3:latest')])

    const known = new Set([
      modelVisibilityKey('local-ollama', 'qwen3:latest'),
      modelVisibilityKey('local-ollama', 'llama3.2:latest')
    ])

    const visible = effectiveVisibleKeys(stored, [provider('local-ollama', ['qwen3:latest', 'llama3.2:latest'])], known)

    expect(visible.has(modelVisibilityKey('local-ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('local-ollama', 'llama3.2:latest'))).toBe(false)
  })

  it('shows a model that appeared after the user curated its provider, unless the provider is hidden', () => {
    // User curated claude-sub (kept sonnet, hid haiku) and hid all of nous; then a plugin update adds opus.
    const stored = new Set([modelVisibilityKey('claude-sub', 'sonnet'), emptyProviderSentinelKey('nous')])

    const known = new Set([
      modelVisibilityKey('claude-sub', 'sonnet'),
      modelVisibilityKey('claude-sub', 'haiku'),
      modelVisibilityKey('nous', 'hermes-4')
    ])

    const providers = [provider('claude-sub', ['sonnet', 'haiku', 'opus']), provider('nous', ['hermes-4', 'hermes-5'])]
    const visible = effectiveVisibleKeys(stored, providers, known)

    expect(visible.has(modelVisibilityKey('claude-sub', 'opus'))).toBe(true)
    expect(visible.has(modelVisibilityKey('claude-sub', 'haiku'))).toBe(false)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-5'))).toBe(false)

    // No snapshot yet (pre-upgrade store): nothing counts as new, hide choices stay verbatim.
    expect(effectiveVisibleKeys(stored, providers, null).has(modelVisibilityKey('claude-sub', 'opus'))).toBe(false)
  })

  it('preserves hidden-provider sentinel without re-adding defaults', () => {
    // User explicitly hid all models for "nous" — sentinel marks this choice.
    const stored = new Set([emptyProviderSentinelKey('nous')])

    const visible = effectiveVisibleKeys(stored, [
      provider('nous', ['hermes-3-llama-3.1-70b', 'hermes-3-llama-3.1-8b']),
      provider('ollama', ['qwen3:latest'])
    ])

    expect(visible.has(modelVisibilityKey('nous', 'hermes-3-llama-3.1-70b'))).toBe(false)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-3-llama-3.1-8b'))).toBe(false)
    // Sentinel itself is stripped from the result.
    expect(visible.has(emptyProviderSentinelKey('nous'))).toBe(false)
    // Other providers still get defaults.
    expect(visible.has(modelVisibilityKey('ollama', 'qwen3:latest'))).toBe(true)
  })

  it('folds a date-pinned snapshot into its rolling alias when present', () => {
    const families = collapseModelFamilies(['claude-opus-4-5', 'claude-opus-4-5-20251101'])

    expect(families.map(f => f.id)).toEqual(['claude-opus-4-5'])
  })

  it('keeps a date-pinned snapshot standing alone when it has no alias', () => {
    const families = collapseModelFamilies(['claude-opus-4-5-20251101', 'claude-haiku-4-5-20251001'])

    expect(families.map(f => f.id)).toEqual(['claude-opus-4-5-20251101', 'claude-haiku-4-5-20251001'])
  })

  it('sentinel key helper produces correct format', () => {
    expect(emptyProviderSentinelKey('openai')).toBe('openai::')
    expect(isProviderSentinel('openai::')).toBe(true)
    expect(isProviderSentinel('openai::gpt-4o')).toBe(false)
  })

  it('resolveVisibleKeys preserves sentinels that effectiveVisibleKeys strips', () => {
    const stored = new Set([emptyProviderSentinelKey('nous')])
    const providers = [provider('nous', ['hermes-x', 'hermes-y']), provider('ollama', ['qwen3:latest'])]

    const resolved = resolveVisibleKeys(stored, providers)
    expect(resolved.has(emptyProviderSentinelKey('nous'))).toBe(true)
    expect(resolved.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
    // Un-customized providers still expand to their defaults.
    expect(resolved.has(modelVisibilityKey('ollama', 'qwen3:latest'))).toBe(true)

    // Display variant drops the sentinel.
    expect(effectiveVisibleKeys(stored, providers).has(emptyProviderSentinelKey('nous'))).toBe(false)
  })
})

describe('toggleModelVisibility', () => {
  const providers = [provider('openai', ['gpt-a', 'gpt-b']), provider('nous', ['hermes-x', 'hermes-y'])]

  // Drive the handler the way the dialog does: feed each result back in as the
  // next `stored`, so the persisted set is what the next toggle starts from.
  const apply = (stored: Set<string> | null, slug: string, model: string) =>
    toggleModelVisibility(stored, providers, slug, model)

  it('records a hide-all sentinel when the last model of a provider is toggled off', () => {
    let stored: Set<string> | null = null
    stored = apply(stored, 'openai', 'gpt-a')
    stored = apply(stored, 'openai', 'gpt-b')

    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(true)
    expect(effectiveVisibleKeys(stored, providers).has(modelVisibilityKey('openai', 'gpt-a'))).toBe(false)
    expect(effectiveVisibleKeys(stored, providers).has(modelVisibilityKey('openai', 'gpt-b'))).toBe(false)
  })

  it('keeps a hidden provider hidden when a different provider is toggled (regression for #43485)', () => {
    // Hide ALL of nous — its sentinel is now stored.
    let stored: Set<string> | null = null
    stored = apply(stored, 'nous', 'hermes-x')
    stored = apply(stored, 'nous', 'hermes-y')
    expect(stored.has(emptyProviderSentinelKey('nous'))).toBe(true)

    // Toggle a model in another provider. nous must NOT snap back on.
    stored = apply(stored, 'openai', 'gpt-a')

    expect(stored.has(emptyProviderSentinelKey('nous'))).toBe(true)
    const visible = effectiveVisibleKeys(stored, providers)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-y'))).toBe(false)
  })

  it('clears only the toggled provider sentinel when a model is re-enabled', () => {
    let stored: Set<string> | null = new Set([emptyProviderSentinelKey('openai'), emptyProviderSentinelKey('nous')])

    stored = apply(stored, 'openai', 'gpt-a')

    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(false)
    expect(stored.has(emptyProviderSentinelKey('nous'))).toBe(true)
    const visible = effectiveVisibleKeys(stored, providers)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
  })

  it('re-enabling one model of a hidden-all provider restores ONLY that model, not the curated defaults', () => {
    // openai hidden-all, nous untouched.
    let stored: Set<string> | null = new Set([emptyProviderSentinelKey('openai')])

    stored = apply(stored, 'openai', 'gpt-a')

    const visible = effectiveVisibleKeys(stored, providers)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
    // gpt-b is NOT restored — "you hid everything, you get back only what you re-enable".
    expect(visible.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(false)
  })

  it('re-hiding the last re-enabled model re-adds the sentinel (full round-trip)', () => {
    let stored: Set<string> | null = new Set([emptyProviderSentinelKey('openai')])

    // Re-enable gpt-a (clears sentinel, set = {gpt-a}), then toggle it back off.
    stored = apply(stored, 'openai', 'gpt-a')
    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(false)
    stored = apply(stored, 'openai', 'gpt-a')

    expect(stored.has(emptyProviderSentinelKey('openai'))).toBe(true)
    expect(effectiveVisibleKeys(stored, providers).has(modelVisibilityKey('openai', 'gpt-a'))).toBe(false)
  })

  it('toggling from an empty (non-null) stored set adds the model without expanding defaults', () => {
    // Empty-but-not-null = "everything hidden". resolveVisibleKeys short-circuits to {}.
    const stored = new Set<string>()

    const next = apply(stored, 'openai', 'gpt-a')

    expect(next.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
    // No curated defaults were expanded for any provider.
    expect(next.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(false)
    expect(next.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
  })

  it('toggling off one default model from null stored keeps the rest of the curated defaults', () => {
    // null = "never customized": resolveVisibleKeys expands all defaults first.
    const next = apply(null, 'openai', 'gpt-a')

    expect(next.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(false)
    expect(next.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(true)
    expect(next.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(true)
    // Other models remain, so no sentinel.
    expect(next.has(emptyProviderSentinelKey('openai'))).toBe(false)
  })

  it('tolerates a provider with zero models (defensive — dialog filters these out)', () => {
    const ps = [provider('empty', []), provider('openai', ['gpt-a'])]
    const next = toggleModelVisibility(new Set([modelVisibilityKey('openai', 'gpt-a')]), ps, 'empty', 'ghost')

    // No crash; the phantom key is recorded but no defaults are invented.
    expect([...next].some(k => k.startsWith('empty::') && !isProviderSentinel(k))).toBe(true)
    expect(next.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
  })
})

describe('resolveVisibleKeys', () => {
  const providers = [provider('openai', ['gpt-a', 'gpt-b']), provider('nous', ['hermes-x', 'hermes-y'])]

  it('returns the curated defaults verbatim for null stored', () => {
    expect(resolveVisibleKeys(null, providers)).toEqual(defaultVisibleKeys(providers))
  })

  it('returns an empty set for an empty (non-null) stored set', () => {
    expect([...resolveVisibleKeys(new Set(), providers)]).toEqual([])
  })
})

describe('featured defaults', () => {
  const featuredProvider = (slug: string, models: string[], featured_models: string[]): ModelOptionProvider => ({
    featured_models,
    models,
    name: slug,
    slug
  })

  it('defaults to the featured shortlist when a provider publishes one', () => {
    const nous = featuredProvider(
      'nous',
      ['anthropic/opus', 'anthropic/haiku', 'google/gemini', 'x-ai/grok'],
      ['anthropic/opus', 'google/gemini', 'x-ai/grok']
    )

    const visible = defaultVisibleKeys([nous])

    // Featured are visible; the non-featured model is hidden by default.
    expect(visible.has(modelVisibilityKey('nous', 'anthropic/opus'))).toBe(true)
    expect(visible.has(modelVisibilityKey('nous', 'google/gemini'))).toBe(true)
    expect(visible.has(modelVisibilityKey('nous', 'x-ai/grok'))).toBe(true)
    expect(visible.has(modelVisibilityKey('nous', 'anthropic/haiku'))).toBe(false)
  })

  it('falls back to top-N when a provider ships no featured list', () => {
    const plain = provider('ollama', ['qwen3:latest', 'llama3.2:latest'])

    const visible = defaultVisibleKeys([plain])

    // No featured_models → every model stays a default (top-N, N ≫ 2 here).
    expect(visible.has(modelVisibilityKey('ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('ollama', 'llama3.2:latest'))).toBe(true)
  })

  it('ignores an empty featured list and falls back to top-N', () => {
    const plain = featuredProvider('ollama', ['qwen3:latest', 'llama3.2:latest'], [])

    const visible = defaultVisibleKeys([plain])

    expect(visible.has(modelVisibilityKey('ollama', 'qwen3:latest'))).toBe(true)
    expect(visible.has(modelVisibilityKey('ollama', 'llama3.2:latest'))).toBe(true)
  })
})

describe('setProviderVisibility', () => {
  const providers = [provider('openai', ['gpt-a', 'gpt-b']), provider('nous', ['hermes-x', 'hermes-y'])]

  it('enabling a provider makes every one of its models visible', () => {
    // Start from a hidden-all openai; flip it on.
    const stored = new Set([emptyProviderSentinelKey('openai')])

    const next = setProviderVisibility(stored, providers, 'openai', true)

    const visible = effectiveVisibleKeys(next, providers)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(true)
    // Sentinel is cleared.
    expect(next.has(emptyProviderSentinelKey('openai'))).toBe(false)
  })

  it('disabling a provider hides all its models and records the sentinel', () => {
    const next = setProviderVisibility(null, providers, 'openai', false)

    expect(next.has(emptyProviderSentinelKey('openai'))).toBe(true)
    const visible = effectiveVisibleKeys(next, providers)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(false)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-b'))).toBe(false)
  })

  it('leaves other providers untouched (their sentinels survive)', () => {
    const stored = new Set([emptyProviderSentinelKey('nous')])

    // Turn openai fully on; nous must stay hidden.
    const next = setProviderVisibility(stored, providers, 'openai', true)

    expect(next.has(emptyProviderSentinelKey('nous'))).toBe(true)
    const visible = effectiveVisibleKeys(next, providers)
    expect(visible.has(modelVisibilityKey('nous', 'hermes-x'))).toBe(false)
    expect(visible.has(modelVisibilityKey('openai', 'gpt-a'))).toBe(true)
  })

  it('round-trips: enable then disable returns to a clean hidden-all', () => {
    const enabled = setProviderVisibility(null, providers, 'openai', true)
    const disabled = setProviderVisibility(enabled, providers, 'openai', false)

    expect(disabled.has(emptyProviderSentinelKey('openai'))).toBe(true)
    // No stray real keys left for the provider.
    expect([...disabled].some(k => k.startsWith('openai::') && !isProviderSentinel(k))).toBe(false)
  })

  it('collapses model families to one key per family when enabling', () => {
    // A base + its -fast sibling collapse to a single family row/key.
    const ps = [provider('nous', ['model', 'model-fast'])]

    const next = setProviderVisibility(null, ps, 'nous', true)

    expect(next.has(modelVisibilityKey('nous', 'model'))).toBe(true)
    // The -fast sibling is represented by its base family, not its own key.
    expect(next.has(modelVisibilityKey('nous', 'model-fast'))).toBe(false)
  })

  afterEach(() => {
    $connection.set(null)
  })

  it('namespaces model choices by connection without affecting local or sibling connections', () => {
    // Seed pre-existing unsuffixed local key in localStorage
    const localKey = modelVisibilityKey('openai', 'gpt-4o')
    window.localStorage.setItem('hermes.desktop.visible-models', JSON.stringify([localKey]))
    $connection.set({ connectionId: 'local', mode: 'local' } as never)

    // Switch to connection A
    $connection.set({ connectionId: 'conn-a', mode: 'remote' } as never)
    const connAKey = modelVisibilityKey('anthropic', 'claude-3-5-sonnet')

    const nextA = toggleModelVisibility(
      new Set(),
      [provider('anthropic', ['claude-3-5-sonnet'])],
      'anthropic',
      'claude-3-5-sonnet'
    )

    setVisibleModels(nextA)

    // Verify conn-a persisted under namespaced key
    expect(window.localStorage.getItem('hermes.desktop.visible-models::conn-a')).toContain(connAKey)
    // Verify local unsuffixed key was NOT modified
    expect(window.localStorage.getItem('hermes.desktop.visible-models')).toEqual(JSON.stringify([localKey]))

    // Switch to connection B
    $connection.set({ connectionId: 'conn-b', mode: 'remote' } as never)
    expect($visibleModels.get()).toBeNull()

    const connBKey = modelVisibilityKey('google', 'gemini-1.5-pro')

    const nextB = toggleModelVisibility(
      new Set(),
      [provider('google', ['gemini-1.5-pro'])],
      'google',
      'gemini-1.5-pro'
    )

    setVisibleModels(nextB)

    // Verify conn-b persisted under namespaced key
    expect(window.localStorage.getItem('hermes.desktop.visible-models::conn-b')).toContain(connBKey)
    // Conn A's key still unchanged
    expect(window.localStorage.getItem('hermes.desktop.visible-models::conn-a')).toContain(connAKey)
    expect(window.localStorage.getItem('hermes.desktop.visible-models::conn-a')).not.toContain(connBKey)

    // Switching back to connection A restores A's set
    $connection.set({ connectionId: 'conn-a', mode: 'remote' } as never)
    expect($visibleModels.get()?.has(connAKey)).toBe(true)
    expect($visibleModels.get()?.has(connBKey)).toBe(false)

    // Switching back to local connection restores local set and reads unsuffixed key
    $connection.set({ connectionId: 'local', mode: 'local' } as never)
    expect($visibleModels.get()?.has(localKey)).toBe(true)
    expect($visibleModels.get()?.has(connAKey)).toBe(false)
    expect($visibleModels.get()?.has(connBKey)).toBe(false)
    expect(window.localStorage.getItem('hermes.desktop.visible-models')).toEqual(JSON.stringify([localKey]))
  })
})

describe('seen-set newest model leads', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearSeenModels()
  })

  it('seeds the first catalog without reordering and promotes only models added later until shown', () => {
    $seenModels.set(null)
    const initial = provider('anthropic', ['claude-opus-4-5'])
    seedSeenModels([initial])
    expect(collapseModelFamilies(initial.models ?? []).map(f => f.id)).toEqual(['claude-opus-4-5'])

    const updated = provider('anthropic', ['claude-opus-4-5', 'claude-opus-5-5'])
    expect(collapseModelFamilies(updated.models ?? []).map(f => f.id)).toEqual(['claude-opus-5-5', 'claude-opus-4-5'])

    markProvidersSeen([updated])
    expect(collapseModelFamilies(updated.models ?? []).map(f => f.id)).toEqual(['claude-opus-4-5', 'claude-opus-5-5'])
  })

  it('sorts an unseen model in a family to the top of that family (unseen newest first)', () => {
    // claude-opus-4-5 has been seen; claude-opus-5-5 is a newly released unseen model in the same Opus family
    setSeenModels(new Set(['claude-opus-4-5']))

    const models = ['claude-opus-4-5', 'claude-opus-5-5']
    const families = collapseModelFamilies(models)

    expect(families.map(f => f.id)).toEqual(['claude-opus-5-5', 'claude-opus-4-5'])
  })

  it('sorts an unseen claude-sonnet-5-5 ahead of claude-sonnet-5 in the sonnet family', () => {
    setSeenModels(new Set(['claude-sonnet-5']))

    const models = ['claude-sonnet-5', 'claude-sonnet-5-5']
    const families = collapseModelFamilies(models)

    expect(families.map(f => f.id)).toEqual(['claude-sonnet-5-5', 'claude-sonnet-5'])
  })

  it('preserves the family relative position among other families while promoting unseen within the family', () => {
    setSeenModels(new Set(['claude-sonnet-4-6', 'claude-opus-4-5', 'claude-haiku-4-5']))

    const models = ['claude-sonnet-4-6', 'claude-opus-4-5', 'claude-opus-5-5', 'claude-haiku-4-5']
    const families = collapseModelFamilies(models)

    // claude-opus-5-5 leads the Opus family, but Sonnet remains first
    expect(families.map(f => f.id)).toEqual([
      'claude-sonnet-4-6',
      'claude-opus-5-5',
      'claude-opus-4-5',
      'claude-haiku-4-5'
    ])
  })

  it('retains default curated order once the model is marked seen after display', () => {
    setSeenModels(new Set(['claude-opus-4-5']))

    const models = ['claude-opus-4-5', 'claude-opus-5-5']
    const initial = collapseModelFamilies(models)
    expect(initial.map(f => f.id)).toEqual(['claude-opus-5-5', 'claude-opus-4-5'])

    // Marked seen after display / selection
    markModelSeen('claude-opus-5-5')

    const afterDisplay = collapseModelFamilies(models)
    expect(afterDisplay.map(f => f.id)).toEqual(['claude-opus-4-5', 'claude-opus-5-5'])
  })

  it('persists seen models across reload via the existing persistence mock', async () => {
    setSeenModels(new Set(['claude-opus-4-5']))
    markModelSeen('claude-opus-5-5')

    // Simulate reload via vi.resetModules()
    vi.resetModules()
    const reloaded = await import('./model-visibility')

    expect(reloaded.$seenModels.get()?.has('claude-opus-5-5')).toBe(true)

    const models = ['claude-opus-4-5', 'claude-opus-5-5']
    const families = reloaded.collapseModelFamilies(models)
    expect(families.map(f => f.id)).toEqual(['claude-opus-4-5', 'claude-opus-5-5'])
  })
})
