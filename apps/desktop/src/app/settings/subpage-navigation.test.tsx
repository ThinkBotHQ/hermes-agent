import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $connectionsRegistry } from '@/store/connections'
import { $connection } from '@/store/session'

import { SettingsSubpageHeader } from './subpage-navigation'

describe('SettingsSubpageHeader', () => {
  afterEach(() => {
    $connection.set(null)
    $connectionsRegistry.set(null)
  })

  it('renders remote connection badge when on remote connection, and no badge for local', () => {
    const group = { active: true, icon: () => null, id: 'config:model', label: 'Model', onSelect: () => {} }

    // Local connection: no badge
    $connection.set({ connectionId: 'local', mode: 'local' } as never)
    $connectionsRegistry.set({
      connections: [
        {
          authMode: 'token',
          id: 'local',
          kind: 'local',
          label: 'This device',
          tokenPreview: null,
          tokenSet: false
        }
      ],
      lastUsed: 'local',
      launchMode: 'primary',
      primary: 'local',
      secureTokenStorage: false,
      version: 2
    })

    const { rerender } = render(<SettingsSubpageHeader group={group} />)
    expect(screen.queryByText('Fly Host')).toBeNull()

    // Remote connection: renders badge with label
    $connection.set({ connectionId: 'fly', mode: 'remote' } as never)
    $connectionsRegistry.set({
      connections: [
        {
          authMode: 'token',
          id: 'local',
          kind: 'local',
          label: 'This device',
          tokenPreview: null,
          tokenSet: false
        },
        {
          authMode: 'token',
          id: 'fly',
          kind: 'remote',
          label: 'Fly Host',
          tokenPreview: null,
          tokenSet: false
        }
      ],
      lastUsed: 'fly',
      launchMode: 'primary',
      primary: 'local',
      secureTokenStorage: false,
      version: 2
    })

    rerender(<SettingsSubpageHeader group={group} />)
    expect(screen.getByText('Fly Host')).toBeTruthy()
  })
})
