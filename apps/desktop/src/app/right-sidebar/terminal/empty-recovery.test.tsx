import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $terminalTakeover } from '../store'

import { EmptyTerminalRecovery } from './empty-recovery'
import { $activeTerminalId, $terminals } from './terminals'

describe('EmptyTerminalRecovery', () => {
  afterEach(() => {
    cleanup()
    $terminals.set([])
    $activeTerminalId.set(null)
  })

  it('offers a new terminal when none is left, and creates one for the pane', () => {
    $terminals.set([])
    render(<EmptyTerminalRecovery />)

    fireEvent.click(screen.getByRole('button', { name: 'New terminal' }))

    expect($terminals.get()).toHaveLength(1)
    expect($activeTerminalId.get()).toBe($terminals.get()[0].id)
    expect($terminalTakeover.get()).toBe(true)
    expect(screen.queryByRole('button', { name: 'New terminal' })).toBeNull()
  })

  it('renders nothing while a terminal exists', () => {
    $terminals.set([{ auto: true, cwd: '/repo', id: 'term-1', kind: 'user', title: 'zsh' }])
    render(<EmptyTerminalRecovery />)

    expect(screen.queryByRole('button', { name: 'New terminal' })).toBeNull()
  })
})
