import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TerminalPaneChrome } from './chrome'
import { $activeTerminalId, $terminals } from './terminals'

vi.mock('./persistent', () => ({ TerminalSlot: () => null }))
vi.mock('./rail', () => ({ TerminalRail: () => <div data-testid="rail" /> }))

describe('TerminalPaneChrome', () => {
  afterEach(() => {
    cleanup()
    $terminals.set([])
    $activeTerminalId.set(null)
  })

  it('offers a new terminal when the pane is on screen with none left (remote shell ended on a connection switch)', () => {
    $terminals.set([])
    render(<TerminalPaneChrome />)

    expect(screen.queryByTestId('rail')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'New terminal' }))

    expect($terminals.get()).toHaveLength(1)
    expect($activeTerminalId.get()).toBe($terminals.get()[0].id)
    expect(screen.queryByRole('button', { name: 'New terminal' })).toBeNull()
    expect(screen.getByTestId('rail')).toBeTruthy()
  })

  it('shows no empty-state control while a terminal exists', () => {
    $terminals.set([{ auto: true, cwd: '/repo', id: 'term-1', kind: 'user', title: 'zsh' }])
    render(<TerminalPaneChrome />)

    expect(screen.queryByRole('button', { name: 'New terminal' })).toBeNull()
  })
})
