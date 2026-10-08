import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'

import { setTerminalTakeover } from '../store'

import { $terminals, createTerminal } from './terminals'

/** The way back from a terminal pane that is on screen with no terminal left
 *  (a window's terminals are closed when it switches connection, and the tab
 *  rail, which holds the only "+", goes with the last tab).
 *
 *  Rendered INSIDE the persistent overlay: that overlay paints over the pane
 *  slot and takes its pointer events, so a control in the pane's own DOM would
 *  be covered. Same action as the "New terminal" keybind. */
export function EmptyTerminalRecovery() {
  const terminals = useStore($terminals)
  const { t } = useI18n()

  if (terminals.length > 0) {
    return null
  }

  return (
    <div className="flex min-h-0 flex-1 items-center justify-center">
      <button
        className="rounded-md border border-current/20 px-3 py-1.5 text-xs text-(--ui-text-secondary) hover:opacity-80"
        onClick={() => {
          createTerminal()
          setTerminalTakeover(true)
        }}
        type="button"
      >
        {t.rightSidebar.terminalNew}
      </button>
    </div>
  )
}
