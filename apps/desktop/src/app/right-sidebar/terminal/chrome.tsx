import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'

import { TerminalSlot } from './persistent'
import { TerminalRail } from './rail'
import { $terminals, createTerminal } from './terminals'

/** Pane-side terminal chrome: the body slot (which the persistent overlay chases)
 *  plus the always-on tab rail. Lives in the real pane DOM — NOT the z-4 terminal
 *  overlay — so the rail sits above the collapsed sidebars' z-30 hover-reveal
 *  triggers (z-40, like the thread timeline) and suppresses them while hovered.
 *  The rail is always shown when a terminal exists (even one), so every tab keeps
 *  its close affordance; closing the last one hides the pane (reopen re-creates).
 *
 *  A pane can still be on screen with no terminal left: a remote shell ends when
 *  its window switches connection, and its tab goes with it. The rail (and its
 *  "+") is gone then, so the empty pane offers the way back itself. */
export function TerminalPaneChrome() {
  const terminals = useStore($terminals)
  const { t } = useI18n()

  return (
    <div className="flex min-h-0 min-w-0 flex-1">
      <div className="relative flex min-h-0 min-w-0 flex-1 flex-col">
        <TerminalSlot />
        {terminals.length === 0 && (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <button
              className="rounded-md border border-current/20 px-3 py-1.5 text-xs text-(--ui-text-secondary) hover:opacity-80"
              onClick={() => createTerminal()}
              type="button"
            >
              {t.rightSidebar.terminalNew}
            </button>
          </div>
        )}
      </div>
      {terminals.length > 0 && <TerminalRail />}
    </div>
  )
}
