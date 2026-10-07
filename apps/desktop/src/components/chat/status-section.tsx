import './status-stack.css'

import { type ReactNode, useState } from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'

interface StatusSectionProps {
  /** Optional right-aligned actions (text links / micro buttons). Pass
   *  `Button` with `size="micro"` + `variant="text"` or `"link"`. */
  accessory?: ReactNode
  children: ReactNode
  collapsed?: boolean
  /** Optional inline status next to the label (running spinner, etc). */
  collapsedIndicator?: ReactNode
  defaultCollapsed?: boolean
  /** Optional glyph between the caret and the label (e.g. a `Codicon`). */
  icon?: ReactNode
  label: ReactNode
  onToggle?: () => void
  /** Compact live content stays visible while the full roster is collapsed. */
  preview?: ReactNode
}

/**
 * One collapsible group inside the composer status stack. Pure chrome — header
 * (caret + label) + body — styled to match the queue exactly so every status
 * (queue, subagents, background) reads as one piece. The stack supplies the
 * outer card and the dividers between groups; this owns only its own collapse.
 */
export function StatusSection({
  accessory,
  children,
  collapsed: controlledCollapsed,
  collapsedIndicator,
  defaultCollapsed = true,
  icon,
  label,
  onToggle,
  preview
}: StatusSectionProps) {
  const [uncontrolledCollapsed, setUncontrolledCollapsed] = useState(defaultCollapsed)
  const isControlled = typeof controlledCollapsed === 'boolean'
  const collapsed = isControlled ? controlledCollapsed : uncontrolledCollapsed

  const handleToggle = () => {
    if (onToggle) {
      onToggle()
    }

    if (!isControlled) {
      setUncontrolledCollapsed(open => !open)
    }
  }

  return (
    <div data-slot="status-section">
      <div className="status-section-header flex items-center gap-1 pr-1">
        <button
          aria-expanded={!collapsed}
          className="status-section-trigger flex min-w-0 flex-1 items-center gap-1.5 px-2 py-1 text-left text-xs font-normal text-muted-foreground/92 transition-colors hover:text-foreground/90"
          onClick={handleToggle}
          type="button"
        >
          <DisclosureCaret className="shrink-0" open={!collapsed} size="1em" />
          {icon && <span className="status-section-icon flex shrink-0 items-center">{icon}</span>}
          <span className="min-w-0 truncate">{label}</span>
          {collapsedIndicator && <span className="flex shrink-0 items-center">{collapsedIndicator}</span>}
        </button>
        {accessory && <div className="flex shrink-0 items-center gap-1">{accessory}</div>}
      </div>
      {(!collapsed || preview) && <div className="status-section-body">{collapsed ? preview : children}</div>}
    </div>
  )
}
