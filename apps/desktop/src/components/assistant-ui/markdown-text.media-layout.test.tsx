import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { MessageTextContent } from './markdown-text'

// The shapes of two real turns (prose and MEDIA lines alternating), trimmed:
// a wide 1800x750 check strip, then an mp4 followed by a square png.
const WIDE_TURN = [
  'So I composited you into the splat at eye level with a short camera slide.',
  'MEDIA:/Users/me/splats/renders/splat_justin2_check.jpg',
  'The middle frame is the first time real you and the real splat look like one image.'
].join('\n\n')

const VIDEO_THEN_SQUARE_TURN = [
  'The vertical version is a native 4:5 crop of the 4K file.',
  'MEDIA:/Users/me/splats/jeep/jeep_reveal_v2_4x5.mp4',
  'Voxel apartment in Blender Cycles.',
  'MEDIA:/Users/me/splats/blender/voxel_final_35.png',
  'HeyGen is connected and waiting for your sign-in.'
].join('\n\n')

// No blank line: the prose starts on the line right under the card.
const TIGHT_TURN = 'Portrait check:\nMEDIA:/Users/me/splats/portrait.png\nThe feet are clipped by the frame edge.'

// Tailwind Typography's media-margin rule, as the plugin builds it (`inWhere`
// in @tailwindcss/typography). It ties the cards' `m-0` on specificity and
// wins on order, so a matching <img> sits 2em below the top of its frame while
// `max-h-full` still lets it fill the frame: it overflows the reserved box by
// 2em and paints over the next line of text.
const TYPOGRAPHY_MEDIA_MARGIN = '.prose :where(img, video):not(:where([class~="not-prose"],[class~="not-prose"] *))'

const CARD_SELECTOR = '[data-slot="aui_markdown-image"], video'

async function renderTurn(text: string, cards: number) {
  const { container } = render(<MessageTextContent text={text} />)

  // Settled once every card exists and every image card holds its <img>.
  await waitFor(() => {
    expect(container.querySelectorAll(CARD_SELECTOR)).toHaveLength(cards)
    container.querySelectorAll('[data-slot="aui_markdown-image"]').forEach(card => {
      expect(card.querySelector('img')).not.toBeNull()
    })
  })

  return container
}

function proseNode(container: HTMLElement, snippet: string): Node {
  const walker = container.ownerDocument.createTreeWalker(container, NodeFilter.SHOW_TEXT)

  while (walker.nextNode()) {
    if (walker.currentNode.textContent?.includes(snippet)) {
      return walker.currentNode
    }
  }

  throw new Error(`prose missing: ${snippet}`)
}

const follows = (a: Node, b: Node) => Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING)

describe('MEDIA cards in a turn', () => {
  afterEach(cleanup)

  it.each([
    ['a wide image', WIDE_TURN, 1],
    ['a video then a square image', VIDEO_THEN_SQUARE_TURN, 2],
    ['a portrait image with prose on the next line', TIGHT_TURN, 1]
  ])('renders one card per MEDIA line with the prose around each one, in order (%s)', async (_label, text, count) => {
    const container = await renderTurn(text, count)
    const cards = [...container.querySelectorAll(CARD_SELECTOR)]

    const prose = text
      .split('\n')
      .filter(line => line.trim() && !line.startsWith('MEDIA:'))
      .map(line => proseNode(container, line.slice(0, 24)))

    expect(container.textContent).not.toContain('MEDIA:')
    expect(prose).toHaveLength(cards.length + 1)

    cards.forEach((card, index) => {
      expect(follows(prose[index]!, card)).toBe(true)
      expect(follows(card, prose[index + 1]!)).toBe(true)
    })
  })

  it.each([
    ['a wide image', WIDE_TURN, 1],
    ['a video then a square image', VIDEO_THEN_SQUARE_TURN, 2],
    ['a portrait image with prose on the next line', TIGHT_TURN, 1]
  ])('keeps every card inside its own box so the next line is not covered (%s)', async (_label, text, count) => {
    const container = await renderTurn(text, count)
    const media = [...container.querySelectorAll('img, video')]

    // Not vacuous: the turn really renders inside the Typography scope.
    expect(container.querySelector('.prose')).not.toBeNull()
    expect(media).toHaveLength(count)

    for (const element of media) {
      expect(element.closest('.prose')).not.toBeNull()
      expect(element.matches(TYPOGRAPHY_MEDIA_MARGIN)).toBe(false)
    }
  })
})
