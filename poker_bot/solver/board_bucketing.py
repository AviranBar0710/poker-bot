"""Board texture classification and bucketing.

Reduces the combinatorial explosion of board textures into ~12 buckets
for efficient pre-computed strategy lookup. Also provides stack and SPR
bucketing for strategy selection.
"""

from __future__ import annotations

from poker_bot.strategy.decision_maker import BoardTexture


def bucket_board(texture: BoardTexture) -> str:
    """Classify a board texture into a bucket category.

    Returns one of ~12 categories that capture the strategically
    relevant features of the board:
      - dry_high_rainbow: High cards, no draws (e.g. AK7r)
      - dry_low_rainbow: Low cards, no draws (e.g. 742r)
      - wet_connected: Straight draw heavy (e.g. JT8)
      - wet_two_tone: Flush draw present (e.g. Ks Js 7d)
      - monotone_high: Three to a flush with high cards
      - monotone_low: Three to a flush with low cards
      - paired_high: Paired board with high card (e.g. KK7)
      - paired_low: Paired board with low card (e.g. 773)
      - broadway_heavy: Multiple broadway cards, connected
      - connected_low: Low connected cards (e.g. 654)
      - dry_medium: Medium cards, no draws
      - dynamic: Multiple draw possibilities

    Args:
        texture: BoardTexture analysis from decision_maker.analyze_board.

    Returns:
        String bucket name.
    """
    high = texture.high_card_rank

    # Monotone boards (all one suit)
    if texture.is_monotone:
        if high >= 10:
            return "monotone_high"
        return "monotone_low"

    # Paired boards
    if texture.is_paired:
        if high >= 10:
            return "paired_high"
        return "paired_low"

    # Dynamic: both flush draw and straight draw
    if texture.has_flush_draw and texture.has_straight_draw:
        return "dynamic"

    # Connected boards
    if texture.is_connected:
        if texture.is_two_tone:
            return "wet_two_tone"
        if high >= 10 and texture.num_broadway >= 2:
            return "broadway_heavy"
        if high < 8:
            return "connected_low"
        return "wet_connected"

    # Two-tone but not connected
    if texture.is_two_tone:
        return "wet_two_tone"

    # Rainbow, not connected, not paired
    if texture.is_rainbow:
        if high >= 10:
            return "dry_high_rainbow"
        if high <= 8:
            return "dry_low_rainbow"
        return "dry_medium"

    # Default
    return "dry_medium"


def bucket_stack(stack_bb: float) -> str:
    """Classify stack depth into a bucket for strategy lookup.

    Five tiers matching common strategic thresholds:
      - critical: <10bb (push/fold territory)
      - very_short: 10-19bb (reshove territory)
      - short: 20-39bb (limited post-flop play)
      - medium: 40-99bb (standard play)
      - deep: 100bb+ (full post-flop game)

    Args:
        stack_bb: Stack size in big blinds.

    Returns:
        String bucket name.
    """
    if stack_bb < 10:
        return "critical"
    if stack_bb < 20:
        return "very_short"
    if stack_bb < 40:
        return "short"
    if stack_bb < 100:
        return "medium"
    return "deep"


def bucket_spr(spr: float) -> str:
    """Classify stack-to-pot ratio into a bucket.

    Three tiers:
      - low: SPR <4 (committed or near-committed)
      - medium: SPR 4-10 (standard postflop play)
      - high: SPR >10 (deep relative to pot)

    Args:
        spr: Stack-to-pot ratio.

    Returns:
        String bucket name.
    """
    if spr < 4:
        return "low"
    if spr <= 10:
        return "medium"
    return "high"
