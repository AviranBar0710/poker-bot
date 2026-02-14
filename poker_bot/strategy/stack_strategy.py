"""Stack-depth specific strategy adjustments.

Adjusts pre-flop ranges and actions based on effective stack depth.
Deep stacks favor speculative hands and implied odds; short stacks
shift toward push/fold.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_bot.core.game_context import GameContext
from poker_bot.strategy.preflop_ranges import Range, expand_notation
from poker_bot.utils.constants import Position


@dataclass(frozen=True)
class StackAdjustment:
    """Describes how to modify a range for a given stack depth."""

    add_hands: str  # Comma-separated notation to add
    remove_hands: str  # Comma-separated notation to remove
    description: str

    def apply(self, base_range: Range) -> Range:
        """Apply this adjustment to a base range, returning a new Range."""
        new_range = Range(hands=set(base_range.hands))
        if self.remove_hands:
            new_range.remove(self.remove_hands)
        if self.add_hands:
            new_range.add(self.add_hands)
        return new_range


# ---------------------------------------------------------------------------
# Push/Fold Charts (< 10bb)
#
# Simplified Nash equilibrium push ranges by position.
# At critical stacks, the decision is binary: shove or fold.
# ---------------------------------------------------------------------------

PUSH_FOLD_RANGES: dict[float, dict[Position, Range]] = {
    # ~8-10 bb push ranges
    10: {
        Position.UTG: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,"
            "KQs,KJs,KTs,"
            "KQo"
        ),
        Position.MP: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,"
            "KQs,KJs,KTs,K9s,"
            "KQo,KJo,"
            "QJs,QTs,"
            "JTs"
        ),
        Position.CO: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,A7o,"
            "KQs,KJs,KTs,K9s,K8s,"
            "KQo,KJo,KTo,"
            "QJs,QTs,Q9s,"
            "QJo,"
            "JTs,J9s,"
            "T9s,"
            "98s"
        ),
        Position.BTN: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,A3o,A2o,"
            "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,"
            "KQo,KJo,KTo,K9o,"
            "QJs,QTs,Q9s,Q8s,Q7s,"
            "QJo,QTo,"
            "JTs,J9s,J8s,J7s,"
            "JTo,"
            "T9s,T8s,T7s,"
            "98s,97s,"
            "87s,86s,"
            "76s,75s,"
            "65s,"
            "54s"
        ),
        Position.SB: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,A3o,A2o,"
            "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,"
            "KQo,KJo,KTo,K9o,K8o,"
            "QJs,QTs,Q9s,Q8s,Q7s,Q6s,Q5s,"
            "QJo,QTo,Q9o,"
            "JTs,J9s,J8s,J7s,J6s,"
            "JTo,J9o,"
            "T9s,T8s,T7s,T6s,"
            "T9o,"
            "98s,97s,96s,"
            "87s,86s,85s,"
            "76s,75s,"
            "65s,64s,"
            "54s,53s,"
            "43s"
        ),
    },
    # ~5-7 bb push ranges (tighter)
    5: {
        Position.UTG: Range().add(
            "AA,KK,QQ,JJ,TT,99,"
            "AKs,AQs,AJs,ATs,A9s,"
            "AKo,AQo,"
            "KQs"
        ),
        Position.MP: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,"
            "AKo,AQo,AJo,"
            "KQs,KJs,"
            "QJs"
        ),
        Position.CO: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,"
            "KQs,KJs,KTs,"
            "KQo,"
            "QJs,QTs,"
            "JTs"
        ),
        Position.BTN: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,"
            "KQs,KJs,KTs,K9s,K8s,"
            "KQo,KJo,"
            "QJs,QTs,Q9s,"
            "QJo,"
            "JTs,J9s,"
            "T9s,"
            "98s,"
            "87s"
        ),
        Position.SB: Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,"
            "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
            "AKo,AQo,AJo,ATo,A9o,A8o,A7o,"
            "KQs,KJs,KTs,K9s,K8s,K7s,"
            "KQo,KJo,KTo,"
            "QJs,QTs,Q9s,Q8s,"
            "QJo,QTo,"
            "JTs,J9s,J8s,"
            "JTo,"
            "T9s,T8s,"
            "98s,97s,"
            "87s,86s,"
            "76s,"
            "65s,"
            "54s"
        ),
    },
}


# ---------------------------------------------------------------------------
# Stack-depth adjustments for standard open ranges
# ---------------------------------------------------------------------------

# Hands to ADD when deep-stacked (100bb+) — more speculative hands
_DEEP_STACK_ADD = (
    "55,44,33,22,"       # Small pairs for set mining
    "A9s-A2s,"           # Suited aces
    "K9s-K6s,"           # Suited kings
    "Q9s-Q7s,"           # Suited queens
    "J9s-J7s,"           # Suited jacks
    "T9s-T7s,"           # Suited connectors
    "98s-96s,87s-85s,76s-74s,65s-63s,54s-53s,43s"
)

# Hands to REMOVE when medium-stacked (40-100bb) — drop weakest speculative hands
_MEDIUM_STACK_REMOVE = (
    "43s,53s,64s,63s,74s,75s,85s,86s,96s,97s,"
    "T7s,J7s,Q7s,K6s,K7s"
)

# Hands to REMOVE when short-stacked (20-40bb) — significantly tighter
_SHORT_STACK_REMOVE = (
    "22,33,44,"
    "A2s,A3s,A4s,A6s,A7s,A8s,"
    "K8s,K9s,"
    "Q8s,Q9s,"
    "J8s,J9s,"
    "T8s,T9s,"
    "97s,98s,"
    "86s,87s,"
    "75s,76s,"
    "65s,64s,"
    "54s,53s,43s"
)


def _get_stack_adjustment(stack_category: str) -> StackAdjustment | None:
    """Get the range adjustment for a stack category."""
    match stack_category:
        case "deep":
            return StackAdjustment(
                add_hands=_DEEP_STACK_ADD,
                remove_hands="",
                description="Deep stack: wider with speculative hands",
            )
        case "medium":
            return StackAdjustment(
                add_hands="",
                remove_hands=_MEDIUM_STACK_REMOVE,
                description="Medium stack: tighter, fewer speculative hands",
            )
        case "short":
            return StackAdjustment(
                add_hands="",
                remove_hands=_SHORT_STACK_REMOVE,
                description="Short stack: significantly tighter",
            )
        case _:
            return None  # Very short / critical use push/fold


def get_push_fold_range(stack_bb: float, position: Position) -> Range | None:
    """Get the push/fold range for a given stack and position.

    Returns None if the stack is too deep for push/fold.

    Args:
        stack_bb: Stack size in big blinds.
        position: Player's table position.
    """
    if stack_bb > 15:
        return None

    # Use the closest chart
    if stack_bb <= 7:
        chart = PUSH_FOLD_RANGES.get(5, {})
    else:
        chart = PUSH_FOLD_RANGES.get(10, {})

    return chart.get(position)


def adjust_range_for_stack(
    base_range: Range,
    context: GameContext,
) -> Range:
    """Adjust a pre-flop range based on stack depth.

    For very short and critical stacks, returns a push/fold range
    instead of an adjusted open range.

    Args:
        base_range: The standard (100bb) opening range.
        context: Current game context.

    Returns:
        Adjusted range appropriate for the stack depth.
    """
    category = context.stack_category

    # Push/fold territory
    if category in ("very_short", "critical"):
        # Push/fold ranges are returned directly by get_push_fold_range
        # This function adjusts standard open ranges, so for push/fold
        # we return a tighter version of the base range
        adjustment = StackAdjustment(
            add_hands="",
            remove_hands=_SHORT_STACK_REMOVE,
            description=f"{category} stack: near push/fold territory",
        )
        return adjustment.apply(base_range)

    adjustment = _get_stack_adjustment(category)
    if adjustment is None:
        return base_range

    return adjustment.apply(base_range)
