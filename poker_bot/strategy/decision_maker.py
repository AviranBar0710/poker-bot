"""Core decision engine for Texas Hold'em.

Produces optimal actions by combining pre-flop range analysis,
post-flop equity calculation, pot odds, bet sizing, and ICM
adjustments into a single decision pipeline.

Architecture:
  GameContext + Position + HoleCards + ActionHistory
    → Pre-flop module (range check, action selection)
    → Post-flop module (equity vs range, pot odds, board texture)
    → Tournament adjustment layer (ICM, bubble factor)
    → Decision(action, amount, reasoning)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from poker_bot.core.equity_calculator import EquityCalculator
from poker_bot.core.game_context import GameContext, TournamentPhase
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.core.hand_evaluator import HandEvaluator, HandResult
from poker_bot.strategy.preflop_ranges import (
    HandNotation,
    HandType,
    Range,
    _RANK_INDEX,
    get_3bet_range,
    get_4bet_range,
    get_call_vs_raise_range,
    get_opening_range,
)
from poker_bot.strategy.stack_strategy import get_push_fold_range
from poker_bot.strategy.tournament_strategy import survival_premium
from poker_bot.utils.card import Card
from poker_bot.utils.constants import (
    RANK_VALUES,
    Action,
    HandRanking,
    Position,
    Rank,
    Street,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ActionType(StrEnum):
    """The action types the decision engine can recommend."""

    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    RAISE = "RAISE"
    ALL_IN = "ALL_IN"


@dataclass(frozen=True)
class PriorAction:
    """A single action in the hand's action history."""

    position: Position
    action: Action
    amount: float = 0.0


@dataclass(frozen=True)
class Decision:
    """The engine's recommended action with reasoning."""

    action: ActionType
    amount: float  # 0 for fold/check, call amount for call, raise-to for raise
    reasoning: str
    equity: float = 0.0  # Hero's estimated equity [0, 1]
    pot_odds: float = 0.0  # Required equity to call [0, 1]


@dataclass
class BoardTexture:
    """Analysis of the community card texture."""

    is_monotone: bool = False  # All one suit
    is_two_tone: bool = False  # Two suits
    is_rainbow: bool = False  # Three suits (flop)
    is_paired: bool = False  # Board has a pair
    is_connected: bool = False  # Cards within 2 ranks
    high_card_rank: int = 0
    num_broadway: int = 0  # Cards T or higher
    has_flush_draw: bool = False  # 3 of same suit
    has_straight_draw: bool = False  # 3 connected cards


# ---------------------------------------------------------------------------
# Board texture analysis
# ---------------------------------------------------------------------------


def analyze_board(community_cards: list[Card]) -> BoardTexture:
    """Analyze the texture of the community cards."""
    if not community_cards:
        return BoardTexture()

    suits = [c.suit for c in community_cards]
    values = sorted([c.value for c in community_cards], reverse=True)
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1

    max_suit_count = max(suit_counts.values())
    unique_suits = len(suit_counts)

    # Connectedness: check if any 3 cards are within a 5-card window
    connected = False
    if len(values) >= 3:
        for i in range(len(values) - 2):
            if values[i] - values[i + 2] <= 4:
                connected = True
                break

    # Paired board
    rank_counts = {}
    for c in community_cards:
        rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1
    is_paired = any(v >= 2 for v in rank_counts.values())

    num_broadway = sum(1 for v in values if v >= 10)

    return BoardTexture(
        is_monotone=unique_suits == 1,
        is_two_tone=max_suit_count == 2 and unique_suits == 2,
        is_rainbow=unique_suits >= 3,
        is_paired=is_paired,
        is_connected=connected,
        high_card_rank=values[0] if values else 0,
        num_broadway=num_broadway,
        has_flush_draw=max_suit_count >= 3,
        has_straight_draw=connected,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hand_in_range(card1: Card, card2: Card, hand_range: Range) -> bool:
    """Check if a specific hand is within a range."""
    return hand_range.contains(card1, card2)


def _hand_to_notation(card1: Card, card2: Card) -> HandNotation:
    """Convert two specific cards to their notation form."""
    r1, r2 = card1.rank, card2.rank
    # Ensure rank1 is the higher rank
    if _RANK_INDEX[r1] > _RANK_INDEX[r2]:
        r1, r2 = r2, r1

    if r1 == r2:
        return HandNotation(r1, r2, HandType.PAIR)
    if card1.suit == card2.suit:
        return HandNotation(r1, r2, HandType.SUITED)
    return HandNotation(r1, r2, HandType.OFFSUIT)


def _calculate_pot_odds(call_amount: float, pot: float) -> float:
    """Calculate pot odds: the equity needed to make calling profitable.

    pot_odds = call / (pot + call)
    """
    total = pot + call_amount
    if total <= 0:
        return 0.0
    return call_amount / total


def _min_raise_amount(current_bet: float, big_blind: float) -> float:
    """Calculate the minimum legal raise amount (raise TO, not BY)."""
    return max(current_bet * 2, big_blind * 2)


# ---------------------------------------------------------------------------
# Bet sizing
# ---------------------------------------------------------------------------


def compute_raise_size(
    pot: float,
    current_bet: float,
    hero_stack: float,
    big_blind: float,
    street: Street,
    hand_strength: float,
    board_texture: BoardTexture,
    spr: float,
) -> float:
    """Compute optimal raise/bet size.

    Args:
        pot: Current pot size.
        current_bet: Current bet hero must respond to (0 if opening).
        hero_stack: Hero's remaining chips.
        big_blind: Big blind amount.
        street: Current street.
        hand_strength: Estimated hand strength [0, 1].
        board_texture: Board texture analysis.
        spr: Stack-to-pot ratio.

    Returns:
        Raise-to amount (capped at hero's stack).
    """
    min_raise = _min_raise_amount(current_bet, big_blind)

    if street == Street.PREFLOP:
        # Standard open: 2.5bb from most positions, 3bb from EP
        if current_bet <= big_blind:
            open_size = big_blind * 2.5
        else:
            # 3-bet or 4-bet: ~3x the previous raise
            open_size = current_bet * 3.0
        return min(max(open_size, min_raise), hero_stack)

    # Post-flop sizing based on pot geometry and texture
    effective_pot = pot + current_bet  # Pot after villain's bet is included

    if current_bet > 0:
        # Facing a bet — raise to ~3x
        raise_size = current_bet * 3.0
    elif spr <= 1.5:
        # Low SPR: bet large to set up all-in
        raise_size = effective_pot * 0.75
    elif board_texture.is_monotone or board_texture.has_flush_draw:
        # Wet board: bet larger to deny equity
        raise_size = effective_pot * 0.75
    elif board_texture.is_connected or board_texture.has_straight_draw:
        # Connected board: medium-large sizing
        raise_size = effective_pot * 0.66
    elif board_texture.is_paired and not board_texture.is_connected:
        # Dry paired board: smaller sizing
        raise_size = effective_pot * 0.33
    elif board_texture.is_rainbow and not board_texture.is_connected:
        # Dry rainbow board: smaller sizing
        raise_size = effective_pot * 0.33
    else:
        # Default: half pot
        raise_size = effective_pot * 0.50

    return min(max(raise_size, min_raise), hero_stack)


# ---------------------------------------------------------------------------
# Pre-flop decision logic
# ---------------------------------------------------------------------------


class PreflopEngine:
    """Pre-flop decision making based on range analysis."""

    @staticmethod
    def decide(
        card1: Card,
        card2: Card,
        position: Position,
        context: GameContext,
        pot: float,
        current_bet: float,
        hero_stack: float,
        big_blind: float,
        action_history: list[PriorAction],
    ) -> Decision:
        """Make a pre-flop decision.

        Determines the action based on position, ranges, action history,
        and game context.
        """
        facing_raise = current_bet > big_blind
        facing_3bet = _count_raises(action_history) >= 2
        facing_4bet = _count_raises(action_history) >= 3

        # Push/fold for very short stacks
        if context.stack_category in ("very_short", "critical"):
            return PreflopEngine._push_fold_decision(
                card1, card2, position, context, pot, hero_stack, big_blind
            )

        # Facing a 4-bet+: very tight continue range
        if facing_4bet:
            return PreflopEngine._facing_4bet(
                card1, card2, position, context, pot, current_bet,
                hero_stack, big_blind,
            )

        # Facing a 3-bet: 4-bet, call, or fold
        if facing_3bet:
            return PreflopEngine._facing_3bet(
                card1, card2, position, context, pot, current_bet,
                hero_stack, big_blind,
            )

        # Facing an open raise: 3-bet, call, or fold
        if facing_raise:
            return PreflopEngine._facing_raise(
                card1, card2, position, context, pot, current_bet,
                hero_stack, big_blind,
            )

        # No raise yet: open or fold
        return PreflopEngine._open_action(
            card1, card2, position, context, pot, current_bet,
            hero_stack, big_blind,
        )

    @staticmethod
    def _push_fold_decision(
        card1: Card,
        card2: Card,
        position: Position,
        context: GameContext,
        pot: float,
        hero_stack: float,
        big_blind: float,
    ) -> Decision:
        pf_range = get_push_fold_range(context.stack_depth_bb, position)
        if pf_range and _hand_in_range(card1, card2, pf_range):
            # Apply tournament tightening
            if context.is_tournament:
                premium = survival_premium(context)
                # For push/fold, use a simple threshold
                hand = _hand_to_notation(card1, card2)
                strength = _hand_notation_strength(hand)
                threshold = 1.0 - premium  # Higher premium = lower threshold
                if strength < threshold:
                    return Decision(
                        action=ActionType.FOLD,
                        amount=0.0,
                        reasoning=(
                            f"Push/fold: {hand} in range but ICM pressure "
                            f"(survival premium={premium:.2f}) dictates fold"
                        ),
                    )
            return Decision(
                action=ActionType.ALL_IN,
                amount=hero_stack,
                reasoning=f"Push/fold: {_hand_to_notation(card1, card2)} is in "
                          f"push range for {position} at {context.stack_depth_bb:.0f}bb",
            )
        return Decision(
            action=ActionType.FOLD,
            amount=0.0,
            reasoning=f"Push/fold: {_hand_to_notation(card1, card2)} not in "
                      f"push range for {position} at {context.stack_depth_bb:.0f}bb",
        )

    @staticmethod
    def _open_action(
        card1: Card,
        card2: Card,
        position: Position,
        context: GameContext,
        pot: float,
        current_bet: float,
        hero_stack: float,
        big_blind: float,
    ) -> Decision:
        open_range = get_opening_range(position, context)
        hand = _hand_to_notation(card1, card2)

        if _hand_in_range(card1, card2, open_range):
            raise_size = compute_raise_size(
                pot, current_bet, hero_stack, big_blind,
                Street.PREFLOP, 0.5, BoardTexture(), hero_stack / pot if pot > 0 else 100,
            )
            return Decision(
                action=ActionType.RAISE,
                amount=raise_size,
                reasoning=f"Open raise: {hand} is in {position} opening range "
                          f"({open_range.percentage:.1f}% of hands)",
            )

        # In BB with no raise, check instead of fold
        if position == Position.BB and current_bet <= big_blind:
            return Decision(
                action=ActionType.CHECK,
                amount=0.0,
                reasoning=f"BB check: {hand} not strong enough to raise",
            )

        return Decision(
            action=ActionType.FOLD,
            amount=0.0,
            reasoning=f"Fold: {hand} not in {position} opening range",
        )

    @staticmethod
    def _facing_raise(
        card1: Card,
        card2: Card,
        position: Position,
        context: GameContext,
        pot: float,
        current_bet: float,
        hero_stack: float,
        big_blind: float,
    ) -> Decision:
        hand = _hand_to_notation(card1, card2)

        # Check 3-bet range first (stronger action)
        three_bet_range = get_3bet_range(position, context)
        if _hand_in_range(card1, card2, three_bet_range):
            raise_size = compute_raise_size(
                pot, current_bet, hero_stack, big_blind,
                Street.PREFLOP, 0.7, BoardTexture(), hero_stack / pot if pot > 0 else 100,
            )
            return Decision(
                action=ActionType.RAISE,
                amount=raise_size,
                reasoning=f"3-bet: {hand} is in {position} 3-bet range",
            )

        # Check calling range
        call_range = get_call_vs_raise_range(position, context)
        if _hand_in_range(card1, card2, call_range):
            call_amount = current_bet
            pot_odds = _calculate_pot_odds(call_amount, pot)
            return Decision(
                action=ActionType.CALL,
                amount=call_amount,
                reasoning=f"Call raise: {hand} is in {position} calling range "
                          f"(pot odds: {pot_odds:.1%})",
                pot_odds=pot_odds,
            )

        return Decision(
            action=ActionType.FOLD,
            amount=0.0,
            reasoning=f"Fold vs raise: {hand} not in {position} 3-bet or call range",
        )

    @staticmethod
    def _facing_3bet(
        card1: Card,
        card2: Card,
        position: Position,
        context: GameContext,
        pot: float,
        current_bet: float,
        hero_stack: float,
        big_blind: float,
    ) -> Decision:
        hand = _hand_to_notation(card1, card2)

        # Check 4-bet range
        four_bet_range = get_4bet_range(position, context)
        if _hand_in_range(card1, card2, four_bet_range):
            raise_size = compute_raise_size(
                pot, current_bet, hero_stack, big_blind,
                Street.PREFLOP, 0.8, BoardTexture(), hero_stack / pot if pot > 0 else 100,
            )
            # If 4-bet would commit >40% of stack, just shove
            if raise_size > hero_stack * 0.4:
                return Decision(
                    action=ActionType.ALL_IN,
                    amount=hero_stack,
                    reasoning=f"4-bet all-in: {hand} — raise would commit "
                              f"too much of stack",
                )
            return Decision(
                action=ActionType.RAISE,
                amount=raise_size,
                reasoning=f"4-bet: {hand} is in {position} 4-bet range",
            )

        # Call with strong hands that aren't in 4-bet range (flatting range)
        # Roughly: hands in 3-bet range but not 4-bet range
        three_bet_range = get_3bet_range(position, context)
        if _hand_in_range(card1, card2, three_bet_range):
            call_amount = current_bet
            pot_odds = _calculate_pot_odds(call_amount, pot)
            return Decision(
                action=ActionType.CALL,
                amount=call_amount,
                reasoning=f"Call 3-bet: {hand} strong enough to continue "
                          f"(pot odds: {pot_odds:.1%})",
                pot_odds=pot_odds,
            )

        return Decision(
            action=ActionType.FOLD,
            amount=0.0,
            reasoning=f"Fold vs 3-bet: {hand} not in {position} continue range",
        )

    @staticmethod
    def _facing_4bet(
        card1: Card,
        card2: Card,
        position: Position,
        context: GameContext,
        pot: float,
        current_bet: float,
        hero_stack: float,
        big_blind: float,
    ) -> Decision:
        hand = _hand_to_notation(card1, card2)

        # Only continue with the very top of our range
        # AA, KK always continue; QQ, AKs sometimes
        four_bet_range = get_4bet_range(position, context)
        if _hand_in_range(card1, card2, four_bet_range):
            # Premium pairs: shove
            if hand.hand_type == HandType.PAIR and hand.rank1 in (Rank.ACE, Rank.KING):
                return Decision(
                    action=ActionType.ALL_IN,
                    amount=hero_stack,
                    reasoning=f"5-bet all-in vs 4-bet: {hand} — premium pair",
                )
            # AKs: call or shove depending on stack depth
            if context.stack_depth_bb <= 100:
                call_amount = current_bet
                pot_odds = _calculate_pot_odds(call_amount, pot)
                return Decision(
                    action=ActionType.CALL,
                    amount=call_amount,
                    reasoning=f"Call 4-bet: {hand} with {context.stack_depth_bb:.0f}bb "
                              f"(pot odds: {pot_odds:.1%})",
                    pot_odds=pot_odds,
                )
            return Decision(
                action=ActionType.CALL,
                amount=current_bet,
                reasoning=f"Call 4-bet: {hand} in continue range",
            )

        return Decision(
            action=ActionType.FOLD,
            amount=0.0,
            reasoning=f"Fold vs 4-bet: {hand} not strong enough to continue",
        )


def _count_raises(history: list[PriorAction]) -> int:
    """Count the number of raises in the action history."""
    return sum(1 for a in history if a.action in (Action.RAISE, Action.ALL_IN))


def _hand_notation_strength(hand: HandNotation) -> float:
    """Estimate hand strength as a percentile [0, 1] for push/fold decisions.

    1.0 = strongest (AA), 0.0 = weakest (32o).
    """
    r1_val = 13 - _RANK_INDEX[hand.rank1]  # 13 = A, 0 = 2
    r2_val = 13 - _RANK_INDEX[hand.rank2]

    if hand.hand_type == HandType.PAIR:
        # Pairs: AA=1.0, KK≈0.95, ..., 22≈0.5
        return 0.5 + (r1_val / 13) * 0.5
    if hand.hand_type == HandType.SUITED:
        return (r1_val + r2_val) / 26 * 0.8
    # Offsuit
    return (r1_val + r2_val) / 26 * 0.6


# ---------------------------------------------------------------------------
# Post-flop decision logic
# ---------------------------------------------------------------------------


class PostflopEngine:
    """Post-flop decision making based on equity and pot odds."""

    @staticmethod
    def decide(
        hero_cards: list[Card],
        community_cards: list[Card],
        position: Position,
        context: GameContext,
        pot: float,
        current_bet: float,
        hero_stack: float,
        big_blind: float,
        num_opponents: int,
        action_history: list[PriorAction],
        opponent_range: Range | None = None,
    ) -> Decision:
        """Make a post-flop decision.

        Uses hand evaluation, equity calculation, pot odds, and board
        texture to determine the optimal action.
        """
        board_texture = analyze_board(community_cards)
        hand_result = HandEvaluator.evaluate(hero_cards + community_cards)
        hand_strength = PostflopEngine._hand_strength_score(
            hand_result, community_cards
        )
        spr = hero_stack / pot if pot > 0 else float("inf")

        # Estimate opponent range if not provided
        if opponent_range is None:
            opponent_range = PostflopEngine._estimate_opponent_range(
                action_history, context
            )

        # Calculate equity vs opponent range
        equity = PostflopEngine._estimate_equity(
            hero_cards, community_cards, opponent_range, hand_strength
        )

        facing_bet = current_bet > 0

        if facing_bet:
            return PostflopEngine._facing_bet(
                hero_cards, community_cards, position, context,
                pot, current_bet, hero_stack, big_blind,
                hand_result, hand_strength, equity, board_texture, spr,
                num_opponents,
            )

        return PostflopEngine._no_bet(
            hero_cards, community_cards, position, context,
            pot, hero_stack, big_blind,
            hand_result, hand_strength, equity, board_texture, spr,
            num_opponents,
        )

    @staticmethod
    def _facing_bet(
        hero_cards: list[Card],
        community_cards: list[Card],
        position: Position,
        context: GameContext,
        pot: float,
        current_bet: float,
        hero_stack: float,
        big_blind: float,
        hand_result: HandResult,
        hand_strength: float,
        equity: float,
        board_texture: BoardTexture,
        spr: float,
        num_opponents: int,
    ) -> Decision:
        """Decision when facing a bet."""
        call_amount = current_bet
        pot_odds = _calculate_pot_odds(call_amount, pot)

        # Apply ICM adjustment to required equity
        # A survival premium < 1.0 means chips lost cost more than chips
        # gained (ICM tax), so we require more equity to continue.
        required_equity = pot_odds
        if context.is_tournament:
            premium = survival_premium(context)
            if 0 < premium < 1.0:
                required_equity = pot_odds / premium

        # Strong made hand: raise for value
        if hand_strength >= 0.85 and equity > 0.70:
            raise_size = compute_raise_size(
                pot, current_bet, hero_stack, big_blind,
                Street.FLOP,  # Simplified — actual street doesn't affect raise calc much
                hand_strength, board_texture, spr,
            )
            if raise_size >= hero_stack:
                return Decision(
                    action=ActionType.ALL_IN,
                    amount=hero_stack,
                    reasoning=f"All-in for value: {hand_result.ranking.name} "
                              f"with {equity:.0%} equity",
                    equity=equity,
                    pot_odds=pot_odds,
                )
            return Decision(
                action=ActionType.RAISE,
                amount=raise_size,
                reasoning=f"Raise for value: {hand_result.ranking.name} "
                          f"with {equity:.0%} equity",
                equity=equity,
                pot_odds=pot_odds,
            )

        # Enough equity to call
        if equity >= required_equity:
            # Check for implied odds on draws
            implied_odds_bonus = 0.0
            if hand_strength < 0.5 and spr > 3:
                # Drawing hand with good implied odds
                implied_odds_bonus = min(0.10, (spr - 3) * 0.02)

            if equity + implied_odds_bonus >= required_equity:
                return Decision(
                    action=ActionType.CALL,
                    amount=call_amount,
                    reasoning=f"Call: {equity:.0%} equity vs "
                              f"{required_equity:.0%} required "
                              f"({hand_result.ranking.name})"
                              + (f", +{implied_odds_bonus:.0%} implied odds"
                                 if implied_odds_bonus > 0 else ""),
                    equity=equity,
                    pot_odds=pot_odds,
                )

        # Semi-bluff raise with draws on wet boards
        if (
            hand_strength >= 0.3
            and equity >= 0.25
            and (board_texture.has_flush_draw or board_texture.has_straight_draw)
            and spr > 2
        ):
            raise_size = compute_raise_size(
                pot, current_bet, hero_stack, big_blind,
                Street.FLOP, hand_strength, board_texture, spr,
            )
            return Decision(
                action=ActionType.RAISE,
                amount=raise_size,
                reasoning=f"Semi-bluff raise: {equity:.0%} equity with "
                          f"draw on wet board",
                equity=equity,
                pot_odds=pot_odds,
            )

        return Decision(
            action=ActionType.FOLD,
            amount=0.0,
            reasoning=f"Fold: {equity:.0%} equity < "
                      f"{required_equity:.0%} required "
                      f"({hand_result.ranking.name})",
            equity=equity,
            pot_odds=pot_odds,
        )

    @staticmethod
    def _no_bet(
        hero_cards: list[Card],
        community_cards: list[Card],
        position: Position,
        context: GameContext,
        pot: float,
        hero_stack: float,
        big_blind: float,
        hand_result: HandResult,
        hand_strength: float,
        equity: float,
        board_texture: BoardTexture,
        spr: float,
        num_opponents: int,
    ) -> Decision:
        """Decision when checked to us (no bet to face)."""
        # Strong hand: bet for value
        if hand_strength >= 0.65 and equity > 0.55:
            bet_size = compute_raise_size(
                pot, 0, hero_stack, big_blind,
                Street.FLOP, hand_strength, board_texture, spr,
            )
            if bet_size >= hero_stack:
                return Decision(
                    action=ActionType.ALL_IN,
                    amount=hero_stack,
                    reasoning=f"All-in for value: {hand_result.ranking.name} "
                              f"with {equity:.0%} equity, low SPR ({spr:.1f})",
                    equity=equity,
                )
            return Decision(
                action=ActionType.RAISE,
                amount=bet_size,
                reasoning=f"Bet for value: {hand_result.ranking.name} "
                          f"with {equity:.0%} equity",
                equity=equity,
            )

        # Semi-bluff with draws
        if (
            hand_strength >= 0.25
            and equity >= 0.30
            and (board_texture.has_flush_draw or board_texture.has_straight_draw)
        ):
            bet_size = compute_raise_size(
                pot, 0, hero_stack, big_blind,
                Street.FLOP, hand_strength, board_texture, spr,
            )
            return Decision(
                action=ActionType.RAISE,
                amount=bet_size,
                reasoning=f"Semi-bluff bet: {equity:.0%} equity with "
                          f"draw ({hand_result.ranking.name})",
                equity=equity,
            )

        # Weak hand: check
        return Decision(
            action=ActionType.CHECK,
            amount=0.0,
            reasoning=f"Check: {hand_result.ranking.name} with "
                      f"{equity:.0%} equity — not strong enough to bet",
            equity=equity,
        )

    @staticmethod
    def _hand_strength_score(
        hand_result: HandResult,
        community_cards: list[Card],
    ) -> float:
        """Score hand strength on a 0-1 scale relative to the board.

        Takes into account both the absolute hand ranking and how the
        hand connects with the board.
        """
        ranking = hand_result.ranking

        # Base score from hand ranking
        base_scores: dict[HandRanking, float] = {
            HandRanking.HIGH_CARD: 0.10,
            HandRanking.ONE_PAIR: 0.40,
            HandRanking.TWO_PAIR: 0.65,
            HandRanking.THREE_OF_A_KIND: 0.85,
            HandRanking.STRAIGHT: 0.88,
            HandRanking.FLUSH: 0.90,
            HandRanking.FULL_HOUSE: 0.94,
            HandRanking.FOUR_OF_A_KIND: 0.97,
            HandRanking.STRAIGHT_FLUSH: 0.99,
            HandRanking.ROYAL_FLUSH: 1.00,
        }
        score = base_scores.get(ranking, 0.0)

        # Adjust pair strength based on board
        if ranking == HandRanking.ONE_PAIR and community_cards:
            board_values = sorted(
                [c.value for c in community_cards], reverse=True
            )
            best_card_val = max(c.value for c in hand_result.best_cards)
            if best_card_val >= board_values[0]:
                score += 0.25  # Top pair or overpair
            elif len(board_values) >= 2 and best_card_val >= board_values[1]:
                score += 0.10  # Second pair

        return min(score, 1.0)

    @staticmethod
    def _estimate_equity(
        hero_cards: list[Card],
        community_cards: list[Card],
        opponent_range: Range,
        hand_strength: float,
    ) -> float:
        """Estimate equity vs opponent range.

        Uses Monte Carlo simulation for accuracy, with fewer simulations
        for obvious spots to save time.
        """
        # Fast-track obvious spots
        if hand_strength >= 0.95:
            return 0.95
        if hand_strength <= 0.05:
            return 0.05

        # Adjust simulation count based on how close the decision is
        sims = 1000
        if 0.25 <= hand_strength <= 0.75:
            sims = 2000  # More sims for close decisions

        try:
            result = EquityCalculator.hand_vs_range(
                hero_cards, opponent_range,
                board=community_cards,
                simulations=sims,
            )
            return result.equity
        except ValueError:
            # Fallback if range has no valid combos
            return hand_strength

    @staticmethod
    def _estimate_opponent_range(
        action_history: list[PriorAction],
        context: GameContext,
    ) -> Range:
        """Estimate opponent's range based on their actions.

        Starts with a position-based range and narrows based on
        aggression shown.
        """
        raises = _count_raises(action_history)

        if raises >= 3:
            # Very aggressive: narrow to premium hands
            return Range().add("AA,KK,QQ,AKs")
        if raises >= 2:
            # 3-bet pot: strong range
            return Range().add("AA,KK,QQ,JJ,TT,AKs,AQs,AKo")
        if raises >= 1:
            # Single raise: standard opening range (approximate CO)
            return Range().add(
                "AA,KK,QQ,JJ,TT,99,88,77,66,55,"
                "AKs,AQs,AJs,ATs,A9s,A8s,A5s,A4s,"
                "AKo,AQo,AJo,ATo,"
                "KQs,KJs,KTs,"
                "KQo,KJo,"
                "QJs,QTs,"
                "JTs,"
                "T9s,"
                "98s,"
                "87s"
            )

        # Limped/checked: wide range
        return Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
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
            "76s,75s,"
            "65s,64s,"
            "54s"
        )


# ---------------------------------------------------------------------------
# Main decision engine
# ---------------------------------------------------------------------------


class DecisionMaker:
    """Top-level decision engine combining pre-flop and post-flop logic.

    Usage:
        maker = DecisionMaker()
        decision = maker.make_decision(game_state, context, hero_index, action_history)
    """

    def make_decision(
        self,
        game_state: GameState,
        context: GameContext,
        hero_index: int,
        action_history: list[PriorAction] | None = None,
        opponent_range: Range | None = None,
    ) -> Decision:
        """Produce a decision for the hero player.

        Args:
            game_state: Current game state.
            context: Game context (cash/tournament, stack depth, etc).
            hero_index: Index of the hero in game_state.players.
            action_history: Prior actions in this hand.
            opponent_range: Optional explicit opponent range.

        Returns:
            Decision with action, amount, and reasoning.
        """
        hero = game_state.players[hero_index]
        action_history = action_history or []

        if not hero.hole_cards or len(hero.hole_cards) < 2:
            return Decision(
                action=ActionType.FOLD,
                amount=0.0,
                reasoning="No hole cards — cannot make a decision",
            )

        card1, card2 = hero.hole_cards[0], hero.hole_cards[1]

        # Effective call = current bet minus what hero already posted
        effective_bet = max(0.0, game_state.current_bet - hero.current_bet)

        if game_state.current_street == Street.PREFLOP:
            decision = PreflopEngine.decide(
                card1, card2, hero.position, context,
                game_state.pot, effective_bet,
                hero.chips, game_state.big_blind, action_history,
            )
        else:
            decision = PostflopEngine.decide(
                hero.hole_cards, game_state.community_cards,
                hero.position, context,
                game_state.pot, effective_bet,
                hero.chips, game_state.big_blind,
                max(1, game_state.players_in_hand - 1),
                action_history,
                opponent_range,
            )

        # Clamp amounts to legal bounds
        return _clamp_decision(decision, hero.chips, game_state.big_blind)


def _clamp_decision(
    decision: Decision,
    hero_stack: float,
    big_blind: float,
) -> Decision:
    """Ensure decision amounts are legal.

    - Amounts are clamped to [0, hero_stack]
    - Raises that would be all-in are converted to ALL_IN
    - Raises below minimum are bumped up (or converted to ALL_IN)
    - Calls that exceed stack are converted to ALL_IN
    """
    amount = decision.amount

    if decision.action in (ActionType.FOLD, ActionType.CHECK):
        return decision

    # Clamp to stack, never negative
    amount = max(0.0, min(amount, hero_stack))

    # Enforce minimum raise for RAISE actions
    if decision.action == ActionType.RAISE:
        min_raise = big_blind * 2
        if amount < min_raise:
            # Can't legally raise this small — go to min raise or all-in
            amount = min(min_raise, hero_stack)

    # If raise/call would be all-in, convert to ALL_IN
    if decision.action in (ActionType.RAISE, ActionType.CALL) and amount >= hero_stack:
        return Decision(
            action=ActionType.ALL_IN,
            amount=hero_stack,
            reasoning=decision.reasoning + " (all-in)",
            equity=decision.equity,
            pot_odds=decision.pot_odds,
        )

    return Decision(
        action=decision.action,
        amount=amount,
        reasoning=decision.reasoning,
        equity=decision.equity,
        pot_odds=decision.pot_odds,
    )
