"""Tournament-specific strategy adjustments.

Implements ICM (Independent Chip Model) calculations, bubble factors,
and survival premium adjustments for tournament play.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from poker_bot.core.game_context import GameContext, PayoutStructure, TournamentPhase
from poker_bot.strategy.preflop_ranges import Range


@dataclass
class ICMResult:
    """Result of an ICM equity calculation for all players."""

    equities: list[float]  # Tournament equity ($) for each player
    chip_stacks: list[float]  # Original chip stacks

    @property
    def total_equity(self) -> float:
        return sum(self.equities)

    def equity_for(self, player_index: int) -> float:
        return self.equities[player_index]

    def chip_ev(self, player_index: int, total_prize: float) -> float:
        """Chip EV (proportional to stack) for comparison with ICM equity."""
        total_chips = sum(self.chip_stacks)
        if total_chips == 0:
            return 0.0
        return (self.chip_stacks[player_index] / total_chips) * total_prize


def calculate_icm(
    stacks: list[float],
    payouts: list[float],
    iterations: int | None = None,
) -> ICMResult:
    """Calculate ICM equity for each player using the Malmuth-Harville method.

    For small player counts (<=7), computes exact ICM. For larger fields,
    uses Monte Carlo sampling.

    Args:
        stacks: Chip stacks for each player.
        payouts: Payout amounts ordered 1st, 2nd, 3rd, etc.
            Must have at least as many entries as players, but extra
            entries (zero payouts) are fine.

    Returns:
        ICMResult with equity for each player.
    """
    n = len(stacks)
    total_chips = sum(stacks)
    if total_chips == 0:
        return ICMResult(equities=[0.0] * n, chip_stacks=list(stacks))

    # Pad payouts if needed
    payouts_padded = list(payouts) + [0.0] * max(0, n - len(payouts))

    if n <= 7 and iterations is None:
        equities = _icm_exact(stacks, payouts_padded, total_chips)
    else:
        equities = _icm_monte_carlo(
            stacks, payouts_padded, total_chips, iterations or 10_000
        )

    return ICMResult(equities=equities, chip_stacks=list(stacks))


def _icm_exact(
    stacks: list[float],
    payouts: list[float],
    total_chips: float,
) -> list[float]:
    """Exact ICM calculation using recursive probability tree."""
    n = len(stacks)
    equities = [0.0] * n

    def _recurse(
        remaining: list[int],
        place: int,
        prob: float,
    ) -> None:
        if place >= len(payouts) or not remaining:
            return

        remaining_total = sum(stacks[r] for r in remaining)
        if remaining_total == 0:
            return

        for i, player_idx in enumerate(remaining):
            if stacks[player_idx] == 0:
                continue
            # Probability this player finishes in this place
            p = (stacks[player_idx] / remaining_total) * prob
            equities[player_idx] += p * payouts[place]

            # Recurse for remaining places
            next_remaining = remaining[:i] + remaining[i + 1:]
            if next_remaining and place + 1 < len(payouts):
                _recurse(next_remaining, place + 1, p)

    # Filter out zero-stack players from initial recursion
    active = [i for i in range(n) if stacks[i] > 0]
    _recurse(active, 0, 1.0)
    return equities


def _icm_monte_carlo(
    stacks: list[float],
    payouts: list[float],
    total_chips: float,
    iterations: int,
) -> list[float]:
    """Monte Carlo ICM approximation for large fields."""
    n = len(stacks)
    probs = np.array(stacks) / total_chips
    equities = np.zeros(n)

    rng = np.random.default_rng()

    for _ in range(iterations):
        # Simulate a finish order weighted by chip stacks
        remaining = list(range(n))
        remaining_probs = probs.copy()

        for place in range(min(len(payouts), n)):
            if not remaining:
                break
            # Normalize probabilities for remaining players
            r_probs = remaining_probs[remaining]
            r_sum = r_probs.sum()
            if r_sum == 0:
                break
            r_probs = r_probs / r_sum

            # Select who finishes in this place
            chosen_local = rng.choice(len(remaining), p=r_probs)
            chosen = remaining[chosen_local]
            equities[chosen] += payouts[place]
            remaining.pop(chosen_local)

    equities /= iterations
    return equities.tolist()


@dataclass(frozen=True)
class BubbleFactor:
    """Bubble factor for a specific matchup.

    Bubble factor > 1 means chips lost are worth more than chips gained
    (due to ICM). A factor of 2.0 means losing chips costs twice as
    much equity as gaining the same amount.
    """

    risk_factor: float  # How much more costly losing is vs gaining
    description: str

    @property
    def effective_pot_odds_multiplier(self) -> float:
        """Multiply required equity by this to get ICM-adjusted requirement.

        If the bubble factor is 1.5 and you need 40% chip-EV equity,
        you need 40% * 1.5 = 60% to be ICM-profitable.
        """
        return self.risk_factor


def calculate_bubble_factor(
    hero_stack: float,
    villain_stack: float,
    all_stacks: list[float],
    payouts: list[float],
) -> BubbleFactor:
    """Calculate the bubble factor for hero vs villain.

    Compares ICM equity change from winning vs losing a pot.

    Args:
        hero_stack: Hero's current chip stack.
        villain_stack: Villain's current chip stack.
        all_stacks: All players' chip stacks (including hero and villain).
        payouts: Payout structure.

    Returns:
        BubbleFactor indicating the risk/reward ratio.
    """
    hero_idx = all_stacks.index(hero_stack)
    villain_idx = all_stacks.index(villain_stack)

    # Current ICM equity
    current_icm = calculate_icm(all_stacks, payouts)
    current_eq = current_icm.equity_for(hero_idx)

    # Scenario: Hero wins (villain eliminated or loses chips)
    pot_size = min(hero_stack, villain_stack)  # Effective stack
    win_stacks = list(all_stacks)
    win_stacks[hero_idx] += pot_size
    win_stacks[villain_idx] -= pot_size

    # Remove villain if busted
    if win_stacks[villain_idx] <= 0:
        win_stacks_filtered = [s for i, s in enumerate(win_stacks) if i != villain_idx]
        # Adjust payouts for fewer players
        win_icm = calculate_icm(win_stacks_filtered, payouts)
        # Hero's index shifts if villain was before hero
        h_idx = hero_idx if villain_idx > hero_idx else hero_idx - 1
        win_eq = win_icm.equity_for(h_idx)
    else:
        win_icm = calculate_icm(win_stacks, payouts)
        win_eq = win_icm.equity_for(hero_idx)

    # Scenario: Hero loses
    lose_stacks = list(all_stacks)
    lose_stacks[hero_idx] -= pot_size
    lose_stacks[villain_idx] += pot_size

    if lose_stacks[hero_idx] <= 0:
        lose_eq = 0.0  # Hero busted
    else:
        lose_icm = calculate_icm(lose_stacks, payouts)
        lose_eq = lose_icm.equity_for(hero_idx)

    # Bubble factor = equity lost / equity gained
    equity_gained = win_eq - current_eq
    equity_lost = current_eq - lose_eq

    if equity_gained <= 0:
        factor = float("inf")
        desc = "Extremely unfavorable — avoid confrontation"
    elif equity_lost <= 0:
        factor = 0.0
        desc = "Freeroll — no downside risk"
    else:
        factor = equity_lost / equity_gained
        if factor >= 3.0:
            desc = "Extreme bubble pressure — play very tight"
        elif factor >= 2.0:
            desc = "High bubble pressure — play tight"
        elif factor >= 1.5:
            desc = "Moderate bubble pressure — tighten up"
        elif factor >= 1.1:
            desc = "Slight bubble pressure — minor adjustments"
        else:
            desc = "Minimal ICM pressure — play close to chip-EV"

    return BubbleFactor(risk_factor=factor, description=desc)


def survival_premium(context: GameContext) -> float:
    """Calculate a survival premium multiplier for tournament play.

    Returns a multiplier (0.0 to 1.0) representing how much to tighten
    ranges. 1.0 means no tightening (chip-EV play), 0.5 means ranges
    should be approximately halved.

    The premium is highest on the bubble and near big payout jumps.

    Args:
        context: Current game context.

    Returns:
        Multiplier for range width (1.0 = no adjustment, lower = tighter).
    """
    if not context.is_tournament:
        return 1.0

    base = 1.0

    # Phase-based adjustments
    match context.tournament_phase:
        case TournamentPhase.EARLY:
            base = 1.0  # Play close to chip-EV
        case TournamentPhase.MIDDLE:
            base = 0.90  # Slight tightening
        case TournamentPhase.BUBBLE:
            base = 0.65  # Significant tightening
        case TournamentPhase.IN_THE_MONEY:
            base = 0.85  # Some tightening for pay jumps
        case TournamentPhase.FINAL_TABLE:
            base = 0.75  # Pay jumps are large

    # Stack-relative adjustments
    if context.average_stack_bb > 0:
        stack_ratio = context.stack_depth_bb / context.average_stack_bb
        if stack_ratio < 0.5:
            # Short stack relative to field — tighten to survive
            base *= 0.85
        elif stack_ratio > 2.0:
            # Big stack — can afford to be more aggressive
            base = min(1.0, base * 1.15)

    # Near payout jump — extra tightening
    if context.is_near_payout_jump:
        base *= 0.85

    return max(0.3, min(1.0, base))


def adjust_range_for_tournament(
    base_range: Range,
    context: GameContext,
) -> Range:
    """Adjust a range for tournament considerations.

    Applies survival premium to narrow ranges when ICM pressure
    demands tighter play.

    Args:
        base_range: The stack-adjusted range.
        context: Current game context.

    Returns:
        Tournament-adjusted range.
    """
    if not context.is_tournament:
        return base_range

    premium = survival_premium(context)

    if premium >= 0.95:
        return base_range

    # Sort hands by "strength" (pair > suited > offsuit, higher ranks first)
    # and keep only the top (premium * 100)% of hands
    all_hands = sorted(
        base_range.hands,
        key=_hand_strength_key,
        reverse=True,
    )

    keep_count = max(1, int(len(all_hands) * premium))
    kept = set(all_hands[:keep_count])

    return Range(hands=kept)


def _hand_strength_key(hand) -> tuple[int, int, int]:
    """Sort key for hand strength (higher = stronger).

    Ordering: pairs first (by rank), then suited, then offsuit.
    Within each category, higher ranks are stronger.
    """
    from poker_bot.strategy.preflop_ranges import HandType, _RANK_INDEX

    # Type bonus: pairs > suited > offsuit
    type_bonus = {HandType.PAIR: 200, HandType.SUITED: 100, HandType.OFFSUIT: 0}

    r1_val = 13 - _RANK_INDEX[hand.rank1]  # Higher rank = higher value
    r2_val = 13 - _RANK_INDEX[hand.rank2]

    return (type_bonus[hand.hand_type], r1_val, r2_val)
