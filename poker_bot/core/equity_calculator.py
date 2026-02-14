"""Monte Carlo equity calculator for Texas Hold'em.

Calculates equity (winning probability) of hands and ranges by running
simulated runouts of the remaining community cards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from poker_bot.core.hand_evaluator import HandEvaluator, HandResult
from poker_bot.strategy.preflop_ranges import HandCombo, Range
from poker_bot.utils.card import Card, Deck
from poker_bot.utils.constants import Rank, Suit


@dataclass
class EquityResult:
    """Result of an equity calculation."""

    equity: float  # Win probability [0, 1]
    win_count: int
    tie_count: int
    loss_count: int
    simulations: int

    @property
    def win_pct(self) -> float:
        return self.equity * 100

    @property
    def tie_pct(self) -> float:
        return (self.tie_count / self.simulations) * 100 if self.simulations else 0.0

    def __str__(self) -> str:
        return (
            f"Equity: {self.win_pct:.1f}% "
            f"(W: {self.win_count}, T: {self.tie_count}, L: {self.loss_count}, "
            f"sims: {self.simulations})"
        )


def _all_cards() -> list[Card]:
    """Return all 52 cards."""
    return [Card(rank=r, suit=s) for s in Suit for r in Rank]


def _available_deck(dead_cards: set[Card]) -> list[Card]:
    """Return a shuffled list of cards not in dead_cards."""
    available = [c for c in _all_cards() if c not in dead_cards]
    random.shuffle(available)
    return available


class EquityCalculator:
    """Monte Carlo equity calculator."""

    @staticmethod
    def hand_vs_hand(
        hand1: list[Card],
        hand2: list[Card],
        board: list[Card] | None = None,
        simulations: int = 10_000,
    ) -> EquityResult:
        """Calculate equity of hand1 vs hand2.

        Args:
            hand1: Player 1's hole cards (2 cards).
            hand2: Player 2's hole cards (2 cards).
            board: Community cards already dealt (0-4 cards).
            simulations: Number of Monte Carlo simulations.

        Returns:
            EquityResult for hand1.
        """
        board = board or []
        cards_needed = 5 - len(board)
        dead_cards = set(hand1) | set(hand2) | set(board)

        wins = 0
        ties = 0
        losses = 0

        for _ in range(simulations):
            deck = _available_deck(dead_cards)
            runout = board + deck[:cards_needed]

            eval1 = HandEvaluator.evaluate(list(hand1) + runout)
            eval2 = HandEvaluator.evaluate(list(hand2) + runout)

            if eval1 > eval2:
                wins += 1
            elif eval1 == eval2:
                ties += 1
            else:
                losses += 1

        equity = (wins + ties * 0.5) / simulations
        return EquityResult(
            equity=equity,
            win_count=wins,
            tie_count=ties,
            loss_count=losses,
            simulations=simulations,
        )

    @staticmethod
    def hand_vs_range(
        hand: list[Card],
        opponent_range: Range,
        board: list[Card] | None = None,
        simulations: int = 10_000,
    ) -> EquityResult:
        """Calculate equity of a specific hand vs an opponent's range.

        For each simulation, a random hand from the range is selected
        (filtering out combos that conflict with known cards), then a
        random board runout is generated.

        Args:
            hand: Player's hole cards (2 cards).
            opponent_range: Opponent's range of hands.
            board: Community cards already dealt (0-4 cards).
            simulations: Number of Monte Carlo simulations.

        Returns:
            EquityResult for the hand.
        """
        board = board or []
        cards_needed = 5 - len(board)
        known_cards = set(hand) | set(board)

        # Pre-filter combos that don't conflict with known cards
        valid_combos = [
            combo
            for combo in opponent_range.to_combos()
            if combo.card1 not in known_cards and combo.card2 not in known_cards
        ]

        if not valid_combos:
            raise ValueError("No valid combos in opponent range given known cards")

        wins = 0
        ties = 0
        losses = 0

        for _ in range(simulations):
            # Pick a random opponent hand from the range
            combo = random.choice(valid_combos)
            opp_hand = [combo.card1, combo.card2]

            dead_cards = known_cards | {combo.card1, combo.card2}
            deck = _available_deck(dead_cards)
            runout = board + deck[:cards_needed]

            eval1 = HandEvaluator.evaluate(list(hand) + runout)
            eval2 = HandEvaluator.evaluate(opp_hand + runout)

            if eval1 > eval2:
                wins += 1
            elif eval1 == eval2:
                ties += 1
            else:
                losses += 1

        equity = (wins + ties * 0.5) / simulations
        return EquityResult(
            equity=equity,
            win_count=wins,
            tie_count=ties,
            loss_count=losses,
            simulations=simulations,
        )

    @staticmethod
    def range_vs_range(
        range1: Range,
        range2: Range,
        board: list[Card] | None = None,
        simulations: int = 10_000,
    ) -> EquityResult:
        """Calculate equity of range1 vs range2.

        For each simulation, a random hand is drawn from each range
        (ensuring no card conflicts), then a random board runout is
        generated.

        Args:
            range1: First player's range.
            range2: Second player's range.
            board: Community cards already dealt (0-4 cards).
            simulations: Number of Monte Carlo simulations.

        Returns:
            EquityResult for range1.
        """
        board = board or []
        cards_needed = 5 - len(board)
        board_set = set(board)

        combos1 = [
            c for c in range1.to_combos()
            if c.card1 not in board_set and c.card2 not in board_set
        ]
        combos2 = [
            c for c in range2.to_combos()
            if c.card1 not in board_set and c.card2 not in board_set
        ]

        if not combos1 or not combos2:
            raise ValueError("One or both ranges have no valid combos")

        wins = 0
        ties = 0
        losses = 0
        valid_sims = 0

        for _ in range(simulations):
            c1 = random.choice(combos1)
            c2 = random.choice(combos2)

            # Skip if hands share cards
            hand1_cards = {c1.card1, c1.card2}
            hand2_cards = {c2.card1, c2.card2}
            if hand1_cards & hand2_cards:
                continue

            dead_cards = hand1_cards | hand2_cards | board_set
            deck = _available_deck(dead_cards)
            runout = board + deck[:cards_needed]

            eval1 = HandEvaluator.evaluate([c1.card1, c1.card2] + runout)
            eval2 = HandEvaluator.evaluate([c2.card1, c2.card2] + runout)

            if eval1 > eval2:
                wins += 1
            elif eval1 == eval2:
                ties += 1
            else:
                losses += 1
            valid_sims += 1

        if valid_sims == 0:
            raise ValueError("No valid simulations — ranges may fully overlap")

        equity = (wins + ties * 0.5) / valid_sims
        return EquityResult(
            equity=equity,
            win_count=wins,
            tie_count=ties,
            loss_count=losses,
            simulations=valid_sims,
        )
