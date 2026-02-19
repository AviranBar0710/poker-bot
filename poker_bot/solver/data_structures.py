"""Core data structures for the GTO solver engine.

ActionFrequency: A single action with its frequency, amount, and EV.
StrategyNode: A collection of action frequencies forming a mixed strategy.
SolverResult: Complete solver output with strategy, source, and confidence.
SpotKey: Hashable identifier for a specific game situation.
SolverProtocol: Interface that any solver backend must implement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from poker_bot.core.game_context import GameContext
    from poker_bot.core.game_state import GameState
    from poker_bot.strategy.decision_maker import PriorAction
    from poker_bot.strategy.preflop_ranges import Range


@dataclass(frozen=True)
class ActionFrequency:
    """A single action in a mixed strategy with its frequency and EV.

    Attributes:
        action: Action name (e.g. "raise", "call", "fold").
        frequency: How often to take this action [0.0, 1.0].
        amount: Bet/raise amount (0 for fold/check).
        ev: Expected value of this action in big blinds.
    """

    action: str
    frequency: float
    amount: float = 0.0
    ev: float = 0.0


@dataclass
class StrategyNode:
    """A mixed strategy: a collection of weighted actions.

    The frequencies should sum to 1.0 (within floating-point tolerance).
    """

    actions: list[ActionFrequency] = field(default_factory=list)

    @property
    def recommended_action(self) -> ActionFrequency | None:
        """Return the highest-frequency action."""
        if not self.actions:
            return None
        return max(self.actions, key=lambda a: a.frequency)

    def sample_action(self) -> ActionFrequency | None:
        """Randomly sample an action according to the mixed strategy frequencies.

        Used for GTO mixing in live play to be unexploitable.
        """
        if not self.actions:
            return None
        if len(self.actions) == 1:
            return self.actions[0]

        r = random.random()
        cumulative = 0.0
        for action in self.actions:
            cumulative += action.frequency
            if r <= cumulative:
                return action
        # Fallback to last action (handles floating-point rounding)
        return self.actions[-1]

    @property
    def is_pure(self) -> bool:
        """Whether this is a pure (non-mixed) strategy."""
        return len(self.actions) <= 1 or any(
            a.frequency >= 0.99 for a in self.actions
        )

    @property
    def best_ev(self) -> float:
        """Highest EV among all actions."""
        if not self.actions:
            return 0.0
        return max(a.ev for a in self.actions)

    @property
    def weighted_ev(self) -> float:
        """Frequency-weighted EV of the strategy."""
        if not self.actions:
            return 0.0
        return sum(a.frequency * a.ev for a in self.actions)

    def normalized(self) -> StrategyNode:
        """Return a copy with frequencies normalized to sum to 1.0."""
        total = sum(a.frequency for a in self.actions)
        if total <= 0:
            return StrategyNode(actions=list(self.actions))
        return StrategyNode(
            actions=[
                ActionFrequency(
                    action=a.action,
                    frequency=a.frequency / total,
                    amount=a.amount,
                    ev=a.ev,
                )
                for a in self.actions
            ]
        )


@dataclass(frozen=True)
class SpotKey:
    """Hashable identifier for a specific game situation.

    Used as dictionary key for pre-computed strategy lookups.
    """

    street: str  # "preflop", "flop", "turn", "river"
    position: str  # "UTG", "MP", "CO", "BTN", "SB", "BB"
    action_sequence: str  # e.g. "open", "vs_raise", "vs_3bet"
    stack_bucket: str  # e.g. "deep", "medium", "short"
    # Postflop-specific
    board_bucket: str = ""  # e.g. "dry_high_rainbow"
    spr_bucket: str = ""  # e.g. "high", "medium", "low"
    hand_category: str = ""  # e.g. "nuts", "strong_made", "medium_draw"


@dataclass
class SolverResult:
    """Complete output from the solver for a specific spot.

    Attributes:
        strategy: The mixed strategy for this spot.
        source: Where the strategy came from ("preflop_lookup",
                "postflop_lookup", "monte_carlo", "heuristic").
        confidence: How confident the solver is [0.0, 1.0].
                    < 0.5 triggers fallback to heuristic engine.
        ev: Expected value of the strategy in big blinds.
        spot_key: The spot identifier used for lookup.
    """

    strategy: StrategyNode
    source: str
    confidence: float
    ev: float = 0.0
    spot_key: SpotKey | None = None


@runtime_checkable
class SolverProtocol(Protocol):
    """Interface that any solver backend must implement.

    Implementations include the built-in SolverEngine (hybrid lookup +
    heuristic), and future backends like PioSolverEngine or LLMSolverEngine.

    Usage:
        def make_decision(solver: SolverProtocol, ...):
            result = solver.solve(game_state, context, hero_index)
    """

    def solve(
        self,
        game_state: GameState,
        context: GameContext,
        hero_index: int,
        action_history: list[PriorAction] | None = None,
        opponent_range: Range | None = None,
    ) -> SolverResult: ...
