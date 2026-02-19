"""Game context tracking for cash games and tournaments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poker_bot.interface.opponent_tracker import OpponentStats


class GameType(StrEnum):
    CASH = "CASH"
    TOURNAMENT = "TOURNAMENT"
    SIT_AND_GO = "SIT_AND_GO"


class TournamentPhase(StrEnum):
    EARLY = "EARLY"          # Blinds are small relative to stacks
    MIDDLE = "MIDDLE"        # Antes kick in, stacks shrink
    BUBBLE = "BUBBLE"        # One or few eliminations from the money
    IN_THE_MONEY = "IN_THE_MONEY"  # Past the bubble, pay jumps matter
    FINAL_TABLE = "FINAL_TABLE"    # Final table dynamics


@dataclass
class BlindLevel:
    """A single blind level in a tournament structure."""

    small_blind: float
    big_blind: float
    ante: float = 0.0
    level_number: int = 1
    duration_minutes: int = 0

    @property
    def total_pot_preflop(self) -> float:
        """Dead money in pot before any action (blinds + antes for typical 6-max)."""
        return self.small_blind + self.big_blind + self.ante * 6


@dataclass
class PayoutStructure:
    """Tournament payout structure.

    payouts maps finishing position (1-indexed) to the payout amount
    or percentage. For example: {1: 50.0, 2: 30.0, 3: 20.0}
    """

    total_prize_pool: float
    payouts: dict[int, float] = field(default_factory=dict)
    total_entries: int = 0

    @property
    def min_cash_position(self) -> int:
        """Last position that receives a payout."""
        return max(self.payouts.keys()) if self.payouts else 0

    def payout_for(self, position: int) -> float:
        """Get payout for a given finishing position."""
        return self.payouts.get(position, 0.0)

    def remaining_payouts(self, players_left: int) -> dict[int, float]:
        """Get payouts still available given how many players remain."""
        return {
            pos: amt
            for pos, amt in self.payouts.items()
            if pos <= players_left
        }


@dataclass
class GameContext:
    """Complete context for strategic decision-making.

    Tracks game type, stack depths, tournament state, and all
    information needed to adjust strategy beyond just the cards.
    """

    game_type: GameType
    stack_depth_bb: float  # Hero's stack in big blinds
    num_players: int = 6   # Players at the table

    # Tournament-specific fields
    tournament_phase: TournamentPhase | None = None
    players_remaining: int = 0
    payout_structure: PayoutStructure | None = None
    blind_level: BlindLevel | None = None
    average_stack_bb: float = 0.0

    # Table dynamics
    table_stack_sizes_bb: list[float] = field(default_factory=list)

    # Opponent modeling (player name -> stats)
    opponent_stats: dict[str, OpponentStats] = field(default_factory=dict)

    @property
    def is_tournament(self) -> bool:
        return self.game_type in (GameType.TOURNAMENT, GameType.SIT_AND_GO)

    @property
    def is_cash(self) -> bool:
        return self.game_type == GameType.CASH

    @property
    def stack_category(self) -> str:
        """Categorize stack depth for strategy selection."""
        if self.stack_depth_bb >= 100:
            return "deep"
        if self.stack_depth_bb >= 40:
            return "medium"
        if self.stack_depth_bb >= 20:
            return "short"
        if self.stack_depth_bb >= 10:
            return "very_short"
        return "critical"

    @property
    def m_ratio(self) -> float:
        """Harrington's M ratio (tournament only).

        M = stack / (SB + BB + antes). Indicates how many orbits
        you can survive without playing a hand.
        """
        if not self.blind_level:
            # Approximate with just big blind
            return self.stack_depth_bb
        pot = self.blind_level.total_pot_preflop
        if pot == 0:
            return float("inf")
        stack = self.stack_depth_bb * self.blind_level.big_blind
        return stack / pot

    @property
    def is_on_bubble(self) -> bool:
        """Whether the tournament is on the bubble."""
        return self.tournament_phase == TournamentPhase.BUBBLE

    @property
    def is_near_payout_jump(self) -> bool:
        """Whether a significant payout jump is nearby."""
        if not self.payout_structure or not self.players_remaining:
            return False
        ps = self.payout_structure
        current_payout = ps.payout_for(self.players_remaining)
        next_payout = ps.payout_for(self.players_remaining - 1)
        if current_payout == 0:
            return False
        jump_ratio = next_payout / current_payout if current_payout else 0
        return jump_ratio >= 1.5

    @classmethod
    def cash_game(cls, stack_bb: float, num_players: int = 6) -> GameContext:
        """Create a cash game context."""
        return cls(
            game_type=GameType.CASH,
            stack_depth_bb=stack_bb,
            num_players=num_players,
        )

    @classmethod
    def tournament(
        cls,
        stack_bb: float,
        phase: TournamentPhase,
        players_remaining: int,
        blind_level: BlindLevel | None = None,
        payout_structure: PayoutStructure | None = None,
        average_stack_bb: float = 0.0,
        table_stacks_bb: list[float] | None = None,
        num_players: int = 6,
    ) -> GameContext:
        """Create a tournament context."""
        return cls(
            game_type=GameType.TOURNAMENT,
            stack_depth_bb=stack_bb,
            num_players=num_players,
            tournament_phase=phase,
            players_remaining=players_remaining,
            blind_level=blind_level,
            payout_structure=payout_structure,
            average_stack_bb=average_stack_bb or stack_bb,
            table_stack_sizes_bb=table_stacks_bb or [],
        )
