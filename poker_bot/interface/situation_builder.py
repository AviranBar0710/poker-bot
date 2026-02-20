"""Pure function to build GameState and GameContext from input parameters.

Extracted from poker_coach.py for reuse by both the CLI coach and the GUI.
This module has no I/O — it only constructs game objects from validated inputs.
"""

from __future__ import annotations

from poker_bot.core.game_context import (
    GameContext,
    PayoutStructure,
    TournamentPhase,
)
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Position, Street


# Ordered position list for assigning villain seats
_ALL_POSITIONS = [
    Position.UTG, Position.MP, Position.CO,
    Position.BTN, Position.SB, Position.BB,
]


def build_game_objects(
    hero_cards: list[Card],
    position: Position,
    stack_bb: float,
    street: Street,
    pot_bb: float,
    current_bet_bb: float,
    num_opponents: int,
    community_cards: list[Card] | None = None,
    is_tournament: bool = False,
    tournament_phase: TournamentPhase | None = None,
    players_remaining: int = 0,
    total_entries: int = 0,
) -> tuple[GameState, GameContext]:
    """Build GameState and GameContext from validated input parameters.

    Args:
        hero_cards: Exactly 2 cards for the hero.
        position: Hero's table position.
        stack_bb: Hero's stack in big blinds.
        street: Current street.
        pot_bb: Current pot size in big blinds.
        current_bet_bb: Current bet to face in big blinds.
        num_opponents: Number of active opponents.
        community_cards: Board cards (empty for preflop).
        is_tournament: Whether this is a tournament game.
        tournament_phase: Tournament phase (if tournament).
        players_remaining: Players left in tournament.
        total_entries: Total tournament entries.

    Returns:
        Tuple of (GameState, GameContext).

    Raises:
        ValueError: If hero_cards doesn't contain exactly 2 cards.
    """
    if len(hero_cards) != 2:
        raise ValueError(f"Need exactly 2 hero cards, got {len(hero_cards)}")

    if community_cards is None:
        community_cards = []

    # Build hero player
    hero = PlayerState(
        name="Hero",
        chips=stack_bb,
        position=position,
        hole_cards=hero_cards,
        is_active=True,
    )
    # Set hero's current_bet for blind positions
    if position == Position.BB and street == Street.PREFLOP and current_bet_bb <= 1.0:
        hero.current_bet = 1.0  # Already posted BB
    elif position == Position.SB and street == Street.PREFLOP:
        hero.current_bet = 0.5  # Already posted SB

    # Build villain players
    villain_positions = [p for p in _ALL_POSITIONS if p != position]
    players = [hero]
    for i in range(num_opponents):
        players.append(PlayerState(
            name=f"Villain{i + 1}",
            chips=stack_bb,
            position=villain_positions[i % len(villain_positions)],
            hole_cards=[],
            is_active=True,
        ))

    gs = GameState(
        players=players,
        small_blind=0.5,
        big_blind=1.0,
        pot=pot_bb,
        current_bet=current_bet_bb,
        current_street=street,
        community_cards=community_cards,
    )

    # Build context
    if is_tournament:
        phase = tournament_phase or TournamentPhase.MIDDLE
        payout = PayoutStructure(
            total_prize_pool=total_entries * 10.0,
            payouts={1: 0.25, 2: 0.15, 3: 0.10, 4: 0.08, 5: 0.06},
            total_entries=total_entries,
        )
        ctx = GameContext.tournament(
            stack_bb=stack_bb,
            phase=phase,
            players_remaining=players_remaining,
            payout_structure=payout,
            num_players=num_opponents + 1,
        )
    else:
        ctx = GameContext.cash_game(
            stack_bb=stack_bb,
            num_players=num_opponents + 1,
        )

    return gs, ctx
