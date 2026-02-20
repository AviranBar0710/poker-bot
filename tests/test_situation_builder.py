"""Tests for the situation_builder module."""

import pytest

from poker_bot.core.game_context import GameType, TournamentPhase
from poker_bot.interface.situation_builder import build_game_objects
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Position, Street


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


class TestBuildGameObjects:
    """Tests for build_game_objects()."""

    def test_basic_cash_preflop(self):
        gs, ctx = build_game_objects(
            hero_cards=_cards("Ah Ks"),
            position=Position.BTN,
            stack_bb=50.0,
            street=Street.PREFLOP,
            pot_bb=3.0,
            current_bet_bb=2.0,
            num_opponents=2,
        )
        assert len(gs.players) == 3
        assert gs.players[0].name == "Hero"
        assert gs.players[0].position == Position.BTN
        assert len(gs.players[0].hole_cards) == 2
        assert gs.pot == 3.0
        assert gs.current_bet == 2.0
        assert gs.current_street == Street.PREFLOP
        assert ctx.game_type == GameType.CASH
        assert ctx.stack_depth_bb == 50.0
        assert ctx.num_players == 3

    def test_postflop_with_board(self):
        board = _cards("Jh 8d 3c")
        gs, ctx = build_game_objects(
            hero_cards=_cards("As Kd"),
            position=Position.CO,
            stack_bb=100.0,
            street=Street.FLOP,
            pot_bb=6.5,
            current_bet_bb=0.0,
            num_opponents=1,
            community_cards=board,
        )
        assert gs.community_cards == board
        assert gs.current_street == Street.FLOP
        assert len(gs.players) == 2

    def test_bb_current_bet_set(self):
        gs, _ = build_game_objects(
            hero_cards=_cards("Td 9d"),
            position=Position.BB,
            stack_bb=50.0,
            street=Street.PREFLOP,
            pot_bb=3.0,
            current_bet_bb=1.0,
            num_opponents=2,
        )
        assert gs.players[0].current_bet == 1.0

    def test_sb_current_bet_set(self):
        gs, _ = build_game_objects(
            hero_cards=_cards("Qh Js"),
            position=Position.SB,
            stack_bb=50.0,
            street=Street.PREFLOP,
            pot_bb=3.0,
            current_bet_bb=2.0,
            num_opponents=2,
        )
        assert gs.players[0].current_bet == 0.5

    def test_tournament_context(self):
        _, ctx = build_game_objects(
            hero_cards=_cards("Ah Ad"),
            position=Position.BTN,
            stack_bb=30.0,
            street=Street.PREFLOP,
            pot_bb=2.5,
            current_bet_bb=2.0,
            num_opponents=3,
            is_tournament=True,
            tournament_phase=TournamentPhase.BUBBLE,
            players_remaining=20,
            total_entries=100,
        )
        assert ctx.is_tournament
        assert ctx.tournament_phase == TournamentPhase.BUBBLE
        assert ctx.players_remaining == 20
        assert ctx.payout_structure is not None
        assert ctx.num_players == 4

    def test_villain_positions_exclude_hero(self):
        gs, _ = build_game_objects(
            hero_cards=_cards("7h 2c"),
            position=Position.UTG,
            stack_bb=50.0,
            street=Street.PREFLOP,
            pot_bb=1.5,
            current_bet_bb=0.0,
            num_opponents=3,
        )
        villain_positions = [p.position for p in gs.players[1:]]
        assert Position.UTG not in villain_positions

    def test_requires_two_hero_cards(self):
        with pytest.raises(ValueError, match="exactly 2"):
            build_game_objects(
                hero_cards=_cards("Ah"),
                position=Position.BTN,
                stack_bb=50.0,
                street=Street.PREFLOP,
                pot_bb=3.0,
                current_bet_bb=0.0,
                num_opponents=1,
            )

    def test_empty_community_cards_default(self):
        gs, _ = build_game_objects(
            hero_cards=_cards("Kh Qd"),
            position=Position.BTN,
            stack_bb=50.0,
            street=Street.PREFLOP,
            pot_bb=3.0,
            current_bet_bb=0.0,
            num_opponents=1,
        )
        assert gs.community_cards == []

    def test_default_tournament_phase(self):
        _, ctx = build_game_objects(
            hero_cards=_cards("Jc Tc"),
            position=Position.CO,
            stack_bb=40.0,
            street=Street.PREFLOP,
            pot_bb=2.5,
            current_bet_bb=0.0,
            num_opponents=2,
            is_tournament=True,
        )
        assert ctx.tournament_phase == TournamentPhase.MIDDLE
