"""Edge case and boundary condition tests for the decision engine.

Verifies correct behavior at stack extremes, missing data, zero-amount
boundaries, and other degenerate game states where bugs commonly hide.
"""

from __future__ import annotations

import pytest

from poker_bot.core.game_context import GameContext, TournamentPhase, PayoutStructure
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.strategy.decision_maker import (
    ActionType,
    DecisionMaker,
    PriorAction,
    Decision,
)
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Street


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cards(s: str) -> list[Card]:
    """Parse space-separated card strings: 'Ah Kh' -> [Card, Card]."""
    return [Card.from_str(c) for c in s.split()]


def _make_game_state(
    hero_chips: float,
    hero_position: Position,
    hero_cards: list[Card],
    pot: float,
    current_bet: float,
    big_blind: float = 2.0,
    small_blind: float = 1.0,
    community_cards: list[Card] | None = None,
    street: Street = Street.PREFLOP,
    villain_chips: float = 200.0,
    num_villains: int = 1,
) -> GameState:
    """Build a complete GameState with hero at index 0."""
    players = [
        PlayerState(
            name="Hero",
            chips=hero_chips,
            position=hero_position,
            hole_cards=hero_cards,
            is_active=True,
        ),
    ]
    villain_positions = [
        p for p in [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB, Position.BB]
        if p != hero_position
    ]
    for i in range(num_villains):
        players.append(
            PlayerState(
                name=f"Villain{i+1}",
                chips=villain_chips,
                position=villain_positions[i % len(villain_positions)],
                hole_cards=_cards("7d 2c"),  # Placeholder — not used by hero decision
                is_active=True,
            )
        )

    gs = GameState(
        players=players,
        small_blind=small_blind,
        big_blind=big_blind,
        pot=pot,
        current_bet=current_bet,
        current_street=street,
        community_cards=community_cards or [],
    )
    return gs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAllInEdgeCases:
    """Tests for all-in conversions at extreme stack sizes."""

    def test_all_in_when_stack_less_than_min_raise(self) -> None:
        """Hero has 1bb (2 chips) — any raise should become ALL_IN, not RAISE."""
        gs = _make_game_state(
            hero_chips=2.0,
            hero_position=Position.BTN,
            hero_cards=_cards("As Ks"),  # Premium hand — wants to raise
            pot=3.0,
            current_bet=2.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=1.0, num_players=2)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        # With 1bb stack, engine should go all-in (push/fold territory)
        assert decision.action == ActionType.ALL_IN
        assert decision.amount == 2.0

    def test_all_in_call_when_bet_exceeds_stack(self) -> None:
        """Call amount exceeds hero stack — should convert to ALL_IN."""
        gs = _make_game_state(
            hero_chips=5.0,
            hero_position=Position.SB,
            hero_cards=_cards("Ah Kh"),  # Strong hand, wants to continue
            pot=20.0,
            current_bet=10.0,  # Bet is 2x hero's stack
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=2.5, num_players=2)
        maker = DecisionMaker()
        history = [PriorAction(Position.BTN, Action.RAISE, 10.0)]
        decision = maker.make_decision(gs, ctx, hero_index=0, action_history=history)

        # With critical stack AKs, push/fold → should go all-in
        assert decision.action == ActionType.ALL_IN
        assert decision.amount == 5.0

    def test_all_in_value_with_nuts_low_spr(self) -> None:
        """Set on paired board with SPR < 1 — should go all-in for value."""
        community = _cards("Qs Qd 7h")
        gs = _make_game_state(
            hero_chips=8.0,
            hero_position=Position.BTN,
            hero_cards=_cards("Qh Qc"),  # Quads — absolute nuts
            pot=20.0,
            current_bet=5.0,
            big_blind=2.0,
            community_cards=community,
            street=Street.FLOP,
        )
        # SPR = 8 / 20 = 0.4 — very low
        ctx = GameContext.cash_game(stack_bb=4.0, num_players=2)
        maker = DecisionMaker()
        history = [PriorAction(Position.SB, Action.RAISE, 5.0)]
        decision = maker.make_decision(gs, ctx, hero_index=0, action_history=history)

        # With the nuts and SPR < 1, should be all-in or raise (which clamps to all-in)
        assert decision.action in (ActionType.ALL_IN, ActionType.RAISE)
        # Amount must not exceed stack
        assert decision.amount <= 8.0


class TestMissingOrDegenerateData:
    """Tests for degenerate inputs: no cards, zero pot, etc."""

    def test_no_hole_cards_returns_fold(self) -> None:
        """Empty hole cards — engine should fold gracefully."""
        gs = _make_game_state(
            hero_chips=100.0,
            hero_position=Position.BTN,
            hero_cards=[],  # No cards dealt
            pot=3.0,
            current_bet=2.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=50.0, num_players=2)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        assert decision.action == ActionType.FOLD
        assert decision.amount == 0.0
        assert "hole cards" in decision.reasoning.lower() or "no" in decision.reasoning.lower()

    def test_zero_pot(self) -> None:
        """Pot is 0 but hero must act — engine should not crash."""
        gs = _make_game_state(
            hero_chips=100.0,
            hero_position=Position.BTN,
            hero_cards=_cards("As Ah"),  # Premium — still want to raise
            pot=0.0,
            current_bet=0.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=50.0, num_players=2)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        # Should not crash; AA should open-raise
        assert decision.action in (ActionType.RAISE, ActionType.ALL_IN)
        assert decision.amount >= 0


class TestCriticalStackPreflop:
    """Tests for critical and very short stack preflop scenarios."""

    def test_hero_1bb_stack_preflop(self) -> None:
        """1bb stack with a playable hand — must push or fold."""
        gs = _make_game_state(
            hero_chips=2.0,
            hero_position=Position.BTN,
            hero_cards=_cards("Ac Kc"),  # Premium — should push
            pot=3.0,
            current_bet=2.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=1.0, num_players=2)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        assert decision.action in (ActionType.ALL_IN, ActionType.FOLD)
        # AKs at 1bb on BTN should push
        assert decision.action == ActionType.ALL_IN

    def test_hero_exact_min_raise_stack(self) -> None:
        """Stack = exactly 2x BB. A raise equals going all-in."""
        gs = _make_game_state(
            hero_chips=4.0,
            hero_position=Position.CO,
            hero_cards=_cards("Jh Jd"),  # Strong hand
            pot=3.0,
            current_bet=2.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=2.0, num_players=3)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        # At 2bb, JJ should push. Any raise = all-in since stack = min raise.
        assert decision.action == ActionType.ALL_IN
        assert decision.amount == 4.0


class TestAmountBounds:
    """Tests ensuring amounts are never negative and never exceed stack."""

    def test_raise_clamped_to_stack(self) -> None:
        """Verify that raise amount never exceeds hero's stack."""
        gs = _make_game_state(
            hero_chips=15.0,
            hero_position=Position.CO,
            hero_cards=_cards("Ad Kd"),  # Premium — wants a big raise
            pot=50.0,
            current_bet=20.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=7.5, num_players=2)
        maker = DecisionMaker()
        history = [PriorAction(Position.BTN, Action.RAISE, 20.0)]
        decision = maker.make_decision(gs, ctx, hero_index=0, action_history=history)

        assert decision.amount <= 15.0, (
            f"Amount {decision.amount} exceeds hero stack of 15.0"
        )
        assert decision.amount >= 0.0

    def test_amount_never_negative(self) -> None:
        """Verify no negative amounts across various action types."""
        for action_type_cards in [
            (Position.UTG, _cards("7s 2h")),  # Trash — likely fold
            (Position.BTN, _cards("As Ad")),  # Premium — likely raise
            (Position.BB, _cards("Tc 9c")),   # Mediocre
        ]:
            pos, cards = action_type_cards
            gs = _make_game_state(
                hero_chips=50.0,
                hero_position=pos,
                hero_cards=cards,
                pot=6.0,
                current_bet=4.0,
                big_blind=2.0,
            )
            ctx = GameContext.cash_game(stack_bb=25.0, num_players=3)
            maker = DecisionMaker()
            decision = maker.make_decision(gs, ctx, hero_index=0)

            assert decision.amount >= 0.0, (
                f"Negative amount {decision.amount} for {pos} with {cards}"
            )


class TestCheckAndFoldPaths:
    """Tests for the check and fold decision paths."""

    def test_check_available_when_no_bet(self) -> None:
        """BB can check when the pot is unraised (limped to BB)."""
        gs = _make_game_state(
            hero_chips=100.0,
            hero_position=Position.BB,
            hero_cards=_cards("8c 3d"),  # Weak hand — should check not raise
            pot=4.0,
            current_bet=2.0,  # Only BB posted, no raise
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=50.0, num_players=3)
        maker = DecisionMaker()
        # No raises in history — limped pot
        decision = maker.make_decision(gs, ctx, hero_index=0, action_history=[])

        # With a weak hand and no raise, BB should check
        assert decision.action == ActionType.CHECK
        assert decision.amount == 0.0

    def test_fold_when_no_equity_postflop(self) -> None:
        """Worst hand on a dry board facing a bet — should fold."""
        community = _cards("Ah Kd Qs")
        gs = _make_game_state(
            hero_chips=100.0,
            hero_position=Position.BB,
            hero_cards=_cards("3c 2d"),  # Complete air vs AKQ board
            pot=20.0,
            current_bet=15.0,
            big_blind=2.0,
            community_cards=community,
            street=Street.FLOP,
        )
        ctx = GameContext.cash_game(stack_bb=50.0, num_players=2)
        maker = DecisionMaker()
        history = [PriorAction(Position.BTN, Action.RAISE, 15.0)]
        decision = maker.make_decision(gs, ctx, hero_index=0, action_history=history)

        # 32o on AKQ rainbow board facing a large bet — clear fold
        assert decision.action == ActionType.FOLD


class TestMultiwayPot:
    """Tests for multiway pot adjustments."""

    def test_multiway_pot_adjustments(self) -> None:
        """3+ opponents should affect decision — generally tighter play."""
        community = _cards("Jh 8d 3c")
        # Build a game state with 4 active players
        # Use clear air (no pair, no draw) so the decision isn't borderline
        gs = _make_game_state(
            hero_chips=100.0,
            hero_position=Position.UTG,
            hero_cards=_cards("5d 2s"),  # Pure air — no pair, no draw
            pot=24.0,
            current_bet=8.0,
            big_blind=2.0,
            community_cards=community,
            street=Street.FLOP,
            num_villains=3,
        )
        ctx = GameContext.cash_game(stack_bb=50.0, num_players=4)
        maker = DecisionMaker()
        history = [PriorAction(Position.CO, Action.RAISE, 8.0)]
        decision_multiway = maker.make_decision(
            gs, ctx, hero_index=0, action_history=history,
        )

        # Now heads-up with the same hand
        gs_hu = _make_game_state(
            hero_chips=100.0,
            hero_position=Position.UTG,
            hero_cards=_cards("5d 2s"),
            pot=24.0,
            current_bet=8.0,
            big_blind=2.0,
            community_cards=community,
            street=Street.FLOP,
            num_villains=1,
        )
        ctx_hu = GameContext.cash_game(stack_bb=50.0, num_players=2)
        decision_hu = maker.make_decision(
            gs_hu, ctx_hu, hero_index=0, action_history=history,
        )

        # Multiway should be at least as tight as heads-up (fold or lower action)
        action_strength = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.RAISE: 3,
            ActionType.ALL_IN: 4,
        }
        assert action_strength[decision_multiway.action] <= action_strength[decision_hu.action], (
            f"Multiway ({decision_multiway.action}) should not be more aggressive "
            f"than heads-up ({decision_hu.action}) with a marginal hand"
        )


class TestPushFoldBoundary:
    """Tests around the push/fold stack boundary."""

    def test_trash_hand_folds_at_critical_stack(self) -> None:
        """Trash hand at critical stack should fold even though it is push/fold."""
        gs = _make_game_state(
            hero_chips=4.0,
            hero_position=Position.UTG,
            hero_cards=_cards("7h 2c"),  # Absolute worst hand
            pot=3.0,
            current_bet=2.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=2.0, num_players=6)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        # 72o from UTG at 2bb should fold
        assert decision.action == ActionType.FOLD

    def test_premium_pair_pushes_at_5bb(self) -> None:
        """AA at 5bb should always push."""
        gs = _make_game_state(
            hero_chips=10.0,
            hero_position=Position.UTG,
            hero_cards=_cards("As Ah"),
            pot=3.0,
            current_bet=2.0,
            big_blind=2.0,
        )
        ctx = GameContext.cash_game(stack_bb=5.0, num_players=6)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        assert decision.action == ActionType.ALL_IN
        assert decision.amount == 10.0

    def test_postflop_strong_hand_checks_or_bets(self) -> None:
        """Postflop with no bet to face — strong hand bets, weak checks."""
        community = _cards("Kh 9d 4c")
        gs = _make_game_state(
            hero_chips=100.0,
            hero_position=Position.BTN,
            hero_cards=_cards("Kd Ks"),  # Top set
            pot=10.0,
            current_bet=0.0,
            big_blind=2.0,
            community_cards=community,
            street=Street.FLOP,
        )
        ctx = GameContext.cash_game(stack_bb=50.0, num_players=2)
        maker = DecisionMaker()
        decision = maker.make_decision(gs, ctx, hero_index=0)

        # Top set on a dry board — should bet for value
        assert decision.action in (ActionType.RAISE, ActionType.ALL_IN)
        assert decision.amount > 0
