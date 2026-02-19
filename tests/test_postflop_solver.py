"""Tests for postflop solver."""

from unittest.mock import patch

from poker_bot.solver.postflop_solver import PostflopSolver
from poker_bot.strategy.decision_maker import PriorAction
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Street


def _card(s: str) -> Card:
    return Card.from_str(s)


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


class TestPostflopSolver:
    def setup_method(self):
        self.solver = PostflopSolver()

    def test_value_hand_bets_high_frequency(self):
        """Strong made hand should bet at high frequency."""
        result = self.solver.get_strategy(
            hero_cards=_cards("Ah As"),
            community_cards=_cards("Ad Kh 7c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.95,
        )
        # Nuts/strong hands should have high raise frequency
        raise_freq = sum(
            a.frequency for a in result.strategy.actions if a.action == "raise"
        )
        assert raise_freq >= 0.5

    def test_draw_has_mixed_strategy(self):
        """Drawing hands should have mixed bet/check."""
        result = self.solver.get_strategy(
            hero_cards=_cards("Jh Th"),
            community_cards=_cards("9h 8d 2c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.40,
            has_draw=True,
            draw_strength=0.6,
        )
        # Should have multiple actions
        assert len(result.strategy.actions) >= 2
        # Should have both aggressive and passive options
        action_types = {a.action for a in result.strategy.actions}
        assert len(action_types) >= 2

    def test_air_folds_facing_large_bet(self):
        """Air should fold at high frequency."""
        result = self.solver.get_strategy(
            hero_cards=_cards("2h 3d"),
            community_cards=_cards("Ah Kd Qs"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[PriorAction(Position.UTG, Action.RAISE, 7.0)],
            hand_strength=0.05,
        )
        fold_freq = sum(
            a.frequency for a in result.strategy.actions if a.action == "fold"
        )
        assert fold_freq >= 0.4

    def test_solver_result_has_spot_key(self):
        """Result should have populated SpotKey."""
        result = self.solver.get_strategy(
            hero_cards=_cards("Ah Kh"),
            community_cards=_cards("Qh 8d 3c"),
            position=Position.CO,
            pot=10.0,
            hero_stack=50.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.65,
        )
        assert result.spot_key is not None
        assert result.spot_key.street == "flop"
        assert result.spot_key.position == "CO"
        assert result.spot_key.hand_category in (
            "nuts", "strong_made", "medium_made", "weak_made",
            "strong_draw", "medium_draw", "weak_draw", "air",
        )

    def test_confidence_reasonable(self):
        """Confidence should be in valid range."""
        result = self.solver.get_strategy(
            hero_cards=_cards("Ah Kh"),
            community_cards=_cards("Qh 8d 3c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.65,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_frequencies_sum_to_one(self):
        """Strategy frequencies should approximately sum to 1."""
        result = self.solver.get_strategy(
            hero_cards=_cards("Kh Qh"),
            community_cards=_cards("Jh Td 5c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.50,
            has_draw=True,
            draw_strength=0.4,
        )
        total_freq = sum(a.frequency for a in result.strategy.actions)
        assert abs(total_freq - 1.0) < 0.05

    def test_detect_street(self):
        assert PostflopSolver._detect_street([]) == "preflop"
        assert PostflopSolver._detect_street(_cards("Ah Kd 3c")) == "flop"
        assert PostflopSolver._detect_street(_cards("Ah Kd 3c 7h")) == "turn"
        assert PostflopSolver._detect_street(_cards("Ah Kd 3c 7h 2s")) == "river"
