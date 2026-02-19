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


class TestIsInPosition:
    def test_btn_ip_vs_blinds(self):
        assert PostflopSolver._is_in_position("BTN", ["SB", "BB"]) is True

    def test_bb_oop_vs_btn(self):
        assert PostflopSolver._is_in_position("BB", ["BTN"]) is False

    def test_co_ip_vs_utg(self):
        assert PostflopSolver._is_in_position("CO", ["UTG"]) is True

    def test_utg_oop_vs_btn_co(self):
        assert PostflopSolver._is_in_position("UTG", ["CO", "BTN"]) is False

    def test_btn_ip_multiway(self):
        assert PostflopSolver._is_in_position("BTN", ["SB", "BB", "UTG"]) is True


class TestPositionAdjustments:
    """Test that IP/OOP adjustments shift frequencies in the expected direction."""

    def setup_method(self):
        self.solver = PostflopSolver()
        # Common args for a strong made hand on a dry board
        self._common = dict(
            hero_cards=_cards("Ah Kh"),
            community_cards=_cards("As 8d 3c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.85,
        )

    def _get_freq(self, result, action_name):
        return sum(a.frequency for a in result.strategy.actions if a.action == action_name)

    def test_ip_bets_more_than_oop(self):
        """IP should raise more frequently than OOP for the same hand."""
        ip_result = self.solver.get_strategy(**self._common, is_ip=True, num_opponents=1)
        oop_result = self.solver.get_strategy(**self._common, is_ip=False, num_opponents=1)
        assert self._get_freq(ip_result, "raise") > self._get_freq(oop_result, "raise")

    def test_oop_checks_more(self):
        """OOP should check more frequently than IP."""
        ip_result = self.solver.get_strategy(**self._common, is_ip=True, num_opponents=1)
        oop_result = self.solver.get_strategy(**self._common, is_ip=False, num_opponents=1)
        assert self._get_freq(oop_result, "check") > self._get_freq(ip_result, "check")

    def test_ip_oop_frequencies_sum_to_one(self):
        """Both IP and OOP strategies should have frequencies summing to ~1.0."""
        for is_ip in (True, False):
            result = self.solver.get_strategy(**self._common, is_ip=is_ip, num_opponents=1)
            total = sum(a.frequency for a in result.strategy.actions)
            assert abs(total - 1.0) < 0.01, f"is_ip={is_ip}: freqs sum to {total}"


class TestMultiwayAdjustments:
    """Test that multiway pot adjustments shift frequencies correctly."""

    def setup_method(self):
        self.solver = PostflopSolver()

    def _get_freq(self, result, action_name):
        return sum(a.frequency for a in result.strategy.actions if a.action == action_name)

    def _get_raise_amount(self, result):
        for a in result.strategy.actions:
            if a.action == "raise" and a.amount > 0:
                return a.amount
        return 0.0

    def test_multiway_folds_more(self):
        """3-way pot should fold more than heads-up for a weak hand."""
        common = dict(
            hero_cards=_cards("7h 6d"),
            community_cards=_cards("As Kd Qc"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.10,
            is_ip=True,
        )
        hu_result = self.solver.get_strategy(**common, num_opponents=1)
        mw_result = self.solver.get_strategy(**common, num_opponents=3)
        assert self._get_freq(mw_result, "fold") > self._get_freq(hu_result, "fold")

    def test_multiway_bluffs_less(self):
        """Air in multiway should have lower raise frequency than heads-up."""
        common = dict(
            hero_cards=_cards("2h 3d"),
            community_cards=_cards("As Kd 9c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.05,
            is_ip=True,
        )
        hu_result = self.solver.get_strategy(**common, num_opponents=1)
        mw_result = self.solver.get_strategy(**common, num_opponents=3)
        assert self._get_freq(mw_result, "raise") < self._get_freq(hu_result, "raise")

    def test_multiway_sizes_smaller(self):
        """Raise amounts should be smaller in multiway pots."""
        common = dict(
            hero_cards=_cards("Ah Ad"),
            community_cards=_cards("As 8d 3c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.95,
            is_ip=True,
        )
        hu_result = self.solver.get_strategy(**common, num_opponents=1)
        mw_result = self.solver.get_strategy(**common, num_opponents=3)
        hu_amt = self._get_raise_amount(hu_result)
        mw_amt = self._get_raise_amount(mw_result)
        # Only assert if both have raise amounts
        if hu_amt > 0 and mw_amt > 0:
            assert mw_amt < hu_amt

    def test_multiway_frequencies_sum_to_one(self):
        """Multiway strategy frequencies should still sum to ~1.0."""
        result = self.solver.get_strategy(
            hero_cards=_cards("Ah Kh"),
            community_cards=_cards("As 8d 3c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.85,
            is_ip=True,
            num_opponents=4,
        )
        total = sum(a.frequency for a in result.strategy.actions)
        assert abs(total - 1.0) < 0.01
