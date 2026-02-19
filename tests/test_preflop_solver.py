"""Tests for preflop solver."""

from poker_bot.solver.preflop_solver import PreflopSolver
from poker_bot.strategy.decision_maker import PriorAction
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Rank, Suit


def _card(s: str) -> Card:
    return Card.from_str(s)


class TestPreflopSolver:
    def setup_method(self):
        self.solver = PreflopSolver()

    def test_aa_pure_raise_from_all_positions(self):
        """AA should be a pure raise from every position."""
        for pos in [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB]:
            result = self.solver.get_strategy(
                _card("Ah"), _card("As"), pos, [], 100.0,
            )
            rec = result.strategy.recommended_action
            assert rec is not None
            assert rec.action == "raise", f"AA should raise from {pos}"
            assert rec.frequency >= 0.95, f"AA should be pure raise from {pos}"

    def test_72o_mostly_fold(self):
        """72o should be mostly fold from early positions."""
        result = self.solver.get_strategy(
            _card("7h"), _card("2d"), Position.UTG, [], 100.0,
        )
        rec = result.strategy.recommended_action
        assert rec is not None
        assert rec.action == "fold"
        assert rec.frequency >= 0.8

    def test_border_hand_mixed(self):
        """A border hand like 87s from UTG should have mixed frequencies."""
        result = self.solver.get_strategy(
            _card("8h"), _card("7h"), Position.UTG, [], 100.0,
        )
        # 87s is in UTG open range but near the border
        assert len(result.strategy.actions) >= 1
        # Should have raise as an option
        raise_actions = [a for a in result.strategy.actions if a.action == "raise"]
        assert len(raise_actions) > 0

    def test_vs_raise_3bet_range(self):
        """AA facing a raise should 3-bet."""
        history = [PriorAction(Position.UTG, Action.RAISE, 6.0)]
        result = self.solver.get_strategy(
            _card("Ah"), _card("As"), Position.BTN, history, 100.0,
        )
        rec = result.strategy.recommended_action
        assert rec is not None
        assert rec.action == "raise"

    def test_icm_increases_fold_frequency(self):
        """Higher ICM pressure should increase fold frequency."""
        # No ICM
        result_no_icm = self.solver.get_strategy(
            _card("8h"), _card("7h"), Position.CO, [], 100.0,
            survival_premium=1.0,
        )
        # High ICM
        result_icm = self.solver.get_strategy(
            _card("8h"), _card("7h"), Position.CO, [], 100.0,
            survival_premium=0.6,
        )

        def fold_freq(result):
            for a in result.strategy.actions:
                if a.action == "fold":
                    return a.frequency
            return 0.0

        assert fold_freq(result_icm) > fold_freq(result_no_icm)

    def test_lookup_source_for_known_hand(self):
        """Known hands should come from preflop_lookup source."""
        result = self.solver.get_strategy(
            _card("Ah"), _card("Kh"), Position.BTN, [], 100.0,
        )
        assert result.source == "preflop_lookup"
        assert result.confidence > 0.5

    def test_heuristic_source_for_unknown(self):
        """Hands not in the lookup should fall back to heuristic."""
        # 32o from UTG — not in any range
        result = self.solver.get_strategy(
            _card("3h"), _card("2d"), Position.UTG, [], 100.0,
        )
        assert result.source == "heuristic"
        assert result.confidence < 0.5

    def test_spot_key_populated(self):
        """SolverResult should have a populated SpotKey."""
        result = self.solver.get_strategy(
            _card("Ah"), _card("Kh"), Position.BTN, [], 100.0,
        )
        assert result.spot_key is not None
        assert result.spot_key.street == "preflop"
        assert result.spot_key.position == "BTN"
        assert result.spot_key.action_sequence == "open"

    def test_direct_lookup(self):
        """Direct lookup should return a StrategyNode for known hands."""
        node = self.solver.lookup("AKs", "BTN", "open")
        assert node is not None
        assert len(node.actions) >= 1

    def test_direct_lookup_missing(self):
        """Direct lookup for unknown hand returns None."""
        node = self.solver.lookup("32o", "UTG", "open")
        assert node is None
