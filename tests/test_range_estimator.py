"""Tests for range estimator."""

from poker_bot.solver.range_estimator import RangeEstimator
from poker_bot.strategy.decision_maker import PriorAction
from poker_bot.utils.constants import Action, Position, Street


class TestRangeEstimator:
    def test_preflop_single_raise(self):
        """Single raise should give a standard opening range."""
        history = [PriorAction(Position.CO, Action.RAISE, 6.0)]
        range_ = RangeEstimator.estimate_preflop_range(Position.CO, history)
        assert len(range_) > 0
        # CO open range should be reasonably wide
        assert range_.percentage > 15

    def test_preflop_3bet(self):
        """3-bet should narrow to strong range."""
        history = [
            PriorAction(Position.UTG, Action.RAISE, 6.0),
            PriorAction(Position.UTG, Action.RAISE, 18.0),
        ]
        range_ = RangeEstimator.estimate_preflop_range(Position.UTG, history)
        # 3-bet range is narrower than open range
        assert range_.percentage < 10

    def test_preflop_4bet(self):
        """4-bet should be very narrow."""
        history = [
            PriorAction(Position.CO, Action.RAISE, 6.0),
            PriorAction(Position.BTN, Action.RAISE, 18.0),
            PriorAction(Position.CO, Action.RAISE, 45.0),
        ]
        range_ = RangeEstimator.estimate_preflop_range(Position.CO, history)
        assert range_.percentage < 8

    def test_preflop_call(self):
        """Call should keep a medium-wide range."""
        history = [
            PriorAction(Position.CO, Action.RAISE, 6.0),
            PriorAction(Position.BTN, Action.CALL, 6.0),
        ]
        range_ = RangeEstimator.estimate_preflop_range(Position.BTN, history)
        assert range_.percentage > 10

    def test_preflop_wide_range(self):
        """No actions should give very wide range."""
        range_ = RangeEstimator.estimate_preflop_range(Position.BB, [])
        assert range_.percentage > 25

    def test_narrow_for_postflop_raise(self):
        """Postflop raise should narrow range significantly."""
        from poker_bot.strategy.preflop_ranges import Range

        base = Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,AKo,AQo,KQs,JTs,T9s,98s"
        )
        narrowed = RangeEstimator.narrow_for_postflop_action(
            base, Action.RAISE, 0.75, Street.FLOP,
        )
        assert len(narrowed) < len(base)

    def test_narrow_for_postflop_check(self):
        """Check should not narrow much."""
        from poker_bot.strategy.preflop_ranges import Range

        base = Range().add("AA,KK,QQ,JJ,TT,99,AKs,AQs")
        narrowed = RangeEstimator.narrow_for_postflop_action(
            base, Action.CHECK, 0.0, Street.FLOP,
        )
        assert len(narrowed) >= len(base) * 0.7

    def test_narrow_later_streets_tighter(self):
        """Later streets should narrow more than earlier ones."""
        from poker_bot.strategy.preflop_ranges import Range

        base = Range().add(
            "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,AKo,AQo"
        )
        flop_narrow = RangeEstimator.narrow_for_postflop_action(
            base, Action.CALL, 0.5, Street.FLOP,
        )
        river_narrow = RangeEstimator.narrow_for_postflop_action(
            base, Action.CALL, 0.5, Street.RIVER,
        )
        assert len(river_narrow) <= len(flop_narrow)


class TestCategorizeHand:
    def test_nuts(self):
        assert RangeEstimator.categorize_hand(0.98, False) == "nuts"

    def test_strong_made(self):
        assert RangeEstimator.categorize_hand(0.88, False) == "strong_made"

    def test_medium_made(self):
        assert RangeEstimator.categorize_hand(0.70, False) == "medium_made"

    def test_weak_made(self):
        assert RangeEstimator.categorize_hand(0.45, False) == "weak_made"

    def test_strong_draw(self):
        assert RangeEstimator.categorize_hand(0.45, True, 0.6) == "strong_draw"

    def test_medium_draw(self):
        assert RangeEstimator.categorize_hand(0.20, True, 0.3) == "medium_draw"

    def test_weak_draw(self):
        assert RangeEstimator.categorize_hand(0.20, True, 0.1) == "weak_draw"

    def test_air(self):
        assert RangeEstimator.categorize_hand(0.05, False) == "air"
