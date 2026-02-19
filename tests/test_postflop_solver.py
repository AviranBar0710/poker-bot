"""Tests for postflop solver."""

from unittest.mock import patch

from poker_bot.interface.opponent_tracker import OpponentStats
from poker_bot.solver.postflop_solver import (
    PostflopSolver,
    _FULL_CONFIDENCE_MULT,
    _GTO_DEFAULTS,
    _MIN_SAMPLES,
)
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


def _make_stats(
    name: str = "Villain",
    hands_seen: int = 100,
    vpip_count: int = 22,
    pfr_count: int = 18,
    cbet_faced: int = 30,
    fold_to_cbet_count: int = 15,
    aggression_actions: int = 40,
    passive_actions: int = 20,
) -> OpponentStats:
    """Create an OpponentStats with sensible defaults."""
    return OpponentStats(
        name=name,
        hands_seen=hands_seen,
        vpip_count=vpip_count,
        pfr_count=pfr_count,
        cbet_faced=cbet_faced,
        fold_to_cbet_count=fold_to_cbet_count,
        aggression_actions=aggression_actions,
        passive_actions=passive_actions,
    )


class TestEffectiveStat:
    """Test the linear blending helper for confidence gating."""

    def test_below_threshold_returns_gto_default(self):
        """Sample below threshold should return pure GTO default."""
        result = PostflopSolver._effective_stat(40.0, "vpip", 10)
        assert result == _GTO_DEFAULTS["vpip"]

    def test_at_threshold_returns_gto_default(self):
        """Exactly at threshold boundary, weight is 0 so still GTO default."""
        threshold = _MIN_SAMPLES["vpip"]
        result = PostflopSolver._effective_stat(40.0, "vpip", threshold)
        assert result == _GTO_DEFAULTS["vpip"]

    def test_at_full_confidence_returns_observed(self):
        """At 3× threshold, should return the full observed value."""
        threshold = _MIN_SAMPLES["vpip"]
        full_at = threshold * _FULL_CONFIDENCE_MULT
        result = PostflopSolver._effective_stat(40.0, "vpip", full_at)
        assert abs(result - 40.0) < 0.01

    def test_midpoint_blends_halfway(self):
        """Halfway between threshold and full should blend ~50%."""
        threshold = _MIN_SAMPLES["vpip"]
        full_at = threshold * _FULL_CONFIDENCE_MULT
        midpoint = (threshold + full_at) // 2
        result = PostflopSolver._effective_stat(40.0, "vpip", midpoint)
        gto = _GTO_DEFAULTS["vpip"]
        expected = gto + 0.5 * (40.0 - gto)
        assert abs(result - expected) < 1.0  # within 1%

    def test_above_full_confidence_capped_at_observed(self):
        """Sample >> 3× threshold should still return observed, not overshoot."""
        result = PostflopSolver._effective_stat(40.0, "vpip", 500)
        assert abs(result - 40.0) < 0.01

    def test_fold_to_cbet_threshold_uses_cbet_faced(self):
        """fold_to_cbet uses its own threshold (10 instances)."""
        # Below threshold
        result_low = PostflopSolver._effective_stat(70.0, "fold_to_cbet", 5)
        assert result_low == _GTO_DEFAULTS["fold_to_cbet"]
        # Above full confidence
        full_at = _MIN_SAMPLES["fold_to_cbet"] * _FULL_CONFIDENCE_MULT
        result_high = PostflopSolver._effective_stat(70.0, "fold_to_cbet", full_at)
        assert abs(result_high - 70.0) < 0.01


class TestExploitAdjustments:
    """Test exploitative frequency shifts based on opponent tendencies."""

    def setup_method(self):
        self.solver = PostflopSolver()
        self._air_common = dict(
            hero_cards=_cards("2h 3d"),
            community_cards=_cards("As Kd 9c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.05,
            is_ip=True,
            num_opponents=1,
        )
        self._strong_common = dict(
            hero_cards=_cards("Ah Ad"),
            community_cards=_cards("As 8d 3c"),
            position=Position.BTN,
            pot=10.0,
            hero_stack=100.0,
            big_blind=1.0,
            action_history=[],
            hand_strength=0.95,
            is_ip=True,
            num_opponents=1,
        )

    def _get_freq(self, result, action_name):
        return sum(a.frequency for a in result.strategy.actions if a.action == action_name)

    def test_no_stats_matches_gto(self):
        """No opponent_stats should produce the same result as GTO baseline."""
        gto_result = self.solver.get_strategy(**self._air_common)
        stats_result = self.solver.get_strategy(**self._air_common, opponent_stats=None)
        for a1, a2 in zip(gto_result.strategy.actions, stats_result.strategy.actions):
            assert abs(a1.frequency - a2.frequency) < 0.001

    def test_low_sample_matches_gto(self):
        """Opponent with too few hands should not trigger exploit adjustments."""
        low_sample = _make_stats(hands_seen=10, vpip_count=8, cbet_faced=3)
        gto_result = self.solver.get_strategy(**self._air_common)
        exploit_result = self.solver.get_strategy(**self._air_common, opponent_stats=low_sample)
        gto_raise = self._get_freq(gto_result, "raise")
        exploit_raise = self._get_freq(exploit_result, "raise")
        # Should be nearly identical since low samples fall back to GTO defaults
        assert abs(gto_raise - exploit_raise) < 0.01

    def test_high_fold_to_cbet_bluffs_more(self):
        """Vs opponent who folds >60% to c-bets, bluff more with air."""
        high_folder = _make_stats(
            cbet_faced=40, fold_to_cbet_count=30,  # 75% fold
        )
        gto_result = self.solver.get_strategy(**self._air_common)
        exploit_result = self.solver.get_strategy(**self._air_common, opponent_stats=high_folder)
        assert self._get_freq(exploit_result, "raise") > self._get_freq(gto_result, "raise")

    def test_low_fold_to_cbet_bluffs_less(self):
        """Vs opponent who folds <30% to c-bets, bluff less with air."""
        station = _make_stats(
            cbet_faced=40, fold_to_cbet_count=8,  # 20% fold
        )
        gto_result = self.solver.get_strategy(**self._air_common)
        exploit_result = self.solver.get_strategy(**self._air_common, opponent_stats=station)
        assert self._get_freq(exploit_result, "raise") < self._get_freq(gto_result, "raise")

    def test_low_fold_cbet_value_bets_more(self):
        """Vs calling station, value bet more with strong hands."""
        station = _make_stats(
            cbet_faced=40, fold_to_cbet_count=8,  # 20% fold
        )
        gto_result = self.solver.get_strategy(**self._strong_common)
        exploit_result = self.solver.get_strategy(**self._strong_common, opponent_stats=station)
        assert self._get_freq(exploit_result, "raise") > self._get_freq(gto_result, "raise")

    def test_hyper_aggressive_traps_more(self):
        """Vs hyper-aggressive opponent, check more with strong hands (trap)."""
        lag = _make_stats(
            aggression_actions=80, passive_actions=10,  # AF = 8.0
        )
        gto_result = self.solver.get_strategy(**self._strong_common)
        exploit_result = self.solver.get_strategy(**self._strong_common, opponent_stats=lag)
        assert self._get_freq(exploit_result, "check") > self._get_freq(gto_result, "check")

    def test_exploit_frequencies_sum_to_one(self):
        """Exploit-adjusted strategies should still normalize to ~1.0."""
        for stats in [
            _make_stats(cbet_faced=40, fold_to_cbet_count=35),  # high folder
            _make_stats(cbet_faced=40, fold_to_cbet_count=5),   # station
            _make_stats(aggression_actions=80, passive_actions=10),  # LAG
        ]:
            result = self.solver.get_strategy(**self._air_common, opponent_stats=stats)
            total = sum(a.frequency for a in result.strategy.actions)
            assert abs(total - 1.0) < 0.01, f"freqs sum to {total} for {stats.name}"

    def test_gradual_scaling_increases_adjustment(self):
        """As sample grows, adjustment should increase monotonically."""
        bluff_freqs = []
        for hands in [30, 45, 60, 90, 150]:
            # Scale cbet_faced proportionally
            cbet_faced = max(10, hands // 3)
            fold_count = int(cbet_faced * 0.75)  # 75% fold rate
            stats = _make_stats(
                hands_seen=hands,
                cbet_faced=cbet_faced,
                fold_to_cbet_count=fold_count,
            )
            result = self.solver.get_strategy(**self._air_common, opponent_stats=stats)
            bluff_freqs.append(self._get_freq(result, "raise"))

        # Bluff frequency should generally increase as confidence grows
        # (because observed fold-to-cbet 75% > GTO default 50%)
        assert bluff_freqs[-1] >= bluff_freqs[0]
