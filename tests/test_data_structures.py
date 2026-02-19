"""Tests for solver data structures."""

from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverResult,
    SpotKey,
    StrategyNode,
)


class TestActionFrequency:
    def test_creation(self):
        af = ActionFrequency(action="raise", frequency=0.7, amount=6.0, ev=1.5)
        assert af.action == "raise"
        assert af.frequency == 0.7
        assert af.amount == 6.0
        assert af.ev == 1.5

    def test_defaults(self):
        af = ActionFrequency(action="fold", frequency=0.3)
        assert af.amount == 0.0
        assert af.ev == 0.0

    def test_frozen(self):
        af = ActionFrequency(action="fold", frequency=0.3)
        try:
            af.action = "call"
            assert False, "Should raise"
        except AttributeError:
            pass


class TestStrategyNode:
    def test_recommended_action(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 0.7, 6.0, 2.0),
            ActionFrequency("call", 0.2, 2.0, 0.5),
            ActionFrequency("fold", 0.1, 0.0, 0.0),
        ])
        rec = node.recommended_action
        assert rec is not None
        assert rec.action == "raise"

    def test_recommended_action_empty(self):
        node = StrategyNode()
        assert node.recommended_action is None

    def test_sample_action_single(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 1.0, 6.0),
        ])
        for _ in range(10):
            assert node.sample_action().action == "raise"

    def test_sample_action_mixed(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 0.5),
            ActionFrequency("fold", 0.5),
        ])
        actions = {node.sample_action().action for _ in range(100)}
        assert "raise" in actions
        assert "fold" in actions

    def test_sample_action_empty(self):
        node = StrategyNode()
        assert node.sample_action() is None

    def test_is_pure(self):
        pure = StrategyNode(actions=[ActionFrequency("raise", 1.0)])
        assert pure.is_pure

        mixed = StrategyNode(actions=[
            ActionFrequency("raise", 0.7),
            ActionFrequency("fold", 0.3),
        ])
        assert not mixed.is_pure

    def test_best_ev(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 0.5, ev=2.0),
            ActionFrequency("call", 0.3, ev=0.5),
            ActionFrequency("fold", 0.2, ev=0.0),
        ])
        assert node.best_ev == 2.0

    def test_weighted_ev(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 0.5, ev=2.0),
            ActionFrequency("fold", 0.5, ev=0.0),
        ])
        assert abs(node.weighted_ev - 1.0) < 1e-6

    def test_normalized(self):
        node = StrategyNode(actions=[
            ActionFrequency("raise", 3.0),
            ActionFrequency("fold", 1.0),
        ])
        normed = node.normalized()
        assert abs(normed.actions[0].frequency - 0.75) < 1e-6
        assert abs(normed.actions[1].frequency - 0.25) < 1e-6


class TestSpotKey:
    def test_hashable(self):
        k1 = SpotKey("preflop", "BTN", "open", "deep")
        k2 = SpotKey("preflop", "BTN", "open", "deep")
        assert k1 == k2
        assert hash(k1) == hash(k2)
        d = {k1: "value"}
        assert d[k2] == "value"

    def test_different_keys(self):
        k1 = SpotKey("preflop", "BTN", "open", "deep")
        k2 = SpotKey("preflop", "CO", "open", "deep")
        assert k1 != k2


class TestSolverResult:
    def test_creation(self):
        strategy = StrategyNode(actions=[
            ActionFrequency("raise", 0.7, 6.0, 1.5),
            ActionFrequency("fold", 0.3, 0.0, 0.0),
        ])
        result = SolverResult(
            strategy=strategy,
            source="preflop_lookup",
            confidence=0.9,
            ev=1.05,
        )
        assert result.source == "preflop_lookup"
        assert result.confidence == 0.9
        assert result.strategy.recommended_action.action == "raise"
