"""Tests for bet sizing module."""

import math

from poker_bot.solver.bet_sizing import BetSizingTree, SizingOption


class TestBetSizingTree:
    def test_get_sizings_known_texture(self):
        sizings = BetSizingTree.get_sizings("dry_high_rainbow")
        assert len(sizings) >= 1
        assert all(isinstance(s, SizingOption) for s in sizings)
        # Dry boards should have small sizings
        assert sizings[0].fraction <= 0.5

    def test_get_sizings_wet_texture(self):
        sizings = BetSizingTree.get_sizings("wet_connected")
        assert sizings[0].fraction >= 0.5

    def test_get_sizings_unknown_texture(self):
        sizings = BetSizingTree.get_sizings("nonexistent_texture")
        assert len(sizings) >= 1  # Should return defaults

    def test_primary_sizing(self):
        assert BetSizingTree.primary_sizing("dry_high_rainbow") == 0.33
        assert BetSizingTree.primary_sizing("dynamic") == 0.75

    def test_geometric_sizing_one_street(self):
        # 100 pot, 100 stack, 1 street: should bet 100% pot
        sizing = BetSizingTree.geometric_sizing(100, 100, 1)
        assert abs(sizing - 1.0) < 0.01

    def test_geometric_sizing_two_streets(self):
        # 100 pot, 200 stack, 2 streets
        sizing = BetSizingTree.geometric_sizing(100, 200, 2)
        # (1 + 200/100)^(1/2) - 1 = sqrt(3) - 1 ≈ 0.73
        expected = math.sqrt(3) - 1
        assert abs(sizing - expected) < 0.01

    def test_geometric_sizing_three_streets(self):
        sizing = BetSizingTree.geometric_sizing(100, 300, 3)
        # (1 + 3)^(1/3) - 1 ≈ 0.587
        expected = 4 ** (1/3) - 1
        assert abs(sizing - expected) < 0.01

    def test_geometric_sizing_zero_pot(self):
        assert BetSizingTree.geometric_sizing(0, 100, 2) == 0.0

    def test_geometric_sizing_zero_stack(self):
        assert BetSizingTree.geometric_sizing(100, 0, 2) == 0.0

    def test_compute_bet_amount(self):
        amount = BetSizingTree.compute_bet_amount(100, 200, 0.5)
        assert amount == 50.0

    def test_compute_bet_amount_capped_at_stack(self):
        amount = BetSizingTree.compute_bet_amount(100, 30, 0.5)
        assert amount == 30.0

    def test_compute_bet_amount_min_bet(self):
        amount = BetSizingTree.compute_bet_amount(10, 100, 0.1, min_bet=4.0)
        assert amount == 4.0
