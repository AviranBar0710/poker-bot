"""Tests for board bucketing."""

from poker_bot.solver.board_bucketing import bucket_board, bucket_spr, bucket_stack
from poker_bot.strategy.decision_maker import BoardTexture


class TestBucketBoard:
    def test_monotone_high(self):
        t = BoardTexture(
            is_monotone=True, high_card_rank=14, num_broadway=3,
        )
        assert bucket_board(t) == "monotone_high"

    def test_monotone_low(self):
        t = BoardTexture(
            is_monotone=True, high_card_rank=8, num_broadway=0,
        )
        assert bucket_board(t) == "monotone_low"

    def test_paired_high(self):
        t = BoardTexture(is_paired=True, high_card_rank=13)
        assert bucket_board(t) == "paired_high"

    def test_paired_low(self):
        t = BoardTexture(is_paired=True, high_card_rank=7)
        assert bucket_board(t) == "paired_low"

    def test_dry_high_rainbow(self):
        t = BoardTexture(
            is_rainbow=True, high_card_rank=14, num_broadway=2,
        )
        assert bucket_board(t) == "dry_high_rainbow"

    def test_dry_low_rainbow(self):
        t = BoardTexture(
            is_rainbow=True, high_card_rank=7, num_broadway=0,
        )
        assert bucket_board(t) == "dry_low_rainbow"

    def test_wet_connected(self):
        t = BoardTexture(
            is_connected=True, is_rainbow=True, high_card_rank=11,
            has_straight_draw=True,
        )
        assert bucket_board(t) == "wet_connected"

    def test_wet_two_tone(self):
        t = BoardTexture(
            is_two_tone=True, is_connected=True, high_card_rank=9,
            has_straight_draw=True,
        )
        assert bucket_board(t) == "wet_two_tone"

    def test_broadway_heavy(self):
        t = BoardTexture(
            is_connected=True, is_rainbow=True, high_card_rank=13,
            num_broadway=3, has_straight_draw=True,
        )
        assert bucket_board(t) == "broadway_heavy"

    def test_connected_low(self):
        t = BoardTexture(
            is_connected=True, is_rainbow=True, high_card_rank=7,
            has_straight_draw=True,
        )
        assert bucket_board(t) == "connected_low"

    def test_dynamic(self):
        t = BoardTexture(
            has_flush_draw=True, has_straight_draw=True, high_card_rank=11,
        )
        assert bucket_board(t) == "dynamic"

    def test_dry_medium(self):
        t = BoardTexture(is_rainbow=True, high_card_rank=9)
        assert bucket_board(t) == "dry_medium"


class TestBucketStack:
    def test_critical(self):
        assert bucket_stack(5) == "critical"
        assert bucket_stack(9.9) == "critical"

    def test_very_short(self):
        assert bucket_stack(10) == "very_short"
        assert bucket_stack(19) == "very_short"

    def test_short(self):
        assert bucket_stack(20) == "short"
        assert bucket_stack(39) == "short"

    def test_medium(self):
        assert bucket_stack(40) == "medium"
        assert bucket_stack(99) == "medium"

    def test_deep(self):
        assert bucket_stack(100) == "deep"
        assert bucket_stack(200) == "deep"


class TestBucketSPR:
    def test_low(self):
        assert bucket_spr(1.0) == "low"
        assert bucket_spr(3.9) == "low"

    def test_medium(self):
        assert bucket_spr(4.0) == "medium"
        assert bucket_spr(10.0) == "medium"

    def test_high(self):
        assert bucket_spr(10.1) == "high"
        assert bucket_spr(20.0) == "high"
