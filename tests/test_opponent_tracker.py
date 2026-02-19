"""Tests for SQLite-backed OpponentTracker."""

import json
from pathlib import Path

import pytest

from poker_bot.interface.opponent_tracker import OpponentStats, OpponentTracker


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide a temporary DB path for each test."""
    return tmp_path / "opponents.db"


@pytest.fixture
def tracker(db_path: Path) -> OpponentTracker:
    t = OpponentTracker(db_path=db_path)
    yield t
    t.close()


class TestOpponentStatsDataclass:
    def test_vpip_pct_no_hands(self):
        s = OpponentStats(name="X")
        assert s.vpip_pct == 0.0

    def test_vpip_pct_with_hands(self):
        s = OpponentStats(name="X", hands_seen=100, vpip_count=40)
        assert s.vpip_pct == 40.0

    def test_fold_to_cbet_pct(self):
        s = OpponentStats(name="X", cbet_faced=20, fold_to_cbet_count=14)
        assert s.fold_to_cbet_pct == 70.0

    def test_player_type_lag(self):
        s = OpponentStats(name="X", hands_seen=50, vpip_count=25, pfr_count=15)
        assert s.player_type == "LAG (Loose-Aggressive)"

    def test_player_type_unknown_few_hands(self):
        s = OpponentStats(name="X", hands_seen=3)
        assert "Unknown" in s.player_type


class TestTrackerBasicOps:
    def test_update_and_get(self, tracker: OpponentTracker):
        tracker.update_stats("Alice", hands=10, vpip=4, pfr=2)
        p = tracker.get_player("Alice")
        assert p is not None
        assert p.hands_seen == 10
        assert p.vpip_count == 4
        assert p.pfr_count == 2

    def test_incremental_updates(self, tracker: OpponentTracker):
        tracker.update_stats("Bob", hands=5, vpip=2)
        tracker.update_stats("Bob", hands=5, vpip=3)
        p = tracker.get_player("Bob")
        assert p.hands_seen == 10
        assert p.vpip_count == 5

    def test_add_note(self, tracker: OpponentTracker):
        tracker.add_note("Charlie", "Likes to bluff rivers")
        p = tracker.get_player("Charlie")
        assert p is not None
        assert "Likes to bluff rivers" in p.notes

    def test_get_notes_with_timestamps(self, tracker: OpponentTracker):
        tracker.add_note("Dave", "Tight preflop")
        tracker.add_note("Dave", "Opens up on turn")
        notes = tracker.get_notes("Dave")
        assert len(notes) == 2
        assert notes[0][0] == "Tight preflop"
        assert notes[1][0] == "Opens up on turn"
        # Each note has a timestamp string
        assert len(notes[0][1]) > 0

    def test_list_players(self, tracker: OpponentTracker):
        tracker.update_stats("Zoe", hands=1)
        tracker.update_stats("Alice", hands=1)
        players = tracker.list_players()
        assert players == ["Alice", "Zoe"]

    def test_get_unknown_player_returns_none(self, tracker: OpponentTracker):
        assert tracker.get_player("Nobody") is None

    def test_name_stripping(self, tracker: OpponentTracker):
        tracker.update_stats("  Eve  ", hands=5)
        assert tracker.get_player("Eve") is not None

    def test_cbet_stats(self, tracker: OpponentTracker):
        tracker.update_stats("Frank", hands=20, cbet_faced=10, fold_to_cbet=7)
        p = tracker.get_player("Frank")
        assert p.cbet_faced == 10
        assert p.fold_to_cbet_count == 7
        assert p.fold_to_cbet_pct == 70.0


class TestPersistenceAcrossRestarts:
    """Core requirement: stats survive tracker re-initialization."""

    def test_stats_persist_after_restart(self, db_path: Path):
        # Session 1: create and update
        t1 = OpponentTracker(db_path=db_path)
        t1.update_stats("Alice", hands=50, vpip=20, pfr=10,
                        cbet_faced=15, fold_to_cbet=9)
        t1.add_note("Alice", "3-bets light from CO")
        t1.close()

        # Session 2: re-init from same DB
        t2 = OpponentTracker(db_path=db_path)
        p = t2.get_player("Alice")
        assert p is not None
        assert p.hands_seen == 50
        assert p.vpip_count == 20
        assert p.pfr_count == 10
        assert p.cbet_faced == 15
        assert p.fold_to_cbet_count == 9
        assert "3-bets light from CO" in p.notes
        t2.close()

    def test_incremental_across_sessions(self, db_path: Path):
        t1 = OpponentTracker(db_path=db_path)
        t1.update_stats("Bob", hands=30, vpip=10)
        t1.close()

        t2 = OpponentTracker(db_path=db_path)
        t2.update_stats("Bob", hands=20, vpip=8)
        p = t2.get_player("Bob")
        assert p.hands_seen == 50
        assert p.vpip_count == 18
        t2.close()

    def test_notes_accumulate_across_sessions(self, db_path: Path):
        t1 = OpponentTracker(db_path=db_path)
        t1.add_note("Charlie", "Session 1 note")
        t1.close()

        t2 = OpponentTracker(db_path=db_path)
        t2.add_note("Charlie", "Session 2 note")
        p = t2.get_player("Charlie")
        assert len(p.notes) == 2
        assert p.notes[0] == "Session 1 note"
        assert p.notes[1] == "Session 2 note"
        t2.close()


class TestLegacyJsonMigration:
    def test_migrates_json_to_sqlite(self, tmp_path: Path):
        # Create a legacy JSON file
        legacy_data = {
            "Alice": {
                "name": "Alice",
                "notes": ["Loose player", "Tilts easily"],
                "hands_seen": 100,
                "vpip_count": 40,
                "pfr_count": 15,
                "three_bet_count": 5,
                "aggression_actions": 30,
                "passive_actions": 20,
                "cbet_faced": 25,
                "fold_to_cbet_count": 18,
            },
        }
        json_path = tmp_path / "opponents.json"
        json_path.write_text(json.dumps(legacy_data))

        db_path = tmp_path / "opponents.db"
        tracker = OpponentTracker(db_path=db_path)

        p = tracker.get_player("Alice")
        assert p is not None
        assert p.hands_seen == 100
        assert p.vpip_count == 40
        assert p.cbet_faced == 25
        assert "Loose player" in p.notes
        assert "Tilts easily" in p.notes

        # JSON file should be renamed to .bak
        assert not json_path.exists()
        assert (tmp_path / "opponents.json.bak").exists()

        tracker.close()

    def test_migration_skips_if_no_json(self, db_path: Path):
        """No crash when there's no legacy file."""
        tracker = OpponentTracker(db_path=db_path)
        assert tracker.list_players() == []
        tracker.close()

    def test_migration_handles_corrupt_json(self, tmp_path: Path):
        """Corrupt JSON doesn't crash the tracker."""
        json_path = tmp_path / "opponents.json"
        json_path.write_text("{{not valid json")

        db_path = tmp_path / "opponents.db"
        tracker = OpponentTracker(db_path=db_path)
        assert tracker.list_players() == []
        tracker.close()


class TestWalMode:
    def test_wal_mode_enabled(self, db_path: Path):
        tracker = OpponentTracker(db_path=db_path)
        mode = tracker._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode == "wal"
        tracker.close()


class TestCloseMethod:
    def test_close_is_idempotent(self, db_path: Path):
        tracker = OpponentTracker(db_path=db_path)
        tracker.close()
        # Second close should not raise
        tracker.close()

    def test_save_is_noop(self, tracker: OpponentTracker):
        """save() exists for backward compat but does nothing extra."""
        tracker.update_stats("X", hands=1)
        tracker.save()  # Should not raise
        assert tracker.get_player("X").hands_seen == 1
