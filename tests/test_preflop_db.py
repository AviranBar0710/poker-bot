"""Tests for preflop SQLite database and import pipeline."""

import csv
from pathlib import Path

import pytest

from poker_bot.solver.preflop_db import PreflopDB, nearest_stack_bucket, STACK_BUCKETS
from poker_bot.solver.preflop_solver import (
    PreflopSolver,
    _action_sequence,
    _has_limps,
    _count_raises,
    _parse_db_action,
)
from poker_bot.strategy.decision_maker import PriorAction
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position


def _card(s: str) -> Card:
    return Card.from_str(s)


# ---------------------------------------------------------------------------
# nearest_stack_bucket
# ---------------------------------------------------------------------------

class TestNearestStackBucket:
    def test_exact_match(self):
        assert nearest_stack_bucket(100.0) == "100bb"
        assert nearest_stack_bucket(50.0) == "50bb"
        assert nearest_stack_bucket(10.0) == "10bb"

    def test_rounds_to_nearest(self):
        assert nearest_stack_bucket(47.0) == "50bb"
        assert nearest_stack_bucket(12.0) == "10bb"
        assert nearest_stack_bucket(90.0) == "100bb"
        assert nearest_stack_bucket(130.0) == "150bb"

    def test_extreme_low(self):
        assert nearest_stack_bucket(3.0) == "10bb"

    def test_extreme_high(self):
        assert nearest_stack_bucket(500.0) == "200bb"

    def test_midpoint_between_buckets(self):
        # Between 40 and 50, midpoint is 45 — should go to closest
        result = nearest_stack_bucket(45.0)
        assert result in ("40bb", "50bb")


# ---------------------------------------------------------------------------
# PreflopDB
# ---------------------------------------------------------------------------

class TestPreflopDB:
    def test_create_empty_db(self, tmp_path):
        db = PreflopDB(db_path=tmp_path / "test.db")
        assert db.row_count() == 0
        assert db.list_spots() == []
        db.close()

    def test_insert_and_lookup(self, tmp_path):
        db = PreflopDB(db_path=tmp_path / "test.db")
        db.insert("SB", "open", "100bb", "AA", "raise_2.5", 1.0, 0.52)
        db.commit()

        rows = db.lookup("SB", "open", "100bb", "AA")
        assert len(rows) == 1
        assert rows[0] == ("raise_2.5", 1.0, 0.52)
        db.close()

    def test_insert_batch(self, tmp_path):
        db = PreflopDB(db_path=tmp_path / "test.db")
        batch = [
            ("SB", "open", "100bb", "AA", "raise_2.5", 1.0, 0.52),
            ("SB", "open", "100bb", "T7o", "limp", 0.83, -0.08),
            ("SB", "open", "100bb", "T7o", "fold", 0.17, -0.08),
        ]
        db.insert_batch(batch)

        rows = db.lookup("SB", "open", "100bb", "T7o")
        assert len(rows) == 2
        # Sorted by frequency descending
        assert rows[0][0] == "limp"
        assert rows[0][1] == pytest.approx(0.83)
        assert rows[1][0] == "fold"
        db.close()

    def test_lookup_missing(self, tmp_path):
        db = PreflopDB(db_path=tmp_path / "test.db")
        rows = db.lookup("SB", "open", "100bb", "XX")
        assert rows == []
        db.close()

    def test_insert_or_replace(self, tmp_path):
        db = PreflopDB(db_path=tmp_path / "test.db")
        db.insert("SB", "open", "100bb", "AA", "raise_2.5", 0.80, 0.40)
        db.commit()
        db.insert("SB", "open", "100bb", "AA", "raise_2.5", 1.00, 0.52)
        db.commit()

        rows = db.lookup("SB", "open", "100bb", "AA")
        assert len(rows) == 1
        assert rows[0][1] == pytest.approx(1.0)
        db.close()

    def test_row_count(self, tmp_path):
        db = PreflopDB(db_path=tmp_path / "test.db")
        batch = [
            ("SB", "open", "100bb", "AA", "raise_2.5", 1.0, 0.52),
            ("SB", "open", "100bb", "KK", "raise_2.5", 1.0, 0.45),
            ("SB", "open", "100bb", "T7o", "limp", 0.83, -0.08),
        ]
        db.insert_batch(batch)
        assert db.row_count() == 3
        db.close()

    def test_list_spots(self, tmp_path):
        db = PreflopDB(db_path=tmp_path / "test.db")
        batch = [
            ("SB", "open", "100bb", "AA", "raise_2.5", 1.0, 0.52),
            ("SB", "open", "100bb", "KK", "raise_2.5", 1.0, 0.45),
            ("BTN", "open", "100bb", "AA", "raise_2.5", 1.0, 0.55),
        ]
        db.insert_batch(batch)

        spots = db.list_spots()
        assert len(spots) == 2
        # Each spot: (position, action_seq, stack_bucket, num_distinct_hands)
        sb_spot = [s for s in spots if s[0] == "SB"][0]
        assert sb_spot[3] == 2  # AA and KK
        db.close()


# ---------------------------------------------------------------------------
# _parse_db_action
# ---------------------------------------------------------------------------

class TestParseDbAction:
    def test_fold(self):
        assert _parse_db_action("fold") == ("fold", 0.0)

    def test_call(self):
        assert _parse_db_action("call") == ("call", 0.0)

    def test_limp(self):
        assert _parse_db_action("limp") == ("limp", 1.0)

    def test_raise_with_amount(self):
        assert _parse_db_action("raise_2.5") == ("raise", 2.5)
        assert _parse_db_action("raise_3.0") == ("raise", 3.0)

    def test_raise_all_in(self):
        assert _parse_db_action("raise_all_in") == ("all_in", 0.0)

    def test_raise_bare(self):
        assert _parse_db_action("raise") == ("raise", 2.5)

    def test_check(self):
        assert _parse_db_action("check") == ("check", 0.0)


# ---------------------------------------------------------------------------
# _action_sequence / _count_raises / _has_limps
# ---------------------------------------------------------------------------

class TestActionSequenceHelpers:
    def test_open_no_history(self):
        assert _action_sequence([]) == "open"

    def test_vs_raise(self):
        history = [PriorAction(Position.UTG, Action.RAISE, 6.0)]
        assert _action_sequence(history) == "vs_raise"

    def test_vs_3bet(self):
        history = [
            PriorAction(Position.UTG, Action.RAISE, 6.0),
            PriorAction(Position.BTN, Action.RAISE, 18.0),
        ]
        assert _action_sequence(history) == "vs_3bet"

    def test_vs_4bet(self):
        history = [
            PriorAction(Position.UTG, Action.RAISE, 6.0),
            PriorAction(Position.BTN, Action.RAISE, 18.0),
            PriorAction(Position.UTG, Action.RAISE, 50.0),
        ]
        assert _action_sequence(history) == "vs_4bet"

    def test_vs_limp(self):
        history = [PriorAction(Position.UTG, Action.CALL, 2.0)]
        assert _action_sequence(history) == "vs_limp"

    def test_has_limps_true(self):
        history = [PriorAction(Position.UTG, Action.LIMP, 2.0)]
        assert _has_limps(history) is True

    def test_has_limps_false(self):
        history = [PriorAction(Position.UTG, Action.RAISE, 6.0)]
        assert _has_limps(history) is False

    def test_count_raises(self):
        history = [
            PriorAction(Position.UTG, Action.RAISE, 6.0),
            PriorAction(Position.BTN, Action.RAISE, 18.0),
        ]
        assert _count_raises(history) == 2


# ---------------------------------------------------------------------------
# PreflopSolver with SQLite DB
# ---------------------------------------------------------------------------

class TestPreflopSolverDB:
    """Test the three-tier resolution with a real SQLite DB."""

    @pytest.fixture
    def solver_with_db(self, tmp_path):
        """Create a PreflopSolver with a small test DB."""
        db_path = tmp_path / "test_preflop.db"
        db = PreflopDB(db_path=db_path)
        batch = [
            # AA from SB: pure raise
            ("SB", "open", "100bb", "AA", "raise_2.5", 1.0, 0.52),
            # T7o from SB: 83% limp, 17% fold (the GTO example)
            ("SB", "open", "100bb", "T7o", "limp", 0.83, -0.08),
            ("SB", "open", "100bb", "T7o", "fold", 0.17, -0.08),
            # KK from BTN: pure raise
            ("BTN", "open", "100bb", "KK", "raise_2.5", 1.0, 0.45),
        ]
        db.insert_batch(batch)
        db.close()

        return PreflopSolver(db_path=db_path)

    def test_db_hit_aa_sb(self, solver_with_db):
        """AA from SB should come from the DB with confidence 0.95."""
        result = solver_with_db.get_strategy(
            _card("Ah"), _card("As"), Position.SB, [], 100.0,
        )
        assert result.source == "preflop_db"
        assert result.confidence == 0.95
        rec = result.strategy.recommended_action
        assert rec.action == "raise"
        assert rec.frequency == pytest.approx(1.0)

    def test_db_hit_t7o_limp(self, solver_with_db):
        """T7o from SB should recommend limp at 83% from DB."""
        result = solver_with_db.get_strategy(
            _card("Th"), _card("7d"), Position.SB, [], 100.0,
        )
        assert result.source == "preflop_db"
        rec = result.strategy.recommended_action
        assert rec.action == "limp"
        assert rec.frequency == pytest.approx(0.83)

    def test_db_miss_falls_to_json(self, solver_with_db):
        """A hand in JSON but not DB should come from preflop_lookup."""
        # AKs from BTN — in JSON but our test DB doesn't have it for BTN
        result = solver_with_db.get_strategy(
            _card("Ah"), _card("Kh"), Position.BTN, [], 100.0,
        )
        # Should fall through to JSON (the DB only has KK for BTN)
        # AKs/BTN/open is in the legacy JSON
        assert result.source == "preflop_lookup"
        assert result.confidence == 0.85

    def test_db_miss_json_miss_falls_to_heuristic(self, solver_with_db):
        """A hand not in DB or JSON should come from heuristic."""
        # 32o from UTG — not in any lookup
        result = solver_with_db.get_strategy(
            _card("3h"), _card("2d"), Position.UTG, [], 100.0,
        )
        assert result.source == "heuristic"
        assert result.confidence == 0.4

    def test_db_stack_bucket_snapping(self, solver_with_db):
        """Stack of 95bb should snap to 100bb bucket and find DB data."""
        result = solver_with_db.get_strategy(
            _card("Ah"), _card("As"), Position.SB, [], 95.0,
        )
        assert result.source == "preflop_db"

    def test_icm_on_db_result(self, solver_with_db):
        """ICM pressure should increase fold frequency even on DB results."""
        result_normal = solver_with_db.get_strategy(
            _card("Th"), _card("7d"), Position.SB, [], 100.0,
            survival_premium=1.0,
        )
        result_icm = solver_with_db.get_strategy(
            _card("Th"), _card("7d"), Position.SB, [], 100.0,
            survival_premium=0.6,
        )

        def fold_freq(result):
            for a in result.strategy.actions:
                if a.action == "fold":
                    return a.frequency
            return 0.0

        assert fold_freq(result_icm) > fold_freq(result_normal)


# ---------------------------------------------------------------------------
# Import script (parse_filename + import_csv)
# ---------------------------------------------------------------------------

class TestImportScript:
    def test_parse_filename_valid(self):
        from scripts.import_preflop import parse_filename
        result = parse_filename(Path("SB_open_100bb.csv"))
        assert result == ("SB", "open", "100bb")

    def test_parse_filename_complex_action(self):
        from scripts.import_preflop import parse_filename
        result = parse_filename(Path("BB_vs_sb_limp_100bb.csv"))
        assert result == ("BB", "vs_sb_limp", "100bb")

    def test_parse_filename_invalid(self):
        from scripts.import_preflop import parse_filename
        assert parse_filename(Path("random_file.csv")) is None
        assert parse_filename(Path("XX_open_100bb.csv")) is None

    def test_import_csv_roundtrip(self, tmp_path):
        from scripts.import_preflop import import_csv

        # Create a small CSV
        csv_path = tmp_path / "BTN_open_50bb.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hand", "fold", "raise_2.5", "ev"])
            writer.writerow(["AA", "0.00", "1.00", "0.55"])
            writer.writerow(["72o", "0.95", "0.05", "-0.42"])

        db = PreflopDB(db_path=tmp_path / "test.db")
        hands, actions = import_csv(csv_path, db)
        assert hands == 2
        assert actions == 3  # AA raise + 72o fold + 72o raise

        # Verify lookups
        rows = db.lookup("BTN", "open", "50bb", "AA")
        assert len(rows) == 1
        assert rows[0][0] == "raise_2.5"

        rows = db.lookup("BTN", "open", "50bb", "72o")
        assert len(rows) == 2
        db.close()

    def test_import_csv_skips_tiny_frequency(self, tmp_path):
        from scripts.import_preflop import import_csv

        csv_path = tmp_path / "CO_open_100bb.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hand", "fold", "raise_2.5", "ev"])
            writer.writerow(["AA", "0.0005", "0.9995", "0.55"])

        db = PreflopDB(db_path=tmp_path / "test.db")
        hands, actions = import_csv(csv_path, db)
        assert hands == 1
        assert actions == 1  # Only raise (fold < 0.001 threshold)
        db.close()

    def test_import_csv_dry_run(self, tmp_path):
        from scripts.import_preflop import import_csv

        csv_path = tmp_path / "SB_open_100bb.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hand", "fold", "raise_2.5", "ev"])
            writer.writerow(["AA", "0.00", "1.00", "0.55"])

        db = PreflopDB(db_path=tmp_path / "test.db")
        hands, actions = import_csv(csv_path, db, dry_run=True)
        assert hands == 1
        assert actions == 1
        # DB should still be empty
        assert db.row_count() == 0
        db.close()
