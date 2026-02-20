"""SQLite-backed preflop strategy database.

Stores precise GTO preflop frequencies imported from commercial solvers
(PioSolver, GTO Gecko, etc.). Provides O(1) indexed lookups by
(position, action_sequence, stack_bucket, hand).

Schema:
    preflop_strategies(position, action_sequence, stack_bucket, hand, action, frequency, ev)

Usage:
    db = PreflopDB()                          # opens/creates solver/data/preflop.db
    rows = db.lookup("SB", "open", "100bb", "T7o")
    # → [("limp", 0.83, -0.08), ("fold", 0.17, -0.42)]
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).parent / "data" / "preflop.db"

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS preflop_strategies (
    position        TEXT NOT NULL,
    action_sequence TEXT NOT NULL,
    stack_bucket    TEXT NOT NULL,
    hand            TEXT NOT NULL,
    action          TEXT NOT NULL,
    frequency       REAL NOT NULL,
    ev              REAL DEFAULT 0.0,
    PRIMARY KEY (position, action_sequence, stack_bucket, hand, action)
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_spot
    ON preflop_strategies(position, action_sequence, stack_bucket);
"""

_LOOKUP_SQL = """\
SELECT action, frequency, ev
FROM preflop_strategies
WHERE position = ? AND action_sequence = ? AND stack_bucket = ? AND hand = ?
ORDER BY frequency DESC;
"""

_INSERT_SQL = """\
INSERT OR REPLACE INTO preflop_strategies
    (position, action_sequence, stack_bucket, hand, action, frequency, ev)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

# Supported stack buckets (sorted ascending). Solvers export at these depths.
STACK_BUCKETS = ["10bb", "15bb", "20bb", "25bb", "30bb", "40bb", "50bb", "75bb", "100bb", "150bb", "200bb"]

_STACK_VALUES = [int(b.replace("bb", "")) for b in STACK_BUCKETS]


def nearest_stack_bucket(stack_bb: float) -> str:
    """Snap a stack depth to the nearest available bucket.

    >>> nearest_stack_bucket(47.0)
    '50bb'
    >>> nearest_stack_bucket(100.0)
    '100bb'
    >>> nearest_stack_bucket(12.0)
    '10bb'
    """
    best = _STACK_VALUES[0]
    best_dist = abs(stack_bb - best)
    for v in _STACK_VALUES[1:]:
        d = abs(stack_bb - v)
        if d < best_dist:
            best = v
            best_dist = d
    return f"{best}bb"


class PreflopDB:
    """Connection to the preflop strategies SQLite database."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def lookup(
        self,
        position: str,
        action_sequence: str,
        stack_bucket: str,
        hand: str,
    ) -> list[tuple[str, float, float]]:
        """Look up strategy for a specific spot.

        Returns:
            List of (action, frequency, ev) tuples sorted by frequency desc.
            Empty list if no data found.
        """
        rows = self._conn.execute(
            _LOOKUP_SQL, (position, action_sequence, stack_bucket, hand),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def insert(
        self,
        position: str,
        action_sequence: str,
        stack_bucket: str,
        hand: str,
        action: str,
        frequency: float,
        ev: float = 0.0,
    ) -> None:
        """Insert or replace a single strategy row."""
        self._conn.execute(
            _INSERT_SQL,
            (position, action_sequence, stack_bucket, hand, action, frequency, ev),
        )

    def insert_batch(
        self,
        rows: list[tuple[str, str, str, str, str, float, float]],
    ) -> None:
        """Batch insert rows: [(position, action_seq, stack, hand, action, freq, ev)]."""
        self._conn.executemany(_INSERT_SQL, rows)
        self._conn.commit()

    def commit(self) -> None:
        self._conn.commit()

    def row_count(self) -> int:
        """Total number of strategy rows in the database."""
        return self._conn.execute("SELECT COUNT(*) FROM preflop_strategies").fetchone()[0]

    def list_spots(self) -> list[tuple[str, str, str, int]]:
        """List all unique spots with row counts.

        Returns:
            List of (position, action_sequence, stack_bucket, num_hands).
        """
        rows = self._conn.execute(
            "SELECT position, action_sequence, stack_bucket, COUNT(DISTINCT hand) "
            "FROM preflop_strategies "
            "GROUP BY position, action_sequence, stack_bucket "
            "ORDER BY position, action_sequence, stack_bucket"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
