"""Persistent opponent notes and statistics tracker.

Stores opponent data in a SQLite database (~/.poker_coach/opponents.db)
with WAL mode for concurrent-safe reads/writes. An in-memory cache
keeps lookups O(1) during gameplay.

Migrates automatically from the legacy JSON format on first run.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


_DATA_DIR = Path.home() / ".poker_coach"
_DB_FILE = _DATA_DIR / "opponents.db"
_LEGACY_JSON = _DATA_DIR / "opponents.json"

_CREATE_STATS_TABLE = """\
CREATE TABLE IF NOT EXISTS opponent_stats (
    name               TEXT PRIMARY KEY,
    hands_seen         INTEGER NOT NULL DEFAULT 0,
    vpip_count         INTEGER NOT NULL DEFAULT 0,
    pfr_count          INTEGER NOT NULL DEFAULT 0,
    three_bet_count    INTEGER NOT NULL DEFAULT 0,
    aggression_actions INTEGER NOT NULL DEFAULT 0,
    passive_actions    INTEGER NOT NULL DEFAULT 0,
    cbet_faced         INTEGER NOT NULL DEFAULT 0,
    fold_to_cbet_count INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_NOTES_TABLE = """\
CREATE TABLE IF NOT EXISTS opponent_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL REFERENCES opponent_stats(name),
    note       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_UPSERT_STATS = """\
INSERT INTO opponent_stats (
    name, hands_seen, vpip_count, pfr_count, three_bet_count,
    aggression_actions, passive_actions, cbet_faced, fold_to_cbet_count
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(name) DO UPDATE SET
    hands_seen         = excluded.hands_seen,
    vpip_count         = excluded.vpip_count,
    pfr_count          = excluded.pfr_count,
    three_bet_count    = excluded.three_bet_count,
    aggression_actions = excluded.aggression_actions,
    passive_actions    = excluded.passive_actions,
    cbet_faced         = excluded.cbet_faced,
    fold_to_cbet_count = excluded.fold_to_cbet_count;
"""


@dataclass
class OpponentStats:
    """Tracked statistics and notes for a single opponent."""

    name: str
    notes: list[str] = field(default_factory=list)
    hands_seen: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    three_bet_count: int = 0
    aggression_actions: int = 0  # bet + raise
    passive_actions: int = 0     # check + call
    cbet_faced: int = 0          # times faced a continuation bet
    fold_to_cbet_count: int = 0  # times folded to a continuation bet

    @property
    def vpip_pct(self) -> float:
        return (self.vpip_count / self.hands_seen * 100) if self.hands_seen else 0.0

    @property
    def pfr_pct(self) -> float:
        return (self.pfr_count / self.hands_seen * 100) if self.hands_seen else 0.0

    @property
    def three_bet_pct(self) -> float:
        return (self.three_bet_count / self.hands_seen * 100) if self.hands_seen else 0.0

    @property
    def fold_to_cbet_pct(self) -> float:
        return (self.fold_to_cbet_count / self.cbet_faced * 100) if self.cbet_faced else 0.0

    @property
    def aggression_factor(self) -> float:
        return (self.aggression_actions / self.passive_actions) if self.passive_actions else float("inf")

    @property
    def player_type(self) -> str:
        """Classify player based on VPIP and PFR."""
        if self.hands_seen < 5:
            return "Unknown (need more data)"
        vpip = self.vpip_pct
        pfr = self.pfr_pct
        if vpip > 35 and pfr > 25:
            return "LAG (Loose-Aggressive)"
        if vpip > 35 and pfr <= 25:
            return "Loose-Passive (Calling Station)"
        if vpip <= 25 and pfr > 18:
            return "TAG (Tight-Aggressive)"
        if vpip <= 25:
            return "Nit (Tight-Passive)"
        return "Average"

    def summary(self) -> str:
        """One-line summary of this player."""
        if self.hands_seen == 0:
            notes_str = f" | Notes: {len(self.notes)}" if self.notes else ""
            return f"{self.name}: No stats yet{notes_str}"
        return (
            f"{self.name}: {self.player_type} | "
            f"VPIP {self.vpip_pct:.0f}% | PFR {self.pfr_pct:.0f}% | "
            f"3bet {self.three_bet_pct:.0f}% | AF {self.aggression_factor:.1f} | "
            f"{self.hands_seen} hands"
        )


class OpponentTracker:
    """Manages persistent opponent notes and statistics.

    Uses SQLite with WAL mode for safe concurrent access. Stats are
    cached in memory for O(1) lookups during gameplay and written
    through to the database on every update.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DB_FILE
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._create_tables()
        self._migrate_legacy_json()
        self._players: dict[str, OpponentStats] = {}
        self._load()

    def _create_tables(self) -> None:
        self._conn.execute(_CREATE_STATS_TABLE)
        self._conn.execute(_CREATE_NOTES_TABLE)
        self._conn.commit()

    def _migrate_legacy_json(self) -> None:
        """Import data from the old JSON file if it exists."""
        json_path = self._db_path.parent / "opponents.json"
        if not json_path.exists():
            return
        try:
            raw = json.loads(json_path.read_text())
            for name, d in raw.items():
                notes = d.pop("notes", [])
                self._conn.execute(_UPSERT_STATS, (
                    name,
                    d.get("hands_seen", 0),
                    d.get("vpip_count", 0),
                    d.get("pfr_count", 0),
                    d.get("three_bet_count", 0),
                    d.get("aggression_actions", 0),
                    d.get("passive_actions", 0),
                    d.get("cbet_faced", 0),
                    d.get("fold_to_cbet_count", 0),
                ))
                for note in notes:
                    self._conn.execute(
                        "INSERT INTO opponent_notes (name, note) VALUES (?, ?)",
                        (name, note),
                    )
            self._conn.commit()
            # Rename legacy file so migration only runs once
            backup = json_path.with_suffix(".json.bak")
            json_path.rename(backup)
        except (json.JSONDecodeError, TypeError, KeyError, OSError):
            pass  # Skip migration on corrupt/unreadable data

    def _load(self) -> None:
        """Load all stats and notes from the database into memory."""
        self._players.clear()
        rows = self._conn.execute(
            "SELECT name, hands_seen, vpip_count, pfr_count, three_bet_count, "
            "aggression_actions, passive_actions, cbet_faced, fold_to_cbet_count "
            "FROM opponent_stats"
        ).fetchall()
        for row in rows:
            name = row[0]
            self._players[name] = OpponentStats(
                name=name,
                hands_seen=row[1],
                vpip_count=row[2],
                pfr_count=row[3],
                three_bet_count=row[4],
                aggression_actions=row[5],
                passive_actions=row[6],
                cbet_faced=row[7],
                fold_to_cbet_count=row[8],
            )
        # Load notes for each player
        note_rows = self._conn.execute(
            "SELECT name, note FROM opponent_notes ORDER BY id"
        ).fetchall()
        for name, note in note_rows:
            if name in self._players:
                self._players[name].notes.append(note)

    def _save_player(self, name: str) -> None:
        """Write a single player's stats to the database."""
        p = self._players[name]
        self._conn.execute(_UPSERT_STATS, (
            p.name, p.hands_seen, p.vpip_count, p.pfr_count,
            p.three_bet_count, p.aggression_actions, p.passive_actions,
            p.cbet_faced, p.fold_to_cbet_count,
        ))
        self._conn.commit()

    def add_note(self, name: str, note: str) -> None:
        """Add a text note for a player."""
        name = name.strip()
        note = note.strip()
        if name not in self._players:
            self._players[name] = OpponentStats(name=name)
            self._save_player(name)
        self._players[name].notes.append(note)
        self._conn.execute(
            "INSERT INTO opponent_notes (name, note) VALUES (?, ?)",
            (name, note),
        )
        self._conn.commit()

    def get_notes(self, name: str) -> list[tuple[str, str]]:
        """Get all notes for a player with timestamps.

        Returns:
            List of (note, created_at) tuples, ordered by creation time.
        """
        rows = self._conn.execute(
            "SELECT note, created_at FROM opponent_notes "
            "WHERE name = ? ORDER BY id",
            (name.strip(),),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def update_stats(
        self,
        name: str,
        hands: int = 0,
        vpip: int = 0,
        pfr: int = 0,
        three_bet: int = 0,
        aggressive: int = 0,
        passive: int = 0,
        cbet_faced: int = 0,
        fold_to_cbet: int = 0,
    ) -> None:
        """Increment stat counters for a player."""
        name = name.strip()
        if name not in self._players:
            self._players[name] = OpponentStats(name=name)
        p = self._players[name]
        p.hands_seen += hands
        p.vpip_count += vpip
        p.pfr_count += pfr
        p.three_bet_count += three_bet
        p.aggression_actions += aggressive
        p.passive_actions += passive
        p.cbet_faced += cbet_faced
        p.fold_to_cbet_count += fold_to_cbet
        self._save_player(name)

    def get_player(self, name: str) -> OpponentStats | None:
        """Get stats for a player, or None if not tracked."""
        return self._players.get(name.strip())

    def get_range_adjustment(self, name: str) -> str:
        """Get strategy advice based on opponent tendencies."""
        p = self.get_player(name)
        if not p or p.hands_seen < 5:
            return "Not enough data to adjust."

        advice: list[str] = []
        if p.vpip_pct > 40:
            advice.append("Very loose — widen your value betting range, reduce bluffs")
        elif p.vpip_pct > 30:
            advice.append("Loose — value bet thinner, bluff less")
        elif p.vpip_pct < 20:
            advice.append("Very tight — respect their raises, fold marginal hands")

        if p.pfr_pct < 10 and p.vpip_pct > 25:
            advice.append("Calling station — never bluff, value bet relentlessly")
        if p.aggression_factor > 3.0:
            advice.append("Hyper-aggressive — trap with strong hands, call down lighter")
        elif p.aggression_factor < 1.0 and p.hands_seen >= 10:
            advice.append("Passive — their bets/raises mean strength, fold medium hands")

        if p.three_bet_pct > 12:
            advice.append("High 3-bet% — consider 4-betting light or flatting traps")
        elif p.three_bet_pct < 4 and p.hands_seen >= 10:
            advice.append("Low 3-bet% — their 3-bets are premium only, fold wide")

        return " | ".join(advice) if advice else "Plays standard — no major adjustments."

    def list_players(self) -> list[str]:
        """Return all tracked player names."""
        return sorted(self._players.keys())

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    # Keep save() as a no-op alias for backward compatibility
    def save(self) -> None:
        """No-op — writes happen automatically on each update."""
