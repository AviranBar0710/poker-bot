"""Persistent opponent notes and statistics tracker.

Stores opponent data in ~/.poker_coach/opponents.json so it
survives across sessions. Provides tendencies-based advice for
adjusting play against known opponents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


_DATA_DIR = Path.home() / ".poker_coach"
_DATA_FILE = _DATA_DIR / "opponents.json"


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
    """Manages persistent opponent notes and statistics."""

    def __init__(self) -> None:
        self._players: dict[str, OpponentStats] = {}
        self._load()

    def add_note(self, name: str, note: str) -> None:
        """Add a text note for a player."""
        name = name.strip()
        if name not in self._players:
            self._players[name] = OpponentStats(name=name)
        self._players[name].notes.append(note.strip())
        self.save()

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
        self.save()

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

    def save(self) -> None:
        """Persist data to disk."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(stats) for name, stats in self._players.items()}
        _DATA_FILE.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load data from disk if it exists."""
        if not _DATA_FILE.exists():
            return
        try:
            raw = json.loads(_DATA_FILE.read_text())
            for name, d in raw.items():
                self._players[name] = OpponentStats(**d)
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Start fresh on corrupt data
