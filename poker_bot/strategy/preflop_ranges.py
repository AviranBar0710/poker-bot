"""GTO pre-flop range system for Texas Hold'em.

Hand notation:
  - "AA"   → pocket pair (all suit combos)
  - "AKs"  → suited (4 combos)
  - "AKo"  → offsuit (12 combos)
  - "AK"   → both suited and offsuit (16 combos)
  - "JJ+"  → JJ, QQ, KK, AA
  - "ATs+" → ATs, AJs, AQs, AKs
  - "A5s-A2s" → A5s, A4s, A3s, A2s
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations

from poker_bot.utils.card import Card
from poker_bot.utils.constants import Position, Rank, Suit

# Ranks ordered high to low for range expansion
_RANKS_DESCENDING: list[Rank] = [
    Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN,
    Rank.NINE, Rank.EIGHT, Rank.SEVEN, Rank.SIX, Rank.FIVE,
    Rank.FOUR, Rank.THREE, Rank.TWO,
]

_RANK_INDEX: dict[Rank, int] = {r: i for i, r in enumerate(_RANKS_DESCENDING)}


class HandType(StrEnum):
    PAIR = "pair"
    SUITED = "suited"
    OFFSUIT = "offsuit"


@dataclass(frozen=True)
class HandCombo:
    """A specific 2-card hand (e.g. AhKh)."""

    card1: Card
    card2: Card

    def __str__(self) -> str:
        return f"{self.card1}{self.card2}"


@dataclass(frozen=True)
class HandNotation:
    """A hand in standard poker notation (e.g. AKs, JJ, T9o)."""

    rank1: Rank
    rank2: Rank
    hand_type: HandType

    @classmethod
    def from_str(cls, s: str) -> HandNotation:
        """Parse notation like 'AKs', 'JJ', 'T9o'.

        Args:
            s: Hand notation string (2-3 characters).

        Returns:
            HandNotation instance.

        Raises:
            ValueError: If notation is invalid.
        """
        if len(s) < 2 or len(s) > 3:
            raise ValueError(f"Invalid hand notation: '{s}'")

        r1 = Rank(s[0])
        r2 = Rank(s[1])

        if r1 == r2:
            return cls(rank1=r1, rank2=r2, hand_type=HandType.PAIR)

        # Ensure rank1 is the higher rank
        if _RANK_INDEX[r1] > _RANK_INDEX[r2]:
            r1, r2 = r2, r1

        if len(s) == 3:
            if s[2] == "s":
                return cls(rank1=r1, rank2=r2, hand_type=HandType.SUITED)
            elif s[2] == "o":
                return cls(rank1=r1, rank2=r2, hand_type=HandType.OFFSUIT)
            else:
                raise ValueError(f"Invalid suit indicator: '{s[2]}'")

        # 2-char non-pair defaults to both suited + offsuit (handled at range level)
        # Here we treat it as offsuit; callers should add both
        return cls(rank1=r1, rank2=r2, hand_type=HandType.OFFSUIT)

    def to_combos(self) -> list[HandCombo]:
        """Expand this notation into all specific card combinations."""
        suits = list(Suit)

        if self.hand_type == HandType.PAIR:
            return [
                HandCombo(Card(self.rank1, s1), Card(self.rank2, s2))
                for s1, s2 in combinations(suits, 2)
            ]

        if self.hand_type == HandType.SUITED:
            return [
                HandCombo(Card(self.rank1, s), Card(self.rank2, s))
                for s in suits
            ]

        # Offsuit
        return [
            HandCombo(Card(self.rank1, s1), Card(self.rank2, s2))
            for s1 in suits
            for s2 in suits
            if s1 != s2
        ]

    @property
    def combo_count(self) -> int:
        if self.hand_type == HandType.PAIR:
            return 6
        if self.hand_type == HandType.SUITED:
            return 4
        return 12

    def __str__(self) -> str:
        r = f"{self.rank1.value}{self.rank2.value}"
        if self.hand_type == HandType.PAIR:
            return r
        if self.hand_type == HandType.SUITED:
            return r + "s"
        return r + "o"


def expand_notation(notation: str) -> list[HandNotation]:
    """Expand range notation into a list of HandNotation objects.

    Supports:
      - Single hands: "AKs", "JJ", "T9o"
      - Plus notation: "JJ+" → JJ,QQ,KK,AA
      - Plus on non-pairs: "ATs+" → ATs,AJs,AQs,AKs
      - Dash ranges: "JJ-88" → JJ,TT,99,88
      - Dash on non-pairs: "A5s-A2s" → A5s,A4s,A3s,A2s

    Args:
        notation: Range notation string.

    Returns:
        List of HandNotation objects.
    """
    notation = notation.strip()

    # Dash range: "JJ-88" or "A5s-A2s"
    if "-" in notation:
        parts = notation.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range notation: '{notation}'")
        return _expand_dash_range(parts[0].strip(), parts[1].strip())

    # Plus notation: "JJ+" or "ATs+"
    if notation.endswith("+"):
        return _expand_plus(notation[:-1])

    # Single hand
    hand = notation
    r1 = Rank(hand[0])
    r2 = Rank(hand[1])

    if len(hand) == 2 and r1 != r2:
        # "AK" means both AKs and AKo
        if _RANK_INDEX[r1] > _RANK_INDEX[r2]:
            r1, r2 = r2, r1
        return [
            HandNotation(r1, r2, HandType.SUITED),
            HandNotation(r1, r2, HandType.OFFSUIT),
        ]

    return [HandNotation.from_str(hand)]


def _expand_plus(base: str) -> list[HandNotation]:
    """Expand 'JJ+' or 'ATs+' style notation."""
    hand = HandNotation.from_str(base)

    if hand.hand_type == HandType.PAIR:
        # JJ+ → JJ, QQ, KK, AA
        idx = _RANK_INDEX[hand.rank1]
        return [
            HandNotation(_RANKS_DESCENDING[i], _RANKS_DESCENDING[i], HandType.PAIR)
            for i in range(idx + 1)  # from AA down to the base pair
        ]

    # ATs+ → ATs, AJs, AQs, AKs (gap narrows toward rank1)
    high = hand.rank1
    low_idx = _RANK_INDEX[hand.rank2]
    high_idx = _RANK_INDEX[high]
    results = []
    for i in range(high_idx + 1, low_idx + 1):
        results.append(HandNotation(high, _RANKS_DESCENDING[i], hand.hand_type))
    return results


def _expand_dash_range(start: str, end: str) -> list[HandNotation]:
    """Expand 'JJ-88' or 'A5s-A2s' style notation."""
    h_start = HandNotation.from_str(start)
    h_end = HandNotation.from_str(end)

    if h_start.hand_type == HandType.PAIR and h_end.hand_type == HandType.PAIR:
        idx_start = _RANK_INDEX[h_start.rank1]
        idx_end = _RANK_INDEX[h_end.rank1]
        lo, hi = sorted([idx_start, idx_end])
        return [
            HandNotation(_RANKS_DESCENDING[i], _RANKS_DESCENDING[i], HandType.PAIR)
            for i in range(lo, hi + 1)
        ]

    # Non-pair range: same high card, varying low card
    if h_start.rank1 != h_end.rank1:
        raise ValueError(
            f"Non-pair dash ranges must share the high card: '{start}-{end}'"
        )
    if h_start.hand_type != h_end.hand_type:
        raise ValueError(
            f"Dash range endpoints must have same type (s/o): '{start}-{end}'"
        )

    idx_start = _RANK_INDEX[h_start.rank2]
    idx_end = _RANK_INDEX[h_end.rank2]
    lo, hi = sorted([idx_start, idx_end])
    return [
        HandNotation(h_start.rank1, _RANKS_DESCENDING[i], h_start.hand_type)
        for i in range(lo, hi + 1)
    ]


@dataclass
class Range:
    """A collection of hands representing a player's range.

    Hands are stored as a set of HandNotation objects. Use add() with
    standard poker notation strings.
    """

    hands: set[HandNotation] = field(default_factory=set)

    def add(self, notation: str) -> Range:
        """Add hands using range notation. Returns self for chaining.

        Args:
            notation: Comma-separated range notation.
                Examples: "AA,KK", "ATs+", "JJ-88", "AKs,AQs,KQs"
        """
        for part in notation.split(","):
            part = part.strip()
            if part:
                for hand in expand_notation(part):
                    self.hands.add(hand)
        return self

    def remove(self, notation: str) -> Range:
        """Remove hands using range notation. Returns self for chaining."""
        for part in notation.split(","):
            part = part.strip()
            if part:
                for hand in expand_notation(part):
                    self.hands.discard(hand)
        return self

    def to_combos(self) -> list[HandCombo]:
        """Expand the range into all specific card combinations."""
        combos: list[HandCombo] = []
        for hand in self.hands:
            combos.extend(hand.to_combos())
        return combos

    @property
    def combo_count(self) -> int:
        """Total number of specific card combinations in this range."""
        return sum(h.combo_count for h in self.hands)

    @property
    def percentage(self) -> float:
        """Percentage of all possible starting hands (1326 total combos)."""
        return (self.combo_count / 1326) * 100

    def contains(self, card1: Card, card2: Card) -> bool:
        """Check if a specific hand is within this range."""
        for hand in self.hands:
            for combo in hand.to_combos():
                if (
                    (combo.card1 == card1 and combo.card2 == card2)
                    or (combo.card1 == card2 and combo.card2 == card1)
                ):
                    return True
        return False

    def __contains__(self, item: HandNotation) -> bool:
        return item in self.hands

    def __len__(self) -> int:
        return len(self.hands)

    def __str__(self) -> str:
        return ", ".join(sorted(str(h) for h in self.hands))


# ---------------------------------------------------------------------------
# GTO Pre-Flop Ranges
#
# These are approximate GTO ranges for 6-max No-Limit Hold'em (100bb deep).
# Ranges are conservative estimates based on solver outputs.
# ---------------------------------------------------------------------------


def _build_range(notation: str) -> Range:
    """Build a Range from a comma-separated notation string."""
    return Range().add(notation)


# === OPENING RANGES (RFI — Raise First In) ===

UTG_OPEN = _build_range(
    "AA,KK,QQ,JJ,TT,99,88,77,"
    "AKs,AQs,AJs,ATs,A5s,A4s,"
    "AKo,AQo,AJo,"
    "KQs,KJs,KTs,"
    "KQo,"
    "QJs,QTs,"
    "JTs,"
    "T9s,"
    "98s,"
    "87s"
)

MP_OPEN = _build_range(
    "AA,KK,QQ,JJ,TT,99,88,77,66,"
    "AKs,AQs,AJs,ATs,A9s,A5s,A4s,A3s,"
    "AKo,AQo,AJo,ATo,"
    "KQs,KJs,KTs,K9s,"
    "KQo,"
    "QJs,QTs,Q9s,"
    "JTs,J9s,"
    "T9s,T8s,"
    "98s,"
    "87s,"
    "76s"
)

CO_OPEN = _build_range(
    "AA,KK,QQ,JJ,TT,99,88,77,66,55,"
    "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
    "AKo,AQo,AJo,ATo,A9o,"
    "KQs,KJs,KTs,K9s,K8s,"
    "KQo,KJo,KTo,"
    "QJs,QTs,Q9s,Q8s,"
    "QJo,"
    "JTs,J9s,J8s,"
    "JTo,"
    "T9s,T8s,"
    "98s,97s,"
    "87s,86s,"
    "76s,75s,"
    "65s,"
    "54s"
)

BTN_OPEN = _build_range(
    "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
    "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
    "AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,A3o,A2o,"
    "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,"
    "KQo,KJo,KTo,K9o,"
    "QJs,QTs,Q9s,Q8s,Q7s,Q6s,"
    "QJo,QTo,Q9o,"
    "JTs,J9s,J8s,J7s,"
    "JTo,J9o,"
    "T9s,T8s,T7s,"
    "T9o,"
    "98s,97s,96s,"
    "87s,86s,"
    "76s,75s,"
    "65s,64s,"
    "54s,53s,"
    "43s"
)

SB_OPEN = _build_range(
    "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
    "AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
    "AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,"
    "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,"
    "KQo,KJo,KTo,K9o,K8o,"
    "QJs,QTs,Q9s,Q8s,Q7s,Q6s,Q5s,"
    "QJo,QTo,Q9o,"
    "JTs,J9s,J8s,J7s,J6s,"
    "JTo,J9o,"
    "T9s,T8s,T7s,T6s,"
    "T9o,"
    "98s,97s,96s,"
    "98o,"
    "87s,86s,85s,"
    "76s,75s,"
    "65s,64s,"
    "54s,53s,"
    "43s"
)

# BB has no open range (last to act preflop, already has money in)


# === 3-BET RANGES (vs open raise) ===

UTG_3BET = _build_range(
    "AA,KK,QQ,"
    "AKs,"
    "AKo"
)

MP_3BET = _build_range(
    "AA,KK,QQ,JJ,"
    "AKs,AQs,"
    "AKo,"
    "A5s,A4s"  # Bluff 3-bets
)

CO_3BET = _build_range(
    "AA,KK,QQ,JJ,TT,"
    "AKs,AQs,AJs,"
    "AKo,AQo,"
    "A5s,A4s,"  # Bluff 3-bets
    "KQs"
)

BTN_3BET = _build_range(
    "AA,KK,QQ,JJ,TT,"
    "AKs,AQs,AJs,ATs,"
    "AKo,AQo,"
    "A5s,A4s,A3s,"  # Bluff 3-bets
    "KQs,KJs,"
    "QJs,"
    "76s,65s,54s"  # Suited connector bluffs
)

SB_3BET = _build_range(
    "AA,KK,QQ,JJ,TT,99,"
    "AKs,AQs,AJs,ATs,A9s,"
    "AKo,AQo,AJo,"
    "A5s,A4s,A3s,A2s,"  # Bluff 3-bets
    "KQs,KJs,KTs,"
    "QJs,QTs,"
    "JTs,"
    "T9s,"
    "98s,"
    "87s,76s,65s,54s"  # Suited connector bluffs
)

BB_3BET = _build_range(
    "AA,KK,QQ,JJ,TT,"
    "AKs,AQs,AJs,"
    "AKo,AQo,"
    "A5s,A4s,A3s,"  # Bluff 3-bets
    "KQs,"
    "76s,65s,54s"  # Suited connector bluffs
)


# === 4-BET RANGES (vs 3-bet) ===

UTG_4BET = _build_range(
    "AA,KK,"
    "AKs,"
    "A5s"  # Bluff
)

MP_4BET = _build_range(
    "AA,KK,QQ,"
    "AKs,"
    "AKo,"
    "A5s,A4s"  # Bluffs
)

CO_4BET = _build_range(
    "AA,KK,QQ,"
    "AKs,AQs,"
    "AKo,"
    "A5s,A4s"  # Bluffs
)

BTN_4BET = _build_range(
    "AA,KK,QQ,"
    "AKs,AQs,"
    "AKo,"
    "A5s,A4s,A3s"  # Bluffs
)

SB_4BET = _build_range(
    "AA,KK,QQ,JJ,"
    "AKs,AQs,"
    "AKo,"
    "A5s,A4s"  # Bluffs
)

BB_4BET = _build_range(
    "AA,KK,QQ,"
    "AKs,"
    "AKo,"
    "A5s,A4s"  # Bluffs
)


# === CALLING RANGES (vs open raise — cold call or BB defend) ===

# Cold-call ranges (calling an open raise without initiative)
# UTG typically does not cold-call (3-bet or fold)

MP_CALL_VS_RAISE = _build_range(
    "TT,99,88,77,66,"
    "AQs,AJs,ATs,A9s,"
    "AQo,"
    "KQs,KJs,"
    "QJs,QTs,"
    "JTs,"
    "T9s,"
    "98s,"
    "87s"
)

CO_CALL_VS_RAISE = _build_range(
    "TT,99,88,77,66,55,"
    "AQs,AJs,ATs,A9s,A8s,"
    "AQo,AJo,"
    "KQs,KJs,KTs,"
    "KQo,"
    "QJs,QTs,Q9s,"
    "JTs,J9s,"
    "T9s,T8s,"
    "98s,97s,"
    "87s,86s,"
    "76s,"
    "65s"
)

BTN_CALL_VS_RAISE = _build_range(
    "TT,99,88,77,66,55,44,"
    "AQs,AJs,ATs,A9s,A8s,A7s,A6s,"
    "AQo,AJo,ATo,"
    "KQs,KJs,KTs,K9s,"
    "KQo,KJo,"
    "QJs,QTs,Q9s,"
    "QJo,"
    "JTs,J9s,J8s,"
    "JTo,"
    "T9s,T8s,"
    "T9o,"
    "98s,97s,"
    "87s,86s,"
    "76s,75s,"
    "65s,64s,"
    "54s"
)

# SB typically 3-bets or folds (no cold-call in GTO from SB)

BB_CALL_VS_RAISE = _build_range(
    "TT,99,88,77,66,55,44,33,22,"
    "AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,"
    "AJo,ATo,A9o,A8o,A7o,A6o,A5o,"
    "KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,"
    "KQo,KJo,KTo,K9o,"
    "QJs,QTs,Q9s,Q8s,Q7s,"
    "QJo,QTo,Q9o,"
    "JTs,J9s,J8s,J7s,"
    "JTo,J9o,"
    "T9s,T8s,T7s,"
    "T9o,T8o,"
    "98s,97s,96s,"
    "98o,"
    "87s,86s,85s,"
    "87o,"
    "76s,75s,"
    "65s,64s,"
    "54s,53s,"
    "43s"
)


# === Lookup Tables ===

OPENING_RANGES: dict[Position, Range] = {
    Position.UTG: UTG_OPEN,
    Position.MP: MP_OPEN,
    Position.CO: CO_OPEN,
    Position.BTN: BTN_OPEN,
    Position.SB: SB_OPEN,
}

THREE_BET_RANGES: dict[Position, Range] = {
    Position.UTG: UTG_3BET,
    Position.MP: MP_3BET,
    Position.CO: CO_3BET,
    Position.BTN: BTN_3BET,
    Position.SB: SB_3BET,
    Position.BB: BB_3BET,
}

FOUR_BET_RANGES: dict[Position, Range] = {
    Position.UTG: UTG_4BET,
    Position.MP: MP_4BET,
    Position.CO: CO_4BET,
    Position.BTN: BTN_4BET,
    Position.SB: SB_4BET,
    Position.BB: BB_4BET,
}

CALL_VS_RAISE_RANGES: dict[Position, Range] = {
    Position.MP: MP_CALL_VS_RAISE,
    Position.CO: CO_CALL_VS_RAISE,
    Position.BTN: BTN_CALL_VS_RAISE,
    Position.BB: BB_CALL_VS_RAISE,
}


def get_opening_range(position: Position) -> Range:
    """Get the GTO opening range for a position.

    Args:
        position: Player's table position.

    Returns:
        The opening range, or an empty Range if no open range exists.
    """
    return OPENING_RANGES.get(position, Range())


def get_3bet_range(position: Position) -> Range:
    """Get the GTO 3-bet range for a position."""
    return THREE_BET_RANGES.get(position, Range())


def get_4bet_range(position: Position) -> Range:
    """Get the GTO 4-bet range for a position."""
    return FOUR_BET_RANGES.get(position, Range())


def get_call_vs_raise_range(position: Position) -> Range:
    """Get the calling range vs a raise for a position."""
    return CALL_VS_RAISE_RANGES.get(position, Range())
