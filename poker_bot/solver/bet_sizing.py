"""Geometric bet sizing trees for GTO play.

Provides standard sizing recommendations by street and board texture,
plus multi-street geometric sizing for planning bet-fold or bet-bet-shove
lines across streets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizingOption:
    """A bet sizing option with fraction of pot and description."""

    fraction: float  # As fraction of pot (0.33 = 1/3 pot)
    label: str


# Standard sizing options by board texture category
TEXTURE_SIZINGS: dict[str, list[SizingOption]] = {
    "dry_high_rainbow": [
        SizingOption(0.33, "1/3 pot (dry board c-bet)"),
        SizingOption(0.25, "1/4 pot (probe bet)"),
    ],
    "dry_low_rainbow": [
        SizingOption(0.33, "1/3 pot (dry board c-bet)"),
        SizingOption(0.25, "1/4 pot (probe bet)"),
    ],
    "dry_medium": [
        SizingOption(0.33, "1/3 pot (dry board)"),
        SizingOption(0.50, "1/2 pot (standard)"),
    ],
    "wet_connected": [
        SizingOption(0.66, "2/3 pot (protect vs draws)"),
        SizingOption(0.75, "3/4 pot (heavy protection)"),
    ],
    "wet_two_tone": [
        SizingOption(0.66, "2/3 pot (protect vs flush draw)"),
        SizingOption(0.50, "1/2 pot (standard)"),
    ],
    "monotone_high": [
        SizingOption(0.75, "3/4 pot (polarized on monotone)"),
        SizingOption(0.33, "1/3 pot (block bet)"),
    ],
    "monotone_low": [
        SizingOption(0.75, "3/4 pot (polarized on monotone)"),
        SizingOption(0.33, "1/3 pot (block bet)"),
    ],
    "paired_high": [
        SizingOption(0.33, "1/3 pot (dry paired board)"),
        SizingOption(0.50, "1/2 pot (standard)"),
    ],
    "paired_low": [
        SizingOption(0.33, "1/3 pot (dry paired board)"),
        SizingOption(0.25, "1/4 pot (probe)"),
    ],
    "broadway_heavy": [
        SizingOption(0.50, "1/2 pot (broadway texture)"),
        SizingOption(0.66, "2/3 pot (protection)"),
    ],
    "connected_low": [
        SizingOption(0.66, "2/3 pot (protect vs draws)"),
        SizingOption(0.50, "1/2 pot (standard)"),
    ],
    "dynamic": [
        SizingOption(0.75, "3/4 pot (dynamic board)"),
        SizingOption(1.0, "pot (overbet for polarization)"),
    ],
}

# Default sizing when texture bucket is unknown
_DEFAULT_SIZINGS = [
    SizingOption(0.50, "1/2 pot (default)"),
    SizingOption(0.66, "2/3 pot (default)"),
]


class BetSizingTree:
    """Provides bet sizing recommendations based on board texture and street."""

    @staticmethod
    def get_sizings(board_bucket: str) -> list[SizingOption]:
        """Get recommended sizings for a board texture bucket.

        Args:
            board_bucket: Board texture category string.

        Returns:
            List of SizingOption from most preferred to least.
        """
        return TEXTURE_SIZINGS.get(board_bucket, _DEFAULT_SIZINGS)

    @staticmethod
    def primary_sizing(board_bucket: str) -> float:
        """Get the primary (most common) sizing for a texture.

        Args:
            board_bucket: Board texture category string.

        Returns:
            Bet size as fraction of pot.
        """
        sizings = TEXTURE_SIZINGS.get(board_bucket, _DEFAULT_SIZINGS)
        return sizings[0].fraction if sizings else 0.5

    @staticmethod
    def geometric_sizing(
        pot: float,
        stack: float,
        streets_remaining: int,
    ) -> float:
        """Calculate geometric bet size to get all-in over remaining streets.

        Finds the fraction x such that betting x * pot on each remaining
        street results in being all-in by the river.

        pot * (1 + x)^n = pot + stack
        x = (1 + stack/pot)^(1/n) - 1

        Args:
            pot: Current pot size.
            stack: Remaining stack.
            streets_remaining: Number of streets left to bet (1-3).

        Returns:
            Bet size as fraction of pot per street.
        """
        if pot <= 0 or stack <= 0 or streets_remaining <= 0:
            return 0.0

        ratio = 1.0 + stack / pot
        per_street = ratio ** (1.0 / streets_remaining) - 1.0

        # Clamp to reasonable range [0.2, 2.0]
        return max(0.2, min(2.0, per_street))

    @staticmethod
    def compute_bet_amount(
        pot: float,
        stack: float,
        sizing_fraction: float,
        min_bet: float = 0.0,
    ) -> float:
        """Compute actual bet amount from pot and sizing fraction.

        Args:
            pot: Current pot size.
            stack: Hero's remaining stack.
            sizing_fraction: Bet size as fraction of pot.
            min_bet: Minimum legal bet size.

        Returns:
            Bet amount clamped to [min_bet, stack].
        """
        amount = pot * sizing_fraction
        amount = max(amount, min_bet)
        return min(amount, stack)
