"""Card and Deck classes for poker."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import total_ordering

from poker_bot.utils.constants import RANK_VALUES, Rank, Suit


@total_ordering
@dataclass(frozen=True)
class Card:
    """Represents a single playing card."""

    rank: Rank
    suit: Suit

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Create a Card from a 2-character string like 'Ah' or 'Td'.

        Args:
            s: A 2-character string where the first char is the rank
               and the second is the suit.

        Returns:
            A new Card instance.

        Raises:
            ValueError: If the string is not exactly 2 characters or
                       contains invalid rank/suit characters.
        """
        if len(s) != 2:
            raise ValueError(f"Card string must be 2 characters, got '{s}'")
        try:
            rank = Rank(s[0])
        except ValueError:
            raise ValueError(f"Invalid rank character: '{s[0]}'")
        try:
            suit = Suit(s[1])
        except ValueError:
            raise ValueError(f"Invalid suit character: '{s[1]}'")
        return cls(rank=rank, suit=suit)

    @property
    def value(self) -> int:
        """Numeric value of the card's rank (2-14)."""
        return RANK_VALUES[self.rank]

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"

    def __repr__(self) -> str:
        return f"Card('{self}')"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.value < other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


class Deck:
    """Standard 52-card deck with shuffle and deal operations."""

    def __init__(self) -> None:
        self._cards: list[Card] = []
        self._dealt: list[Card] = []
        self.reset()

    def reset(self) -> None:
        """Reset and shuffle the deck."""
        self._cards = [
            Card(rank=rank, suit=suit) for suit in Suit for rank in Rank
        ]
        self._dealt = []
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the remaining cards in the deck."""
        random.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        """Deal n cards from the top of the deck.

        Args:
            n: Number of cards to deal.

        Returns:
            List of dealt cards.

        Raises:
            ValueError: If not enough cards remain.
        """
        if n > len(self._cards):
            raise ValueError(
                f"Cannot deal {n} cards, only {len(self._cards)} remaining"
            )
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        self._dealt.extend(dealt)
        return dealt

    def deal_one(self) -> Card:
        """Deal a single card from the top of the deck."""
        return self.deal(1)[0]

    @property
    def remaining(self) -> int:
        """Number of cards remaining in the deck."""
        return len(self._cards)

    def remove(self, cards: list[Card]) -> None:
        """Remove specific cards from the deck (for setting up known boards).

        Args:
            cards: Cards to remove from the deck.

        Raises:
            ValueError: If a card is not in the deck.
        """
        for card in cards:
            if card not in self._cards:
                raise ValueError(f"Card {card} not in deck")
            self._cards.remove(card)
            self._dealt.append(card)
