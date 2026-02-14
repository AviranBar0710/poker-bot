"""Game state tracking for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass, field

from poker_bot.utils.card import Card, Deck
from poker_bot.utils.constants import Position, Street


@dataclass
class PlayerState:
    """State of a single player at the table."""

    name: str
    chips: float
    position: Position
    hole_cards: list[Card] = field(default_factory=list)
    is_active: bool = True
    current_bet: float = 0.0
    is_all_in: bool = False

    def reset_for_hand(self) -> None:
        """Reset player state for a new hand."""
        self.hole_cards = []
        self.is_active = self.chips > 0
        self.current_bet = 0.0
        self.is_all_in = False


@dataclass
class GameState:
    """Complete state of a Texas Hold'em hand."""

    players: list[PlayerState]
    small_blind: float
    big_blind: float
    deck: Deck = field(default_factory=Deck)
    community_cards: list[Card] = field(default_factory=list)
    pot: float = 0.0
    current_street: Street = Street.PREFLOP
    current_bet: float = 0.0
    dealer_position: int = 0

    def deal_hole_cards(self) -> None:
        """Deal 2 hole cards to each active player."""
        for player in self.players:
            if player.is_active:
                player.hole_cards = self.deck.deal(2)

    def deal_community_cards(self, n: int) -> list[Card]:
        """Deal n community cards (flop=3, turn=1, river=1).

        Args:
            n: Number of community cards to deal.

        Returns:
            The newly dealt community cards.
        """
        cards = self.deck.deal(n)
        self.community_cards.extend(cards)
        return cards

    def next_street(self) -> Street:
        """Advance to the next street and deal community cards.

        Returns:
            The new street.

        Raises:
            ValueError: If already on the river.
        """
        match self.current_street:
            case Street.PREFLOP:
                self.current_street = Street.FLOP
                self.deal_community_cards(3)
            case Street.FLOP:
                self.current_street = Street.TURN
                self.deal_community_cards(1)
            case Street.TURN:
                self.current_street = Street.RIVER
                self.deal_community_cards(1)
            case Street.RIVER:
                raise ValueError("Cannot advance past the river")

        self.current_bet = 0.0
        for player in self.players:
            player.current_bet = 0.0

        return self.current_street

    def reset(self) -> None:
        """Reset the game state for a new hand."""
        self.deck.reset()
        self.community_cards = []
        self.pot = 0.0
        self.current_street = Street.PREFLOP
        self.current_bet = 0.0
        for player in self.players:
            player.reset_for_hand()

    @property
    def active_players(self) -> list[PlayerState]:
        """Return list of players still active in the hand."""
        return [p for p in self.players if p.is_active]

    @property
    def players_in_hand(self) -> int:
        """Number of players still in the hand."""
        return len(self.active_players)
