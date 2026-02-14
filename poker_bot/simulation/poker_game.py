"""Single-hand poker game engine for simulation.

Orchestrates dealing, betting rounds, and showdown using the
DecisionMaker for the bot and random actions for villains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from poker_bot.core.game_context import GameContext
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.core.hand_evaluator import HandEvaluator, HandResult
from poker_bot.strategy.decision_maker import (
    ActionType,
    DecisionMaker,
    PriorAction,
)
from poker_bot.utils.card import Deck
from poker_bot.utils.constants import Action, HandRanking, Position, Street


# ---------------------------------------------------------------------------
# Position ordering
# ---------------------------------------------------------------------------

_PREFLOP_ORDER = [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB, Position.BB]
_POSTFLOP_ORDER = [Position.SB, Position.BB, Position.UTG, Position.MP, Position.CO, Position.BTN]

# 6-max seat assignment based on dealer index
_SEATS_FROM_DEALER: list[Position] = [
    Position.BTN, Position.SB, Position.BB, Position.UTG, Position.MP, Position.CO,
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class HandRecord:
    """Record of a single played hand."""

    hand_number: int
    winner_name: str
    pot_size: float
    community_cards: list[str]
    player_hands: dict[str, list[str]]
    winning_hand_ranking: HandRanking | None
    actions_summary: list[str]
    hero_profit: float = 0.0


# ---------------------------------------------------------------------------
# Poker game engine
# ---------------------------------------------------------------------------


class PokerGame:
    """Single-hand game engine that orchestrates a full hand of poker."""

    def __init__(
        self,
        players: list[PlayerState],
        small_blind: float,
        big_blind: float,
    ) -> None:
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.decision_maker = DecisionMaker()
        self.dealer_index = 0
        self.hand_number = 0

    def play_hand(self) -> HandRecord:
        """Play a complete hand and return the record."""
        self.hand_number += 1

        # Assign positions based on dealer
        self._assign_positions()

        # Build game state
        gs = GameState(
            players=self.players,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            dealer_position=self.dealer_index,
        )
        gs.reset()

        # Deal hole cards
        gs.deal_hole_cards()

        # Track starting chips for profit calc
        starting_chips = {p.name: p.chips for p in self.players}

        actions_summary: list[str] = []
        action_history: list[PriorAction] = []

        # Post blinds
        sb_player = self._player_at_position(Position.SB)
        bb_player = self._player_at_position(Position.BB)

        sb_amount = min(self.small_blind, sb_player.chips)
        bb_amount = min(self.big_blind, bb_player.chips)

        sb_player.chips -= sb_amount
        sb_player.current_bet = sb_amount
        bb_player.chips -= bb_amount
        bb_player.current_bet = bb_amount
        gs.pot = sb_amount + bb_amount
        gs.current_bet = bb_amount

        actions_summary.append(f"{sb_player.name} ({Position.SB}) posts SB {sb_amount}")
        actions_summary.append(f"{bb_player.name} ({Position.BB}) posts BB {bb_amount}")

        # Run betting rounds
        for street in [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]:
            if gs.current_street != street:
                if len([p for p in gs.players if p.is_active and not p.is_all_in]) <= 1:
                    # All but one folded or all-in — deal remaining boards
                    break
                gs.next_street()
                actions_summary.append(f"--- {street} --- Board: {' '.join(str(c) for c in gs.community_cards)}")

            if street == Street.PREFLOP:
                actions_summary.append(f"--- PREFLOP ---")

            # Only run betting if 2+ players can still act
            active_non_allin = [p for p in gs.players if p.is_active and not p.is_all_in]
            if len(active_non_allin) < 2:
                continue

            action_history_street: list[PriorAction] = []
            finished = self._run_betting_round(
                gs, street, action_history_street, actions_summary,
            )
            action_history.extend(action_history_street)

            if len([p for p in gs.players if p.is_active]) <= 1:
                break

        # Deal remaining community cards if needed (all-in before board complete)
        active_count = len([p for p in gs.players if p.is_active])
        if active_count > 1:
            while len(gs.community_cards) < 5:
                if gs.current_street == Street.PREFLOP:
                    gs.next_street()  # Deals flop (3 cards)
                elif gs.current_street == Street.FLOP:
                    gs.next_street()  # Deals turn (1 card)
                elif gs.current_street == Street.TURN:
                    gs.next_street()  # Deals river (1 card)
                else:
                    break

        # Showdown or last-man-standing
        active = [p for p in gs.players if p.is_active]
        winner_name = ""
        winning_ranking: HandRanking | None = None
        player_hands: dict[str, list[str]] = {}

        for p in gs.players:
            if p.hole_cards:
                player_hands[p.name] = [str(c) for c in p.hole_cards]

        community_strs = [str(c) for c in gs.community_cards]

        if len(active) == 1:
            winner = active[0]
            winner_name = winner.name
            winner.chips += gs.pot
            actions_summary.append(f"{winner.name} wins pot of {gs.pot:.0f} (everyone else folded)")
        else:
            # Showdown — evaluate hands
            best_result: HandResult | None = None
            winner = active[0]
            for p in active:
                if len(p.hole_cards) >= 2 and len(gs.community_cards) >= 3:
                    result = HandEvaluator.evaluate(p.hole_cards + gs.community_cards)
                    if best_result is None or result > best_result:
                        best_result = result
                        winner = p
                        winning_ranking = result.ranking

            winner_name = winner.name
            winner.chips += gs.pot
            hand_desc = winning_ranking.name if winning_ranking else "unknown"
            actions_summary.append(
                f"Showdown: {winner.name} wins {gs.pot:.0f} with {hand_desc}"
            )

        hero_profit = self.players[0].chips - starting_chips[self.players[0].name]

        # Rotate dealer
        self.dealer_index = (self.dealer_index + 1) % len(self.players)

        return HandRecord(
            hand_number=self.hand_number,
            winner_name=winner_name,
            pot_size=gs.pot,
            community_cards=community_strs,
            player_hands=player_hands,
            winning_hand_ranking=winning_ranking,
            actions_summary=actions_summary,
            hero_profit=hero_profit,
        )

    def _assign_positions(self) -> None:
        """Assign positions to players based on dealer index."""
        n = len(self.players)
        for i in range(n):
            seat = (i - self.dealer_index) % n
            self.players[i].position = _SEATS_FROM_DEALER[seat % len(_SEATS_FROM_DEALER)]

    def _player_at_position(self, pos: Position) -> PlayerState:
        """Find the player at a given position."""
        for p in self.players:
            if p.position == pos:
                return p
        raise ValueError(f"No player at position {pos}")

    def _run_betting_round(
        self,
        gs: GameState,
        street: Street,
        action_history: list[PriorAction],
        actions_summary: list[str],
    ) -> bool:
        """Run a single betting round. Returns True if hand should end."""
        order = _PREFLOP_ORDER if street == Street.PREFLOP else _POSTFLOP_ORDER

        # Build ordered list of players for this round
        ordered_players = []
        for pos in order:
            for p in gs.players:
                if p.position == pos and p.is_active and not p.is_all_in:
                    ordered_players.append(p)

        if len(ordered_players) < 2:
            return False

        # Track who has acted and the last raiser
        last_raiser_name: str | None = None
        players_to_act = list(ordered_players)
        acted: set[str] = set()

        while players_to_act:
            player = players_to_act.pop(0)

            if not player.is_active or player.is_all_in:
                continue

            # If player already acted and no new raise to respond to, skip
            if player.name in acted and (last_raiser_name is None or last_raiser_name == player.name):
                continue

            is_hero = (player == self.players[0])
            effective_bet = max(0.0, gs.current_bet - player.current_bet)

            if is_hero:
                action_type, amount = self._hero_action(
                    gs, player, action_history
                )
            else:
                action_type, amount = self._villain_action(
                    gs, player, effective_bet
                )

            # Apply action
            self._apply_action(gs, player, action_type, amount)
            acted.add(player.name)

            action_str = Action(action_type.value)
            action_history.append(PriorAction(player.position, action_str, amount))
            actions_summary.append(
                f"  {player.name} ({player.position}): {action_type.value}"
                + (f" {amount:.0f}" if amount > 0 else "")
            )

            # If a raise happened, everyone else needs to act again
            if action_type in (ActionType.RAISE, ActionType.ALL_IN) and amount > effective_bet:
                last_raiser_name = player.name
                # Re-add other active non-allin players who haven't responded to this raise
                for p in ordered_players:
                    if p != player and p.is_active and not p.is_all_in and p not in players_to_act:
                        players_to_act.append(p)

            # Check if only 1 active player remains
            if len([p for p in gs.players if p.is_active]) <= 1:
                return True

        return False

    def _hero_action(
        self,
        gs: GameState,
        hero: PlayerState,
        action_history: list[PriorAction],
    ) -> tuple[ActionType, float]:
        """Get the bot's decision via DecisionMaker."""
        hero_index = self.players.index(hero)
        stack_bb = hero.chips / self.big_blind
        ctx = GameContext.cash_game(stack_bb=stack_bb, num_players=len([p for p in gs.players if p.is_active]))

        decision = self.decision_maker.make_decision(
            gs, ctx, hero_index=hero_index, action_history=action_history,
        )
        return decision.action, decision.amount

    def _villain_action(
        self,
        gs: GameState,
        villain: PlayerState,
        effective_bet: float,
    ) -> tuple[ActionType, float]:
        """Simple random villain strategy."""
        can_check = effective_bet == 0

        if can_check:
            # Never fold when check is free
            roll = random.random()
            if roll < 0.70:
                return ActionType.CHECK, 0.0
            else:
                # Bet ~50% pot
                bet_size = max(gs.pot * 0.5, self.big_blind * 2)
                bet_size = min(bet_size, villain.chips)
                if bet_size >= villain.chips:
                    return ActionType.ALL_IN, villain.chips
                return ActionType.RAISE, bet_size
        else:
            roll = random.random()
            if roll < 0.30:
                return ActionType.FOLD, 0.0
            elif roll < 0.80:
                call_amount = min(effective_bet, villain.chips)
                if call_amount >= villain.chips:
                    return ActionType.ALL_IN, villain.chips
                return ActionType.CALL, call_amount
            else:
                # Raise 2-3x the current bet
                raise_mult = random.uniform(2.0, 3.0)
                raise_to = gs.current_bet * raise_mult
                raise_to = max(raise_to, self.big_blind * 2)
                raise_to = min(raise_to, villain.chips)
                if raise_to >= villain.chips:
                    return ActionType.ALL_IN, villain.chips
                return ActionType.RAISE, raise_to

    def _apply_action(
        self,
        gs: GameState,
        player: PlayerState,
        action_type: ActionType,
        amount: float,
    ) -> None:
        """Apply an action to the game state."""
        match action_type:
            case ActionType.FOLD:
                player.is_active = False

            case ActionType.CHECK:
                pass

            case ActionType.CALL:
                call_amount = min(max(0.0, gs.current_bet - player.current_bet), player.chips)
                player.chips -= call_amount
                player.current_bet += call_amount
                gs.pot += call_amount
                if player.chips <= 0:
                    player.is_all_in = True

            case ActionType.RAISE:
                # amount is raise-to
                raise_diff = max(0.0, amount - player.current_bet)
                raise_diff = min(raise_diff, player.chips)
                player.chips -= raise_diff
                player.current_bet += raise_diff
                gs.pot += raise_diff
                gs.current_bet = player.current_bet
                if player.chips <= 0:
                    player.is_all_in = True

            case ActionType.ALL_IN:
                allin_amount = player.chips
                player.current_bet += allin_amount
                gs.pot += allin_amount
                player.chips = 0.0
                player.is_all_in = True
                if player.current_bet > gs.current_bet:
                    gs.current_bet = player.current_bet
