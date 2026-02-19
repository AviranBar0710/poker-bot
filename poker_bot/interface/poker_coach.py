"""Interactive real-time poker coaching tool.

CLI-based personal poker coach that provides instant GTO-based
recommendations, hand review, and opponent tracking.

Usage:
    python -m poker_bot.interface.poker_coach

Example session:
    ══════════════════════════════════════════
      POKER COACH — Main Menu
    ══════════════════════════════════════════
      1. Live coaching
      2. Hand review
      3. Opponent notes
      4. Quit
    > 1

    Game type (cash/tournament): cash
    Your hand (e.g. AhKs): AhKs
    Position (UTG/MP/CO/BTN/SB/BB): BTN
    Stack in bb: 50
    Street (preflop/flop/turn/river): preflop
    Pot size (in bb): 3
    Current bet to face (in bb, 0 if none): 2
    Action history (e.g. "UTG raise 6, MP call 6" or blank):
    Number of opponents: 2

    ══════════════════════════════════════════
      RECOMMENDATION: RAISE to 5.0 bb
    ══════════════════════════════════════════
      Hand:       Ah Ks (AKs)
      Position:   BTN
      Stack:      50.0 bb

      ── Why this play? ──
      Open raise: AKs is in BTN opening range (42.0% of hands)
    ══════════════════════════════════════════
"""

from __future__ import annotations

import sys

from poker_bot.core.game_context import (
    BlindLevel,
    GameContext,
    PayoutStructure,
    TournamentPhase,
)
from poker_bot.core.game_state import GameState, PlayerState
from poker_bot.core.hand_evaluator import HandEvaluator
from poker_bot.interface.opponent_tracker import OpponentTracker
from poker_bot.solver.engine import SolverEngine
from poker_bot.strategy.decision_maker import (
    ActionType,
    DecisionMaker,
    PostflopEngine,
    PriorAction,
    _hand_to_notation,
)
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Action, Position, Street


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_cards(s: str) -> list[Card]:
    """Parse card string: 'AhKs' or 'Ah Ks' or 'Ah Ks Td' -> list[Card].

    Supports both concatenated (2-char groups) and space-separated formats.
    """
    s = s.strip()
    if not s:
        return []
    if " " in s:
        return [Card.from_str(c.strip()) for c in s.split() if c.strip()]
    # Concatenated: split into 2-char chunks
    if len(s) % 2 != 0:
        raise ValueError(f"Invalid card string: '{s}' (odd length)")
    return [Card.from_str(s[i:i+2]) for i in range(0, len(s), 2)]


def _parse_position(s: str) -> Position:
    """Parse position string, case-insensitive."""
    return Position(s.strip().upper())


def _parse_street(s: str) -> Street:
    """Parse street string, case-insensitive."""
    return Street(s.strip().upper())


def _parse_action_history(s: str) -> list[PriorAction]:
    """Parse action history string.

    Format: "UTG raise 6, MP call 6, CO fold"
    Each entry: "POSITION ACTION [AMOUNT]"
    """
    s = s.strip()
    if not s:
        return []
    actions = []
    for part in s.split(","):
        tokens = part.strip().split()
        if len(tokens) < 2:
            continue
        pos = _parse_position(tokens[0])
        action = Action(tokens[1].strip().upper())
        amount = float(tokens[2]) if len(tokens) >= 3 else 0.0
        actions.append(PriorAction(pos, action, amount))
    return actions


def _hand_display(cards: list[Card]) -> str:
    """Display cards with notation: 'Ah Ks (AKs)'."""
    card_str = " ".join(str(c) for c in cards)
    if len(cards) == 2:
        notation = _hand_to_notation(cards[0], cards[1])
        return f"{card_str} ({notation})"
    return card_str


def _prompt(msg: str, default: str = "") -> str:
    """Print a prompt and read user input."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return val if val else default


def _prompt_float(msg: str, default: float = 0.0) -> float:
    """Prompt for a float value."""
    val = _prompt(msg, str(default))
    try:
        return float(val)
    except ValueError:
        print(f"    Invalid number, using {default}")
        return default


def _prompt_int(msg: str, default: int = 0) -> int:
    """Prompt for an integer value."""
    val = _prompt(msg, str(default))
    try:
        return int(val)
    except ValueError:
        print(f"    Invalid number, using {default}")
        return default


# ---------------------------------------------------------------------------
# Build game state from user input
# ---------------------------------------------------------------------------


def _build_situation(big_blind_chips: float = 2.0) -> tuple[GameState, GameContext, int, list[PriorAction]] | None:
    """Interactively collect situation details. Returns None on abort."""
    print()

    # Game type
    game_type = _prompt("Game type (cash/tournament)", "cash").lower()
    is_tournament = game_type.startswith("t")

    # Hero hand
    hand_str = _prompt("Your hand (e.g. AhKs)")
    if not hand_str:
        print("    No hand provided, aborting.")
        return None
    try:
        hero_cards = _parse_cards(hand_str)
        if len(hero_cards) != 2:
            print(f"    Need exactly 2 cards, got {len(hero_cards)}")
            return None
    except ValueError as e:
        print(f"    {e}")
        return None

    # Position
    try:
        position = _parse_position(_prompt("Position (UTG/MP/CO/BTN/SB/BB)", "BTN"))
    except ValueError:
        print("    Invalid position.")
        return None

    # Stack
    stack_bb = _prompt_float("Stack in bb", 50.0)

    # Street
    try:
        street = _parse_street(_prompt("Street (preflop/flop/turn/river)", "preflop"))
    except ValueError:
        print("    Invalid street.")
        return None

    # Board cards
    community_cards: list[Card] = []
    if street != Street.PREFLOP:
        board_str = _prompt("Board cards (e.g. Jh 8d 3c)")
        if board_str:
            try:
                community_cards = _parse_cards(board_str)
            except ValueError as e:
                print(f"    {e}")
                return None

    # Pot and bet
    pot_bb = _prompt_float("Pot size (in bb)", 3.0)
    current_bet_bb = _prompt_float("Current bet to face (in bb, 0 if none)", 0.0)

    # Action history
    history_str = _prompt("Action history (e.g. 'UTG raise 6, MP call 6' or blank)")
    try:
        action_history = _parse_action_history(history_str)
    except ValueError as e:
        print(f"    {e}")
        action_history = []

    # Opponents
    num_opponents = _prompt_int("Number of opponents", 2)

    # Build game state (amounts in bb — use bb as chip unit)
    hero = PlayerState(
        name="Hero",
        chips=stack_bb,
        position=position,
        hole_cards=hero_cards,
        is_active=True,
    )
    # Set hero's current_bet for blind positions
    if position == Position.BB and street == Street.PREFLOP and current_bet_bb <= 1.0:
        hero.current_bet = 1.0  # Already posted BB
    elif position == Position.SB and street == Street.PREFLOP:
        hero.current_bet = 0.5  # Already posted SB

    players = [hero]
    villain_positions = [p for p in [Position.UTG, Position.MP, Position.CO, Position.BTN, Position.SB, Position.BB] if p != position]
    for i in range(num_opponents):
        players.append(PlayerState(
            name=f"Villain{i+1}",
            chips=stack_bb,  # Assume similar stacks
            position=villain_positions[i % len(villain_positions)],
            hole_cards=[],
            is_active=True,
        ))

    gs = GameState(
        players=players,
        small_blind=0.5,
        big_blind=1.0,
        pot=pot_bb,
        current_bet=current_bet_bb,
        current_street=street,
        community_cards=community_cards,
    )

    # Build context
    if is_tournament:
        phase_str = _prompt("Tournament phase (early/middle/bubble/itm/final_table)", "middle").lower()
        phase_map = {
            "early": TournamentPhase.EARLY,
            "middle": TournamentPhase.MIDDLE,
            "bubble": TournamentPhase.BUBBLE,
            "itm": TournamentPhase.IN_THE_MONEY,
            "final_table": TournamentPhase.FINAL_TABLE,
        }
        phase = phase_map.get(phase_str, TournamentPhase.MIDDLE)
        players_left = _prompt_int("Players remaining", 50)
        total_entries = _prompt_int("Total entries", 100)

        # Simple payout structure
        payout = PayoutStructure(
            total_prize_pool=total_entries * 10.0,
            payouts={1: 0.25, 2: 0.15, 3: 0.10, 4: 0.08, 5: 0.06},
            total_entries=total_entries,
        )
        ctx = GameContext.tournament(
            stack_bb=stack_bb,
            phase=phase,
            players_remaining=players_left,
            payout_structure=payout,
            num_players=num_opponents + 1,
        )
    else:
        ctx = GameContext.cash_game(
            stack_bb=stack_bb,
            num_players=num_opponents + 1,
        )

    return gs, ctx, 0, action_history


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


_DIVIDER = "\n" + "=" * 50


def _print_recommendation(
    decision: "Decision",
    hero_cards: list[Card],
    position: Position,
    street: Street,
    stack_bb: float,
    community_cards: list[Card],
    opponent_advice: str = "",
    solver_result=None,
) -> None:
    """Print a formatted recommendation."""
    from poker_bot.strategy.decision_maker import Decision

    # Action display
    if decision.action == ActionType.RAISE:
        action_str = f"RAISE to {decision.amount:.1f} bb"
    elif decision.action == ActionType.CALL:
        action_str = f"CALL {decision.amount:.1f} bb"
    elif decision.action == ActionType.ALL_IN:
        action_str = f"ALL-IN ({decision.amount:.1f} bb)"
    else:
        action_str = decision.action.value

    print(_DIVIDER)
    print(f"  RECOMMENDATION: {action_str}")
    print("=" * 50)
    print(f"  Hand:       {_hand_display(hero_cards)}")
    print(f"  Position:   {position}")
    print(f"  Street:     {street}")
    if community_cards:
        print(f"  Board:      {' '.join(str(c) for c in community_cards)}")
    print(f"  Stack:      {stack_bb:.1f} bb")

    # Mixed strategy display from solver
    if solver_result is not None:
        print()
        print("  -- GTO Mixed Strategy --")
        print(f"  Source:     {solver_result.source} (confidence: {solver_result.confidence:.0%})")
        if solver_result.ev != 0:
            print(f"  EV:         {solver_result.ev:+.2f} bb")
        for af in sorted(solver_result.strategy.actions, key=lambda a: -a.frequency):
            if af.frequency >= 0.01:
                if af.amount > 0:
                    print(f"    {af.action:<8} {af.frequency:5.1%}  ({af.amount:.1f} bb)")
                else:
                    print(f"    {af.action:<8} {af.frequency:5.1%}")

    # Analysis section
    if decision.equity > 0 or decision.pot_odds > 0:
        print()
        print("  -- Analysis --")
        if decision.equity > 0:
            print(f"  Equity:     {decision.equity:.1%} vs estimated range")
        if decision.pot_odds > 0:
            print(f"  Pot odds:   {decision.pot_odds:.1%} (need {decision.pot_odds:.1%} to call)")

    # Hand strength for post-flop
    if street != Street.PREFLOP and community_cards and len(hero_cards) >= 2:
        try:
            hand_result = HandEvaluator.evaluate(hero_cards + community_cards)
            hand_strength = PostflopEngine._hand_strength_score(hand_result, community_cards)
            print(f"  Hand:       {hand_result.ranking.name} (strength: {hand_strength:.2f})")
        except ValueError:
            pass

    # Reasoning
    print()
    print("  -- Why this play? --")
    print(f"  {decision.reasoning}")

    # Opponent context
    if opponent_advice:
        print()
        print("  -- Opponent context --")
        print(f"  {opponent_advice}")

    print("=" * 50)
    print()


# ---------------------------------------------------------------------------
# Mode 1: Live coaching
# ---------------------------------------------------------------------------


def _live_coaching(tracker: OpponentTracker) -> None:
    """Interactive live coaching mode."""
    result = _build_situation()
    if result is None:
        return

    gs, ctx, hero_index, action_history = result
    hero = gs.players[hero_index]

    # Use solver-enhanced decision maker
    solver = SolverEngine()
    maker = DecisionMaker(solver=solver)
    decision, solver_result = maker.make_decision_detailed(
        gs, ctx, hero_index=hero_index, action_history=action_history,
    )

    # Check for opponent notes
    opponent_advice = ""
    opp_name = _prompt("Opponent name (for notes lookup, or blank)", "")
    if opp_name:
        advice = tracker.get_range_adjustment(opp_name)
        player = tracker.get_player(opp_name)
        if player:
            opponent_advice = f"{player.summary()}\n  Adjustment: {advice}"

    _print_recommendation(
        decision,
        hero.hole_cards,
        hero.position,
        gs.current_street,
        ctx.stack_depth_bb,
        gs.community_cards,
        opponent_advice,
        solver_result=solver_result,
    )


# ---------------------------------------------------------------------------
# Mode 2: Hand review
# ---------------------------------------------------------------------------

_ACTION_RANK = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 2,
    ActionType.RAISE: 3,
    ActionType.ALL_IN: 4,
}


def _grade_play(
    user_action: ActionType,
    user_amount: float,
    optimal: "Decision",
) -> tuple[str, str]:
    """Grade the user's play vs optimal. Returns (grade, explanation)."""
    from poker_bot.strategy.decision_maker import Decision

    opt_rank = _ACTION_RANK[optimal.action]
    usr_rank = _ACTION_RANK[user_action]
    diff = abs(opt_rank - usr_rank)

    if user_action == optimal.action:
        if optimal.action in (ActionType.FOLD, ActionType.CHECK):
            return "Optimal", "Correct play."
        # Check amount accuracy for raise/call
        if optimal.amount > 0:
            ratio = user_amount / optimal.amount if optimal.amount else 1.0
            if 0.8 <= ratio <= 1.2:
                return "Optimal", "Correct action and sizing."
            elif 0.6 <= ratio <= 1.5:
                return "Acceptable", f"Right action, sizing slightly off (optimal: {optimal.amount:.1f} bb)."
            else:
                return "Acceptable", f"Right action, but sizing needs work (optimal: {optimal.amount:.1f} bb)."
        return "Optimal", "Correct play."

    if diff == 1:
        # One step off
        if usr_rank < opt_rank:
            return "Mistake", f"Too passive. {optimal.reasoning}"
        return "Mistake", f"Too aggressive. {optimal.reasoning}"

    if diff >= 2:
        if usr_rank < opt_rank:
            return "Blunder", f"Way too passive. {optimal.reasoning}"
        return "Blunder", f"Way too aggressive. {optimal.reasoning}"

    return "Mistake", optimal.reasoning


def _hand_review(tracker: OpponentTracker) -> None:
    """Hand review mode: compare user's play vs optimal."""
    result = _build_situation()
    if result is None:
        return

    gs, ctx, hero_index, action_history = result
    hero = gs.players[hero_index]

    # Get optimal decision
    maker = DecisionMaker()
    optimal = maker.make_decision(
        gs, ctx, hero_index=hero_index, action_history=action_history,
    )

    # Ask what user actually did
    print()
    user_action_str = _prompt(
        "What did you do? (fold/check/call/raise/allin)", "call"
    ).upper().replace("-", "_").replace(" ", "_")

    # Normalize "ALLIN" to "ALL_IN"
    if user_action_str == "ALLIN":
        user_action_str = "ALL_IN"

    try:
        user_action = ActionType(user_action_str)
    except ValueError:
        print(f"    Unknown action: {user_action_str}")
        return

    user_amount = 0.0
    if user_action in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
        user_amount = _prompt_float("Amount (in bb)", 0.0)

    # Grade
    grade, explanation = _grade_play(user_action, user_amount, optimal)

    # Display
    print(_DIVIDER)
    print("  HAND REVIEW")
    print("=" * 50)
    print(f"  Hand:       {_hand_display(hero.hole_cards)}")
    print(f"  Position:   {hero.position}")
    print(f"  Street:     {gs.current_street}")
    if gs.community_cards:
        print(f"  Board:      {' '.join(str(c) for c in gs.community_cards)}")
    print()

    # User's play
    if user_amount > 0:
        print(f"  Your play:     {user_action.value} {user_amount:.1f} bb")
    else:
        print(f"  Your play:     {user_action.value}")

    # Optimal
    if optimal.amount > 0:
        print(f"  Optimal play:  {optimal.action.value} {optimal.amount:.1f} bb")
    else:
        print(f"  Optimal play:  {optimal.action.value}")

    # Grade with color-like emphasis
    grade_marks = {
        "Optimal": "[OK]",
        "Acceptable": "[~]",
        "Mistake": "[!]",
        "Blunder": "[!!]",
    }
    print()
    print(f"  Grade: {grade_marks.get(grade, '')} {grade}")
    print(f"  {explanation}")

    if optimal.equity > 0:
        print()
        print(f"  Equity: {optimal.equity:.1%} | Pot odds: {optimal.pot_odds:.1%}")

    print("=" * 50)
    print()


# ---------------------------------------------------------------------------
# Mode 3: Opponent notes
# ---------------------------------------------------------------------------


def _opponent_notes(tracker: OpponentTracker) -> None:
    """Opponent tracking sub-menu."""
    while True:
        print()
        print("  -- Opponent Notes --")
        print("  1. Add note")
        print("  2. View player")
        print("  3. List all players")
        print("  4. Update stats")
        print("  5. Back to main menu")
        choice = _prompt(">", "5")

        if choice == "1":
            name = _prompt("Player name")
            if not name:
                continue
            note = _prompt("Note")
            if note:
                tracker.add_note(name, note)
                print(f"    Note added for {name}.")

        elif choice == "2":
            name = _prompt("Player name")
            if not name:
                continue
            player = tracker.get_player(name)
            if not player:
                print(f"    No data for '{name}'.")
                continue
            print()
            print(f"  {player.summary()}")
            print(f"  Type:       {player.player_type}")
            adjustment = tracker.get_range_adjustment(name)
            print(f"  Adjustment: {adjustment}")
            if player.notes:
                print(f"  Notes:")
                for i, note in enumerate(player.notes, 1):
                    print(f"    {i}. {note}")

        elif choice == "3":
            players = tracker.list_players()
            if not players:
                print("    No players tracked yet.")
            else:
                print()
                for name in players:
                    p = tracker.get_player(name)
                    if p:
                        print(f"  {p.summary()}")

        elif choice == "4":
            name = _prompt("Player name")
            if not name:
                continue
            hands = _prompt_int("Hands to add", 0)
            vpip = _prompt_int("VPIP hands to add", 0)
            pfr = _prompt_int("PFR hands to add", 0)
            three_bet = _prompt_int("3-bet hands to add", 0)
            tracker.update_stats(
                name, hands=hands, vpip=vpip, pfr=pfr, three_bet=three_bet,
            )
            print(f"    Stats updated for {name}.")

        elif choice == "5":
            break


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


def run() -> None:
    """Main entry point for the poker coach."""
    tracker = OpponentTracker()

    print()
    print("=" * 50)
    print("  POKER COACH")
    print("=" * 50)

    while True:
        print()
        print("  1. Live coaching")
        print("  2. Hand review")
        print("  3. Opponent notes")
        print("  4. Quit")
        choice = _prompt(">", "4")

        if choice == "1":
            _live_coaching(tracker)
        elif choice == "2":
            _hand_review(tracker)
        elif choice == "3":
            _opponent_notes(tracker)
        elif choice == "4":
            print("  Good luck at the tables!")
            break


if __name__ == "__main__":
    run()
