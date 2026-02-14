"""Run poker simulations to validate the DecisionMaker.

Plays 1000 hands of the bot vs 5 random villains and prints
summary statistics plus 3 interesting example hands.
"""

from __future__ import annotations

from poker_bot.core.game_state import PlayerState
from poker_bot.simulation.poker_game import HandRecord, PokerGame
from poker_bot.utils.constants import Action, Position


def run_simulation(num_hands: int = 1000) -> None:
    """Run a simulation of num_hands and print results."""
    # Create 6 players: 1 bot + 5 random villains
    starting_chips = 1000.0
    players = [
        PlayerState(name="Bot", chips=starting_chips, position=Position.BTN),
    ]
    for i in range(5):
        players.append(
            PlayerState(
                name=f"Villain{i+1}",
                chips=starting_chips,
                position=Position.BTN,  # Will be reassigned each hand
            )
        )

    game = PokerGame(
        players=players,
        small_blind=5.0,
        big_blind=10.0,
    )

    # Track stats
    records: list[HandRecord] = []
    hands_won = 0
    vpip_count = 0  # Voluntarily Put money In Pot
    pfr_count = 0   # Pre-Flop Raise
    total_pots: list[float] = []
    profits: list[float] = []

    for i in range(num_hands):
        # Reset chips each hand to avoid elimination (keep it simple)
        for p in players:
            p.chips = starting_chips

        record = game.play_hand()
        records.append(record)

        hero_profit = record.hero_profit
        profits.append(hero_profit)
        total_pots.append(record.pot_size)

        if record.winner_name == "Bot":
            hands_won += 1

        # Check VPIP and PFR from actions
        hero_acted_preflop = False
        hero_raised_preflop = False
        in_preflop = True
        for action_line in record.actions_summary:
            if "---" in action_line and "PREFLOP" not in action_line:
                in_preflop = False
            if in_preflop and "Bot" in action_line:
                if "CALL" in action_line or "RAISE" in action_line or "ALL_IN" in action_line:
                    hero_acted_preflop = True
                if "RAISE" in action_line or "ALL_IN" in action_line:
                    hero_raised_preflop = True

        if hero_acted_preflop:
            vpip_count += 1
        if hero_raised_preflop:
            pfr_count += 1

    # Find interesting hands
    bot_wins = [r for r in records if r.winner_name == "Bot" and r.pot_size > 0]
    bot_losses = [r for r in records if r.winner_name != "Bot" and r.hero_profit < 0]
    showdown_hands = [r for r in records if r.winning_hand_ranking is not None]

    biggest_win = max(bot_wins, key=lambda r: r.pot_size) if bot_wins else None
    biggest_loss = min(bot_losses, key=lambda r: r.hero_profit) if bot_losses else None
    best_showdown = max(
        showdown_hands, key=lambda r: (r.winning_hand_ranking or 0)
    ) if showdown_hands else None

    # Print results
    print("=" * 60)
    print(f"  POKER BOT SIMULATION — {num_hands} hands")
    print("=" * 60)
    print()
    print(f"  Hands played:     {num_hands}")
    print(f"  Hands won:        {hands_won} ({hands_won / num_hands:.1%})")
    print(f"  Total profit:     {sum(profits):+.0f}")
    print(f"  Avg profit/hand:  {sum(profits) / num_hands:+.1f}")
    print()
    print(f"  VPIP:             {vpip_count / num_hands:.1%}")
    print(f"  PFR:              {pfr_count / num_hands:.1%}")
    print(f"  Avg pot size:     {sum(total_pots) / len(total_pots):.0f}")
    print(f"  Biggest win:      {max(profits):+.0f}")
    print(f"  Biggest loss:     {min(profits):+.0f}")
    print()

    interesting = [
        ("Biggest pot won by Bot", biggest_win),
        ("Biggest loss for Bot", biggest_loss),
        ("Best showdown hand", best_showdown),
    ]

    for title, record in interesting:
        if record is None:
            continue
        print("-" * 60)
        print(f"  {title} (Hand #{record.hand_number})")
        print(f"  Bot cards: {record.player_hands.get('Bot', ['?', '?'])}")
        print(f"  Board:     {' '.join(record.community_cards) if record.community_cards else '(none)'}")
        print(f"  Pot:       {record.pot_size:.0f}")
        print(f"  Winner:    {record.winner_name}", end="")
        if record.winning_hand_ranking:
            print(f" ({record.winning_hand_ranking.name})")
        else:
            print()
        print(f"  Bot profit: {record.hero_profit:+.0f}")
        print(f"  Actions:")
        for line in record.actions_summary:
            print(f"    {line}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
