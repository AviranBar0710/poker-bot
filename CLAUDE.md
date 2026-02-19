# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run all tests (241 tests, ~35s)
pytest -v

# Run a single test file
pytest tests/test_decision_maker.py -v

# Run a single test
pytest tests/test_decision_maker.py::TestPreflopOpen::test_utg_open_with_aces -v

# Run with coverage
pytest --cov=poker_bot --cov-report=term-missing
```

## Architecture

The decision engine follows a layered pipeline: **GameState + GameContext → PreflopEngine / PostflopEngine → Decision**.

### Module dependency flow (no circular imports)

```
utils/constants.py          # Enums: Rank, Suit, Position, Action, Street, HandRanking
utils/card.py               # Card (frozen dataclass), Deck
    ↓
core/hand_evaluator.py      # HandEvaluator.evaluate(cards) → HandResult (best 5 from up to 7)
core/game_state.py          # PlayerState, GameState (tracks pot, board, players, street)
core/game_context.py        # GameContext: game type, stack depth, tournament phase, ICM data
    ↓
core/equity_calculator.py   # Monte Carlo: hand_vs_hand, hand_vs_range, range_vs_range
strategy/preflop_ranges.py  # GTO ranges by position, notation parser ("JJ+", "ATs+", "A5s-A2s")
strategy/stack_strategy.py  # Stack-depth adjustments, push/fold charts (<15bb)
strategy/tournament_strategy.py  # ICM calculator (Malmuth-Harville), bubble factor, survival premium
    ↓
strategy/decision_maker.py  # DecisionMaker.make_decision() → Decision(action, amount, reasoning)
```

### Key design decisions

- **Decisions are data**: `Decision` is a frozen dataclass with `action`, `amount`, `reasoning`, `equity`, `pot_odds`. No side effects.
- **Range system**: `Range` class holds `set[HandNotation]`. Notation like `"AA,AKs,JJ+"` expands to specific card combos. Context-aware: `get_opening_range(position, context)` chains stack + tournament adjustments.
- **Effective call**: `make_decision` computes `effective_bet = game_state.current_bet - hero.current_bet` before passing to engines.
- **ICM adjustment**: Post-flop calling threshold is `pot_odds / survival_premium` where premium ∈ [0.3, 1.0]. Lower premium = tighter play.
- **Push/fold**: Stacks <15bb bypass normal range logic and use position-specific push/fold charts (5bb and 10bb tiers). BB has no push chart — handled separately.
- **Equity estimation**: 1000 sims for clear spots, 2000 for close decisions (hand_strength 0.25–0.75). Fast-tracked at extremes (≥0.95 or ≤0.05).

### Post-flop hand strength scores (thresholds that drive decisions)

Betting threshold: `hand_strength ≥ 0.65 and equity > 0.55`. Value-raise threshold: `hand_strength ≥ 0.85 and equity > 0.70`. Key scores: HIGH_CARD=0.10, ONE_PAIR=0.40 (+0.25 for top pair), TWO_PAIR=0.65, THREE_OF_A_KIND=0.85, FLUSH=0.90, FULL_HOUSE=0.94.

## Development workflow

For changes to the decision engine, follow the 5-step pipeline in `.claude/workflows/decision_engine_workflow.md`:
1. Implement → 2. Test (agent: `.claude/agents/testing_agent.md`) → 3. Code review (agent: `.claude/agents/code_review_agent.md`) → 4. Validate GTO (agent: `.claude/agents/validation_agent.md`) → 5. Iterate until all pass.

## Testing patterns

- Post-flop tests mock `PostflopEngine._estimate_equity` to avoid Monte Carlo stochasticity
- Assert on `ActionType` (FOLD/CHECK/CALL/RAISE/ALL_IN), not exact equity values
- Use extreme matchups (AA vs 72o, sets vs air) for deterministic outcomes
- Helper: `_cards("Ah Kh")` parses to `[Card, Card]`, used across all test files
- Tests build `GameState` with `PlayerState` list; hero is typically at index 0

## Known limitations

- No pure bluffing or trapping/slow-play logic in PostflopEngine
- `Range.contains()` is O(n*m) per check — no combo caching yet
