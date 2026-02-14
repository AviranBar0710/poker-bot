# Decision Engine Development Workflow

## Overview
This workflow ensures every change to the decision engine is correct, tested, and performant. Each feature goes through a 5-step pipeline before being considered complete.

## Prerequisites
- All existing tests pass (`pytest -v` — currently 99 tests)
- Working virtual environment with dependencies installed
- Understanding of the current module structure:
  - `poker_bot/core/` — hand_evaluator, game_state, game_context, equity_calculator
  - `poker_bot/strategy/` — preflop_ranges, stack_strategy, tournament_strategy
  - `poker_bot/models/` — (opponent modeling, to be built)

---

## Step 1: Design & Implement

### Goal
Implement the feature with clean architecture and correct poker logic.

### Process
1. **Read agent config**: Review `.claude/agents/code_review_agent.md` for coding standards
2. **Design the interface first**: Define public methods, parameters, return types
3. **Implement core logic**: Follow these principles:
   - Decisions are data: return `Decision` dataclass, not side effects
   - Separate "what to do" from "how much" (action selection vs bet sizing)
   - Every decision must be explainable (attach reasoning to the Decision)
   - Use the existing module hierarchy (GameContext → Range → Equity → Decision)
4. **Keep functions small**: Max ~40 lines per function, decompose complex logic
5. **Add type hints everywhere**: Python 3.10+ syntax

### Decision Engine Architecture
```
GameContext + Position + HoleCards
        ↓
  [Pre-flop Module]
   ├── Check range membership
   ├── Determine action type (open/3bet/4bet/call/fold)
   └── Calculate raise sizing
        ↓
  [Post-flop Module]  (if applicable)
   ├── Evaluate hand strength on board
   ├── Calculate equity vs estimated opponent range
   ├── Consider pot odds and implied odds
   └── Determine action and sizing
        ↓
  [Tournament Adjustment Layer]  (if tournament)
   ├── Apply ICM pressure
   ├── Apply bubble factor
   └── Adjust thresholds
        ↓
     Decision(action, amount, reasoning)
```

### Output
- Implementation in `poker_bot/strategy/decision_maker.py`
- Any supporting modules needed

---

## Step 2: Testing Agent Writes Tests

### Goal
Achieve >90% coverage with comprehensive tests covering every decision path.

### Process
1. **Read agent config**: Review `.claude/agents/testing_agent.md` for test standards
2. **Write tests BEFORE confirming implementation is complete** (TDD where practical)
3. **Create test files**:
   - `tests/test_decision_maker.py` — Unit tests for each decision type
   - `tests/test_decision_edge_cases.py` — Edge cases and boundary conditions
   - `tests/test_decision_tournament.py` — ICM and tournament-specific decisions
   - `tests/test_decision_integration.py` — End-to-end hand simulations
4. **Run tests and iterate**: Fix failures, add missing coverage

### Test Checklist
- [ ] Every public method has ≥1 test
- [ ] Every Action enum value appears in ≥1 test
- [ ] Every Position enum value appears in ≥1 test
- [ ] Every Street enum value appears in ≥1 test
- [ ] Every stack category (deep/medium/short/very_short/critical) tested
- [ ] Every tournament phase tested
- [ ] Edge cases: all-in, 1bb stack, zero equity, nut hand
- [ ] Property: bet_size is always legal (≥ min_raise, ≤ stack)
- [ ] Property: stronger hands get equal or stronger actions

### Output
- Test files with all tests passing
- Coverage report showing >90% line coverage

---

## Step 3: Code Review

### Goal
Verify correctness, performance, type safety, and GTO adherence.

### Process
1. **Read agent config**: Review `.claude/agents/code_review_agent.md` for review criteria
2. **Review implementation against checklist**:
   - [ ] Pot odds math is correct
   - [ ] Equity calculations use correct opponent ranges
   - [ ] ICM adjustments are applied properly
   - [ ] Bet sizing respects constraints
   - [ ] No division by zero possible
   - [ ] No unchecked None/empty list access
   - [ ] Performance: equity calcs don't run unnecessarily
   - [ ] GTO: ranges are balanced, not exploitably transparent
3. **Check performance hot paths**:
   - Decision should complete in <100ms for a single spot
   - Equity calculations should be lazy (only when needed)
   - Range expansion should be cached
4. **Document findings**: Categorize as Critical / Performance / Suggestion

### Output
- Review document with categorized findings
- All Critical issues must be resolved before proceeding

---

## Step 4: Validation

### Goal
Verify decisions match poker theory and mathematical expectations.

### Process
1. **Read agent config**: Review `.claude/agents/validation_agent.md` for validation scenarios
2. **Run known GTO spots**:
   - AA pre-flop: always raise (never fold, rarely flat)
   - 72o UTG: always fold
   - AKs vs QQ: ~46% equity
   - Pot-sized bet: requires 33% equity to call
3. **Validate ICM**:
   - Equal stacks → equal equity
   - Chip leader ICM < chip-proportional
   - Total ICM equity = prize pool
4. **Monotonicity checks**:
   - Better hand → equal or better action
   - Better position → equal or wider range
   - Less ICM pressure → equal or wider range
5. **Simulation runs**:
   - 1000 random pre-flop scenarios: verify range percentages
   - Verify no crashes with edge-case inputs

### Output
- Validation report with PASS / FAIL / MARGINAL for each scenario
- All FAIL scenarios must be resolved before proceeding

---

## Step 5: Iterate Until Approved

### Goal
All three agents approve the implementation.

### Process
1. Collect outputs from Steps 2-4
2. Address all Critical issues and test failures
3. Re-run the pipeline:
   - Run `pytest -v` — all tests must pass
   - Re-check any modified code against review criteria
   - Re-validate any scenarios affected by changes
4. Final checklist:
   - [ ] All tests pass
   - [ ] No Critical review issues remain
   - [ ] All validation scenarios PASS
   - [ ] Coverage >90%
   - [ ] Performance: <100ms per decision
5. Commit with descriptive message

### Commit Convention
```
feat(decision): <short description>

- Bullet points of key changes
- Reference any GTO theory implemented
- Note any known limitations

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Quick Reference: Running the Pipeline

```bash
# Step 1: Implement
# (write code in poker_bot/strategy/decision_maker.py)

# Step 2: Test
pytest tests/test_decision_maker.py tests/test_decision_edge_cases.py \
       tests/test_decision_tournament.py tests/test_decision_integration.py -v

# Step 3: Review
# (run code review agent against decision_maker.py)

# Step 4: Validate
pytest tests/test_decision_validation.py -v

# Step 5: Full suite
pytest -v --tb=short

# Coverage check
pytest --cov=poker_bot/strategy/decision_maker --cov-report=term-missing
```

---

## Appendix: Key Files

| File | Purpose |
|------|---------|
| `poker_bot/strategy/decision_maker.py` | Core decision engine |
| `poker_bot/core/game_context.py` | Game state and context |
| `poker_bot/core/equity_calculator.py` | Monte Carlo equity |
| `poker_bot/core/hand_evaluator.py` | Hand ranking |
| `poker_bot/strategy/preflop_ranges.py` | GTO pre-flop ranges |
| `poker_bot/strategy/stack_strategy.py` | Stack-depth adjustments |
| `poker_bot/strategy/tournament_strategy.py` | ICM and tournament play |
| `.claude/agents/testing_agent.md` | Testing standards |
| `.claude/agents/code_review_agent.md` | Code review criteria |
| `.claude/agents/validation_agent.md` | Validation scenarios |
