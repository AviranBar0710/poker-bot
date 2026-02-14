# Poker Bot

An advanced Texas Hold'em poker bot built in Python.

## Features

- **Hand Evaluation** — Fast and accurate 5/7-card hand ranking
- **Game State Tracking** — Full game state management for Texas Hold'em
- **Strategy Engine** — GTO-based decision making (planned)
- **Opponent Modeling** — Adaptive play based on opponent tendencies (planned)

## Project Structure

```
poker_bot/
├── core/          # Game state, hand evaluation, equity calculation
├── strategy/      # Ranges, GTO engine, decision making
├── models/        # Opponent modeling
└── utils/         # Cards, constants, helpers
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest
```
