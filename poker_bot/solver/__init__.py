"""Hybrid GTO solver engine for live MTT play.

Provides mixed-strategy GTO solutions using pre-computed lookups for
common spots and real-time Monte Carlo fallback for unusual ones.
Designed for <2s per decision in live tournament play.

Key public API:
    SolverProtocol  -- Interface for swappable solver backends
    SolverEngine    -- Built-in hybrid solver (lookup + heuristic)
    SolverResult    -- Solver output (strategy, source, confidence, EV)
"""

from poker_bot.solver.data_structures import SolverProtocol, SolverResult
from poker_bot.solver.engine import SolverEngine

__all__ = ["SolverProtocol", "SolverResult", "SolverEngine"]
