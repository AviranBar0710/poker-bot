"""Abstract bridge interface for external GTO solvers.

Defines the protocol for communicating with headless CFR solver engines
(TexasSolver, PioSolver). All data types are solver-agnostic intermediates
that decouple our internal GameState from solver-specific wire formats.

Zero heuristic mandate: if the solver cannot provide a mathematically
proven GTO solution, SolverError is raised — never a guess.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from poker_bot.solver.data_structures import (
    SolverResult,
    StrategyNode,
)
from poker_bot.utils.card import Card

logger = logging.getLogger("poker_bot.solver.external")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverConfig:
    """Configuration for an external GTO solver binary."""

    solver_type: str  # "texassolver" or "piosolver"
    binary_path: Path
    working_dir: Path | None = None
    thread_count: int = 4
    accuracy: float = 0.5  # Convergence target (% of pot)
    max_solve_seconds: int = 120
    extra_options: dict[str, str] = field(default_factory=dict)


def load_solver_config(config_path: Path | None = None) -> SolverConfig | None:
    """Load solver configuration from JSON file.

    Default path: ~/.poker_coach/solver_config.json

    Returns None if the config file does not exist, allowing the caller
    to decide how to handle the absence (GTO_UNAVAILABLE).

    Expected JSON format:
        {
            "solver_type": "texassolver",
            "binary_path": "/path/to/console_solver",
            "thread_count": 8,
            "accuracy": 0.5,
            "max_solve_seconds": 60
        }
    """
    path = config_path or Path.home() / ".poker_coach" / "solver_config.json"
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read solver config at %s: %s", path, e)
        return None

    required = ("solver_type", "binary_path")
    for key in required:
        if key not in data:
            logger.warning("Solver config missing required key: %s", key)
            return None

    return SolverConfig(
        solver_type=data["solver_type"],
        binary_path=Path(data["binary_path"]),
        working_dir=Path(data["working_dir"]) if "working_dir" in data else None,
        thread_count=data.get("thread_count", 4),
        accuracy=data.get("accuracy", 0.5),
        max_solve_seconds=data.get("max_solve_seconds", 120),
        extra_options=data.get("extra_options", {}),
    )


# ---------------------------------------------------------------------------
# Solver I/O data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverInput:
    """Standardized input for an external solver solve request.

    All values are in the solver's chip/bb scale. The bridge implementation
    is responsible for formatting these into solver-specific wire formats.
    """

    board: list[Card]  # 3 (flop), 4 (turn), or 5 (river) community cards
    hero_cards: list[Card]  # Hero's 2 hole cards
    pot: float  # Current pot size
    effective_stack: float  # Effective stack remaining (min of hero/villain)
    hero_is_ip: bool  # Whether hero is in position
    ip_range_str: str  # IP player's range in solver format
    oop_range_str: str  # OOP player's range in solver format
    street: str  # "flop", "turn", "river"


@dataclass
class SolverOutput:
    """Parsed output from an external solver.

    hero_strategy maps our internal action names (raise, call, fold, check)
    to their GTO frequencies. These frequencies come directly from the CFR
    solver — zero modification, zero interpretation.
    """

    hero_strategy: dict[str, float]  # action_name -> frequency [0.0, 1.0]
    hero_ev: float  # Expected value in chips/bb
    converged: bool  # Whether the solver reached the accuracy target
    exploitability: float = 0.0  # Exploitability in % of pot


# ---------------------------------------------------------------------------
# Error and sentinel
# ---------------------------------------------------------------------------

class SolverError(Exception):
    """Raised when an external solver fails to produce a GTO solution.

    This is NOT a fallback trigger — it propagates as GTO_UNAVAILABLE
    to the caller. No heuristic guesses are ever substituted.
    """

    pass


# Sentinel result: returned when no GTO solution can be computed.
# Empty strategy, zero confidence. The GUI renders this as an explicit
# "solver not available" warning — never as a fake recommendation.
GTO_UNAVAILABLE = SolverResult(
    strategy=StrategyNode(actions=[]),
    source="gto_unavailable",
    confidence=0.0,
    ev=0.0,
)


# ---------------------------------------------------------------------------
# Abstract bridge
# ---------------------------------------------------------------------------

class SolverBridge(ABC):
    """Abstract interface for external GTO solver backends.

    Implementations handle the solver-specific details of:
    - Validating the binary exists and is executable
    - Formatting SolverInput into the solver's wire format
    - Running the solver (subprocess or stdin/stdout)
    - Parsing the solver's output into SolverOutput

    The bridge is a pure data relay. It never modifies, adjusts, or
    interprets the solver's computed frequencies.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the solver binary exists and is executable."""
        ...

    @abstractmethod
    def solve(self, solver_input: SolverInput) -> SolverOutput:
        """Run the solver and return parsed GTO output.

        This is a blocking call. Timeouts are enforced internally.

        Args:
            solver_input: Standardized game state for the solver.

        Returns:
            SolverOutput with raw GTO frequencies.

        Raises:
            SolverError: If the solver fails for any reason.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (temp files, running processes)."""
        ...
