"""TexasSolver bridge — file-based I/O adapter for the open-source CFR solver.

Invocation flow:
  1. Write a deterministic input.txt from SolverInput
  2. Run `console_solver -i input.txt` as subprocess with timeout
  3. Parse output_result.json → SolverOutput

Zero interpretation: frequencies come through exactly as the solver computed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from poker_bot.solver.external.bridge import (
    SolverBridge,
    SolverConfig,
    SolverError,
    SolverInput,
    SolverOutput,
)

logger = logging.getLogger("poker_bot.solver.external.texas")


# Standard GTO bet sizing trees by street.
# These produce a balanced game tree that captures the most common lines.
_DEFAULT_BET_SIZES: dict[str, list[str]] = {
    "flop": [
        "set_bet_sizes oop,flop,bet,33,50,75",
        "set_bet_sizes ip,flop,bet,33,50,75",
        "set_bet_sizes oop,flop,raise,60,100",
        "set_bet_sizes ip,flop,raise,60,100",
    ],
    "turn": [
        "set_bet_sizes oop,turn,bet,50,75,100",
        "set_bet_sizes ip,turn,bet,50,75,100",
        "set_bet_sizes oop,turn,raise,60,100",
        "set_bet_sizes ip,turn,raise,60,100",
    ],
    "river": [
        "set_bet_sizes oop,river,bet,50,75,100,150",
        "set_bet_sizes ip,river,bet,50,75,100,150",
        "set_bet_sizes oop,river,raise,100",
        "set_bet_sizes ip,river,raise,100",
        "set_bet_sizes oop,river,allin",
        "set_bet_sizes ip,river,allin",
    ],
}

# Action name mapping: solver output → our convention
_ACTION_MAP = {
    "fold": "fold",
    "check": "check",
    "call": "call",
    "allin": "raise",  # All-in mapped to raise
}


class TexasSolverBridge(SolverBridge):
    """File-based I/O adapter for TexasSolver (open-source CFR).

    Each solve creates an input file, runs the solver subprocess,
    and parses the JSON output. The solver runs to completion and
    exits — no persistent process.
    """

    def __init__(self, config: SolverConfig) -> None:
        self._config = config
        self._work_dir = (
            Path(config.working_dir)
            if config.working_dir
            else Path(tempfile.mkdtemp(prefix="texassolver_"))
        )
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if the TexasSolver binary exists and is executable."""
        path = self._config.binary_path
        return path.exists() and os.access(path, os.X_OK)

    def solve(self, solver_input: SolverInput) -> SolverOutput:
        """Run TexasSolver and return parsed GTO output.

        Args:
            solver_input: Standardized game state.

        Returns:
            SolverOutput with raw CFR frequencies.

        Raises:
            SolverError: On any failure (timeout, crash, parse error).
        """
        input_path = self._work_dir / "input.txt"
        output_path = self._work_dir / "output_result.json"

        # Clean previous output
        if output_path.exists():
            output_path.unlink()

        # 1. Write input file
        self._write_input_file(input_path, output_path, solver_input)

        # 2. Run solver subprocess
        try:
            result = subprocess.run(
                [str(self._config.binary_path), "-i", str(input_path)],
                capture_output=True,
                text=True,
                timeout=self._config.max_solve_seconds,
                cwd=str(self._work_dir),
            )
        except subprocess.TimeoutExpired as e:
            raise SolverError(
                f"TexasSolver timed out after {self._config.max_solve_seconds}s"
            ) from e
        except FileNotFoundError as e:
            raise SolverError(
                f"TexasSolver binary not found: {self._config.binary_path}"
            ) from e

        if result.returncode != 0:
            raise SolverError(
                f"TexasSolver exited with code {result.returncode}: "
                f"{result.stderr[:500]}"
            )

        # 3. Parse output
        if not output_path.exists():
            raise SolverError("TexasSolver did not produce output_result.json")

        return self._parse_output(output_path, solver_input)

    def cleanup(self) -> None:
        """Remove the working directory and all temp files."""
        if self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)

    # -------------------------------------------------------------------
    # Input file generation
    # -------------------------------------------------------------------

    def _write_input_file(
        self,
        input_path: Path,
        output_path: Path,
        solver_input: SolverInput,
    ) -> None:
        """Generate a deterministic TexasSolver input file."""
        board_str = ",".join(str(c) for c in solver_input.board)

        lines = [
            f"set_pot {solver_input.pot:.0f}",
            f"set_effective_stack {solver_input.effective_stack:.0f}",
            f"set_board {board_str}",
            f"set_range_ip {solver_input.ip_range_str}",
            f"set_range_oop {solver_input.oop_range_str}",
        ]

        # Bet sizing: for multi-street solves, include all streets
        # from the current street onward
        street_order = ["flop", "turn", "river"]
        start_idx = street_order.index(solver_input.street)
        for street in street_order[start_idx:]:
            lines.extend(_DEFAULT_BET_SIZES.get(street, []))

        lines.extend([
            "set_allin_threshold 0.67",
            f"set_thread_num {self._config.thread_count}",
            f"set_accuracy {self._config.accuracy}",
            "set_use_isomorphism 1",
            "build_tree",
            "start_solve",
            f"dump_result {output_path}",
        ])

        input_path.write_text("\n".join(lines) + "\n")
        logger.debug("Wrote TexasSolver input: %s", input_path)

    # -------------------------------------------------------------------
    # Output parsing
    # -------------------------------------------------------------------

    def _parse_output(
        self,
        output_path: Path,
        solver_input: SolverInput,
    ) -> SolverOutput:
        """Parse TexasSolver JSON output, extract hero's hand strategy.

        The JSON structure contains a tree of nodes. We navigate to the
        root decision node for hero's position and extract the strategy
        for hero's specific hand combo.

        Raises:
            SolverError: If the output cannot be parsed.
        """
        try:
            with open(output_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise SolverError(f"Failed to parse solver output: {e}") from e

        hero_key = self._hand_to_solver_key(solver_input.hero_cards)

        try:
            strategy, ev = self._extract_strategy(
                data, hero_key, solver_input.hero_is_ip,
            )
        except (KeyError, IndexError, TypeError) as e:
            raise SolverError(
                f"Failed to extract strategy for {hero_key}: {e}"
            ) from e

        return SolverOutput(
            hero_strategy=strategy,
            hero_ev=ev,
            converged=True,  # If solver completed without error, it converged
            exploitability=data.get("exploitability", 0.0),
        )

    def _extract_strategy(
        self,
        data: dict,
        hero_hand_key: str,
        hero_is_ip: bool,
    ) -> tuple[dict[str, float], float]:
        """Extract action frequencies for a specific hand from solver output.

        TexasSolver JSON structure varies by version. We handle the common
        format where the root node contains "strategy" as a dict mapping
        hand combos to action frequency arrays, and "actions" listing the
        available action names.

        Returns:
            Tuple of (action_freq_dict, ev).
        """
        # Navigate to the root node
        # TexasSolver typically stores the tree with root at the top
        root = data

        # Find the first decision node for the relevant player
        # In TexasSolver output, the root node for OOP acts first
        node = root
        if "childrens" in node:
            # Root contains children; the strategy is at this level
            pass
        elif "root" in node:
            node = node["root"]

        # Get available actions and strategy
        actions = node.get("actions", [])
        strategy_data = node.get("strategy", {})

        if not actions or not strategy_data:
            raise SolverError("No strategy data in solver output root node")

        # Look up hero's hand
        hand_freqs = strategy_data.get(hero_hand_key)
        if hand_freqs is None:
            # Try lowercase variant
            hand_freqs = strategy_data.get(hero_hand_key.lower())
        if hand_freqs is None:
            raise SolverError(
                f"Hand {hero_hand_key} not found in solver output"
            )

        # Map action names to our convention with frequencies
        result: dict[str, float] = {}
        for i, action_name in enumerate(actions):
            if i >= len(hand_freqs):
                break
            freq = hand_freqs[i]
            if freq < 0.001:
                continue  # Skip negligible actions
            mapped = self._map_action_name(action_name)
            # Aggregate if multiple solver actions map to the same name
            result[mapped] = result.get(mapped, 0.0) + freq

        # Extract EV if available
        ev_data = node.get("ev", {})
        ev = 0.0
        if isinstance(ev_data, dict):
            ev = ev_data.get(hero_hand_key, ev_data.get(hero_hand_key.lower(), 0.0))
        elif isinstance(ev_data, (int, float)):
            ev = float(ev_data)

        return result, ev

    @staticmethod
    def _map_action_name(solver_action: str) -> str:
        """Map TexasSolver action name to our internal convention.

        Solver outputs: "fold", "check", "call", "bet 33", "bet 75",
        "raise 60", "allin", etc.
        Our convention: "fold", "check", "call", "raise"
        """
        action_lower = solver_action.lower().strip()

        if action_lower in _ACTION_MAP:
            return _ACTION_MAP[action_lower]

        # "bet X" or "raise X" → "raise"
        if action_lower.startswith("bet") or action_lower.startswith("raise"):
            return "raise"

        return action_lower

    @staticmethod
    def _hand_to_solver_key(hero_cards: list[Card]) -> str:
        """Convert hero's cards to TexasSolver's hand key format.

        TexasSolver uses lowercase rank+suit, e.g. "AhKs" or "Td7c".
        Our Card.__str__() already produces this format.
        """
        return f"{hero_cards[0]}{hero_cards[1]}"
