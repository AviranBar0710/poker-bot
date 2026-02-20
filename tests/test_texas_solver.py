"""Tests for TexasSolverBridge with mocked subprocess calls."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from poker_bot.solver.external.bridge import (
    SolverConfig,
    SolverError,
    SolverInput,
)
from poker_bot.solver.external.texas_solver import TexasSolverBridge
from poker_bot.utils.card import Card


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


@pytest.fixture
def config(tmp_path):
    """Create a SolverConfig pointing to a fake binary in tmp_path."""
    binary = tmp_path / "console_solver"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return SolverConfig(
        solver_type="texassolver",
        binary_path=binary,
        working_dir=tmp_path / "work",
        thread_count=4,
        accuracy=0.5,
        max_solve_seconds=30,
    )


@pytest.fixture
def bridge(config):
    return TexasSolverBridge(config)


@pytest.fixture
def solver_input():
    return SolverInput(
        board=_cards("Qs Jh 2h"),
        hero_cards=_cards("Ah Kh"),
        pot=50.0,
        effective_stack=200.0,
        hero_is_ip=True,
        ip_range_str="AA,KK,QQ,AKs",
        oop_range_str="JJ,TT,99,AQs",
        street="flop",
    )


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_available_with_executable(self, bridge):
        assert bridge.is_available() is True

    def test_not_available_missing_binary(self, tmp_path):
        config = SolverConfig(
            solver_type="texassolver",
            binary_path=tmp_path / "nonexistent",
        )
        b = TexasSolverBridge(config)
        assert b.is_available() is False

    def test_not_available_non_executable(self, tmp_path):
        binary = tmp_path / "solver"
        binary.write_text("not executable")
        binary.chmod(0o644)  # Not executable
        config = SolverConfig(
            solver_type="texassolver",
            binary_path=binary,
        )
        b = TexasSolverBridge(config)
        assert b.is_available() is False


# ---------------------------------------------------------------------------
# Input file generation
# ---------------------------------------------------------------------------


class TestInputFileGeneration:
    def test_input_file_contains_pot(self, bridge, solver_input, tmp_path):
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, solver_input)

        content = input_path.read_text()
        assert "set_pot 50" in content

    def test_input_file_contains_stack(self, bridge, solver_input):
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, solver_input)

        content = input_path.read_text()
        assert "set_effective_stack 200" in content

    def test_input_file_contains_board(self, bridge, solver_input):
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, solver_input)

        content = input_path.read_text()
        assert "set_board" in content

    def test_input_file_contains_ranges(self, bridge, solver_input):
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, solver_input)

        content = input_path.read_text()
        assert "set_range_ip AA,KK,QQ,AKs" in content
        assert "set_range_oop JJ,TT,99,AQs" in content

    def test_input_file_contains_bet_sizes(self, bridge, solver_input):
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, solver_input)

        content = input_path.read_text()
        # Flop bet sizes should be present
        assert "set_bet_sizes oop,flop,bet" in content
        assert "set_bet_sizes ip,flop,bet" in content

    def test_input_file_multi_street_sizes(self, bridge, solver_input):
        """Flop input should include flop, turn, and river bet sizes."""
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, solver_input)

        content = input_path.read_text()
        assert "flop,bet" in content
        assert "turn,bet" in content
        assert "river,bet" in content

    def test_input_file_turn_excludes_flop(self, bridge):
        """Turn input should NOT include flop bet sizes."""
        turn_input = SolverInput(
            board=_cards("Qs Jh 2h 7c"),
            hero_cards=_cards("Ah Kh"),
            pot=80.0,
            effective_stack=170.0,
            hero_is_ip=True,
            ip_range_str="AA,KK",
            oop_range_str="QQ,JJ",
            street="turn",
        )
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, turn_input)

        content = input_path.read_text()
        assert "flop,bet" not in content
        assert "turn,bet" in content
        assert "river,bet" in content

    def test_input_file_contains_solver_commands(self, bridge, solver_input):
        input_path = bridge._work_dir / "input.txt"
        output_path = bridge._work_dir / "output.json"
        bridge._write_input_file(input_path, output_path, solver_input)

        content = input_path.read_text()
        assert "build_tree" in content
        assert "start_solve" in content
        assert "dump_result" in content
        assert "set_allin_threshold 0.67" in content
        assert "set_use_isomorphism 1" in content


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


class TestOutputParsing:
    def _write_canned_output(self, bridge, solver_input, strategy_data=None):
        """Write a canned JSON output and parse it."""
        output_path = bridge._work_dir / "output_result.json"
        data = {
            "actions": ["fold", "check", "bet 50"],
            "strategy": strategy_data or {
                "AhKh": [0.10, 0.25, 0.65],
            },
            "ev": {"AhKh": 5.2},
            "exploitability": 0.35,
        }
        output_path.write_text(json.dumps(data))
        return bridge._parse_output(output_path, solver_input)

    def test_parses_action_frequencies(self, bridge, solver_input):
        result = self._write_canned_output(bridge, solver_input)
        assert "fold" in result.hero_strategy
        assert "raise" in result.hero_strategy  # "bet 50" maps to "raise"
        assert abs(result.hero_strategy["raise"] - 0.65) < 0.001

    def test_parses_ev(self, bridge, solver_input):
        result = self._write_canned_output(bridge, solver_input)
        assert abs(result.hero_ev - 5.2) < 0.001

    def test_skips_negligible_frequencies(self, bridge, solver_input):
        output_path = bridge._work_dir / "output_result.json"
        data = {
            "actions": ["fold", "check", "bet 50"],
            "strategy": {"AhKh": [0.0005, 0.40, 0.5995]},
            "ev": {"AhKh": 3.0},
        }
        output_path.write_text(json.dumps(data))
        result = bridge._parse_output(output_path, solver_input)
        assert "fold" not in result.hero_strategy  # < 0.001

    def test_converged_true_on_success(self, bridge, solver_input):
        result = self._write_canned_output(bridge, solver_input)
        assert result.converged is True

    def test_exploitability_parsed(self, bridge, solver_input):
        result = self._write_canned_output(bridge, solver_input)
        assert abs(result.exploitability - 0.35) < 0.001

    def test_hand_not_found_raises(self, bridge, solver_input):
        output_path = bridge._work_dir / "output_result.json"
        data = {
            "actions": ["fold", "check"],
            "strategy": {"TsTc": [0.50, 0.50]},  # No AhKh
            "ev": {},
        }
        output_path.write_text(json.dumps(data))
        with pytest.raises(SolverError, match="not found"):
            bridge._parse_output(output_path, solver_input)

    def test_invalid_json_raises(self, bridge, solver_input):
        output_path = bridge._work_dir / "output_result.json"
        output_path.write_text("not json{{{")
        with pytest.raises(SolverError, match="Failed to parse"):
            bridge._parse_output(output_path, solver_input)

    def test_root_node_navigation(self, bridge, solver_input):
        """Handles JSON with 'root' key wrapping the strategy."""
        output_path = bridge._work_dir / "output_result.json"
        data = {
            "root": {
                "actions": ["fold", "call", "bet 75"],
                "strategy": {"AhKh": [0.05, 0.30, 0.65]},
                "ev": {"AhKh": 4.0},
            },
            "exploitability": 0.20,
        }
        output_path.write_text(json.dumps(data))
        result = bridge._parse_output(output_path, solver_input)
        assert "raise" in result.hero_strategy
        assert abs(result.hero_ev - 4.0) < 0.001


# ---------------------------------------------------------------------------
# Action name mapping
# ---------------------------------------------------------------------------


class TestActionMapping:
    def test_fold(self):
        assert TexasSolverBridge._map_action_name("fold") == "fold"

    def test_check(self):
        assert TexasSolverBridge._map_action_name("check") == "check"

    def test_call(self):
        assert TexasSolverBridge._map_action_name("call") == "call"

    def test_bet_maps_to_raise(self):
        assert TexasSolverBridge._map_action_name("bet 50") == "raise"
        assert TexasSolverBridge._map_action_name("bet 75") == "raise"

    def test_raise_maps_to_raise(self):
        assert TexasSolverBridge._map_action_name("raise 100") == "raise"

    def test_allin_maps_to_raise(self):
        assert TexasSolverBridge._map_action_name("allin") == "raise"

    def test_case_insensitive(self):
        assert TexasSolverBridge._map_action_name("FOLD") == "fold"
        assert TexasSolverBridge._map_action_name("BET 33") == "raise"


# ---------------------------------------------------------------------------
# Hand key conversion
# ---------------------------------------------------------------------------


class TestHandKey:
    def test_hand_to_solver_key(self):
        cards = _cards("Ah Ks")
        key = TexasSolverBridge._hand_to_solver_key(cards)
        assert key == "AhKs"

    def test_hand_to_solver_key_order_preserved(self):
        cards = _cards("7c Ad")
        key = TexasSolverBridge._hand_to_solver_key(cards)
        assert key == "7cAd"


# ---------------------------------------------------------------------------
# Solve with mocked subprocess
# ---------------------------------------------------------------------------


class TestSolveSubprocess:
    def test_timeout_raises_solver_error(self, bridge, solver_input):
        import subprocess

        with patch("poker_bot.solver.external.texas_solver.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="solver", timeout=30,
            )
            with pytest.raises(SolverError, match="timed out"):
                bridge.solve(solver_input)

    def test_nonzero_exit_raises_solver_error(self, bridge, solver_input):
        with patch("poker_bot.solver.external.texas_solver.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="segfault")
            with pytest.raises(SolverError, match="exited with code 1"):
                bridge.solve(solver_input)

    def test_missing_binary_raises_solver_error(self, bridge, solver_input):
        with patch("poker_bot.solver.external.texas_solver.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("No such file")
            with pytest.raises(SolverError, match="binary not found"):
                bridge.solve(solver_input)

    def test_missing_output_raises_solver_error(self, bridge, solver_input):
        with patch("poker_bot.solver.external.texas_solver.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Don't create the output file
            with pytest.raises(SolverError, match="did not produce"):
                bridge.solve(solver_input)

    def test_successful_solve(self, bridge, solver_input):
        """Full solve flow with mocked subprocess producing canned output."""
        output_data = {
            "actions": ["fold", "check", "bet 75"],
            "strategy": {"AhKh": [0.10, 0.30, 0.60]},
            "ev": {"AhKh": 4.5},
            "exploitability": 0.25,
        }

        def fake_run(*args, **kwargs):
            # Write the canned output file
            output_path = bridge._work_dir / "output_result.json"
            output_path.write_text(json.dumps(output_data))
            return MagicMock(returncode=0, stderr="")

        with patch("poker_bot.solver.external.texas_solver.subprocess.run", side_effect=fake_run):
            result = bridge.solve(solver_input)

        assert result.converged is True
        assert "raise" in result.hero_strategy
        assert abs(result.hero_strategy["raise"] - 0.60) < 0.001
        assert abs(result.hero_ev - 4.5) < 0.001


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_work_dir(self, bridge):
        assert bridge._work_dir.exists()
        bridge.cleanup()
        assert not bridge._work_dir.exists()

    def test_cleanup_idempotent(self, bridge):
        bridge.cleanup()
        bridge.cleanup()  # Should not raise
