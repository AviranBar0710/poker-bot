"""Tests for solver bridge types, range conversion, and GTO_UNAVAILABLE sentinel."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from poker_bot.solver.data_structures import SolverProtocol, StrategyNode
from poker_bot.solver.external.bridge import (
    GTO_UNAVAILABLE,
    SolverConfig,
    SolverError,
    SolverInput,
    SolverOutput,
    load_solver_config,
)
from poker_bot.solver.external.range_converter import RangeConverter
from poker_bot.strategy.preflop_ranges import Range
from poker_bot.utils.card import Card
from poker_bot.utils.constants import Rank, Suit


def _card(s: str) -> Card:
    return Card.from_str(s)


def _cards(s: str) -> list[Card]:
    return [Card.from_str(c) for c in s.split()]


# ---------------------------------------------------------------------------
# GTO_UNAVAILABLE sentinel
# ---------------------------------------------------------------------------


class TestGTOUnavailable:
    def test_source_is_gto_unavailable(self):
        assert GTO_UNAVAILABLE.source == "gto_unavailable"

    def test_confidence_is_zero(self):
        assert GTO_UNAVAILABLE.confidence == 0.0

    def test_ev_is_zero(self):
        assert GTO_UNAVAILABLE.ev == 0.0

    def test_strategy_is_empty(self):
        assert GTO_UNAVAILABLE.strategy.actions == []

    def test_no_recommended_action(self):
        assert GTO_UNAVAILABLE.strategy.recommended_action is None


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------


class TestSolverConfig:
    def test_config_creation(self):
        config = SolverConfig(
            solver_type="texassolver",
            binary_path=Path("/usr/local/bin/console_solver"),
            thread_count=8,
            accuracy=0.5,
            max_solve_seconds=60,
        )
        assert config.solver_type == "texassolver"
        assert config.thread_count == 8
        assert config.accuracy == 0.5

    def test_config_defaults(self):
        config = SolverConfig(
            solver_type="texassolver",
            binary_path=Path("/bin/solver"),
        )
        assert config.thread_count == 4
        assert config.accuracy == 0.5
        assert config.max_solve_seconds == 120
        assert config.working_dir is None

    def test_config_is_frozen(self):
        config = SolverConfig(
            solver_type="texassolver",
            binary_path=Path("/bin/solver"),
        )
        with pytest.raises(AttributeError):
            config.thread_count = 16


# ---------------------------------------------------------------------------
# load_solver_config
# ---------------------------------------------------------------------------


class TestLoadSolverConfig:
    def test_missing_file_returns_none(self, tmp_path):
        result = load_solver_config(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_config_loads(self, tmp_path):
        config_path = tmp_path / "solver.json"
        config_path.write_text(json.dumps({
            "solver_type": "texassolver",
            "binary_path": "/usr/local/bin/console_solver",
            "thread_count": 8,
            "accuracy": 0.25,
            "max_solve_seconds": 30,
        }))
        result = load_solver_config(config_path)
        assert result is not None
        assert result.solver_type == "texassolver"
        assert result.thread_count == 8
        assert result.accuracy == 0.25
        assert result.max_solve_seconds == 30

    def test_missing_required_key_returns_none(self, tmp_path):
        config_path = tmp_path / "solver.json"
        config_path.write_text(json.dumps({
            "solver_type": "texassolver",
            # Missing binary_path
        }))
        result = load_solver_config(config_path)
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        config_path = tmp_path / "solver.json"
        config_path.write_text("not json{{{")
        result = load_solver_config(config_path)
        assert result is None

    def test_minimal_config(self, tmp_path):
        config_path = tmp_path / "solver.json"
        config_path.write_text(json.dumps({
            "solver_type": "texassolver",
            "binary_path": "/bin/solver",
        }))
        result = load_solver_config(config_path)
        assert result is not None
        assert result.thread_count == 4  # default
        assert result.accuracy == 0.5  # default


# ---------------------------------------------------------------------------
# SolverInput / SolverOutput
# ---------------------------------------------------------------------------


class TestSolverIO:
    def test_solver_input_creation(self):
        inp = SolverInput(
            board=_cards("Qs Jh 2h"),
            hero_cards=_cards("Ah Kh"),
            pot=50.0,
            effective_stack=200.0,
            hero_is_ip=True,
            ip_range_str="AA,KK,QQ",
            oop_range_str="JJ,TT,99",
            street="flop",
        )
        assert len(inp.board) == 3
        assert inp.pot == 50.0
        assert inp.street == "flop"

    def test_solver_output_creation(self):
        out = SolverOutput(
            hero_strategy={"raise": 0.65, "call": 0.25, "fold": 0.10},
            hero_ev=3.5,
            converged=True,
            exploitability=0.3,
        )
        assert out.hero_strategy["raise"] == 0.65
        assert out.converged is True


# ---------------------------------------------------------------------------
# SolverError
# ---------------------------------------------------------------------------


class TestSolverError:
    def test_error_message(self):
        err = SolverError("Solver timed out")
        assert str(err) == "Solver timed out"

    def test_is_exception(self):
        assert issubclass(SolverError, Exception)


# ---------------------------------------------------------------------------
# RangeConverter
# ---------------------------------------------------------------------------


class TestRangeConverterTexas:
    def test_empty_range(self):
        result = RangeConverter.to_texas_solver(Range())
        assert result == ""

    def test_single_hand(self):
        r = Range().add("AA")
        result = RangeConverter.to_texas_solver(r)
        assert "AA" in result

    def test_multiple_hands(self):
        r = Range().add("AA,KK,QQ")
        result = RangeConverter.to_texas_solver(r)
        parts = result.split(",")
        assert len(parts) == 3
        assert "AA" in parts
        assert "KK" in parts
        assert "QQ" in parts

    def test_suited_and_offsuit(self):
        r = Range().add("AKs,AKo")
        result = RangeConverter.to_texas_solver(r)
        assert "AKo" in result
        assert "AKs" in result

    def test_deterministic_output(self):
        r = Range().add("AA,KK,QQ,AKs")
        result1 = RangeConverter.to_texas_solver(r)
        result2 = RangeConverter.to_texas_solver(r)
        assert result1 == result2


class TestRangeConverterPio:
    def test_empty_range_all_zeros(self):
        result = RangeConverter.to_piosolver(Range())
        assert len(result) == 1326
        assert sum(result) == 0.0

    def test_has_correct_length(self):
        r = Range().add("AA")
        result = RangeConverter.to_piosolver(r)
        assert len(result) == 1326

    def test_nonzero_entries_for_aa(self):
        r = Range().add("AA")
        result = RangeConverter.to_piosolver(r)
        # AA has 6 combos (4 choose 2)
        assert sum(result) == 6.0

    def test_dead_card_removal(self):
        r = Range().add("AA")
        dead = [Card(Rank.ACE, Suit.HEARTS)]
        result = RangeConverter.to_piosolver(r, dead_cards=dead)
        # With Ah dead, only combos not involving Ah remain: 3 choose 2 = 3
        assert sum(result) == 3.0
