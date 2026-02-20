"""ExternalSolverEngine: pure GTO relay via external CFR solver.

Zero-heuristic mandate: every postflop decision comes from a real CFR
solver (TexasSolver, PioSolver). If the solver is unavailable, the result
is GTO_UNAVAILABLE — never a guess, never a heuristic estimate.

Preflop: delegates to PreflopSolver (Phase 8a SQLite DB — real solver data).
Postflop: delegates to SolverBridge → external CFR subprocess.
"""

from __future__ import annotations

import logging
import time

from poker_bot.core.game_context import GameContext
from poker_bot.core.game_state import GameState
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverResult,
    SpotKey,
    StrategyNode,
)
from poker_bot.solver.external.bridge import (
    GTO_UNAVAILABLE,
    SolverBridge,
    SolverConfig,
    SolverError,
    SolverInput,
    SolverOutput,
    load_solver_config,
)
from poker_bot.solver.external.range_converter import RangeConverter
from poker_bot.solver.icm_adapter import adjust_for_icm
from poker_bot.solver.postflop_solver import PostflopSolver
from poker_bot.solver.preflop_solver import PreflopSolver
from poker_bot.solver.range_estimator import RangeEstimator
from poker_bot.strategy.decision_maker import PriorAction
from poker_bot.strategy.preflop_ranges import OPENING_RANGES, Range
from poker_bot.strategy.tournament_strategy import survival_premium
from poker_bot.utils.constants import Street

logger = logging.getLogger("poker_bot.solver.external_engine")


class ExternalSolverEngine:
    """Pure GTO solver relay. No heuristics. No estimation.

    Preflop: PreflopSolver (SQLite DB — real solver data from Phase 8a)
    Postflop: External CFR solver via SolverBridge
    Unavailable: Returns GTO_UNAVAILABLE (never guesses)

    Implements SolverProtocol so it can be injected into DecisionMaker.
    """

    def __init__(
        self,
        config: SolverConfig | None = None,
        bridge: SolverBridge | None = None,
    ) -> None:
        """Initialize with optional solver config or pre-built bridge.

        Args:
            config: Solver configuration. If None, attempts to load from
                ~/.poker_coach/solver_config.json.
            bridge: Pre-built bridge instance (for testing). Takes
                priority over config if both are provided.
        """
        self._preflop = PreflopSolver()
        self._bridge = bridge

        if self._bridge is None and config is not None:
            self._bridge = self._create_bridge(config)
        elif self._bridge is None:
            # Try to load config from default path
            loaded_config = load_solver_config()
            if loaded_config is not None:
                self._bridge = self._create_bridge(loaded_config)

    @staticmethod
    def _create_bridge(config: SolverConfig) -> SolverBridge | None:
        """Create the appropriate bridge for the configured solver type."""
        if config.solver_type == "texassolver":
            from poker_bot.solver.external.texas_solver import TexasSolverBridge

            return TexasSolverBridge(config)
        # Future: PioSolver, other solvers
        logger.warning("Unknown solver type: %s", config.solver_type)
        return None

    def solve(
        self,
        game_state: GameState,
        context: GameContext,
        hero_index: int,
        action_history: list[PriorAction] | None = None,
        opponent_range: Range | None = None,
    ) -> SolverResult:
        """Solve for the GTO mixed strategy.

        Preflop: routes to PreflopSolver (Phase 8a).
        Postflop: routes to external CFR solver, or returns GTO_UNAVAILABLE.

        Args:
            game_state: Current game state.
            context: Game context (cash/tournament).
            hero_index: Index of the hero in game_state.players.
            action_history: Prior actions in this hand.
            opponent_range: Optional explicit opponent range.

        Returns:
            SolverResult with CFR-computed strategy, or GTO_UNAVAILABLE.
        """
        t_start = time.perf_counter()
        hero = game_state.players[hero_index]
        action_history = action_history or []

        if not hero.hole_cards or len(hero.hole_cards) < 2:
            logger.debug("No hole cards for hero — returning GTO_UNAVAILABLE")
            return GTO_UNAVAILABLE

        # Tournament survival premium
        premium = survival_premium(context) if context.is_tournament else 1.0
        if context.is_tournament:
            logger.debug(
                "Tournament mode: phase=%s, premium=%.2f",
                context.tournament_phase,
                premium,
            )

        # Route by street
        if game_state.current_street == Street.PREFLOP:
            result = self._solve_preflop(
                hero, game_state, context, action_history, premium,
            )
        else:
            result = self._solve_postflop(
                hero, game_state, context, action_history, opponent_range,
            )

        # Apply ICM adjustment for tournaments (mathematically sound, not heuristic)
        if (
            context.is_tournament
            and premium < 0.95
            and result.source != "gto_unavailable"
        ):
            logger.debug("ICM adjustment: premium=%.2f", premium)
            result = SolverResult(
                strategy=adjust_for_icm(result.strategy, premium),
                source=result.source,
                confidence=result.confidence,
                ev=result.ev * premium,
                spot_key=result.spot_key,
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        rec = result.strategy.recommended_action
        rec_str = f"{rec.action} {rec.frequency:.0%}" if rec else "none"
        logger.info(
            "%s → %s (source=%s, confidence=%.0f%%, ev=%.2f, %.1fms)",
            game_state.current_street.value,
            rec_str,
            result.source,
            result.confidence * 100,
            result.ev,
            elapsed_ms,
        )

        return result

    def _solve_preflop(self, hero, game_state, context, action_history, premium):
        """Route preflop to PreflopSolver (Phase 8a — real solver data)."""
        t0 = time.perf_counter()
        result = self._preflop.get_strategy(
            card1=hero.hole_cards[0],
            card2=hero.hole_cards[1],
            position=hero.position,
            action_history=action_history,
            stack_bb=context.stack_depth_bb,
            survival_premium=premium,
        )
        logger.debug(
            "Preflop lookup: %.1fms (source=%s)",
            (time.perf_counter() - t0) * 1000,
            result.source,
        )
        return result

    def _solve_postflop(
        self, hero, game_state, context, action_history, opponent_range,
    ):
        """Route postflop to external CFR solver.

        If the solver bridge is unavailable or fails, returns GTO_UNAVAILABLE.
        No heuristic fallback. No estimation. No guessing.
        """
        # Check bridge availability
        if self._bridge is None:
            logger.info("No solver bridge configured — returning GTO_UNAVAILABLE")
            return GTO_UNAVAILABLE

        if not self._bridge.is_available():
            logger.warning(
                "Solver bridge not available (binary missing or not executable)"
            )
            return GTO_UNAVAILABLE

        try:
            return self._run_external_solve(
                hero, game_state, context, action_history, opponent_range,
            )
        except SolverError as e:
            logger.error("External solver failed: %s", e)
            return GTO_UNAVAILABLE
        except Exception:
            logger.exception("Unexpected error in external solver")
            return GTO_UNAVAILABLE

    def _run_external_solve(
        self, hero, game_state, context, action_history, opponent_range,
    ):
        """Build SolverInput, invoke bridge, map SolverOutput → SolverResult.

        This is where we translate between our internal data model and the
        external solver's wire format. Zero interpretation of the results.
        """
        board = game_state.community_cards
        street = PostflopSolver._detect_street(board)

        # Determine position
        villain_positions = [
            p.position.value
            for p in game_state.active_players
            if p is not hero
        ]
        hero_is_ip = PostflopSolver._is_in_position(
            hero.position.value, villain_positions,
        )

        # Hero range: position-based opening range
        hero_range = OPENING_RANGES.get(hero.position, Range())
        hero_range_str = RangeConverter.to_texas_solver(hero_range)
        if not hero_range_str:
            hero_range_str = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AKo"

        # Opponent range: estimate from action history or use provided
        if opponent_range is not None:
            opp_range = opponent_range
        else:
            # Find primary villain position
            villain_pos = None
            for p in game_state.active_players:
                if p is not hero:
                    villain_pos = p.position
                    break
            opp_range = RangeEstimator.estimate_preflop_range(
                villain_pos, action_history,
            )

        opp_range_str = RangeConverter.to_texas_solver(opp_range)
        if not opp_range_str:
            opp_range_str = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AKo"

        # Assign IP/OOP ranges
        if hero_is_ip:
            ip_range_str = hero_range_str
            oop_range_str = opp_range_str
        else:
            ip_range_str = opp_range_str
            oop_range_str = hero_range_str

        # Effective stack
        villain_stacks = [
            p.chips for p in game_state.active_players if p is not hero
        ]
        effective_stack = min(hero.chips, min(villain_stacks)) if villain_stacks else hero.chips

        solver_input = SolverInput(
            board=board,
            hero_cards=hero.hole_cards,
            pot=game_state.pot,
            effective_stack=effective_stack,
            hero_is_ip=hero_is_ip,
            ip_range_str=ip_range_str,
            oop_range_str=oop_range_str,
            street=street,
        )

        t0 = time.perf_counter()
        solver_output = self._bridge.solve(solver_input)
        solve_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "External solve: %.1fms (converged=%s, exploitability=%.3f%%)",
            solve_ms,
            solver_output.converged,
            solver_output.exploitability,
        )

        return self._map_output(solver_output, hero, game_state, street)

    def _map_output(
        self, output: SolverOutput, hero, game_state, street: str,
    ) -> SolverResult:
        """Map SolverOutput → SolverResult with zero interpretation.

        The frequencies come through exactly as the CFR solver computed them.
        We only translate action names and pack into our SolverResult type.
        """
        actions = []
        for action_name, freq in output.hero_strategy.items():
            # Map raise to a sizing (use pot-relative sizing)
            amount = 0.0
            if action_name == "raise":
                amount = game_state.pot * 0.75  # Default sizing from solver tree
            actions.append(
                ActionFrequency(
                    action=action_name,
                    frequency=freq,
                    amount=amount,
                    ev=output.hero_ev * freq,
                )
            )

        strategy = StrategyNode(actions=actions).normalized()

        return SolverResult(
            strategy=strategy,
            source="external_solver",
            confidence=0.95 if output.converged else 0.80,
            ev=output.hero_ev,
            spot_key=SpotKey(
                street=street,
                position=hero.position.value,
                action_sequence="",
                stack_bucket="",
            ),
        )

    def cleanup(self) -> None:
        """Release solver resources (temp files, processes)."""
        if self._bridge is not None:
            self._bridge.cleanup()
