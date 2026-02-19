"""SolverEngine: top-level orchestrator for the hybrid GTO solver.

Routes preflop to PreflopSolver, postflop to PostflopSolver,
applies tournament adjustments, and provides a single solve() entry point.

Includes structured logging for decision transparency and a resilience
wrapper that falls back to a safe default on unexpected errors.
"""

from __future__ import annotations

import logging
import time

from poker_bot.core.game_context import GameContext
from poker_bot.core.game_state import GameState
from poker_bot.solver.board_bucketing import bucket_stack
from poker_bot.solver.data_structures import (
    ActionFrequency,
    SolverResult,
    StrategyNode,
)
from poker_bot.solver.icm_adapter import adjust_for_icm
from poker_bot.solver.postflop_solver import PostflopSolver
from poker_bot.solver.preflop_solver import PreflopSolver
from poker_bot.strategy.decision_maker import (
    PostflopEngine,
    PriorAction,
    analyze_board,
)
from poker_bot.strategy.preflop_ranges import Range
from poker_bot.strategy.tournament_strategy import survival_premium
from poker_bot.utils.constants import Street

logger = logging.getLogger("poker_bot.solver")

# Safe fallback result when the solver fails unexpectedly
_FALLBACK_RESULT = SolverResult(
    strategy=StrategyNode(actions=[
        ActionFrequency("fold", 1.0, 0.0, 0.0),
    ]),
    source="fallback",
    confidence=0.0,
)


class SolverEngine:
    """Top-level solver orchestrator.

    Implements SolverProtocol so it can be injected into DecisionMaker
    and swapped for other backends (PioSolver, LLM, etc.).

    Usage:
        engine = SolverEngine()
        result = engine.solve(game_state, context, hero_index, action_history)
    """

    def __init__(self) -> None:
        self._preflop = PreflopSolver()
        self._postflop = PostflopSolver()

    def solve(
        self,
        game_state: GameState,
        context: GameContext,
        hero_index: int,
        action_history: list[PriorAction] | None = None,
        opponent_range: Range | None = None,
    ) -> SolverResult:
        """Solve for the optimal mixed strategy.

        Resilience: if any step raises an unexpected exception, logs
        the error and returns a safe fallback (fold with confidence=0)
        so DecisionMaker falls back to heuristics.

        Args:
            game_state: Current game state.
            context: Game context (cash/tournament, stack depth, etc).
            hero_index: Index of the hero in game_state.players.
            action_history: Prior actions in this hand.
            opponent_range: Optional explicit opponent range.

        Returns:
            SolverResult with mixed strategy, source, and confidence.
        """
        t_start = time.perf_counter()

        try:
            result = self._solve_inner(
                game_state, context, hero_index,
                action_history, opponent_range,
            )
        except Exception:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            logger.exception(
                "Solver failed after %.1fms — returning safe fallback",
                elapsed_ms,
            )
            return _FALLBACK_RESULT

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        rec = result.strategy.recommended_action
        rec_str = f"{rec.action} {rec.frequency:.0%}" if rec else "none"
        logger.info(
            "%s %s → %s (source=%s, confidence=%.0f%%, ev=%.2f, %.1fms)",
            result.spot_key.street if result.spot_key else "unknown",
            result.spot_key.position if result.spot_key else "?",
            rec_str,
            result.source,
            result.confidence * 100,
            result.ev,
            elapsed_ms,
        )

        return result

    def _solve_inner(
        self,
        game_state: GameState,
        context: GameContext,
        hero_index: int,
        action_history: list[PriorAction] | None,
        opponent_range: Range | None,
    ) -> SolverResult:
        """Core solve logic, separated for clean error handling."""
        hero = game_state.players[hero_index]
        action_history = action_history or []

        if not hero.hole_cards or len(hero.hole_cards) < 2:
            logger.debug("No hole cards for hero — returning fold")
            return SolverResult(
                strategy=StrategyNode(actions=[
                    ActionFrequency("fold", 1.0, 0.0, 0.0),
                ]),
                source="no_cards",
                confidence=0.0,
            )

        # Get ICM premium for tournament adjustment
        premium = survival_premium(context) if context.is_tournament else 1.0
        if context.is_tournament:
            logger.debug(
                "Tournament mode: phase=%s, premium=%.2f",
                context.tournament_phase,
                premium,
            )

        if game_state.current_street == Street.PREFLOP:
            result = self._solve_preflop(
                hero, game_state, context, action_history, premium,
            )
        else:
            result = self._solve_postflop(
                hero, game_state, context, action_history,
                opponent_range, premium,
            )

        # Apply ICM adjustment to final strategy
        if context.is_tournament and premium < 0.95:
            logger.debug(
                "Applying ICM adjustment: premium=%.2f", premium,
            )
            result = SolverResult(
                strategy=adjust_for_icm(result.strategy, premium),
                source=result.source,
                confidence=result.confidence,
                ev=result.ev * premium,
                spot_key=result.spot_key,
            )

        return result

    def _solve_preflop(self, hero, game_state, context, action_history, premium):
        """Route to preflop solver."""
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
        self, hero, game_state, context, action_history,
        opponent_range, premium,
    ):
        """Route to postflop solver."""
        from poker_bot.core.hand_evaluator import HandEvaluator

        t0 = time.perf_counter()

        hand_result = HandEvaluator.evaluate(
            hero.hole_cards + game_state.community_cards,
        )
        hand_strength = PostflopEngine._hand_strength_score(
            hand_result, game_state.community_cards,
        )

        # Detect draws
        texture = analyze_board(game_state.community_cards)
        has_draw = texture.has_flush_draw or texture.has_straight_draw
        draw_strength = 0.0
        if has_draw:
            draw_strength = 0.5 if texture.has_flush_draw else 0.35

        effective_bet = max(0.0, game_state.current_bet - hero.current_bet)
        pot = game_state.pot

        # Compute positional and multiway info
        num_opponents = max(1, game_state.players_in_hand - 1)
        villain_positions = [
            p.position.value
            for p in game_state.active_players
            if p is not hero
        ]
        is_ip = PostflopSolver._is_in_position(
            hero.position.value, villain_positions,
        )

        # Look up opponent stats for the primary villain (first active opponent)
        villain_stats = None
        if context.opponent_stats:
            for p in game_state.active_players:
                if p is not hero and p.name in context.opponent_stats:
                    villain_stats = context.opponent_stats[p.name]
                    break

        result = self._postflop.get_strategy(
            hero_cards=hero.hole_cards,
            community_cards=game_state.community_cards,
            position=hero.position,
            pot=pot,
            hero_stack=hero.chips,
            big_blind=game_state.big_blind,
            action_history=action_history,
            hand_strength=hand_strength,
            has_draw=has_draw,
            draw_strength=draw_strength,
            opponent_range=opponent_range,
            num_opponents=num_opponents,
            is_ip=is_ip,
            opponent_stats=villain_stats,
        )

        logger.debug(
            "Postflop solve: %.1fms (source=%s, hand_strength=%.2f, "
            "category=%s, board=%s)",
            (time.perf_counter() - t0) * 1000,
            result.source,
            hand_strength,
            result.spot_key.hand_category if result.spot_key else "?",
            result.spot_key.board_bucket if result.spot_key else "?",
        )
        return result
