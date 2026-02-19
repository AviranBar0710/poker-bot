"""SolverEngine: top-level orchestrator for the hybrid GTO solver.

Routes preflop to PreflopSolver, postflop to PostflopSolver,
applies tournament adjustments, and provides a single solve() entry point.
"""

from __future__ import annotations

from poker_bot.core.game_context import GameContext
from poker_bot.core.game_state import GameState
from poker_bot.solver.board_bucketing import bucket_stack
from poker_bot.solver.data_structures import SolverResult, StrategyNode, ActionFrequency
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


class SolverEngine:
    """Top-level solver orchestrator.

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

        Args:
            game_state: Current game state.
            context: Game context (cash/tournament, stack depth, etc).
            hero_index: Index of the hero in game_state.players.
            action_history: Prior actions in this hand.
            opponent_range: Optional explicit opponent range.

        Returns:
            SolverResult with mixed strategy, source, and confidence.
        """
        hero = game_state.players[hero_index]
        action_history = action_history or []

        if not hero.hole_cards or len(hero.hole_cards) < 2:
            return SolverResult(
                strategy=StrategyNode(actions=[
                    ActionFrequency("fold", 1.0, 0.0, 0.0),
                ]),
                source="no_cards",
                confidence=0.0,
            )

        # Get ICM premium for tournament adjustment
        premium = survival_premium(context) if context.is_tournament else 1.0

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
        return self._preflop.get_strategy(
            card1=hero.hole_cards[0],
            card2=hero.hole_cards[1],
            position=hero.position,
            action_history=action_history,
            stack_bb=context.stack_depth_bb,
            survival_premium=premium,
        )

    def _solve_postflop(
        self, hero, game_state, context, action_history,
        opponent_range, premium,
    ):
        """Route to postflop solver."""
        # Compute hand strength using existing PostflopEngine logic
        from poker_bot.core.hand_evaluator import HandEvaluator

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

        return self._postflop.get_strategy(
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
        )
