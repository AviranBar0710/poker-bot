"""Background solver execution using QThread + signals/slots.

SolverWorker runs DecisionMaker on a background QThread so the UI
never freezes during Monte Carlo equity calculations. EngineAdapter
is the main-thread interface that manages the worker lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, Signal

from poker_bot.core.game_context import GameContext
from poker_bot.core.game_state import GameState
from poker_bot.solver.data_structures import SolverResult
from poker_bot.solver.engine import SolverEngine
from poker_bot.strategy.decision_maker import Decision, DecisionMaker, PriorAction


@dataclass
class SolveRequest:
    """All parameters needed for a single solve."""

    game_state: GameState
    context: GameContext
    hero_index: int = 0
    action_history: list[PriorAction] | None = None


@dataclass
class SolveResponse:
    """Result returned from the background solver."""

    decision: Decision
    solver_result: SolverResult | None


class SolverWorker(QObject):
    """Runs solver computations on a background thread.

    Communicate via signals only — never call methods directly
    from the main thread after moveToThread().
    """

    solve_requested = Signal(object)  # SolveRequest
    finished = Signal(object)  # SolveResponse
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._solver = SolverEngine()
        self._maker = DecisionMaker(solver=self._solver)
        self.solve_requested.connect(self._do_solve)

    def _do_solve(self, request: SolveRequest) -> None:
        """Execute the solve on the worker thread."""
        try:
            decision, solver_result = self._maker.make_decision_detailed(
                request.game_state,
                request.context,
                hero_index=request.hero_index,
                action_history=request.action_history,
            )
            self.finished.emit(SolveResponse(decision, solver_result))
        except Exception as e:
            self.error.emit(str(e))


class EngineAdapter(QObject):
    """Main-thread adapter that manages the background solver worker.

    Usage:
        adapter = EngineAdapter()
        adapter.solving_started.connect(on_start)
        adapter.solving_finished.connect(on_result)
        adapter.solving_error.connect(on_error)
        adapter.request_solve(request)
    """

    solving_started = Signal()
    solving_finished = Signal(object)  # SolveResponse
    solving_error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._thread = QThread()
        self._worker = SolverWorker()
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def request_solve(self, request: SolveRequest) -> None:
        """Submit a solve request to the background thread."""
        self.solving_started.emit()
        self._worker.solve_requested.emit(request)

    def _on_finished(self, response: SolveResponse) -> None:
        self.solving_finished.emit(response)

    def _on_error(self, message: str) -> None:
        self.solving_error.emit(message)

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._thread.quit()
        self._thread.wait()
