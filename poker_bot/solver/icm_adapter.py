"""ICM-to-strategy adjustment layer.

Adjusts mixed strategy frequencies for ICM (Independent Chip Model)
pressure in tournaments. Higher ICM pressure increases fold frequency
and decreases aggressive actions.
"""

from __future__ import annotations

from poker_bot.solver.data_structures import ActionFrequency, StrategyNode


def adjust_for_icm(
    strategy: StrategyNode,
    survival_premium: float,
) -> StrategyNode:
    """Adjust a mixed strategy for ICM tournament pressure.

    When survival_premium < 1.0, we:
    1. Increase fold frequency by `(1 - survival_premium) * 0.3`
    2. Decrease aggressive action frequencies proportionally
    3. Renormalize to sum to 1.0

    Args:
        strategy: The base mixed strategy.
        survival_premium: ICM survival premium [0.3, 1.0].
            1.0 = chip-EV (no adjustment), lower = more ICM pressure.

    Returns:
        ICM-adjusted StrategyNode with normalized frequencies.
    """
    if survival_premium >= 0.95 or not strategy.actions:
        return strategy

    icm_factor = 1.0 - survival_premium  # 0.0 to 0.7

    adjusted = []
    for af in strategy.actions:
        if af.action == "fold":
            # Increase fold frequency
            new_freq = af.frequency + icm_factor * 0.3
            adjusted.append(ActionFrequency(
                af.action, new_freq, af.amount, af.ev,
            ))
        elif af.action in ("raise", "all_in"):
            # Decrease aggressive actions more
            new_freq = max(0.0, af.frequency * (1.0 - icm_factor * 0.5))
            adjusted.append(ActionFrequency(
                af.action, new_freq, af.amount,
                af.ev * survival_premium,  # EV reduced by ICM tax
            ))
        elif af.action == "call":
            # Decrease calling somewhat less
            new_freq = max(0.0, af.frequency * (1.0 - icm_factor * 0.3))
            adjusted.append(ActionFrequency(
                af.action, new_freq, af.amount,
                af.ev * survival_premium,
            ))
        else:
            # Check — no adjustment
            adjusted.append(af)

    return StrategyNode(actions=adjusted).normalized()
