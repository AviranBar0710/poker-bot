"""Hybrid GTO solver engine for live MTT play.

Provides mixed-strategy GTO solutions using pre-computed lookups for
common spots and real-time Monte Carlo fallback for unusual ones.
Designed for <2s per decision in live tournament play.
"""
