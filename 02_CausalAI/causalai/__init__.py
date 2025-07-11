"""
CausalAI package for soccer substitution analysis.

This package provides tools for simulating and analyzing the causal impact
of player substitutions on soccer team performance.
"""

from .simulation import simulate_matches

__version__ = "0.1.0"
__author__ = "Alexandre Andorra"

__all__ = [
    "simulate_matches",
]
