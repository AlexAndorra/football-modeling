# Causal Impact of Player Substitutions on Soccer Team Performance

## Executive Summary

This project applies **Causal AI** techniques to isolate the true impact of player substitutions on soccer team performance, accounting for confounding factors like game context and tactical situations. While substitutes often show higher per-minute production than starters, understanding the *causal effect* of substitutions is complicated by the fact that teams typically make attacking substitutions when losing - when they would naturally increase attacking output out of necessity.

**Key Innovation:** Using controlled simulation and modern causal inference methods to disentangle the true substitution effect from observational correlations. The project simulates realistic soccer match scenarios with known ground-truth causal relationships, then applies deep learning models to estimate treatment effects while controlling for confounders like scoreline and team strength.

**Methodology:** 
- Simulates 10,000+ match scenarios with realistic player attributes, game states, and substitution decisions
- Uses neural networks and causal inference techniques (e.g backdoor adjustment) to estimate Average Treatment Effects
- Incorporates player characteristics to model heterogeneous substitution effects based on player type and game context

**Practical Application:** Develops an interactive dashboard for coaches to test "what-if" substitution scenarios, providing data-driven insights for tactical decision-making. The tool can predict how different substitution strategies affect outcomes like shots, goals, and possession in the final 30 minutes of matches.

**Context:** Designed for submission to MIT Sloan Sports Analytics Conference 2026, leveraging the expanded substitution rules (5 subs) for the upcoming FIFA World Cup.

## Environment Setup

To reproduce the results, install the virtual environment for this project:

```bash
conda env create -f environment.yml
conda activate soccer_sub
```

The environment includes key dependencies for causal inference (PyMC, NumPyro), deep learning (PyTorch, JAX), and data science workflows (pandas, scikit-learn, matplotlib).
