# Improved model vs consensus books — robustness

*Negative ΔlogLik / positive ΔRPS ⇒ the books are better. Renorm + LOFO-calibrated; paired Bayesian bootstrap.*

## By forecasting semantics (all folds)

| variant | n | ΔlogLik (model−book) [95% CI] | ΔRPS [95% CI] | P(model better) | PIT p |
|---|--:|--:|--:|--:|--:|
| sequential (primary) | 1709 | -0.0586 [-0.0760, -0.0415] | +0.0160 [+0.0112, +0.0208] | 0.000 / 0.000 | 0.440 |
| strict-holdout | 1709 | -0.1103 [-0.1309, -0.0904] | +0.0305 [+0.0249, +0.0364] | 0.000 / 0.000 | 0.337 |

## By competition (sequential variant) — the finals are the actual event

| competition | n | ΔlogLik (model−book) [95% CI] | ΔRPS [95% CI] | P(model better) |
|---|--:|--:|--:|--:|
| qualifiers | 1581 | -0.0641 [-0.0824, -0.0466] | +0.0174 [+0.0124, +0.0223] | 0.000 / 0.000 |
| finals | 128 | +0.0084 [-0.0567, +0.0775] | -0.0005 [-0.0197, +0.0183] | 0.591 / 0.516 |
