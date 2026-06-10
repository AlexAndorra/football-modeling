# Strict-holdout robustness — improved model vs consensus books

*Negative ΔlogLik / positive ΔRPS ⇒ the books are better. Both variants are renorm + LOFO-calibrated; paired Bayesian bootstrap.*

| variant | n | ΔlogLik (model−book) [95% CI] | ΔRPS [95% CI] | P(model better) | PIT p |
|---|--:|--:|--:|--:|--:|
| sequential (primary) | 1709 | -0.0586 [-0.0760, -0.0415] | +0.0160 [+0.0112, +0.0208] | 0.000 / 0.000 | 0.440 |
| strict-holdout | 1709 | -0.1103 [-0.1309, -0.0904] | +0.0305 [+0.0249, +0.0364] | 0.000 / 0.000 | 0.337 |
