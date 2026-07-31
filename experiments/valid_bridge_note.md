# Note: the valid-bridge proximal baseline still fails at heavy tails

Context: `nonlinear_compare_to_gmm.py`, PROXY graph, design where confounding enters Y
nonlinearly (`Y = e_Y + beta*X + w*W^2 + f*U^2`, with `W = e_W + b*U^2`).

The plain `LinearProximal` baseline (`proxy`) is biased here because no *linear* bridge
h(W) solves the bridge equation — the confounding channel is ~U^4 while a linear bridge
is ~U^2. A quadratic bridge h(W) = c1*W^2 + c2*W solves it exactly, so we added the
baseline `proxy_valid_bridge`: `LinearProximal("X","Y",["Z","Z2"],["W","W2"])` on data
augmented with W2=W^2, Z2=Z^2.

**Result: even this correctly-specified bridge is inconsistent for tail index alpha <= 2.**
Sweep (30 reps, fraction of runs with |beta_hat - 1| > 0.1, true beta = 1):

| alpha | n=10k | n=100k | n=300k |
|-------|-------|--------|--------|
| 1.5   | 0.50  | 0.93   | 0.97   |
| 2.0   | 0.20  | 0.43   | 0.73   |
| 2.2   | 0.00  | 0.10   | 0.40   |
| 2.5   | 0.07  | 0.03   | 0.10   |
| 3.0   | 0.10  | 0.00   | 0.03   |

The spread *grows* with n for alpha <= 2 (q10/q90 reach +-15 at alpha=1.5).

**Mechanism (sketch):** the transformed regressors (W^2, Z^2 ~ index alpha/4; W, Z ~ alpha/2)
are not a linear SEM in iid innovations (cross terms like 2b*e_W*U^2), so the stable-limit
argument that keeps linear 2SLS consistent under infinite variance does not apply. The
normalized Gram matrix is dominated by the single largest U-jump (asymptotically rank one),
so beta is identified by bulk averages — and for alpha <= 2 the transformed regressors have
infinite *mean*, making even those averages single-jump-dominated and random. Transition is
gradual (alpha=2.2 degrades slowly); don't claim a sharp threshold without proof.

Takeaway for the paper: bridge/moment methods fail here not by misspecification but because
they *average over* extremes; the certificate estimator conditions on the extreme and stays
at ~1.00 across all alpha (with `demean=False` — note `fit_or_null` currently forces
`demean=True`, which inverts the QuantileBall safeguard in infinite-mean regimes and must
be fixed before final certificate runs).

Reproduce: session scratchpad scripts `verify_bridge.py`, `verify_bridge_threshold.py`
(2026-07-31), or refit `proxy_valid_bridge` in `nonlinear_compare_to_gmm.py`.
