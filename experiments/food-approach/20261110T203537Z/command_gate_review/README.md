# Read-only command gate algebra review

Only files in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/command_gate_review/` were created. Existing model, teacher data, corrective training and captured source identities were unchanged. No optimizer update, physics integration, candidate head training or model artifact occurred.

## Compared mathematical families

The existing residual is `tanh(v_forward² + v_turn²) * tanh(F(features))`, where `v_forward=u_forward/.15`, `v_turn=u_turn`, and the per-output latent bound is1. The hypothetical neural alternative is `u_turn*tanh(Fturn(features)) + (-u_forward/.15)*tanh(Fspeed(features))`, with two neural42-output heads. The actual four teacher datasets occupy separate command axes, so its per-output allowable latent radius is `|v_forward|+|v_turn|`. Both families preserve exact zero correction at neutral.

This script compares exact bounded-action intervals `[tanh(base_mean-radius), tanh(base_mean+radius)]` against actual teacherY. It does not assign a finite latent target to exact teacher±1 via atanh. A lower bound permits independently optimal head outputs at every input and therefore does not establish attainable finite-network fit, generalization or physical control.

## Whole9.6s post-prime action-MSE floors

| Actual qualified teacher | Existing norm gate | Hypothetical linear heads | Old/new outside fraction |
|---|---:|---:|---:|
| left_full | 1.46259e-6 | 1.18437e-7 | .017374 / .004142 |
| right_full | 1.28475e-6 | 6.93531e-8 | .017200 / .003832 |
| straight_085 | .000160050 | .0000609445 | .020784 / .005134 |
| straight_090 | .000168293 | .0000317505 | .027245 / .006796 |

The equal-episode floor changes from8.27725e-5 to2.32207e-5. Maximum unavoidable action errors change respectively from .050874/.050012/.599935/.499995 to .028795/.015342/.480783/.385438. Speed-related reachability restrictions remain even for the candidate.

These differences are partly a **range expansion**: full normalized command allows radius1 instead of tanh(1)=.761594; speed090 allows radius.666667 instead of tanh((2/3)²)=.417322. A future experiment cannot attribute any improvement solely to gradient conditioning if it also adopts these larger intervals.

## First-second scaling and neutral mismatch

`GATE_REVIEW.json` records first-second samples, 0–.2/.2–.5/.5–1s subwindows, later1–4.8/4.8–9.6s and twelve fixed.8s provenance bins. These windows are diagnostic only, never actor inputs or action scheduling.

At post.1s, full-turn ramp magnitude is.028: existing gate radius is only.000784 versus candidate.028. Actual left teacher/base bounded-action difference has RMS.009264 and max.027898. Existing per-row floor is7.55765e-5 versus zero for the hypothetical linear interval at that input. Right has analogous RMS.009483; speed085 RMS.009697. Speed090 at.1s has normalized command.018667, norm radius.0003484 versus linear.018667 and teacher/base max.028178; even the linear interval still has floor2.73686e-6 there.

Over the first.2s the old floor is5.36–6.25e-5; the hypothetical linear floor is3.09–5.41e-6. Thus the current quadratic suppression is particularly restrictive during small-command ramping, an algebraic observation rather than proof that it causes observed slip.

At the very first post-prime decision all four actual commands are zero. The verified base differs from teacherY with action-MSE2.2044051984e-5, RMSapproximately.00469511 and **maximum absolute difference.01534208**. The .0047 value is RMS, not maximum error and not the whole-window floor. Both neutral-preserving gates necessarily retain that one-row mismatch. Teacher is not an exact target for the already verified neutral actor; forcing exact teacher matching at zero command conflicts with strict neutral retention at these inputs.

## Derivatives and sign representation

The old norm gate is O(||v||²), so its command derivative at neutral is zero for bounded smooth heads. The candidate command derivatives at neutral can be nonzero: `d correction/d u_turn = tanh(Fturn(features0))`, `d correction/d u_forward = -tanh(Fspeed(features0))/.15`. This requires nonzero head outputs; identically zero heads also give zero command derivative initially. Gradients of head parameters from an exactly zero command remain zero in both families. Deterministic behavior-gradient bootstrap and policy gradients using nonzero stochastic command samples are distinct issues.

A signed linear turn branch with a command-independent head forces an odd correction. Desired corrections may contain even/symmetric components. Allowing the head to read command is a valid neural implementation, but the algebraic envelope alone does not prove that shared weights learn the required dependence.

A range-matched comparison could use signed coefficients `sign(v)*tanh(v²)` on each separate axis while giving heads the command; that retains the old envelope but still has zero derivative at neutral. For uniformly bounded smooth heads, exact O(v²) output radius and nonzero first derivative at0 cannot both hold. A controlled design should declare whether it tests sign parameterization, local slope, interval capacity or several at once. This review recommends no concurrent architecture change to the active corrective-data experiment.

## Reproduction and verification

Executed once with provided Python3.12 environment, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .../env/bin/python -B .../command_gate_review/review_gate.py`, exit0. It uses authenticated original four full2880-decision teacher streams, sequential frozen baseline hidden computation and actual bounded teacher targets; sample NPZ files are read once per dataset by `fast_data.prepare`. Source/data hashes are verified before/after through the original loader and included in the result. A temporary untrained FastMotor is constructed solely to reuse frozen-base cache conventions; no new head weights or model checkpoint are saved. Output is `GATE_REVIEW.json` and this report plus the script.

Conclusion scope: signed linear neural heads are a plausible separately registered parameterization/range experiment. The current evidence establishes only algebraic lower bounds and derivative facts, not improved learning, stable locomotion, food approach, biological validity or a fixed root cause.
