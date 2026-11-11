# Read-only feature-conditioning diagnosis

Inputs: the frozen qualified original four teacher episodes and fixed global3000 checkpoint `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/continuation3000/TRAIN/checkpoint_003000.pt`, SHA-256 `17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8`.

`analyze.py` loads each episode's NPZ once through the existing authenticated `fast_data.prepare`, computes the original frozen GRU history causally, and never constructs a physical Core, optimizer, training update, or new candidate checkpoint. Only the original qualified four episodes are used; rejected mixed data224/225 are absent. The full saved sequence comprises11520 rows; statistics use only the7680 postprime TRAIN rows. No holdout claim is made.

Managed job `bash-231` completed with exit0 and was collected. Six numerical equivalence checks passed. Full per-feature statistics, group contributions, correlations, parameter-gradient norms, per-case activations and before/after input hashes are in `DIAGNOSIS.json`.

## What the measurements say

There is strong feature-scale heterogeneity, but **widespread tanh saturation is not observed** on this teacher distribution:

| Measurement | Encoder tanh | Hidden tanh |
|---|---:|---:|
| Fraction absolute activation >=.95 | .0180613 | .0446564 |
| Fraction absolute activation >=.99 | .00275777 | .0125580 |
| Mean local tanh derivative | .764524 | .726291 |

These measured fractions do not establish that saturation causes the actual physical slip. Normalization alone cannot change these current activations when the transform preserves the function.

The hypothesis that the base hidden variation is tiny relative to physical inputs is contradicted here:

| Group | Median feature std | Encoder weight Frobenius norm | Centered encoder contribution variance, sum over128 neurons |
|---|---:|---:|---:|
|112 physical features|.0192142|23.8161|1.38264|
|128 base-hidden features|.744271|6.07048|43.0356|
|2 normalized command features|.425982|3.39160|3.64152|

Physical features have a wider spread internally, including one constant channel and many small-variance channels. The residual learned a larger encoder weight norm on that block. Base hidden variation dominates the centered encoder drive on these saved teacher states. The physical-versus-hidden contribution cosine is.088666, so the dominance is not simply a large exact cancellation between those groups. The sum of separate group variances is48.0598, versus49.5107 for their actual sum; cross terms are explicitly retained in the report.

The full feature correlation matrix remains nontrivial: absolute pair correlation median.217422, p95.741735,114 distinct pairs above.95. Diagonal normalization will not whiten this representation, so reduced variance spread is not proof of a well-conditioned full covariance or Hessian. No covariance condition-number claim is made.

## Exact function-preserving standardization test

The tested transform is:

```
mu = mean(TRAIN4 postprime features)
s = max(std(TRAIN4 postprime features), .01)
z = (f - mu) / s
W_normalized = W_original * s
b_normalized = b_original + W_original @ mu
```

All downstream weights, the gate, base network, target inhibitory connections and command normalization remain unchanged. Nothing is optimized or saved as a candidate.35 features use the declared.01 floor; one has exactly zero measured variance.

Across **all11520 saved rows**, including priming rows excluded from fitting statistics:

- Largest residual-output difference: `1.609823385706477e-15`.
- Largest encoder-preactivation difference: `5.773159728050814e-15`.
- Largest gradient difference with respect to the same original feature coordinates: `6.217248937900877e-14`.
- Full TRAIN postprime bounded-action loss is identical in float64: `.0005453285125308438`.
- Encoder parameter-gradient chain-rule comparison has maximum error `3.670127586003879e-18`.

All are below the predeclared1e-12 tolerance. This is numerical equivalence, not a bitexact statement. Original-input gradients use the differentiable scalar sum of squared residual outputs across every row, with the recorded gate held fixed in both parameterizations. They check this full feature-to-output mapping; they are not the full closed-loop sensory Jacobian or a claim about perturbing command and recomputing its gate.

The **positive feature-variance diagonal** changes as follows:

| Statistic | Raw | Standardized with floor.01 |
|---|---:|---:|
|max/min positive variance|9,004,341.94|986.308|
|p95/p05 positive variance|24,322.37|3.09031|
|constant features|1|1|

The transformed encoder weight-gradient L2 norm is.000302843, versus raw.0000800557; bias-gradient norms remain.00000294111. This is a change of parameter coordinates, so a larger gradient norm does not itself mean faster or safer learning. The report includes every feature-column gradient norm and correct chain-rule mapping back to the original weights.

## Recommendation

A **controlled normalization experiment is justified by measured heterogeneous feature scales**, rather than by saturation or a lack of hidden-state amplitude. It is a reasonable next fitting-conditioning hypothesis after the existing physical failures; it is not evidence that the teacher-fit or real foot slip will improve.

If the parent proceeds, keep the same qualified original four training episodes, actual sensory interface, base freeze, model width, gate/bound, training budget and final-only physical selection. Derive mu/std exclusively from the TRAIN4 postprime rows and freeze them. Initialize the standardized model using the tested function-preserving algebra and recheck original-input outputs and gradients, with all actual physical failures retained.

Optimizer handling must be declared explicitly: Adam first/second moments cannot simply be copied unchanged into these new parameter coordinates, particularly with the bias/weight coupling from centering. A fair test can compare raw and standardized parameterizations from the same represented function with equally fresh optimizers and the same preregistered budget; or it must provide and verify a justified moment-coordinate transformation. This audit authorizes neither approach and performs neither. Do not silently call it an exact Adam continuation.

Keep the exact zero neutral gate and direct target-position inhibition. Statistics and arrays are engineering training artifacts, not biological receptor calibration. The analysis cannot identify a causal source of actual slip, cannot rescue rejected teacher data, and cannot establish a usable command range, food approach, or neural deceleration.

## Reproduction

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/feature_conditioning/analyze.py
```

The script hashes all captured training sources, data and checkpoint before and after analysis. No earlier source was edited. No temporary files or outstanding jobs remain.
