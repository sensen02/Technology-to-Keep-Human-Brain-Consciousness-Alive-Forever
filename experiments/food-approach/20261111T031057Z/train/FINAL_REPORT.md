# Conditional RH stance-weighted readout continuation — training complete

This task performed read-only feasibility analysis and, because the preregistered rule passed, the fixed 2000→8000 continuation. No physical trajectory or food behavior was executed by this task.

Final checkpoint:
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/train/TRAIN/checkpoint_008000.pt`

SHA-256: `65e3635ec966323f22ce30b18f74cef0ccb6445499dbed0e6d1264dce75d15ba`

Public loader:
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/train/model_io.py`

`load_checked(checkpoint, sha, require_final=True)` returns the SAME immutable legacy `ReadoutMotor` architecture and a payload containing `format_version`, `identity`, `state_dict`, `optimizer`, `config`, `curve`, and `rng`. Config kind is `fixed_RHstance_weighted_readout8000`; `physical_eval_only_step` is 8000. Intermediate 5000 is accepted only with `require_final=False` for integrity verification, never physical selection.

## Read-only feasibility, fixed before optimization

Root and train preregistrations fixed the criterion before analysis: post-prime RH-loaded squared-action-error energy must be at least 0.01 of total post-prime error energy, and the weighted gradient must be finite with norm at least `1e-10`. Otherwise no training was allowed.

Results (`FEASIBILITY.json`, managed `bash-11`, exit0, collected):

- RH-loaded energy fraction: **0.09186122344193763**.
- Weighted head-gradient norm: **5.253824328843305e-5**.
- Uniform/weighted gradient cosine: **0.3707201816698251**.
- Initial uniform action MSE: **8.720976209497367e-5**.
- Initial weighted objective: **7.716301624602138e-5**.
- Explicit nested-row and NumPy weighted-loss calculations agreed with Torch exactly, maximum error **0**.
- All five feasibility checks passed. Optimizer updates and physical integration were both zero.

This establishes that the selected loss intervention is meaningful and computable under the declared rule. It does not establish that weighting will improve physical slip, nor that uniform MSE caused the failure.

## Actual before-action load alignment

Every one of the 2880 decision timestamps is matched to the same teacher's saved physical endpoint at index `999 + 5*k`. That is the endpoint BEFORE the action, including row999 at the first decision after 1s settle. Both nominal decision time `1+k*.005` and saved physical/sample timestamps are checked within `1e-8`.

Actual frozen physics order is `LF,LH,LM,RF,RH,RM`, so RH load is column4. Output rows28:35 are bound to the actual seven RH joint names. The mapping is additionally anchored to the known sealed earlier runtime-inspection manifest by `TRAIN/MAPPING_ANCHOR_CHECK.json`; no Core or integration was needed here. Frozen native `physics.W` supplies the threshold `.1*W`.

`TRAIN/LOSS_WEIGHTS.npz` and `TRAIN/ALIGNMENT.json` bind the actual mask, before-action indices, weights, W, clocks and trace SHA. Weight is 4 only for RH's seven action errors when the actual before-action load is at least 0.1W, and 1 elsewhere. Prime960 rows are excluded. The objective is the sum of weighted squared bounded-action errors divided by the sum of actual post-prime weights. Neither the mask, source, teacher phase, nor time enters the actor or resets its hidden state.

The immutable predecessor prepares full X/U/Y and causal hidden-state caches. TRACE is read once per preparation invocation. The predecessor reads the sample fields once and this adapter performs a separate bounded timestamp-field read; there is no NPZ decompression inside any optimization or neural-decision loop.

## Production and gradient preflight

`PREFLIGHT.json` (`bash-12`, exit0, collected) passed **15/15** checks:

- Full actual 2880-row production means versus cached means within `1e-12`.
- Both causal baseline128 and residual64 histories exact.
- ALL head-weighted gradients versus full production sequence within `1e-12`; maximum error **5.251604272976662e-20**.
- Full production weighted objective matches cached objective.
- Head gradients finite and nonzero; frozen parent receives no gradient.
- Neutral and right full2880 means/actions/both states exact.
- Original direct target-position Jacobian remains the exact negative gain diagonal.

No optimizer update was performed by this preflight.

## Fixed training and saved-state verification

Managed `bash-13` ran fixed **6000** updates, global steps2001…8000, and completed exit0; output was collected. Updates consumed **20.30494s**, overall **27.86595s** including independent save checks.

The actual parent2000 model, two Adam states with counters2000, moments, RNG, and full recorded curve1…2000 were restored exactly and persisted to `TRAIN/RESUME_STATE.pt`. Only optimizer lr changed from `.003` to `.001`; the loss objective changed as separately declared. Architecture, 2730 trained head parameters, 194878 frozen motor parameters, bounds, positive ReLU gate, actual left-teacher data and both continuous memories remained unchanged.

Final metrics:

| Metric | Initial2000 | Final8000 |
|---|---:|---:|
| Weighted post-prime objective | 7.7163016246e-5 | 7.2952978538e-5 |
| Uniform post-prime action MSE | 8.7209762095e-5 | 8.3590677012e-5 |

Save5000 and save8000 each passed **23/23 fresh-process checks**: weights and actual optimizer reload, both counters and moments, parameter shapes/order/coverage and hyperparameters, RNG restore roundtrip, full contiguous curve with exact2000 prefix, all194878 frozen parameters, all2880 neutral/right means/actions/192 states, exact fixed mask and before-action alignment, full left production/cache agreement and checkpoint immutability.

Live weights, optimizer, RNG, config and complete curve were compared exactly against the saved payload before the fresh process. The live RNG was restored after constructor-based checking. Only final8000 is eligible for the parent's physical evaluation.

All top-level train Python/JSON identity sources are now frozen. The final report is Markdown only and is outside those captured source globs. There are no active jobs or temporary unit directories.

## Metadata and limits

New config records immediate parent2000 `eeced...` and the frozen motor1500 `c6b...` separately and accurately. The ancestor-path metadata error in the earlier original2000 payload was not rewritten; its external `PARENT_METADATA_CORRECTION.json` is hash-bound. The inherited curve begins at1; no fictitious step0 metric was introduced.

This candidate changes loss weighting and learning rate together. No equal-budget continued-uniform control is run in this trial, so even a later improvement cannot establish a weighting-only causal effect. Lower teacher-distribution loss and saved-state integrity are NOT physical motor qualification. Original all-five gates, all-six cycles, strict nonfoot zero, full 9.6s scoring window and yaw criterion remain mandatory in the parent's actual neutral/left/right tests. No speed, food approach, ingestion, satiety, connectome or biological validity claim is made.
