# Prepared student-state label collection — no physical run

This directory contains code preparation only. No physical integration, optimization or actual collection has been executed. The collector is an option for a newly registered experiment after the parent inspects the fixed3000 results; it is not a completed recovery controller or admitted dataset.

## Exact proposed timing

Use the unchanged full-body Core, finite42 PD, native12.5µs, target1ms and decisions5ms. Settle1000ms at zero rate, then original conditional teacher with neutral reference for4800ms. Actor base128 state primes on every actual observation, regardless of which action is applied. Thereafter collect4800ms as24 periods:25ms pure student (five decisions) followed by175ms conditional teacher recovery (35 decisions). No body/actuator/history/hidden reset occurs at teacher withdrawal or burst boundaries.

Construct the conditional teacher exactly once after settling, using speed_scale1 and yaw request+.008 or−.008, prime_s4.8. Query it on the actual current state every decision, including student bursts; its private IK/desired-state history therefore evolves on every query at the declared clock, including queries whose command is not applied. Save all labels, applying only recovery/prime labels. A label is the current teacher suggestion on that actual state, not proof of recoverability or a future optimal action. The teacher's reference schedule ramps the separate command smoothly for1s after prime. Policy sees actual238 plus command2 only; teacher phase is separately stored provenance, never policy input. Teacher calls therefore persist in collection and this is explicitly training-only mixed control, never an autonomous walking pass.

Full expected output:10600 physical1ms endpoints,1920 decisions (960prime,120student,840recovery), teacher queried1920 times, neural state advanced1920 times. Source counts:0=1000settle,2=4800prime,3=600student,5=4200recovery endpoints.

## Execution guard and future fixed-run registration

Importing `collect.py` does not load Torch, MuJoCo or the physical backend. An actual launch requires both `--run-diagnostic` and a separate future `--run-plan` whose status is `REGISTERED_FOR_ACTUAL_DIAGNOSTIC`, `draft_sha256` equals the current `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/recovery_data/PREREG_DRAFT.json` SHA, and `runs` contains the exact absolute checkpoint path, SHA, condition and seed. No such registered plan has been created here.

Example future plan entry:

```json
{"checkpoint":"/ABS/checkpoint_003000.pt","checkpoint_sha256":"EXACT_SHA256","condition":"left","seed":20261110}
```

Future CLI, only after parent registration:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/recovery_data/collect.py --checkpoint /ABS/checkpoint_003000.pt --sha256 EXACT_SHA --condition left --seed 20261110 --run-diagnostic --run-plan /ABS/fixed_run_plan.json --out /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T203537Z/recovery_data/NEW_OUTPUT
```

Only left/right are accepted; speed085/speed090 are deliberately rejected, since no speed-collection execution is registered. Fixed global3000 is enforced in the loaded payload. Seeds initialize RNG but do not add physical perturbations or establish independent initial-state coverage.

## Evidence and boundaries

`SAMPLES_5MS.npz` records actual before-command X238, command2, teacher42 label, student mean/action, actual applied action, before/after128 hidden, source and separately labelled phase. `trace_1ms.npz` uses the existing full physical trace. `BOUNDARIES.json` persists before/after state hashes at withdrawal and each burst switch. All controller states remain continuous; the cycle observer alone resets at the start of the collection diagnostic. `IDENTITY.json` binds the original runtime/compiled arrays, actual source modules, frozen teacher, selected checkpoint and the future fixed-run plan.

Stop after100 consecutive actual ms of COM<.8 or nonfoot>.5W; exceptions preserve all available arrays in finally. If a failed step throws, the attempted decision may outnumber completed endpoints; `FAILURE.json` states both rather than fabricating completion. NaN/invalid output is rejected by existing bounded-action/no-external checks.

Whole collection safety admission requires complete10600 endpoints/1920 decisions, no exception/stop, strict nonfoot0, footloadmean≥.90, COMmin≥1.20 and original slip≤.5. Original full five gates are still reported independently. Cycle events are diagnostics, not admission proof for tiny bursts. Source-stratified summaries concatenate disjoint endpoints with gaps in elapsed time, explicitly labelled diagnostic; never interpret these as continuous autonomous windows.

`RECOVERY_ENDPOINTS.json` records raw state/load endpoints after each student and recovery segment. It does not establish improvement: differing segment durations, teacher action and absence of a matched no-recovery counterfactual prevent that inference. Parent must independently verify any actual collected data and recovery effects before deciding whether to train on it. Unsafe labels may remain for evidence but are never silently called valid data.

## Offline verification

`check_prepared.py` validates exact schedules/counts, future-plan binding and guards, whole-window admission rejection cases, pure mock recurrent continuity, and AST call sites confirming one actor/teacher call per decision with no in-loop hidden reset. It does not construct Core or load a neural checkpoint. `OFFLINE_CHECKS.json` records all source hashes, zero physical/optimization steps and the precise limitation. These checks cannot establish real integration correctness before a registered physical diagnostic.
