# Fixed8000 RH-weighted readout physical evaluation — code preparation

All files in this directory are a new scoped evaluator binding. No old evaluator, trainer, weight, body or result file is modified. Preparing or importing these modules performs no physics integration. Actual execution remains guarded by final-checkpoint identity, successful strong READINESS and explicit parent `--training-complete`.

The root experiment is conditional on its registered readonly feasibility test. If feasibility fails or fixed8000 training is not completed, this evaluator stays unused and does not establish any qualification.

## Strict model boundary

The only candidate path is:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/train/TRAIN/checkpoint_008000.pt`

The lazy public loader is:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/train/model_io.py`

Its `load_checked(path, sha, require_final=True)` must return the original `ReadoutMotor` and authenticated payload with final curve step8000, config `physical_eval_only_step=8000`, source/data hashes, original frozen motor and unchanged positive readout architecture. No loss-mask, load threshold, phase, time, food coordinate or training metadata is appended to the deployment observations. The RH before-action mask is training loss metadata only.

All authenticated config `source_hashes` and `data_hashes` paths, including the loss-mask evidence sidecar, are added to the actual run identity. Their existence is checked before Core construction. All actual evaluator source paths—including the existing root PREREG—are checked before Core, preventing missing-source path errors after physics starts.

## Fixed physical protocol and criteria

Exactly three conditions: neutral, left and right. Each is a fresh physical episode with1000 zero-rate settle milliseconds,4800 actual algorithm-teacher priming milliseconds, and9600 pure neural milliseconds. The neural command is zero during priming, then the same fixed one-second smoothstep to left+1 or right−1; forward remains0. Explicit128+64 memory remains continuous through withdrawal, with no root/state/pose corrections.

The unchanged original complete71 opaque mesh body uses42 finite-PD actuators, native12.5µs physics,1ms target integration and5ms neural decisions. Adhesion and applied extra forces remain zero. The full9.6s window, including the ramp, must pass original five engineering gates, all six sensed cycles, strict nonfoot zero and appropriate signed whole-window yaw≥.01rad. No trim, denominator change or gate relaxation occurs.

Neutral and right must also reproduce the already qualified c6b1500 motor's actual complete physical/neural/geometry arrays bitexactly. The reference remains:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/EVAL_right/`

with its independent audit:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/VERIFY_right.json`

Old and new source identities are retained separately; array equivalence never pretends that evaluation sources are identical.

## Preparation and launch

Structural/code checks, requiring no final model, have been executed successfully:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/evaluation/check_prepared.py
```

They write `CODE_CHECKS.json`; `READINESS.json` remains false until the actual final model passes strong checks. Once the parent provides the final SHA:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/evaluation/check_prepared.py --sha256 FINAL8000_SHA
```

This replays all2880 actual saved neutral/right decisions with the new final checkpoint, validates all192 hidden dimensions, strict8000 path/step guards, command formulas, scoring/physical-loop AST, full reference comparator and intentional hidden-state corruption rejection. After the LAST source edit, rerun this actual check, independently compare every readiness source SHA, and then freeze all Python/JSON files before physical execution.

Parent-only actual launch, after explicit authorization:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T031057Z/evaluation/pipeline.py --training-complete --sha256 FINAL8000_SHA
```

It runs actual neutral plus independent saved audit first; only a passed and audited neutral allows left/right in at most two CPU-single-thread processes. All outputs are exclusive new `EVAL_neutral`, `EVAL_left`, `EVAL_right`, `VERIFY_*.json`, and `FINAL_SUMMARY.json` under this directory. Partial runs persist saved endpoint/decision evidence and failure details; HANDOVER hashes are written before the first autonomous step.

An individual evaluator accepts optional `--native` if the parent later explicitly requests a native record. Recording native data alone is not full native acceptance.

## Independent actual audit

`verify_actual.py --run ABS_RUN --out ABS_NEW_JSON` loads each saved archive once, reconstructs all2880 checkpoint means, actions and both memory chains, checks15400 endpoints/source phases/clock/previous actions/held controls/forces/runtime geometry and original prefix, recomputes whole9600-window summary and cycles, yaw and eight observable HANDOVER hashes, and checks full right-parent preservation. It constructs no Core and performs no integration.

The physical and scoring code is reused from SHA-pinned immutable predecessors. Changes are explicit final8000 identity/path metadata, source/data evidence capture and pre-Core existence guards. The actual simulation/inference loops, numerical body, actuator, command generation and score functions remain unchanged.

## Scope

A passed fixed-condition test is engineering evidence only. Speed control, command switching, initial-state holdouts, food approach/arrival and sensory-dependence controls, ingestion/satiety, biological/connectome validity and formal native D1–D8 certification remain separate pending objectives. This delegation prepares code only and does not claim any actual physical run or motor pass.
