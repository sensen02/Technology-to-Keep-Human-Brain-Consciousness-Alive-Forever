# Final5000 nonlinear positive readout: prepared physical evaluation

This directory is code preparation only. It does not execute training or physical integration on import. Actual execution belongs to the parent after the final5000 checkpoint SHA and frozen `READINESS.json` have passed the actual final-model preflight.

All paths below are absolute. The model loader is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/train/model_io.py`, API `load_checked(checkpoint,sha,require_final=True) -> NonlinearMotor,payload`. The one eligible model is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/train/TRAIN/checkpoint_005000.pt`. The loader checks source/data identities and final step5000; evaluator independently enforces exact path and curve/config step.

The evaluator reuses the SHA-pinned immutable physical/scoring implementation from `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/evaluation/`. Only checkpoint-step/path, descriptive nonlinear-model metadata and a pre-Core source-path existence guard are changed. Actual new and ancestor source identities remain distinct. The c6b1500 right-turn reference stays fixed and is not replaced by the immediate eeced2000 frozen neural parent.

Actual cases are neutral, left and right. Every case retains1000ms zero settle,4800ms actual original teacher priming and9600ms pureANN; no body/actuator/128+64 hidden-state reset occurs at withdrawal. The high-level turn command follows the same fixed1s smoothstep as the qualified teacher records, and forward input stays0. All original five engineering gates, strict nonfoot force0, all six sensed cycles and proper signed whole-window yaw≥.01rad remain required. No initial window is trimmed. The full71opaque-mesh body,42 finite PD actuators, native12.5µs,1ms target integration and5ms neural decisions remain unchanged; adhesion, extra forces and root writes are forbidden.

`evaluate_motor.py` immediately persists HANDOVER hashes, saves all15400 endpoints and2880 decisions with full192-state before/after chains, and preserves partial records on exceptions. Geometry includes actual head/root yaw and audited sampling-point poses. Optional `--native` records the original native stream but does not itself grant complete native D1–D8 certification.

`verify_actual.py` loads each NPZ once, constructs no `Core`, and performs no physical integration. It replays every actual checkpoint mean/action/dual-hidden update, checks full source/time/action/previous-input alignment, all phase counts, actual zero extra forces, original prefix, compiled geometry/71opaque identity, complete whole-window summary/cycles/yaw and derived HANDOVER hashes. Right-turn measured body/neural/geometry arrays must all match the independently passed c6b1500 reference. Neutral arrays must reproduce the original walking baseline.

Before launch, run the real final checkpoint preflight:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/evaluation/check_prepared.py --sha256 FINAL5000_SHA
```

This verifies actual5000 neutral and right production-sized full2880 inference, tuple dimensions, frozen197608-parent/new22378-MLP counts, source paths beforeCore, wrong-checkpoint rejection, teacher commands, unchanged scoring and a deliberately corrupted right-state rejection. Every source edit requires rerunning this check; do not update hashes alone. No physical model or optimizer is constructed by these checks.

Parent-only actual launch:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/evaluation/pipeline.py --training-complete --sha256 FINAL5000_SHA
```

The pipeline first evaluates and audits a new neutral run, then runs mirrored turns with at most two CPU-single-thread processes. Outputs are fresh `EVAL_neutral`, `EVAL_left`, `EVAL_right`, `VERIFY_*.json` and `FINAL_SUMMARY.json` within this directory. Existing outputs are never overwritten.

These are fixed-condition replacement-ANN engineering checks. Even all three passing does not establish switching, arbitrary initial conditions, speed control, food approach, feeding, satiety, original connectome or biological validity. Old failed left and ancestor metadata correction records remain unchanged. Training teacher fit and offline saved-array invariance cannot substitute for actual physical results.
