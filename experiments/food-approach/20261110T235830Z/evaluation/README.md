# Round2 fixed1500 slow-residual evaluation — CODE ONLY

These files prepare actual evaluation and independent saved-evidence verification. No physical simulation or training has been executed by this preparation task.

## Fixed interface

The evaluator loads only `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train/TRAIN/checkpoint_001500.pt`, with its explicit SHA-256, through the actual new `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train/model_io.py` API `load_checked(checkpoint,sha,require_final=True)`. It also checks actual `payload['curve'][-1]['step']==1500` and `config['physical_eval_only_step']==1500`. Architecture remains the original `ResidualMotor`: explicit `(base_hidden128,residual_hidden64)` tuple. Both states continue through preheat/withdrawal without reset. Flattening to192 is recording only; inference continues to receive the tuple.

`evaluate_motor.py` handles exactly `neutral`, `left`, `right`. Forward modulation is always0. Turn input follows the identical one-second smoothstep used by qualified teacher data. The entire9.6s autonomous window, including ramp, is scored using unchanged original all5 gates, all6 sensed cycles and strict nonfoot force0. Left signed whole-window yaw must be≥.01rad; right≤−.01rad. Neutral has no yaw criterion.

Timing is actual1s zero settle +4.8s original teacher/neutral sensory neural priming +9.6s pure ANN, with the original full71opaque-mesh body,42finite-PD motors, native12.5µs,1ms target integration and5ms decisions. No extra force, adhesion, root-pose writes or teacher calls occur during autonomous inference.

## Execution after parent authorization

Preparation does not authorize execution. After training completes and the parent binds the final SHA:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/pipeline.py --training-complete --sha256 EXACT_FINAL1500_SHA
```

Pipeline first runs neutral plus its independent audit. Only if both pass does it run left/right with at most2 CPU-single-thread processes. No checkpoint shopping or additional profiles. Results are fresh `evaluation/EVAL_neutral`, `EVAL_left`, `EVAL_right`; audits are `VERIFY_*.json`; final status is `FINAL_SUMMARY.json`. A failed neutral is explicitly recorded and mirrored turns are not run.

Individual evaluator:

```text
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/evaluate_motor.py --training-complete --checkpoint /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train/TRAIN/checkpoint_001500.pt --sha256 EXACT_SHA --condition left --out /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/EVAL_left
```

Optional `--native` records original native contact chunks for a separately selected final confirmation. Recorder completion and chunk hashes are audited, but this alone does not perform complete original D1–D8/native physical certification.

Auditor:

```text
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/verify_actual.py --run /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/EVAL_left --out /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/VERIFY_left.json
```

## Evidence

HANDOVER.json persists before the first autonomous tick, including both recurrent memories in192 logged dimensions. Complete endpoints and decisions are saved in finally; errors preserve available arrays with `FAILURE.json`, never manufacture a completed run.

Actual source identity includes new evaluator/auditor/pipeline/prereg/preparation checks, `READINESS.json`, the actual new loader and top-level training Python/JSON files, parent ROUND2_PLAN, old head geometry audit and all imported old physical modules. Runtime geometry and every compiled model array are bound through the original `identity.capture/verify_runtime`; source hashes are rechecked after execution. Old sources are never copied and silently relabeled.

`trace_1ms.npz` stores full15400 endpoints and actual control sources. `neural_decisions.npz` stores all2880 actual pre-command physical observations, commands, means, policy/applied actions,192 before/after memories. `geometry_1ms.npz` stores measured thorax/head yaw, actual Head pose, and diagnostic head-fixed engineering sensor points. These points are not policy inputs or anatomical receptor claims. All71mesh opacity is checked.

The independent auditor loads each archive once, constructs no Core and performs no physical integration. It replays all2880 actual checkpoint means/actions/both hidden states, chains and source/previous-action/hold alignment; reconstructs eight observable handover hashes including192 hidden; recomputes all9600 whole-window summary and cycle events; checks original5800 physical prefix and960 neural prime; checks actual saved-quaternion thorax yaw and head geometry. Neutral compares all shared15400 original physical arrays plus2880 mean/action/observations and base128 hidden bitexactly; residual64 has no old counterpart and is checked against actual checkpoint inference instead.

## Offline preparation verification

`check_prepared.py` tests parser/launch guards, JSON/scoring rejection cases, complete2880 command formula versus both actual qualified teacher streams, original tuple-state flatten/split equivalence across both2880 input streams, and neutral invariance across2880 inputs with a nonzero residual memory. It verifies actual15400 quaternion yaw against old saved records. It loads the immutable old500 checkpoint solely for these offline architecture checks and never constructs Core, imports MuJoCo or runs an optimizer. This does not test the not-yet-trained1500 checkpoint or establish physical success. `READINESS.json` binds final frozen preparation source hashes; actual execution refuses drift.

Engineering acceptance, saved-data integrity, original native certification and food behavior are separate conclusions. Even all3 fixed-condition engineering passes would still require later parent-registered command-switch and holdout tests before food deployment. No speed behavior is trained or certified here.
