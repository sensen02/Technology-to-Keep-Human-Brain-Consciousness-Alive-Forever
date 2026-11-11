# Prepared six-case steering qualification — CODE ONLY

All files are confined to `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/steering_holdouts/`. No physical integration, optimization, neural model load or Core construction was executed during preparation. The final offline check passed **67/67**, exit code 0, without importing Torch/MuJoCo/physics/student_wave.

The completed nonlinear5000 candidate currently fails left slip. Therefore these tests **cannot be launched for that candidate**: the actual failed summary was tested and rejected by the qualification guard. This is prepared future machinery, not a new registration or claim of qualification. A changed later candidate (including the parent's 024558 candidate) is also rejected by the exact nonlinear5000 path/loader boundary; using it requires an explicit new loader/ancestry/qualification amendment and fresh source identities, not automatic substitution.

## Fixed cases and timelines

Exactly these six cases are declared in `PREREG_DRAFT.json`:

| Case | Neural turn stimulus | Initial motor pulse |
|---|---|---|
| halfleft | 0→+.5 over first1s | none |
| halfright | 0→−.5 over first1s | none |
| left_to_right | 0→+.5 first1s, +.5→−.5 smoothly from4.8s to5.8s | none |
| right_to_left | mirror | none |
| pulse_left | 0→+1 over first1s | seed20261112, normalstd.02, clipped42commands, held20ms |
| pulse_right | 0→−1 over first1s | seed20261113, otherwise same |

Forward command is structurally zero. Profiles are test inputs to the neural actor, not gait/leg sequences or deployed clock inputs. Pulses use the unchanged finite42 motor-command interface; no qpos/root writes, external forces or adhesion are introduced. For pulse cases, the pulse occurs BEFORE1s actual zero settling. The teacher is constructed only afterwards, calibrating rest to the actual physical state. This preserves the required order of pulse→settle→4.8s teacher prime→9.6s ANN.

Normal trajectories have15400 actual1ms endpoints; pulse trajectories15420. All have2880 actual5ms decisions (960prime+1920ANN). Model states128+64 initialize once before priming and remain continuous across withdrawal and the4.8s test midpoint. The midpoint snapshot is saved for every case, and it is the true command-switch start for the two switching cases. HANDOVER and midpoint before/after hashes are saved immediately in `BOUNDARIES.json`; no state reset occurs.

All original five gates, six-leg cycle completion and strict nonfoot0 apply to the full9.6s autonomous window. Half-turn signed whole-window yaw must be≥.005rad; switching requires each4.8s signed yaw≥.002rad including transition; pulse full turns require≥.01rad. Original world-X forward and slip denominator remain unchanged; no first-second trimming or additional short-window gate exists.

## Fail-closed authorization and qualification

A future launch requires BOTH an explicit `--launch-authorized` and a separately created `--run-plan` JSON. No RUN_PLAN was created here. The required plan schema is:

```json
{
  "status": "REGISTERED_FOR_ACTUAL_QUALIFICATION",
  "draft_sha256": "SHA256 of PREREG_DRAFT.json",
  "checkpoint": "/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/train/TRAIN/checkpoint_005000.pt",
  "checkpoint_sha256": "EXACT_SHA",
  "qualification": "/ABS/three-condition-FINAL_SUMMARY.json",
  "cases": ["halfleft", "halfright", "left_to_right", "right_to_left", "pulse_left", "pulse_right"]
}
```

The same-checkpoint qualification must have exactly neutral/left/right, `all_conditions_eligible=true`, `saved_trajectory_verification=true`; each condition requires actual engineering and independent audit pass flags, all audit checks, exact checkpoint path/SHA and result/audit hashes. Saved trace/neural/geometry/runtime hashes and evaluation sources are rechecked. Actual original runtime identities are verified before physical construction. The new strict model loader independently checks training source/data/ancestry. The current nonlinear5000 summary fails these requirements and is rejected before backend/model import.

An already qualified neutral is bound as evidence; no redundant neutral physical run is added. Unperturbed profiles independently compare all5800physical-prefix endpoints and960neural-prime rows with that neutral. Pulse cases intentionally have a different prefix, so the auditor verifies the seeded pulse, exact20ms count, full clocks and phase/source counts rather than falsely claiming prefix equality.

## Future CLI — not executed

One case:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/steering_holdouts/evaluate_holdout.py --launch-authorized --qualification /ABS/qualified-summary.json --checkpoint /ABS/final5000.pt --sha256 EXACT_SHA --run-plan /ABS/fixed-plan.json --profile halfleft --out /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/steering_holdouts/EVAL_halfleft
```

After a registered actual run, independent audit:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/steering_holdouts/verify_saved.py --run /ABS/EVAL_halfleft --out /ABS/VERIFY_halfleft.json
```

Exactly-six-case managed runner, at most two CPU1 subprocesses at once:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/steering_holdouts/pipeline.py --launch-authorized --qualification /ABS/qualified-summary.json --checkpoint /ABS/final5000.pt --sha256 EXACT_SHA --run-plan /ABS/fixed-plan.json
```

The example's checkpoint must satisfy the exact path boundary stated above; arbitrary `/ABS/final5000.pt` is a placeholder and will be rejected.

## Evidence, verification and limits

`evaluate_holdout.py` records the unchanged physical trace, all actual observation238/command2/mean42/policy/applied actions, flattened before/after192hidden states, sources and actual yaw. Source hashes and future plan/qualification digests bind the run. The original model runtime and all71 opaque meshes are checked. Teacher calls occur only during960 priming decisions, and the decision inputs never include elapsed time or source metadata beyond the supplied neural command stimulus. Exceptions save available rows and a FAILURE record; no incomplete trajectory can qualify.

`verify_saved.py` once-loads archives and independently replays all2880 actual checkpoint outputs/actions/both memory states, checks stored chain/action alignment/previous-action inputs/held actions/source counts/clocks/pulse/zero applied force, compiled71mesh arrays, whole9600sample summary, full sensed cycle events and yaw criteria. It derives eight observable boundary hashes at withdrawal and midpoint from actual saved arrays. It imports the frozen physics summary implementation but constructs no Core and performs no integration. Raw endpoint checks are engineering evidence; full native contact/force audit remains separate.

Offline tests cover all schedules, exact counts, mirror/switch continuity and endpoints, seeded pulse reproducibility, proper and zero-yaw controls, invalid schedules, launch-plan guards, actual failed qualification rejection, AST one actor-step/teacher-call sites and no in-loop hidden reset. These tests do not establish actual collector integration correctness or robustness. Source hashes are in `OFFLINE_CHECKS.json`. No physical test, stochastic-policy training, food source, ingestion or satiety was run or claimed.
