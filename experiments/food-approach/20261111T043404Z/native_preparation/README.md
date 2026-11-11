# Selected LEFT native preparation — CODEONLY, no physical result

This directory prepares one selected CEM3-RH-bias LEFT native recording, an independent current-record D1—D8 audit, and a separately authorized same-run renderer. Preparation does not construct `Core`, load a candidate neural model, integrate physics, or optimize anything. At handoff there is no selected candidate or native result. `actual_native_ready` stays false.

The method declarations are in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/METHODS_DRAFT.json`. They reuse the pinned original array-only native physical evaluator and the pure force/D7 diagnostic function. They do **not** reuse the old complete audit wrapper, which constructs Core, assumes a different checkpoint, and hardcodes128 neural state. The current actor has explicit128+64 state and2880 saved decisions.

## What is prepared

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/contract.py`: fail-closed same-candidate full neutral/left/right raw result/audit and immutable CEM selection/proposal admission. Uses only the standard library. It expects the parent's current FINAL_SUMMARY fields `selected`, `selected_checkpoint`, `selected_checkpoint_sha256`, `selected_left`, `selected_proof`, `confirmations`, `selected_all_three_pass`. Selected-null, partial engineering, mismatched SHA or missing artifacts cannot launch.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/execute_native.py`: unchanged parent's full physical rollout, native recorder enabled once. Recorder-only copied tags add actual source, neural command, logical/decision alignment and body rotation at every12.5us step. Before first integration, the actual compiled mass/gravity/W/heading/XML/friction/opacity snapshot is captured in the run identity. These diagnostics never enter the actor or control loop.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/verify_native.py`: no Core construction. Replays all2880 current-checkpoint means/actions/192-state chains, validates the complete15400 endpoints, forces/sources/cycles/handover, compares all logical arrays and HANDOVER bitexact to the selected engineering LEFT run, and verifies identical compiled physical/actuator array hashes. Then it recomputes current-record D1—D8, native gates, all-step yaw and current-body-heading diagnostics.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/render_selected.py`: separate authorized rendering after native PASS. Uses saved states only and disables `mj_step`, `mj_step1`, `mj_step2` before rendering construction. Requires71opaque meshes, identical compiled arrays, all770 decoded frames and in-frame silhouettes. No render runs during preparation.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/check_prepared.py`: syntax, frozen-source, schedule/count, no-backend, admission and missing-selection guards. These checks are code preparation, not a physical success or proof runtime wrappers are already exercised.

## Verified CODEONLY command

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/pipeline.py --prepare-only
```

The saved23-check evidence is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/CODE_PREPARATION.json`; its `actual_native_ready` is false. No registered `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/RUN_PLAN.json` is created by this task. The draft template is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/RUN_PLAN.template.json`.

## Future activation — parent only after selection and explicit authorization

The parent must inspect actual finalized CEM evidence and register the scoped RUN_PLAN, setting status `REGISTERED_SELECTED_LEFT_NATIVE`, exact selected checkpoint/SHA, original engineering summary hash, immutable SELECTED hash and all seven current code/method hashes from CODE_PREPARATION. The plan must bind the same neural SHA as all three actually passed engineering conditions. It is not enough to provide a passing LEFT score or a candidate ID. The actual candidate public loader is the parent's immutable CEM3bias model loader and authenticates proposal RNG/order, exact three scalar changes and every other tensor.

Then, with `SELECTED_CHECKPOINT_ABSOLUTE_PATH` and `SELECTED_SHA256` replaced by the actual registered values:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/pipeline.py --engineering-summary /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/FINAL_SUMMARY.json --checkpoint SELECTED_CHECKPOINT_ABSOLUTE_PATH --sha256 SELECTED_SHA256 --run-plan /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/RUN_PLAN.json --launch-authorized
```

This first independently replays the selected engineering LEFT's actual2880-neural sequence without Core. It writes fresh selected-native readiness and source barriers only after full admission, and runs exactly one native recording followed by independent audit. It refuses an existing selected_native directory and preserves failed logs/partial recording. It never records all optimizer candidates or silently retries. Use a managed background job and collect its exit code. A successful recording is not automatically a native PASS; native audit may fail or leave criteria UNKNOWN.

Expected future outputs:

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/selected_native/EVAL_left/native/`:602 compressed native chunks,2048 rows except final1152,1232000 actual steps. Full raw before/after qpos/qvel, contact forces and velocities, loads, COM, torque/targets and zero-force evidence; source/control/heading tags every native step.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/selected_native/EVAL_left/NATIVE_STATIC.json`: actual compiled contract captured before integration and hashed by the execution identity.
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/NATIVE_AUDIT.json`: current physical/neural conjunction, preserved frozen legacy D7 uncertainty, fresh conservative D7 sufficient proof, force sensitivities and exact engineering/native parity. `full_goal_status` remains NOT_ESTABLISHED.

Native recording of the same existing body previously occupied approximately5.8GiB compressed. This wrapper reserves at least12GiB free before one selected run, not per optimizer rollout. The chunk cache bounds native array RAM; audit retains scalar quantiles and loads one native chunk at a time. Actual future disk/RAM/runtime still need observation; source estimates are not a new measured run.

## Important measurement limits

The same full1s settle +4.8s actual teacher +9.6s ANN interval is retained. Native current-record D1 nonfoot must be exactly0, D2/D6 continuous causal1ms methods and sensitivities follow the declared original freeze, and no source or filter reset is allowed at takeover. Current D7 uses the conservative median >=3 legs at >=.1W as sufficient evidence; a missing proof remains UNKNOWN. Historical legacy UNKNOWN is preserved. Teacher/AUX references are the same actual run, not copied from the original walking motor.

The pinned native evaluator uses **signed initial-heading forward speed** as the slip denominator and reports historical XY-displacement denominator separately. This can be slightly stricter than the engineering optimizer's legacy XY metric; retain both and never substitute the latter to force PASS. All six actual cycles, strict nonfoot-zero and signed LEFTyaw>=.01 remain mandatory. Current-body-heading velocity projection is diagnostic only and does not change the frozen gates.

All input/source/XML/assets/compiled runtime/candidate/proposal/selection hashes are checked before and after. Mutable CURRENT_STATE/LIVE_PROGRESS pointers must not be sealed or added to execution sources. Preparation code never opens or modifies the parent's root status, optimizer, old scientific files or prior manifests.

## Later same-run rendering and visual review

Rendering requires the current NATIVE_AUDIT PASS and a separate explicit parent `--render-authorized` instruction. Example after native qualification:

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/render_selected.py --run-plan /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/RUN_PLAN.json --checkpoint SELECTED_CHECKPOINT_ABSOLUTE_PATH --sha256 SELECTED_SHA256 --run /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/selected_native/EVAL_left --native-audit /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/NATIVE_AUDIT.json --out /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T043404Z/native_preparation/videos/SELECTED_LEFT_V1 --camera-distance 10 --render-authorized
```

770bin-end frames cover actual.020 through15.400s, including50settle/240teacher/480ANN frames at30fps and1.667x nominal slow playback. This preserves the final physical endpoint. Camera is fixed in world coordinates; declared distance10..16 may be chosen in a fresh rendering output if an earlier output clips. Keep a failed render record, do not change native arrays or sealed renderer source. Every frame's fly silhouette must remain away from text/image edges. Six decoded stage frames and a montage are saved; the parent must actually open them and record review separately. No code-preparation result can certify that future video has already been viewed.

Even selected LEFT native PASS plus full engineering N/R is only bounded motor evidence. It does not establish arbitrary command switches/initial states/durations, neural speed control, actual food arrival/feeding, original connectome fidelity, independent SI or biological validity. Those remain NOT_ESTABLISHED.
