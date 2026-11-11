# Draft closed-loop food evaluator — CODE ONLY, NOT EXECUTED

Directory: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/food_evaluation/`.

This is a prepared evaluator contract. No physical jobs or training were run; no motor/behavior checkpoint is asserted qualified. `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/food_evaluation/PREREG_DRAFT.json` is not the final24-layout or PPO registration. Parent must select motor, freeze behavior and actual layouts before execution.

## Future CLI

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/food_evaluation/evaluate_food.py --motor-checkpoint /ABS/SELECTED_MOTOR.pt --motor-sha256 EXACT_MOTOR_SHA --behavior-checkpoint /ABS/FROZEN_BEHAVIOR.pt --behavior-sha256 EXACT_BEHAVIOR_SHA --qualification /ABS/QUALIFICATION.json --layout /ABS/FROZEN_LAYOUT.json --control normal --out /ABS/NEW_OUTPUT_DIRECTORY
```

`--control` is exactly normal/noodor/swap/replay. For replay also require `--replay-file /ABS/FROZEN_ORIGINAL_NEURAL_DECISIONS.npz --replay-sha256 EXACT_SHA`. A replay file must originate from this evaluator schema, including physical-prefix digest; arbitrary42-rate arrays are rejected. No future sensory shuffle exists.

All path arguments must be absolute. Outputs must be fresh. `--diagnostic` allows explicitly unqualified motor diagnostics, but acceptance is always false in that mode. Only qualified NORMAL can report acceptance; no-odor/swap/replay outcomes remain controls. Replay never counts as neural autonomy.

## Qualification producer schema

An explicit absolute JSON file must contain at least:

```json
{"all_conditions_eligible": true, "checkpoint": "/ABS/EXACT_SELECTED_MOTOR.pt", "checkpoint_sha256": "64_lowercase_hex"}
```

Both path and digest must match the CLI. This artifact is provided by the parent candidate-selection stage. The evaluator does not synthesize qualification from a loss or one passing condition. Motor loading uses `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/evaluate_candidate.py:load_checked`, including its fixed500/source/data requirements. Original actual compiled/runtime ancestry is separately validated through baseline sw.load_student and exact frozen base tensors.

## Future behavior trainer checkpoint schema

Torch checkpoint, format_version integer1, kind exact string `food_steering_behavior`, identity EXACTLY `SteeringBehavior().identity()` from `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/steering_behavior.py`, state_dict exact strict architecture, all finite tensors. Required config fields:

- source_hashes: nonempty mapping absolute filename→SHA256. MUST include actual `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/steering_behavior.py`. Include actual trainer, configuration, preregistration and other training sources.
- data_hashes: nonempty mapping absolute filename→SHA256 for frozen training inputs/layouts/manifests.

All listed sources/data are rehashed before deployment and included in run identity for final verification. Additional optimizer/RNG/curve fields are allowed and should be independently verified by future trainer; this loader checks inference identity rather than claiming training reproducibility. Current identity's `training` string is a fixed architecture disclaimer and is compared literally; no code changes are made to claim model qualification.

## Frozen single-layout schema

Exact keys, no silent defaults:

- layout_id: nonempty string.
- offset_forward, offset_lateral: finite declared model lengths, relative to ACTUAL head sampling midpoint at handover. Horizontal head +x defines forward, perpendicular +y defines left; source is then fixed in world XY.
- patch_radius, sigma, amplitude, rise_s, arrival_dwell_s: finite positive numbers. Arrival dwell <=9.6s. Initial sampling midpoint must start outside patch.
- noise_std: finite >=0.
- onset_elapsed_s: finite [0,9.6), measured from actual withdrawal timestamp. Source onset is withdrawal+this value.
- seed: explicit nonnegative integer.

A layout contains one nondepleting Gaussian engineering stimulus. Finite amplitude and explicit patch do NOT establish a finite ingestion reservoir. No ingestion or satiety is modeled. Physical units remain engineering/NOT_ESTABLISHED SI. No example layout is published as a reachable or final evaluation layout.

## Control and evidence contracts

Actual1s settle, original4.8s teacher preheat,9.6s autonomous. Behavior/motor states initialize at physical reset. Behavior state remains zero during preheat, motor receives real physical observations and neutral commands; no state resets at handover. FoodObservation is constructed at withdrawal and returns causal stored response z(t), integrating only for next decision. Source coordinates/distances/time never enter policy inputs. Turn-only behavior enforces forward output0. Actual policy action flows to unchanged42-rate motor/Core.

No-odor supplies zeros to behavior while preserving the same source for geometric scoring; swap exchanges only current local receptor channels. Replay still evaluates neural outputs for diagnostics but applies frozen original actions with its own source code `food_replay=max(original SOURCE)+1`. All960 preheat observations/actions/motor hidden before/after must be bitexact to replay file. Additionally, `physical_prefix_sha256` proves all5800 actual1ms settle+teacher trace rows bitexact across time/qpos/qvel/target/actualctrl/action/control_source. Frozen file must have exactly2880 finite rows and bounded applied actions. Autonomous replay is never attributed to policy.

Outputs: IDENTITY.json, trace_1ms.npz (15400 full physical endpoints), neural_decisions.npz (2880 before/after motor192 and behavior32 states, actual238obs, raw/control receptors, commands, predicted/appliedactions andsource), geometry_1ms.npz (9600 actual autonomous sensor midpoint/distances/source diagnostics), RESULT.json. Saved neural file contains scalar physical_prefix_sha256 for future matched replay. Full1ms traces retain settle and preheat. Geometry is scoring-only; patch+dwell arrival uses actual sampling midpoint every1ms and excludes pre-onset occupancy.

Original whole-window all5 engineering gates and all6 sensed cycles remain unchanged. Acceptance additionally requires arrival, qualified NORMAL, nondiagnostic, exact actual phase/source counts, continuous hidden chain, direct action outputs, all71 flymesh alpha==1, zero external forces before/after every tick, and verified immutable sources/compiled runtime. Source/environment code performs no pose writes or physics changes. Full native D1–D8 audit and actual same-run video remain separate work.

On exceptions, finalization attempts partial trace/decisions/geometry save and independently verifies sources/runtime even if evidence save fails; FAILURE.json records failure and forbids acceptance. Files remain available for diagnosis. This draft has not been physically executed, so physical failure handling and runtime compatibility are prepared code, not certified outcomes.

## Offline checks performed

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/food_evaluation/check_contract.py` ran with the stated interpreter and -B, exit0,33checks PASS. Report: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/food_evaluation/OFFLINE_CHECKS.json`.

Checks cover layout parser and actual-head frame transform, negative/NaN/timing bounds, qualification exact binding, causal arrival dwell/no pre-onset credit, physical prefix hashing including final interval, replay argument requirements, strict behavior checkpoint roundtrip and wrong kind/digest/data rejection, structuralzero forward and syntax. Temporary fake checkpoints were automatically removed. No MuJoCo/physics/student_wave import, no physical integration, no training updates. Superseded earlier30-check report was removed and replaced by final33-check result.
