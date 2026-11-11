# Read-only motor/odor integration review

Scope: requested six new files in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/`: `conditional_motor.py`, `train_motor.py`, `evaluate_motor.py`, `run_motor_stage.py`, `odor_loop.py`, and `MOTOR_TRAINING_DESIGN.json`. Supporting baseline, teacher, sensor implementation and saved teacher trajectories were read to resolve contracts. No captured sources modified. No training or physical integration run. Initial review created this report only. A subsequent explicit parent request authorized adding `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/verify_training.py` and running its offline adapter100 audit, saved as `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/VERIFY_ADAPTER_100.json`.

## Findings and current disposition before behavior training

- Timing issue below: parent corrected the uncaptured odor integration source using pending_response; reviewer reread current lines34–45 and confirmed actions now receive the stored current-time state. Parent records amendment in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/ODOR_TIMING_AMENDMENT.json`; new integration check running at last notification. The original finding below describes the initially reviewed revision.
- Actual eligibility admitted only full-left/full-right plus neutral retention. Half-speed candidates were rejected for cycle completion. Therefore the forward command input has NO training coverage beyond zero, not merely insufficient validation; parent explicitly records this limitation. Do not unlock that channel for behavior training.
- Saved optimizer verification: the newly authorized independent verifier passed adapter100, 34 checks, exit0, including exact 2880-row neutral means/hidden, complete state reload, actual Adam tensors/counters, provenance and RNG restore. Finetune500 verification is prepared but not run by this reviewer, as instructed; parent can run the same CLI when500 is available. Final500 still cannot prove exact numeric moments at handover without a saved handover artifact. No captured source changes were made by this reviewer.

### 1. Receptor endpoint is consumed one control interval early (integration timing bug)

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor_loop.py:35–45` samples at current physical time and current sensor positions with dt=.005, immediately returns the result for policy consumption, and advances its internal sensory clock. `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/food_field.py:100–102,122–147` explicitly defines that result as the END-of-interval receptor response. Consequently action at t would use z(t+.005), while logs label the associated physics pose/time t. Clock contiguity checks do not detect this state/time mismatch.

This is a discretization/timestamp defect, not evidence of target-coordinate leakage: concentration is sampled at t, not at a future body pose. Fix in a new source revision after running identity-protected jobs settle: return stored z(t) for the current action, then advance the receptor over the completed/held interval; alternatively consume the previous interval's held concentration when updating z at t. Make initial state, source onset, and first action timestamp explicit in an offline scheduling check.

### 2. Behavior can emit unvalidated motor commands (coverage gap, not a BC implementation bug)

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/conditional_motor.py:108` emits two tanh values in [-1,1]. Training command semantics in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/MOTOR_TRAINING_DESIGN.json:7` are [speed_scale-1, requested_yaw_rate/.008]. Actual admitted data contain neutral (0,0) and full-speed paired turns only; half-speed candidates were rejected for cycle completion. There are no positive forward-modulation examples, stopped examples (-1,0), or independently varied speed and turn schedules.

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/run_motor_stage.py:18–24` evaluates only forward=0 and turn=0,+1,-1. It never evaluates half-speed command response despite the design's explicit half-speed selection clause. Thus a passing stage report certifies these three fixed commands only, not a usable 2D command square, stopping, or switching.

Before behavior learning, explicitly declare its allowed command region based on measured motor response. Either keep forward at neutral for the first steering-only experiment and qualify intermediate/time-varying turn commands, or add preregistered motor coverage for the additional outputs before allowing them. Positive forward modulation must not be described as a trained speed increase. No additional physical runs were executed by this review.

### 3. Deployment loader does not enforce complete checkpoint training source/data identity (identity verification defect)

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/conditional_motor.py:111–135` stores/verifies only `conditional_motor.py` in top-level source_sha256. The checkpoint config contains training source hashes and data hashes, but load_motor ignores both. `evaluate_motor.py:19–31` checks interface-source hashes and captures current evaluation sources; this does not validate the training provenance embedded in the loaded checkpoint. `train_motor.py:78–85` validates parent's data hashes, but likewise does not verify parent's config.source_hashes before accepting continuation.

Current training does recheck its own sources/data at each checkpoint (train_motor.py:113–114), and the original physical baseline is independently loaded/provenance-checked in evaluation. No actual source drift or wrong checkpoint was observed. The actionable defect is that the loader accepts a checkpoint whose recorded training sources/data no longer match disk, contrary to the advertised source/identity exactness. New revision should verify embedded config source/data digests before inference/continuation, or bind immutable training manifests and verify those identities. Preserve original manifests rather than rewriting historical hashes.

### 4. Production optimizer reload assertion checks groups only (verification gap)

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/train_motor.py:115–120` labels save/reload exactness after testing model tensors and optimizer param_groups only. It does not compare Adam step counters, exp_avg, exp_avg_sq, RNG, config or complete curve. The separate interface unit check tests an adapter-only optimizer once; it does not establish exactness for the actual two-group finetune checkpoint.

No optimizer update or serialization bug was found: adapter moments are loaded before changing lr, base parameters are added in a second group, and base moments appropriately start fresh. Strengthen independent verification of the actual saved checkpoints before release/continuation, without touching active training sources. In particular verify all adapter/base optimizer tensors and counters, group ordering/hyperparameters, saved RNG and curve. The present assertion should not be interpreted as having already done that.

## Saved trajectory diagnosis: COM y sign does not establish wrong food steering

Source paths:

- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/left_full/TRACE_1MS.npz`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/right_full/TRACE_1MS.npz`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/left_half/TRACE_1MS.npz`
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/right_half/TRACE_1MS.npz`
- Sensor offset source: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/odor/HEAD_GEOMETRY_AUDIT.json`.

Offline calculation used saved qpos row5799 (handover boundary) through row15399, baseline Core model construction and MuJoCo mj_kinematics ONLY. No mj_step or integration, no trace modified. Head actual pose was reconstructed; sampling midpoint equals Head position + Head rotation*[.5713176560335599,0,0]. This is the declared sensory/mouth proxy, not a biological mouth claim.

| Teacher | COM delta xy | Head delta xy | Sampling midpoint delta xy | Head yaw delta rad |
|---|---|---|---|---|
| left_full | (.687553380, -.003471172) | (.684712204, +.046485122) | (.683000852, +.081411670) | +.061283799 |
| right_full | (.692539340, +.066538220) | (.690400018, +.002339294) | (.689460547, -.042445355) | -.078512302 |
| left_half | (.359274883, -.015405836) | (.358567289, +.008643443) | (.357964552, +.026346994) | +.031040729 |
| right_half | (.361311206, +.032648534) | (.361099891, -.003265007) | (.361028257, -.027417030) | -.042324313 |

Full-speed sampling midpoint delta-y at 2.4/4.8/7.2/9.6 seconds after handover:

- left_full: +.014273969, +.034894057, +.055476944, +.081411670.
- right_full: -.006884022, -.014034139, -.028526253, -.042445355.

Half-speed right has a small initial positive sampling-midpoint y (+.000927 at2.4s, +.001147 at4.8s), then negative (-.016301 at7.2s, -.027417 at9.6s). These are actual trajectory transients, not gate failures by themselves.

The COM and anterior sampling point have different displacement under body rotation. These trajectories therefore do not show the sensory midpoint steering opposite to the requested side. Correct-sign yaw alone remains insufficient to certify food approach: endpoint layout geometry, actual sampling-point distance reduction/arrival, switching and available turn range must be checked in the behavior stage. Do not infer either failure or success from COM lateral sign alone.

## Items checked with no concrete bug found

- ConditionalMotor's recurrent update and output match the baseline equations; zero command adapter preserves baseline by construction, consistent with the parent's bitexact neutral physical replay.
- Full 2880-step BPTT begins at zero state and includes the actual 960-step preheat observations. No detach, hidden reset at handover, or shuffled recurrent chunks. Warmup targets are distilled from the baseline recurrent sequence; this is explicitly declared retention, not heldout validation.
- Episode losses accumulate gradients with exact equal episode weights before a single optimizer update. Recurrent graphs remain separate between episodes. Frozen baseline weights during adapter training are still differentiable with respect to command input, so adapter learning is not accidentally blocked.
- Physical handover snapshots retain body, actuator history, previous action and recurrent state; only the read-only cycle tracker is reset. Warmup teacher is not queried in autonomous decisions. Remaining time is critic-only in the baseline observations helper, not fed to the motor actor.
- Yaw array indexing is correct: index4799 is the final preheat endpoint; final minus that angle describes the autonomous window. Original fixed-frame forward gates are explicitly labeled; small turn angles do not establish a demonstrated metric bug. Engineering endpoint gates are not represented as full native certification.
- Teacher training commands are ramped, while physical evaluation applies fixed commands abruptly at handover. This is a disclosed/train-test distribution difference, not a wrong recurrent implementation. Passing abrupt constant tests would still not establish safe repeated switching.
- No food coordinates, target direction or distance enter FoodNeuralPolicy's actor path. Geometry diagnostics are separate. Current odor behavior is a zero-output initialized skeleton, not a trained food controller.

Do not edit active captured files to address findings. Use results from current jobs as diagnostics; apply corrections/qualification under new source identities before behavior training.
