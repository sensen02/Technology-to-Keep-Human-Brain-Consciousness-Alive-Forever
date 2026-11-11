# Residual fixed-command profile adapter — prepared, no physical execution

Directory: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_profiles/`.

This new entrypoint reuses `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/command_profiles/evaluate_profile.py` without changing its file, scoring or physics. Parent must finish constant-condition candidate qualification before launching any profile. The offline checkpoint used below is not a qualification claim.

## Launch API after parent qualification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_profiles/evaluate_residual_profile.py --checkpoint /ABS/PARENT_SELECTED_RESIDUAL500.pt --sha256 EXACT_LOWERCASE_SHA256 --profile neutral --out /ABS/NEW_OUTPUT_DIRECTORY
```

Exactly five profiles: neutral, halfleft, halfright, left_to_right, right_to_left. Forward always0. Halfleft/right denote turn±.5, not reduced forward speed. Switching changes sign after exactly960 autonomous decisions (4.8s). All original whole-window all5 engineering gates and all6cycles apply. Intermediate whole signed yaw>=.005rad; switching each4.8s signed yaw>=.002rad including transient. Neutral has no directional yaw criterion. All five profile outcomes must be reported; no best-profile substitution.

## Explicit adapter and capture review

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_profiles/flat_motor.py` exposes FlatResidualMotor(motor): initial_state(B) concatenates128baseline+64residual; step(obs238,command2,state192) splits [:128]/[128:], calls original residual step, and concatenates outputs. No math, ordering, hidden resets or command semantics change. sequence() preserves explicit continuous state.

New entrypoint loads the original evaluator under a distinct module name and explicitly replaces only its in-memory pin_imports hook. Original baseline pinning still runs. The hook returns an explicit loader proxy calling `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/evaluate_candidate.py:load_checked` with selected checkpoint/digest, then returns the flat-state wrapper. Original strict500/source/data/base checks remain intact. The original conditional loader is not used for residual weights.

The capture proxy records the adapter declaration and NEW PREREG content in run config and adds hashes for newentrypoint, flatwrapper, NEW prereg, original profile prereg, root REVIEW.md and strict residual loader. Imported dependencies and actual original physical/runtime identity are captured by unchanged baseline identity code. Checkpoint/training hashes are rechecked by original profile evaluator's final checked-loader call. Do not edit any captured source/report while profiles run.

The rest of original evaluate(args) is unchanged: actual1s settle+4.8s originalteacher neutralpriming+9.6s neural outputs, no teacher after withdrawal; continuity snapshots accept the representation-only192tensor; original source/trace counts, applied action checks, all71 geometry audit and scoring remain as defined by original evaluator. No scripted gait, pose writes or support introduced. Note original profile evaluator records actual head/sensor geometry but does not independently certify opacity; actual model/runtime ancestry stays unchanged, and separate food evaluator checks all71alpha==1 explicitly. No extra opacity gate is inserted into the reused profiles.

## Offline validation completed

Used saved residual500 checkpoint `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/TRAIN_RESIDUAL_500/checkpoint_000500.pt`, SHA256 `040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3`. Strict loader accepted its recorded identities.

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_profiles/check_flat.py` ran using stated interpreter with -B, exit0,15checks PASS. For BOTH actual saved2880-row left_full/right_full teacher observation+command datasets, all means and all192hidden values are bitexact against original tuple motor for fullsequence AND every individual step. Split/resume at960preheat boundary remains exact, wrong191state rejected, profiles unchanged, syntaxchecked. No MuJoCo/physics/student_wave import, no physical model construction, no physical integration, no optimizer updates.

Report: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_profiles/OFFLINE_CHECKS.json`.

Limitation: tests prove state-representation parity on saved observations, not physical switch robustness, food approach, full native certification or candidate qualification. Parent alone selects when to launch fixed profiles.
