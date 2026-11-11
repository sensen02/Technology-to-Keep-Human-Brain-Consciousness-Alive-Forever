# Controlled slow-residual continuation

All outputs are confined to `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train/`. Old source, parent weights and failed data flags are immutable.

`model_io.py` exposes `load_checked(checkpoint, sha, require_final=True) -> (ResidualMotor, payload)`. The returned original model uses `initial_state(B) -> (base128,residual64)`, `step(obs[B,238], command[B,2], state) -> (mean[B,42], state)`, and one final `tanh` for actions. Only actual global1500 is allowed for physical evaluation by default. Passing `require_final=False` is for saved-state validation of1000/1500, not physical selection.

The new payload fields are `format_version`, `identity`, `state_dict`, `optimizer`, `config`, `curve`, `rng`. `config.source_hashes` and `config.data_hashes` are verified during load; original architecture identity and frozen base weights are also checked. `config.physical_eval_only_step` is1500. The final expected artifact is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train/TRAIN/checkpoint_001500.pt`.

Actual parent inspection found EIGHT trainable tensors/Adam states (encoder2,GRU4,head2), each counter500, not six. `PARENT_000500.pt` is an exact local copy; actual original parent SHA remains `040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3`.

`PREREG.json` fixes500→1500, lr.001→.0003 and the addition of two authenticated failed-observation corrective-label trajectories. These are two declared interventions in one engineering candidate, not a single-factor causal comparison. Two original2880-row turn teachers and two corrective1920-row trajectories are equally weighted by each trajectory's own postprime action MSE. All960 actual priming observations remain in the recurrent history. Old corrective `dataset_safety_admitted=false` and slip failures remain false/failed; exact teacher-label provenance is not autonomous success, causal recovery, or oracle optimality.

`engine.py` scripts the original shared encoder/GRU/head parameters, precomputes only pointwise encoder/decoder operations, and scans the full residual recurrence with valid padding. Padding freezes state only after a shorter actual trajectory ends. It is not used in deployment. `PREFLIGHT.json` verifies every valid production mean, both state histories and all eight adapter gradients against the original sequential implementation: maximum errors2.71e-16/2.44e-15/5.96e-19, all below1e-12. Parent/model/Adam and neutral-before/after-update checks passed; one temporary optimizer unit update was discarded.

`check_sealed_data.py` independently confirms the known `46e44c...` corrective-data manifest and all29 artifact hashes/sizes. This additional audit ran after launch; it did not modify captured training sources or inputs. Individual input/audit hashes were already bound by the training preflight.

`continue_train.py` persists exact `RESUME_STATE.pt`, `RESUME_CHECKS.json`, the full inherited0…500 curve, source/data identity and per-update logs before proceeding. It saves only1000 and1500; each is checked live-to-saved and in a fresh `verify_saved.py` process for weights, Adam counters/moments, RNG, full curve and inference. Evaluation belongs to the parent; this trainer performs zero physical integration.

CLI:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train/continue_train.py --out /run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train/TRAIN
```

The managed job is bash-236. Fixed1000 additional updates only; no budget extension, architecture/bound change or intermediate physical checkpoint selection. Training fit never establishes physical turning, food approach or biological validity.
