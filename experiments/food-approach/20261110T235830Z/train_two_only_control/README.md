# Original-two-only budget-matched continuation control

This is the separately preregistered control in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/TWO_TRAJECTORY_CONTROL_AMENDMENT.json`. Only files below this directory were created by this task. Existing primary and legacy sources were not edited.

The exact parent500 is SHA `040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3`. Its original ResidualMotor class, frozen151636 base,43242 slow-GRU64 adapter, bound.5, alpha, command norm gate and explicit (base128,residual64) state are unchanged. Original8 Adam states and global500 counters, moments, model, RNG and full0..500 curve are restored; the declared lr change .001→.0003 matches the primary experiment. Training ends at fixed1500 (1000 additional updates), with integrity saves1000/1500 and only final1500 physically eligible.

`data.py` reads ONLY actual qualified `left_full`/`right_full` original teacher X238/Y42/command2 sequences. Both are2880 actual decisions with960 neutral prime+1920 post. No corrective file or label is read. The loss is the mean of the two postprime bounded-action mean losses, each trajectory weight1/2; actual causal base history is cached through the complete sequence. Batch2 vs primary batch4 rounding is explicitly bounded by preflight, not a change in neural architecture.

`PREFLIGHT.json` checks full2880 production means and residual histories, all8 gradients, shared original parameter objects, all-valid batch2, original primary Scan/make AST, neutral invariance before/after one temporary optimizer unit update, and frozen base. `FULL_HISTORY_CHECKS.json` independently checks ALL2880 base hidden states exactly plus residual histories/means and binds primary/own engine hashes. No persistent training artifact or physics was produced by these tests.

Public API, stable for parent-owned actual evaluation:

- Module: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/model_io.py`.
- `load_checked(checkpoint_absolute_path, expected_sha256, require_final=True)` returns `(original_legacy_ResidualMotor, payload)`.
- Final checkpoint: `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt`.
- `initial_state(B)` returns explicit `(B,128),(B,64)`. `step(obs_B238,command_B2,state)` returns `(pre_tanh_mean_B42,new_state)`. Apply `tanh` once, preserve BOTH states throughout physical priming/withdrawal/command changes.
- Loader validates own new format/kind, original architecture identity/frozen base, actual source/data hashes and exactly2 loss episodes. It rejects intermediate1000 by default; `require_final=False` exists for saved-state integrity verification only.

Training is a managed CPU1 job. `TRAIN/RESUME_STATE.pt` and `RESUME_CHECKS.json` bind the actual before-update state; `VERIFY_001000.json`/`VERIFY_001500.json` are fresh-process checks of weights, actual8 Adam tensors/counters, RNG, fullcurve, parentprefix, neutral2880 and nonzero inference. Logs and final RESULT retain all source/data identities. Loss decrease does not establish physical walking or turning.

No physical evaluation is launched by this task. Parent coordinates the separate wrapper in `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/` and maximum2 actual processes. Two unexecuted duplicate wrapper drafts initially created here were verified absent from the running training identity and actually removed immediately when parent clarified ownership. No captured/scientific source was removed or changed.

All earlier failed trajectories, corrective safetyFALSE flags and original physical gates are untouched. This control does not claim food behavior, independent biological validity, or full-native certification.
