# Original-two-only matched control: fixed training completed, physics not run here

The preregistered exact500→1500 continuation completed. Managed training job bash-4 exited0 and was collected. The full preflight job bash-3 exited0 and was collected. No physical evaluator, optimization beyond the fixed1000 additional updates, or extra checkpoint selection was executed.

Final checkpoint:

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt`

SHA-256: `c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb`

Actual1000 additional update time793.6525s; total including final independent verification796.6461s. Parent500 SHA `040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3` was restored exactly, with all8 Adam states/counter500, actual moments, model, RNG and complete0..500 curve retained. Only declared lr.001→.0003 changed, matching primary. `TRAIN/RESUME_STATE.pt` SHA `a4da66806cfec7307694833f8707076349c06149232d0c9096e4847e4f28bb1f` stores the actual initial state beforeupdates; all resume checks are true.

Only qualified original `left_full` and `right_full` actual2880 sequences were read for optimization. Each postprime1920 actionMSE mean has weight1/2. No corrective label/file is loaded, no speed data are used, and the parent’s inherited source/data ancestry also has no corrective sample paths. Config records loss_episode_count2 and corrective_loss_episodes0. Existing failure/safetyFALSE records were untouched.

Full preflight22checks passed: meansmaxerror2.77556e-16, residual history2.44249e-15, all8gradientmax9.21572e-19; original neural class/alpha/bound.5/normgate/bothstates unchanged. Additional7-check full-history audit verifies all2880 base states BITEXACT, residual/means within1e-12 and primary/control Scan+make AST equality with explicit source hashes. Same original parameter objects are shared by TorchScript. Exactly one temporary optimizer update was used by preflight, discarded; no persistent extra learned model was retained.

| Global step | Left actual teacher action MSE | Right actual teacher action MSE | Purpose |
|---|---:|---:|---|
| parent500 | .000131925913525 | .000123232648223 | Exact frozen parent |
| 1000 | .000115666937303 | .000108719554265 | Saved-state integrity only |
| 1400 | .000108430630586 | .000103812521258 | Training log only, never selected |
| 1500 | .000129452592637 | .000113028267341 | Sole preregistered final candidate |

The final update rebounded relative to earlier logged fitting. That result is retained; earlier1000/1400 is not substituted. Fitting alone is not physical response evidence. Final1500 raw gradientnorm.00206935 is finite. No architecture, bound, input normalization or learning-rate changes were made during the run.

Fresh independent process audits at1000 and1500 each passed20checks: all weights and8 Adam tensors/counters, full curve and exactparentprefix, actual resume state, RNG roundtrip, frozenbase and all2880 neutral means/actions/basehidden, nonzero-command saved inference and checkpoint immutability. All27 captured training source/data identities were rechecked after completion and matched. Physical integration count is0.

Public immutable API: load this directory’s `model_io.py` by explicit path, then `load_checked(checkpoint_absolute_path, exact_sha, require_final=True)`. Return is the original legacy `ResidualMotor` and payload. `initial_state(B)` returns `(base128,residual64)`; `step(obsB238,commandB2,state)` returns meanB42 and both new states. Apply tanh once and preserve both memories. Physical validation is delegated to parent’s separately owned `two_only_evaluation/` under explicit shared resource coordination. No physical launch authorization was requested or exercised here.

Duplicate wrapper preparation was stopped when parent clarified ownership. Two never-executed drafts were confirmed absent from the running source/data identity and actually removed. No captured source or scientific artifact was deleted. All remaining own top-level Python/JSON files are frozen for parent evaluation capture. Future notes may be Markdown only while parent actual runs execute.

This is a matched fixed-budget training control, not a claim of walking, turning, food behavior, body impossibility or native/biological certification.
