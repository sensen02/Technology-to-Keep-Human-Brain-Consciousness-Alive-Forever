# Fixed positive-turn neural readout: training complete, physical qualification pending

Only this `train/` directory was modified. The immutable root preregistration was read and retained unchanged. No physical simulation, food behavior training or old-source/model modification occurred.

The actual frozen parent is `/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt`, SHA `c6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb`. Its 194,878 parameters and both 128+64 recurrent states remain unchanged. Only zero-initialized Linear(64,42), 2,730 parameters, was trained. Its contribution is `relu(turn_command)*.25*tanh(readout(current_residual_hidden))`; forward command is restricted to zero.

The dataset is only the original independently qualified actual left teacher sequence, all 2,880 observations/commands/actions. The complete causal parent sequence generates mean and both state caches; loss is the post-prime 1,920-row bounded-action MSE. No failed/corrective data were used. Fresh Adam lr .003 updates only the two new parameter tensors for exactly 2,000 updates.

## Completed evidence

- `PREFLIGHT.json`: all initial parent neutral/left/right 2,880 outputs and both state histories exact; actual temporary unit update has finite nonzero gradients in both new tensors and no parent gradients/changes; neutral/right remain exact afterward; full cached output and new gradients match production recurrence within 1e-12; target-position inhibitory Jacobian unchanged; temporary serialization checks passed. Job bash-9 exit0 collected.
- `TRAIN/checkpoint_001000.pt`: SHA `fbbda38ece240404b8404cda807c27b77179d4ecccf59e8c9ffbd8973314744c`, never physically selected.
- `TRAIN/checkpoint_002000.pt`: SHA `eeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df`, fixed final candidate. Training bash-10 exit0 collected, about7.15seconds CPU1. Final left teacher action MSE `8.720976209497367e-5`.
- `TRAIN/INDEPENDENT_FINAL_2000.json`: separate 19-check saved-state audit, bash-11 exit0 collected.
- `TRAIN/FINAL_STATE_AUDIT.json`: 25-check final independent state/inference audit, bash-12 exit0 collected. Actual Adam two-state coverage/order/counters2000/moments/shape/hyperparameters/reload and saved RNG restore checked; every frozen parent tensor exactly matches the actual1500 parent; complete recorded curve1..2000; actual neutral/right 2,880 output/action/both-state invariance; left production/cache and independent inference clone checked. Zero physical and optimizer steps in audits.

## Explicit limitations and correction

`PARENT_METADATA_CORRECTION.json` binds the unchanged final checkpoint and records a config metadata error: `config.parent_checkpoint` incorrectly inherits the older500 ancestor path, while `config.parent_sha256`, authoritative `identity.parent_checkpoint/parent_sha256`, strict actual constructor and every frozen tensor correctly bind the actual1500 parent. Original checkpoint/training source/hash were not rewritten.

The recorded training curve starts1, not0. Initial verification at save checked model/optimizer reload but did not preserve a separate full live-before-save RNG/curve equality report; later independent saved-state consistency is established, not a fabricated live-state proof. Constructor reload at step1000 consumed global RNG; updates use no random operations, so no stochastic continuation claim is made. A subsequent attempt to invoke the final-only audit on intermediate1000 was rejected by its strict final2000 loader (bash-13 exit1 before analysis, no output/model updates/physics); this is retained as a verification-scope mistake and is not a model failure or intermediate selection.

All top-level training Python/JSON files are frozen after parent notification. Parent owns the actual final neutral/left/right full-body tests and independent physical audit. Same-observation nonpositive-command invariance does not certify switching after positive-command physical history. Teacher-fit MSE is not physical or food success.
